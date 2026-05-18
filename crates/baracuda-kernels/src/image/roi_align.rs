//! `roi_align` FW plan — Category T.
//!
//! Extract a fixed-size `[pooled_h, pooled_w]` feature from each
//! variable-size RoI by bilinearly sampling the input plane.
//! Trailblazer config matches the PyTorch convention:
//!   - `sampling_ratio == 0` → adaptive (one sample per output cell
//!     per `ceil(bin_h)` × `ceil(bin_w)` grid).
//!   - `aligned == false` → pre-0.6 PyTorch coord convention
//!     (no `0.5` offset before scaling).
//!
//! Layout: NCHW. RoI rows are `(batch_idx, x1, y1, x2, y2)` in INPUT-
//! image pixel coords (will be scaled by `spatial_scale`).
//!
//! Trailblazer dtype coverage: `f32, f64`.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, ImageKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::map_status;

/// Descriptor for `roi_align`.
#[derive(Copy, Clone, Debug)]
pub struct RoiAlignDescriptor {
    /// Batch.
    pub n: i32,
    /// Channels.
    pub c: i32,
    /// Input height.
    pub h: i32,
    /// Input width.
    pub w: i32,
    /// Number of RoIs.
    pub num_rois: i32,
    /// Output pooled height per RoI.
    pub pooled_h: i32,
    /// Output pooled width per RoI.
    pub pooled_w: i32,
    /// RoI coord scale: `roi_pixel = roi_input * spatial_scale`.
    pub spatial_scale: f32,
    /// `0` = adaptive (PyTorch convention).
    pub sampling_ratio: i32,
    /// Use the `aligned=true` half-pixel offset convention.
    pub aligned: bool,
    /// Value element type.
    pub element: ElementKind,
}

/// Args bundle for `roi_align`.
pub struct RoiAlignArgs<'a, T: Element> {
    /// Input `[N, C, H, W]`.
    pub input: TensorRef<'a, T, 4>,
    /// RoIs `[num_rois, 5]` — (batch_idx, x1, y1, x2, y2) in INPUT-pixel coords.
    pub rois: TensorRef<'a, T, 2>,
    /// Output `[num_rois, C, pooled_h, pooled_w]`.
    pub output: TensorMut<'a, T, 4>,
}

/// `roi_align` plan.
///
/// Extract a fixed-size `[pooled_h, pooled_w]` feature from each
/// variable-size RoI by bilinearly sampling the input plane
/// (torchvision `roi_align`).
///
/// **When to use**: forward RoIAlign for detection / instance-
/// segmentation networks. Pair with
/// [`RoiAlignBackwardPlan`](crate::RoiAlignBackwardPlan).
///
/// **Dtypes**: `{f32, f64}` for both input and RoI coords.
///
/// **Shape limits**: rank-4 NCHW input; RoIs `[num_rois, 5]`
/// (rows are `(batch_idx, x1, y1, x2, y2)` in INPUT-image pixel
/// coords, scaled by `spatial_scale`); output
/// `[num_rois, C, pooled_h, pooled_w]`.
///
/// **Config**: `sampling_ratio == 0` → adaptive (one sample per
/// `ceil(bin_h) × ceil(bin_w)` grid). `aligned == false` is the
/// pre-0.6 PyTorch coord convention (no `0.5` offset before scaling).
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable.
pub struct RoiAlignPlan<T: Element> {
    desc: RoiAlignDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> RoiAlignPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &RoiAlignDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::RoiAlignPlan: descriptor element != T",
            ));
        }
        if desc.n < 0
            || desc.c < 0
            || desc.h < 0
            || desc.w < 0
            || desc.num_rois < 0
            || desc.pooled_h < 0
            || desc.pooled_w < 0
            || desc.sampling_ratio < 0
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RoiAlignPlan: extents / sampling_ratio must be non-negative",
            ));
        }
        if !matches!(T::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::RoiAlignPlan: only `f32`, `f64` wired",
            ));
        }
        let precision_guarantee = PrecisionGuarantee {
            math_precision: if T::KIND == ElementKind::F64 {
                MathPrecision::F64
            } else {
                MathPrecision::F32
            },
            accumulator: T::KIND,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Image,
            op: ImageKind::RoiAlign as u16,
            element: T::KIND,
            aux_element: None,
            layout: None,
            epilogue: None,
            arch: ArchSku::Sm80,
            backend: BackendKind::Bespoke,
            precision_guarantee,
        };
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &RoiAlignArgs<'_, T>) -> Result<()> {
        if args.input.shape != [self.desc.n, self.desc.c, self.desc.h, self.desc.w] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RoiAlignPlan: input shape mismatch",
            ));
        }
        if args.rois.shape != [self.desc.num_rois, 5] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RoiAlignPlan: rois must be [num_rois, 5]",
            ));
        }
        if args.output.shape
            != [self.desc.num_rois, self.desc.c, self.desc.pooled_h, self.desc.pooled_w]
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RoiAlignPlan: output shape mismatch",
            ));
        }
        Ok(())
    }

    /// Workspace (zero).
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Identity.
    #[inline]
    pub fn sku(&self) -> KernelSku {
        self.sku
    }

    /// Numerical guarantees.
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee
    }

    /// Launch.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: RoiAlignArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if args.output.numel() == 0 {
            return Ok(());
        }
        let input_ptr = args.input.data.as_raw().0 as *const c_void;
        let rois_ptr = args.rois.data.as_raw().0 as *const c_void;
        let out_ptr = args.output.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let aligned = if self.desc.aligned { 1 } else { 0 };
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_roi_align_f32_run(
                    self.desc.n, self.desc.c, self.desc.h, self.desc.w,
                    self.desc.num_rois, self.desc.pooled_h, self.desc.pooled_w,
                    self.desc.spatial_scale, self.desc.sampling_ratio, aligned,
                    input_ptr, rois_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_roi_align_f64_run(
                    self.desc.n, self.desc.c, self.desc.h, self.desc.w,
                    self.desc.num_rois, self.desc.pooled_h, self.desc.pooled_w,
                    self.desc.spatial_scale, self.desc.sampling_ratio, aligned,
                    input_ptr, rois_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::RoiAlignPlan::run reached unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
