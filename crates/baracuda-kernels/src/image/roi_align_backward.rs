//! `roi_align` BW plan — Category T.
//!
//! Adjoint of [`crate::image::RoiAlignPlan`]: bilinear-weighted
//! atomicAdd of each output cell's gradient back into `dinput`.
//! Caller MUST pre-zero `dinput`.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, ImageKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::map_status;

/// Descriptor for `roi_align_backward`.
#[derive(Copy, Clone, Debug)]
pub struct RoiAlignBackwardDescriptor {
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
    /// Spatial scale (must match FW).
    pub spatial_scale: f32,
    /// Sampling ratio (must match FW).
    pub sampling_ratio: i32,
    /// Aligned flag (must match FW).
    pub aligned: bool,
    /// Value element type.
    pub element: ElementKind,
}

/// Args bundle for `roi_align_backward`.
pub struct RoiAlignBackwardArgs<'a, T: Element> {
    /// Upstream gradient `[num_rois, C, pooled_h, pooled_w]`.
    pub dout: TensorRef<'a, T, 4>,
    /// Saved FW rois `[num_rois, 5]`.
    pub rois: TensorRef<'a, T, 2>,
    /// Gradient w.r.t. input `[N, C, H, W]`. Caller pre-zeros.
    pub dinput: TensorMut<'a, T, 4>,
}

/// `roi_align_backward` plan.
///
/// Adjoint of [`crate::RoiAlignPlan`]: scatter `dout` into `dinput`
/// via the 4 bilinear-sample weights per RoI cell (atomicAdd).
///
/// **When to use**: BW for [`RoiAlignPlan`](crate::RoiAlignPlan).
/// Caller retains FW `rois`.
///
/// **Dtypes**: `{f32, f64}`.
///
/// **Shape limits**: as for FW (rank-4 NCHW, RoIs `[num_rois, 5]`).
///
/// **Workspace**: none. Caller MUST zero `dinput`.
///
/// **Precision guarantee**: **non-deterministic** (atomicAdd).
pub struct RoiAlignBackwardPlan<T: Element> {
    desc: RoiAlignBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> RoiAlignBackwardPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &RoiAlignBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::RoiAlignBackwardPlan: descriptor element != T",
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
                "baracuda-kernels::RoiAlignBackwardPlan: extents must be non-negative",
            ));
        }
        if !matches!(T::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::RoiAlignBackwardPlan: only `f32`, `f64` wired",
            ));
        }
        let precision_guarantee = PrecisionGuarantee {
            math_precision: if T::KIND == ElementKind::F64 {
                MathPrecision::F64
            } else {
                MathPrecision::F32
            },
            accumulator: T::KIND,
            bit_stable_on_same_hardware: false,
            deterministic: false,
        };
        let sku = KernelSku {
            category: OpCategory::Image,
            op: ImageKind::RoiAlignBackward as u16,
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
    pub fn can_implement(&self, args: &RoiAlignBackwardArgs<'_, T>) -> Result<()> {
        if args.dout.shape
            != [self.desc.num_rois, self.desc.c, self.desc.pooled_h, self.desc.pooled_w]
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RoiAlignBackwardPlan: dout shape mismatch",
            ));
        }
        if args.rois.shape != [self.desc.num_rois, 5] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RoiAlignBackwardPlan: rois must be [num_rois, 5]",
            ));
        }
        if args.dinput.shape != [self.desc.n, self.desc.c, self.desc.h, self.desc.w] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RoiAlignBackwardPlan: dinput shape mismatch",
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
        args: RoiAlignBackwardArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if args.dout.numel() == 0 {
            return Ok(());
        }
        let dout_ptr = args.dout.data.as_raw().0 as *const c_void;
        let rois_ptr = args.rois.data.as_raw().0 as *const c_void;
        let din_ptr = args.dinput.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let aligned = if self.desc.aligned { 1 } else { 0 };
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_roi_align_backward_f32_run(
                    self.desc.n, self.desc.c, self.desc.h, self.desc.w,
                    self.desc.num_rois, self.desc.pooled_h, self.desc.pooled_w,
                    self.desc.spatial_scale, self.desc.sampling_ratio, aligned,
                    dout_ptr, rois_ptr, din_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_roi_align_backward_f64_run(
                    self.desc.n, self.desc.c, self.desc.h, self.desc.w,
                    self.desc.num_rois, self.desc.pooled_h, self.desc.pooled_w,
                    self.desc.spatial_scale, self.desc.sampling_ratio, aligned,
                    dout_ptr, rois_ptr, din_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::RoiAlignBackwardPlan::run reached unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
