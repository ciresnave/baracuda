//! `roi_pool` FW plan — Category T.
//!
//! Max-pool variant of [`crate::image::RoiAlignPlan`]. Each output
//! cell is the max value over the (integer-rounded) RoI bin in the
//! input plane. The kernel emits an `argmax` buffer (i32 linear
//! plane-relative index per output cell; `-1` for empty bins) that
//! the BW reads to route gradient.
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

/// Descriptor for `roi_pool`.
#[derive(Copy, Clone, Debug)]
pub struct RoiPoolDescriptor {
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
    /// RoI coord scale.
    pub spatial_scale: f32,
    /// Value element type.
    pub element: ElementKind,
}

/// Args bundle for `roi_pool`.
pub struct RoiPoolArgs<'a, T: Element> {
    /// Input `[N, C, H, W]`.
    pub input: TensorRef<'a, T, 4>,
    /// RoIs `[num_rois, 5]`.
    pub rois: TensorRef<'a, T, 2>,
    /// Output `[num_rois, C, pooled_h, pooled_w]`.
    pub output: TensorMut<'a, T, 4>,
    /// Argmax (i32 plane-relative index per output cell, or -1 for
    /// empty bin) `[num_rois, C, pooled_h, pooled_w]`. Required for BW.
    pub argmax: TensorMut<'a, i32, 4>,
}

/// `roi_pool` plan.
pub struct RoiPoolPlan<T: Element> {
    desc: RoiPoolDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> RoiPoolPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &RoiPoolDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::RoiPoolPlan: descriptor element != T",
            ));
        }
        if desc.n < 0
            || desc.c < 0
            || desc.h < 0
            || desc.w < 0
            || desc.num_rois < 0
            || desc.pooled_h < 0
            || desc.pooled_w < 0
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RoiPoolPlan: extents must be non-negative",
            ));
        }
        if !matches!(T::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::RoiPoolPlan: only `f32`, `f64` wired",
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
            op: ImageKind::RoiPool as u16,
            element: T::KIND,
            aux_element: Some(ElementKind::I32),
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
    pub fn can_implement(&self, args: &RoiPoolArgs<'_, T>) -> Result<()> {
        if args.input.shape != [self.desc.n, self.desc.c, self.desc.h, self.desc.w] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RoiPoolPlan: input shape mismatch",
            ));
        }
        if args.rois.shape != [self.desc.num_rois, 5] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RoiPoolPlan: rois must be [num_rois, 5]",
            ));
        }
        let out_shape = [self.desc.num_rois, self.desc.c, self.desc.pooled_h, self.desc.pooled_w];
        if args.output.shape != out_shape || args.argmax.shape != out_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RoiPoolPlan: output / argmax shape mismatch",
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
        args: RoiPoolArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if args.output.numel() == 0 {
            return Ok(());
        }
        let input_ptr = args.input.data.as_raw().0 as *const c_void;
        let rois_ptr = args.rois.data.as_raw().0 as *const c_void;
        let out_ptr = args.output.data.as_raw().0 as *mut c_void;
        let arg_ptr = args.argmax.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_roi_pool_f32_run(
                    self.desc.n, self.desc.c, self.desc.h, self.desc.w,
                    self.desc.num_rois, self.desc.pooled_h, self.desc.pooled_w,
                    self.desc.spatial_scale,
                    input_ptr, rois_ptr, out_ptr, arg_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_roi_pool_f64_run(
                    self.desc.n, self.desc.c, self.desc.h, self.desc.w,
                    self.desc.num_rois, self.desc.pooled_h, self.desc.pooled_w,
                    self.desc.spatial_scale,
                    input_ptr, rois_ptr, out_ptr, arg_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::RoiPoolPlan::run reached unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
