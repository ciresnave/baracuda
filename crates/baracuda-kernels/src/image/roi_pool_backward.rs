//! `roi_pool` BW plan — Category T.
//!
//! Routes each output cell's gradient (via atomicAdd) to the argmax
//! input cell saved by [`crate::image::RoiPoolPlan`]'s FW pass.
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

/// Descriptor for `roi_pool_backward`.
#[derive(Copy, Clone, Debug)]
pub struct RoiPoolBackwardDescriptor {
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
    /// Pooled height per RoI.
    pub pooled_h: i32,
    /// Pooled width per RoI.
    pub pooled_w: i32,
    /// Value element type.
    pub element: ElementKind,
}

/// Args bundle for `roi_pool_backward`.
pub struct RoiPoolBackwardArgs<'a, T: Element> {
    /// Upstream gradient `[num_rois, C, pooled_h, pooled_w]`.
    pub dout: TensorRef<'a, T, 4>,
    /// Saved FW rois `[num_rois, 5]`.
    pub rois: TensorRef<'a, T, 2>,
    /// Saved FW argmax `[num_rois, C, pooled_h, pooled_w]` (i32).
    pub argmax: TensorRef<'a, i32, 4>,
    /// Gradient w.r.t. input `[N, C, H, W]`. Caller pre-zeros.
    pub dinput: TensorMut<'a, T, 4>,
}

/// `roi_pool_backward` plan.
pub struct RoiPoolBackwardPlan<T: Element> {
    desc: RoiPoolBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> RoiPoolBackwardPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &RoiPoolBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::RoiPoolBackwardPlan: descriptor element != T",
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
                "baracuda-kernels::RoiPoolBackwardPlan: extents must be non-negative",
            ));
        }
        if !matches!(T::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::RoiPoolBackwardPlan: only `f32`, `f64` wired",
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
            op: ImageKind::RoiPoolBackward as u16,
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
    pub fn can_implement(&self, args: &RoiPoolBackwardArgs<'_, T>) -> Result<()> {
        let dout_shape =
            [self.desc.num_rois, self.desc.c, self.desc.pooled_h, self.desc.pooled_w];
        if args.dout.shape != dout_shape || args.argmax.shape != dout_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RoiPoolBackwardPlan: dout / argmax shape mismatch",
            ));
        }
        if args.rois.shape != [self.desc.num_rois, 5] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RoiPoolBackwardPlan: rois must be [num_rois, 5]",
            ));
        }
        if args.dinput.shape != [self.desc.n, self.desc.c, self.desc.h, self.desc.w] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RoiPoolBackwardPlan: dinput shape mismatch",
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
        args: RoiPoolBackwardArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if args.dout.numel() == 0 {
            return Ok(());
        }
        let dout_ptr = args.dout.data.as_raw().0 as *const c_void;
        let rois_ptr = args.rois.data.as_raw().0 as *const c_void;
        let arg_ptr = args.argmax.data.as_raw().0 as *const c_void;
        let din_ptr = args.dinput.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_roi_pool_backward_f32_run(
                    self.desc.n, self.desc.c, self.desc.h, self.desc.w,
                    self.desc.num_rois, self.desc.pooled_h, self.desc.pooled_w,
                    dout_ptr, rois_ptr, arg_ptr, din_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_roi_pool_backward_f64_run(
                    self.desc.n, self.desc.c, self.desc.h, self.desc.w,
                    self.desc.num_rois, self.desc.pooled_h, self.desc.pooled_w,
                    dout_ptr, rois_ptr, arg_ptr, din_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::RoiPoolBackwardPlan::run reached unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
