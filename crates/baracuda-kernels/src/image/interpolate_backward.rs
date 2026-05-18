//! `interpolate` BW plan — Category T (bilinear 2D).
//!
//! Adjoint of [`crate::image::InterpolatePlan`]: distributes each
//! output cell's gradient across the 4 input cells that bilinearly
//! produced it, weighted by the same `wij`. atomicAdd into `dinput`.
//!
//! Caller MUST pre-zero `dinput` before launch.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, ImageKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::interpolate::InterpolateMode;
use super::map_status;

/// Descriptor for an `interpolate_backward` op.
#[derive(Copy, Clone, Debug)]
pub struct InterpolateBackwardDescriptor {
    /// Batch.
    pub n: i32,
    /// Channels.
    pub c: i32,
    /// Input height.
    pub ih: i32,
    /// Input width.
    pub iw: i32,
    /// Output height (FW output).
    pub oh: i32,
    /// Output width (FW output).
    pub ow: i32,
    /// Interpolation mode (must match FW).
    pub mode: InterpolateMode,
    /// Value element type.
    pub element: ElementKind,
}

/// Args bundle for an `interpolate_backward` launch.
pub struct InterpolateBackwardArgs<'a, T: Element> {
    /// Upstream gradient `[N, C, OH, OW]`.
    pub dout: TensorRef<'a, T, 4>,
    /// Gradient w.r.t. input `[N, C, IH, IW]`. Caller pre-zeros.
    pub dinput: TensorMut<'a, T, 4>,
}

/// `interpolate_backward` plan.
pub struct InterpolateBackwardPlan<T: Element> {
    desc: InterpolateBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> InterpolateBackwardPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &InterpolateBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::InterpolateBackwardPlan: descriptor element != T",
            ));
        }
        if !matches!(desc.mode, InterpolateMode::Bilinear2d) {
            return Err(Error::Unsupported(
                "baracuda-kernels::InterpolateBackwardPlan: only Bilinear2d wired",
            ));
        }
        if desc.n < 0 || desc.c < 0 || desc.ih < 0 || desc.iw < 0 || desc.oh < 0 || desc.ow < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::InterpolateBackwardPlan: all extents must be non-negative",
            ));
        }
        if !matches!(T::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::InterpolateBackwardPlan: only `f32`, `f64` wired",
            ));
        }
        let precision_guarantee = PrecisionGuarantee {
            math_precision: if T::KIND == ElementKind::F64 {
                MathPrecision::F64
            } else {
                MathPrecision::F32
            },
            accumulator: T::KIND,
            // atomicAdd scatter — non-deterministic across launches.
            bit_stable_on_same_hardware: false,
            deterministic: false,
        };
        let sku = KernelSku {
            category: OpCategory::Image,
            op: ImageKind::InterpolateBilinear2dBackward as u16,
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
    pub fn can_implement(&self, args: &InterpolateBackwardArgs<'_, T>) -> Result<()> {
        if args.dout.shape != [self.desc.n, self.desc.c, self.desc.oh, self.desc.ow] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::InterpolateBackwardPlan: dout shape mismatch",
            ));
        }
        if args.dinput.shape != [self.desc.n, self.desc.c, self.desc.ih, self.desc.iw] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::InterpolateBackwardPlan: dinput shape mismatch",
            ));
        }
        let dout_numel = args.dout.numel();
        let din_numel = args.dinput.numel();
        if (args.dout.data.len() as i64) < dout_numel {
            return Err(Error::BufferTooSmall {
                needed: dout_numel as usize,
                got: args.dout.data.len(),
            });
        }
        if (args.dinput.data.len() as i64) < din_numel {
            return Err(Error::BufferTooSmall {
                needed: din_numel as usize,
                got: args.dinput.data.len(),
            });
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
        args: InterpolateBackwardArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if args.dout.numel() == 0 {
            return Ok(());
        }
        let dout_ptr = args.dout.data.as_raw().0 as *const c_void;
        let din_ptr = args.dinput.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_interpolate_bilinear_2d_backward_f32_run(
                    self.desc.n, self.desc.c, self.desc.ih, self.desc.iw,
                    self.desc.oh, self.desc.ow,
                    dout_ptr, din_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_interpolate_bilinear_2d_backward_f64_run(
                    self.desc.n, self.desc.c, self.desc.ih, self.desc.iw,
                    self.desc.oh, self.desc.ow,
                    dout_ptr, din_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::InterpolateBackwardPlan::run reached unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
