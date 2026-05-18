//! `grid_sample` BW plan — Category T (2-D).
//!
//! Adjoint of [`crate::image::GridSamplePlan`]: scatters the upstream
//! gradient back into the input via the same bilinear weights
//! (atomicAdd into `dinput`), and accumulates the analytical
//! coordinate derivatives into `dgrid` (also atomic; one (n, oh, ow)
//! cell aggregates contributions across C).
//!
//! Caller MUST pre-zero both `dinput` and `dgrid`.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, ImageKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::map_status;

/// Descriptor for `grid_sample_backward`.
#[derive(Copy, Clone, Debug)]
pub struct GridSampleBackwardDescriptor {
    /// Batch.
    pub n: i32,
    /// Channels.
    pub c: i32,
    /// Input height.
    pub ih: i32,
    /// Input width.
    pub iw: i32,
    /// Output height.
    pub oh: i32,
    /// Output width.
    pub ow: i32,
    /// Value element type.
    pub element: ElementKind,
}

/// Args bundle for a `grid_sample_backward` launch.
pub struct GridSampleBackwardArgs<'a, T: Element> {
    /// Upstream gradient `[N, C, OH, OW]`.
    pub dout: TensorRef<'a, T, 4>,
    /// Saved FW input `[N, C, IH, IW]`.
    pub input: TensorRef<'a, T, 4>,
    /// Saved FW grid `[N, OH, OW, 2]`.
    pub grid: TensorRef<'a, T, 4>,
    /// Gradient w.r.t. input `[N, C, IH, IW]`. Caller pre-zeros.
    pub dinput: TensorMut<'a, T, 4>,
    /// Gradient w.r.t. grid `[N, OH, OW, 2]`. Caller pre-zeros.
    pub dgrid: TensorMut<'a, T, 4>,
}

/// `grid_sample_backward` plan.
pub struct GridSampleBackwardPlan<T: Element> {
    desc: GridSampleBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> GridSampleBackwardPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &GridSampleBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::GridSampleBackwardPlan: descriptor element != T",
            ));
        }
        if desc.n < 0 || desc.c < 0 || desc.ih < 0 || desc.iw < 0 || desc.oh < 0 || desc.ow < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GridSampleBackwardPlan: all extents must be non-negative",
            ));
        }
        if !matches!(T::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::GridSampleBackwardPlan: only `f32`, `f64` wired",
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
            op: ImageKind::GridSample2dBackward as u16,
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
    pub fn can_implement(&self, args: &GridSampleBackwardArgs<'_, T>) -> Result<()> {
        if args.dout.shape != [self.desc.n, self.desc.c, self.desc.oh, self.desc.ow]
            || args.input.shape != [self.desc.n, self.desc.c, self.desc.ih, self.desc.iw]
            || args.grid.shape != [self.desc.n, self.desc.oh, self.desc.ow, 2]
            || args.dinput.shape != [self.desc.n, self.desc.c, self.desc.ih, self.desc.iw]
            || args.dgrid.shape != [self.desc.n, self.desc.oh, self.desc.ow, 2]
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GridSampleBackwardPlan: operand shape mismatch",
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
        args: GridSampleBackwardArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if args.dout.numel() == 0 {
            return Ok(());
        }
        let dout_ptr = args.dout.data.as_raw().0 as *const c_void;
        let input_ptr = args.input.data.as_raw().0 as *const c_void;
        let grid_ptr = args.grid.data.as_raw().0 as *const c_void;
        let din_ptr = args.dinput.data.as_raw().0 as *mut c_void;
        let dgrid_ptr = args.dgrid.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_grid_sample_2d_backward_f32_run(
                    self.desc.n, self.desc.c, self.desc.ih, self.desc.iw,
                    self.desc.oh, self.desc.ow,
                    dout_ptr, input_ptr, grid_ptr,
                    din_ptr, dgrid_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_grid_sample_2d_backward_f64_run(
                    self.desc.n, self.desc.c, self.desc.ih, self.desc.iw,
                    self.desc.oh, self.desc.ow,
                    dout_ptr, input_ptr, grid_ptr,
                    din_ptr, dgrid_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::GridSampleBackwardPlan::run reached unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
