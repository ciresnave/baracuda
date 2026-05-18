//! `affine_grid` plan — Category T.
//!
//! Generate a normalized sampling grid from a 2x3 affine matrix per
//! batch entry. Companion to [`crate::image::GridSamplePlan`] —
//! identical coordinate convention (`align_corners=false`).
//!
//! For each output pixel center (oh, ow), the kernel computes its
//! base normalized coords (`bx, by`) ∈ (-1, 1) and applies the affine:
//!   x = theta[n, 0, 0] * bx + theta[n, 0, 1] * by + theta[n, 0, 2]
//!   y = theta[n, 1, 0] * bx + theta[n, 1, 1] * by + theta[n, 1, 2]
//!
//! Dtype coverage: `f32, f64`.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, ImageKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::map_status;

/// Descriptor for `affine_grid`.
#[derive(Copy, Clone, Debug)]
pub struct AffineGridDescriptor {
    /// Batch.
    pub n: i32,
    /// Output height.
    pub oh: i32,
    /// Output width.
    pub ow: i32,
    /// Value element type.
    pub element: ElementKind,
}

/// Args bundle for `affine_grid`.
pub struct AffineGridArgs<'a, T: Element> {
    /// Affine parameters `[N, 2, 3]`.
    pub theta: TensorRef<'a, T, 3>,
    /// Output grid `[N, OH, OW, 2]`.
    pub grid: TensorMut<'a, T, 4>,
}

/// `affine_grid` plan.
pub struct AffineGridPlan<T: Element> {
    desc: AffineGridDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> AffineGridPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &AffineGridDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::AffineGridPlan: descriptor element != T",
            ));
        }
        if desc.n < 0 || desc.oh < 0 || desc.ow < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::AffineGridPlan: all extents must be non-negative",
            ));
        }
        if !matches!(T::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::AffineGridPlan: only `f32`, `f64` wired",
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
            op: ImageKind::AffineGrid2d as u16,
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
    pub fn can_implement(&self, args: &AffineGridArgs<'_, T>) -> Result<()> {
        if args.theta.shape != [self.desc.n, 2, 3] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::AffineGridPlan: theta must be [N, 2, 3]",
            ));
        }
        if args.grid.shape != [self.desc.n, self.desc.oh, self.desc.ow, 2] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::AffineGridPlan: grid must be [N, OH, OW, 2]",
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
        args: AffineGridArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if args.grid.numel() == 0 {
            return Ok(());
        }
        let theta_ptr = args.theta.data.as_raw().0 as *const c_void;
        let grid_ptr = args.grid.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_affine_grid_2d_f32_run(
                    self.desc.n, self.desc.oh, self.desc.ow,
                    theta_ptr, grid_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_affine_grid_2d_f64_run(
                    self.desc.n, self.desc.oh, self.desc.ow,
                    theta_ptr, grid_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::AffineGridPlan::run reached unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
