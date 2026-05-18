//! `kthvalue_backward` plan — scatter the scalar `dy[batch]` back to
//! the saved-index position in `dx[batch, row_len]`.
//!
//! Reuses the topk-backward kernel (`k = 1` view): we present
//! `dy[batch]` as `[batch, 1]` and `indices[batch]` as `[batch, 1]`,
//! and run `topk_backward` with `k = 1`. Trailblazer dtype coverage:
//! `f32, f64`.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    Element, ElementKind, KernelSku, PlanPreference, PrecisionGuarantee, SortKind, TensorMut,
    TensorRef, Workspace,
};

use super::map_status;
use super::sort::build_sku;
use super::SORT_MAX_ROW;

/// Descriptor for a `kthvalue_backward` op.
#[derive(Copy, Clone, Debug)]
pub struct KthvalueBackwardDescriptor {
    /// Number of independent rows.
    pub batch: i32,
    /// Length of each input row.
    pub row_len: i32,
    /// Grad element type.
    pub element: ElementKind,
}

/// Args bundle for a `kthvalue_backward` launch.
pub struct KthvalueBackwardArgs<'a, T: Element> {
    /// Upstream grad `[batch]` (one cell per row).
    pub dy: TensorRef<'a, T, 1>,
    /// Saved indices from FW `[batch]`.
    pub indices: TensorRef<'a, i32, 1>,
    /// Grad of the input `[batch, row_len]`.
    pub dx: TensorMut<'a, T, 2>,
}

/// `kthvalue_backward` plan.
pub struct KthvalueBackwardPlan<T: Element> {
    desc: KthvalueBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> KthvalueBackwardPlan<T> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &KthvalueBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::KthvalueBackwardPlan: descriptor element != type parameter T",
            ));
        }
        if desc.batch < 0 || desc.row_len < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::KthvalueBackwardPlan: batch / row_len must be non-negative",
            ));
        }
        if desc.row_len > SORT_MAX_ROW {
            return Err(Error::Unsupported(
                "baracuda-kernels::KthvalueBackwardPlan: row_len > 1024 not supported",
            ));
        }
        if !matches!(T::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::KthvalueBackwardPlan: today only f32 / f64 grads supported",
            ));
        }
        let sku = build_sku::<T>(SortKind::KthvalueBackward);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &KthvalueBackwardArgs<'_, T>) -> Result<()> {
        if args.dy.shape != [self.desc.batch] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::KthvalueBackwardPlan: dy shape != [batch]",
            ));
        }
        if args.indices.shape != [self.desc.batch] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::KthvalueBackwardPlan: indices shape != [batch]",
            ));
        }
        if args.dx.shape != [self.desc.batch, self.desc.row_len] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::KthvalueBackwardPlan: dx shape != [batch, row_len]",
            ));
        }
        Ok(())
    }

    /// Workspace size in bytes.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Identity of the kernel this plan picked.
    #[inline]
    pub fn sku(&self) -> KernelSku {
        self.sku
    }

    /// Numerical guarantees for this plan's kernel.
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee
    }

    /// Launch.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: KthvalueBackwardArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if self.desc.batch == 0 || self.desc.row_len == 0 {
            return Ok(());
        }
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let idx_ptr = args.indices.data.as_raw().0 as *const c_void;
        let dx_ptr = args.dx.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let k = 1i32;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_topk_backward_f32_run(
                    self.desc.batch,
                    k,
                    self.desc.row_len,
                    dy_ptr,
                    idx_ptr,
                    dx_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_topk_backward_f64_run(
                    self.desc.batch,
                    k,
                    self.desc.row_len,
                    dy_ptr,
                    idx_ptr,
                    dx_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::KthvalueBackwardPlan::run reached an unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
