//! `sort_backward` plan — scatter `dy` via the saved indices.
//!
//! `dx[batch, indices[batch, i]] = dy[batch, i]`. Launcher zeros `dx`
//! before the scatter. Trailblazer dtype coverage: `f32, f64`.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    Element, ElementKind, KernelSku, PlanPreference, PrecisionGuarantee, SortKind, TensorMut,
    TensorRef, Workspace,
};

use super::map_status;
use super::sort::{build_sku, validate_sort_desc};

/// Descriptor for a `sort_backward` op.
#[derive(Copy, Clone, Debug)]
pub struct SortBackwardDescriptor {
    /// Number of independent rows.
    pub batch: i32,
    /// Length of each row.
    pub row_len: i32,
    /// Value (gradient) element type.
    pub element: ElementKind,
}

/// Args bundle for a `sort_backward` launch.
pub struct SortBackwardArgs<'a, T: Element> {
    /// Upstream grad of sorted-values output `[batch, row_len]`.
    pub dy: TensorRef<'a, T, 2>,
    /// Saved indices from FW `[batch, row_len]`.
    pub indices: TensorRef<'a, i32, 2>,
    /// Grad of the original input `[batch, row_len]` (output).
    pub dx: TensorMut<'a, T, 2>,
}

/// `sort_backward` plan.
///
/// Adjoint of [`crate::SortPlan`]: scatters `d_values[b, p]` to
/// `d_input[b, indices[b, p]]`. Pure index-routed permutation —
/// each input position receives exactly one gradient, so no atomics
/// needed.
///
/// **When to use**: BW for [`SortPlan`](crate::SortPlan). Consumes
/// the FW's saved `indices` verbatim.
///
/// **Dtypes**: `{f32, f64, i32, i64}` (matches FW).
///
/// **Shape limits**: rank-2 `[batch, row_len]`; `row_len ≤ 1024`.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable.
pub struct SortBackwardPlan<T: Element> {
    desc: SortBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> SortBackwardPlan<T> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &SortBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_sort_desc(
            desc.batch,
            desc.row_len,
            desc.element,
            T::KIND,
            "SortBackwardPlan",
        )?;
        if !matches!(T::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::SortBackwardPlan: today only f32 / f64 grads supported",
            ));
        }
        let sku = build_sku::<T>(SortKind::SortBackward);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &SortBackwardArgs<'_, T>) -> Result<()> {
        let expected = [self.desc.batch, self.desc.row_len];
        if args.dy.shape != expected
            || args.indices.shape != expected
            || args.dx.shape != expected
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SortBackwardPlan: tensor shapes != [batch, row_len]",
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
        args: SortBackwardArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if self.desc.batch == 0 || self.desc.row_len == 0 {
            return Ok(());
        }
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let idx_ptr = args.indices.data.as_raw().0 as *const c_void;
        let dx_ptr = args.dx.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_sort_backward_f32_run(
                    self.desc.batch,
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
                baracuda_kernels_sys::baracuda_kernels_sort_backward_f64_run(
                    self.desc.batch,
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
                    "baracuda-kernels::SortBackwardPlan::run reached an unimplemented dtype \
                     — select() should have caught this",
                ));
            }
        };
        map_status(status)
    }
}
