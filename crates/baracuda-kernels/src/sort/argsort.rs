//! `argsort` plan — sorted indices only (no values output).
//!
//! `argsort(x, dim=last, descending)` returns just the i32 permutation
//! that would sort `x` along the last dim. PyTorch `torch.argsort`.
//!
//! Trailblazer dtype coverage: `f32, f64, i32, i64`.
//! Non-differentiable (set-valued indices) — no BW.

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

/// Descriptor for an `argsort` op.
#[derive(Copy, Clone, Debug)]
pub struct ArgsortDescriptor {
    /// Number of independent rows.
    pub batch: i32,
    /// Length of each row. Trailblazer cap: `≤ 1024`.
    pub row_len: i32,
    /// `true` = sort largest-first.
    pub descending: bool,
    /// Value element type (input).
    pub element: ElementKind,
}

/// Args bundle for an `argsort` launch.
pub struct ArgsortArgs<'a, T: Element> {
    /// Input `[batch, row_len]`.
    pub input: TensorRef<'a, T, 2>,
    /// Sorted indices output `[batch, row_len]`.
    pub indices: TensorMut<'a, i32, 2>,
}

/// `argsort` plan.
///
/// Returns only sorted indices along the last axis (PyTorch
/// `torch.argsort`). No values output, no BW (indices are
/// non-differentiable).
///
/// **When to use**: when only the permutation is needed (gather
/// downstream tensors via [`GatherPlan`](crate::GatherPlan) using
/// these indices). For sorted values + indices use
/// [`SortPlan`](crate::SortPlan).
///
/// **Dtypes**: input `{f32, f64, i32, i64}`; output always `i32`.
///
/// **Shape limits**: rank-2 `[batch, row_len]`; `row_len ≤ 1024`.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable.
pub struct ArgsortPlan<T: Element> {
    desc: ArgsortDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> ArgsortPlan<T> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &ArgsortDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_sort_desc(
            desc.batch,
            desc.row_len,
            desc.element,
            T::KIND,
            "ArgsortPlan",
        )?;
        let sku = build_sku::<T>(SortKind::Argsort);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &ArgsortArgs<'_, T>) -> Result<()> {
        let expected = [self.desc.batch, self.desc.row_len];
        if args.input.shape != expected {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ArgsortPlan: input shape != [batch, row_len]",
            ));
        }
        if args.indices.shape != expected {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ArgsortPlan: indices shape != [batch, row_len]",
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
        args: ArgsortArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if self.desc.batch == 0 || self.desc.row_len == 0 {
            return Ok(());
        }
        let in_ptr = args.input.data.as_raw().0 as *const c_void;
        let idx_ptr = args.indices.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let desc_flag = if self.desc.descending { 1 } else { 0 };

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_argsort_f32_run(
                    self.desc.batch,
                    self.desc.row_len,
                    desc_flag,
                    in_ptr,
                    idx_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_argsort_f64_run(
                    self.desc.batch,
                    self.desc.row_len,
                    desc_flag,
                    in_ptr,
                    idx_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::I32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_argsort_i32_run(
                    self.desc.batch,
                    self.desc.row_len,
                    desc_flag,
                    in_ptr,
                    idx_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::I64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_argsort_i64_run(
                    self.desc.batch,
                    self.desc.row_len,
                    desc_flag,
                    in_ptr,
                    idx_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::ArgsortPlan::run reached an unimplemented dtype \
                     — select() should have caught this",
                ));
            }
        };
        map_status(status)
    }
}
