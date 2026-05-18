//! `sort` plan — Category O trailblazer.
//!
//! `sort(x, dim=last, descending)` returns sorted values AND sorted
//! indices along the last dimension of `x`. PyTorch `torch.sort`.
//!
//! Trailblazer dtype coverage: `f32, f64, i32, i64`.
//!
//! **Saved-indices contract.** The FW emits both `values` AND
//! `indices` (i32) in one launch. BW reads the saved indices to route
//! the upstream grad back to the original positions. Callers must
//! retain the indices output for the BW pass.
//!
//! Trailblazer cap: `row_len ≤ 1024` (one CUDA block per row, bitonic
//! network in shared memory). Larger rows return
//! `Error::Unsupported` — a tile-radix follow-up is reserved.
//!
//! BW: see [`crate::sort::SortBackwardPlan`].

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, SortKind, TensorMut, TensorRef, Workspace,
};

use super::{map_status, SORT_MAX_ROW};

/// Descriptor for a `sort` op.
#[derive(Copy, Clone, Debug)]
pub struct SortDescriptor {
    /// Number of independent rows to sort.
    pub batch: i32,
    /// Length of each row. Trailblazer cap: `≤ 1024`.
    pub row_len: i32,
    /// `true` = sort largest-first; `false` = sort smallest-first.
    pub descending: bool,
    /// Value element type.
    pub element: ElementKind,
}

/// Args bundle for a `sort` launch.
pub struct SortArgs<'a, T: Element> {
    /// Input `[batch, row_len]`.
    pub input: TensorRef<'a, T, 2>,
    /// Sorted values output `[batch, row_len]`.
    pub values: TensorMut<'a, T, 2>,
    /// Sorted indices output `[batch, row_len]` — saved for BW.
    pub indices: TensorMut<'a, i32, 2>,
}

/// `sort` plan.
///
/// `sort(x, dim=last, descending)` — returns sorted values AND
/// sorted indices along the last axis (PyTorch `torch.sort`).
///
/// **When to use**: forward row-wise sort. Pair with
/// [`SortBackwardPlan`](crate::SortBackwardPlan) for autograd; for
/// indices-only output use [`ArgsortPlan`](crate::ArgsortPlan).
///
/// **Dtypes**: `{f32, f64, i32, i64}`; indices always `i32`.
///
/// **Shape limits**: rank-2 `[batch, row_len]`; `row_len ≤ 1024`
/// (one CUDA block per row, bitonic network in shared memory).
/// Larger rows return `Unsupported` — tile-radix follow-up reserved.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable. Block-bitonic
/// is a fixed comparator network — no atomics, no reductions.
///
/// **Saved-indices contract**: FW emits both `values` and `indices`
/// in a single launch. BW reads the saved indices verbatim; callers
/// must retain `indices` for autograd.
pub struct SortPlan<T: Element> {
    desc: SortDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> SortPlan<T> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &SortDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_sort_desc(desc.batch, desc.row_len, desc.element, T::KIND, "SortPlan")?;
        let sku = build_sku::<T>(SortKind::Sort);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &SortArgs<'_, T>) -> Result<()> {
        validate_sort_args_2(
            self.desc.batch,
            self.desc.row_len,
            args.input.shape,
            args.values.shape,
            args.indices.shape,
            "SortPlan",
        )
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
        args: SortArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if self.desc.batch == 0 || self.desc.row_len == 0 {
            return Ok(());
        }
        let in_ptr = args.input.data.as_raw().0 as *const c_void;
        let vals_ptr = args.values.data.as_raw().0 as *mut c_void;
        let idx_ptr = args.indices.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let desc_flag = if self.desc.descending { 1 } else { 0 };

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_sort_f32_run(
                    self.desc.batch,
                    self.desc.row_len,
                    desc_flag,
                    in_ptr,
                    vals_ptr,
                    idx_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_sort_f64_run(
                    self.desc.batch,
                    self.desc.row_len,
                    desc_flag,
                    in_ptr,
                    vals_ptr,
                    idx_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::I32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_sort_i32_run(
                    self.desc.batch,
                    self.desc.row_len,
                    desc_flag,
                    in_ptr,
                    vals_ptr,
                    idx_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::I64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_sort_i64_run(
                    self.desc.batch,
                    self.desc.row_len,
                    desc_flag,
                    in_ptr,
                    vals_ptr,
                    idx_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::SortPlan::run reached an unimplemented dtype \
                     — select() should have caught this",
                ));
            }
        };
        map_status(status)
    }
}

// ---- Shared descriptor / args / SKU helpers (used by argsort / msort too) ----

/// Validate descriptor fields shared across sort / argsort / msort / topk.
pub(crate) fn validate_sort_desc(
    batch: i32,
    row_len: i32,
    descriptor_element: ElementKind,
    expected_element: ElementKind,
    _plan_name: &'static str,
) -> Result<()> {
    if descriptor_element != expected_element {
        return Err(Error::Unsupported(
            "baracuda-kernels::sort: descriptor element != type parameter T",
        ));
    }
    if batch < 0 || row_len < 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::sort: batch / row_len must be non-negative",
        ));
    }
    if row_len > SORT_MAX_ROW {
        return Err(Error::Unsupported(
            "baracuda-kernels::sort: row_len > 1024 not supported in the \
             block-bitonic trailblazer (tile-radix follow-up reserved)",
        ));
    }
    if !matches!(
        descriptor_element,
        ElementKind::F32 | ElementKind::F64 | ElementKind::I32 | ElementKind::I64
    ) {
        return Err(Error::Unsupported(
            "baracuda-kernels::sort: today only f32 / f64 / i32 / i64 wired",
        ));
    }
    Ok(())
}

/// Validate value+indices shapes for sort / msort args.
pub(crate) fn validate_sort_args_2(
    batch: i32,
    row_len: i32,
    in_shape: [i32; 2],
    vals_shape: [i32; 2],
    idx_shape: [i32; 2],
    _plan_name: &'static str,
) -> Result<()> {
    let expected = [batch, row_len];
    if in_shape != expected {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::sort: input shape != [batch, row_len]",
        ));
    }
    if vals_shape != expected {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::sort: values shape != [batch, row_len]",
        ));
    }
    if idx_shape != expected {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::sort: indices shape != [batch, row_len]",
        ));
    }
    Ok(())
}

/// Construct a `KernelSku` for a sort-family plan.
pub(crate) fn build_sku<T: Element>(op: SortKind) -> KernelSku {
    let precision_guarantee = PrecisionGuarantee {
        math_precision: if T::KIND == ElementKind::F64 {
            MathPrecision::F64
        } else {
            MathPrecision::F32
        },
        accumulator: T::KIND,
        // Block-bitonic is fully deterministic: per-row work is
        // serialized within one block (no inter-block reduction), and
        // ties are broken (for msort) by original index. Histogram /
        // bincount / unique-consecutive use atomic counters so their
        // output order is not deterministic — those plans re-tag this
        // field through their own builder.
        bit_stable_on_same_hardware: true,
        deterministic: true,
    };
    KernelSku {
        category: OpCategory::Sorting,
        op: op as u16,
        element: T::KIND,
        aux_element: Some(ElementKind::I32),
        layout: None,
        epilogue: None,
        arch: ArchSku::Sm80,
        backend: BackendKind::Bespoke,
        precision_guarantee,
    }
}
