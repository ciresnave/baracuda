//! Sorting / order-statistics op family — Category O.
//!
//! Phase 9 of the baracuda-kernels comprehensive plan. Ships the
//! block-bitonic trailblazer family:
//!
//! - [`SortPlan`] / [`SortBackwardPlan`] / [`ArgsortPlan`] /
//!   [`MsortPlan`] / [`MsortBackwardPlan`] — block-bitonic sort, one
//!   CUDA block per row. Trailblazer cap: `row_len ≤ 1024`. Larger
//!   rows are reserved for a future tile-radix follow-up; the Plan
//!   returns `Unsupported` for `row_len > 1024`.
//! - [`TopkPlan`] / [`TopkBackwardPlan`] / [`KthvaluePlan`] /
//!   [`KthvalueBackwardPlan`] — block-bitonic select; trailblazer
//!   cap: `k ≤ 64` (LLM-inference range).
//! - [`UniquePlan`] / [`UniqueConsecutivePlan`] — set-valued, no BW.
//!   `unique` chains sort + consecutive-dedup at the plan layer.
//! - [`HistogramPlan`] / [`HistogramddPlan`] / [`BincountPlan`] —
//!   atomic-bin accumulation; FW only. `histogramdd` returns
//!   `Unsupported` for `ndim > 1` in the trailblazer.
//! - [`SearchsortedPlan`] — per-query binary search; FW only.
//!
//! **Saved-indices contract for sort / msort / topk / kthvalue BW.**
//! The FW emits both sorted values AND sorted indices in a single
//! launch (FW Args carry `values` and `indices` as required outputs).
//! BW Args receive the **saved indices** verbatim — no recomputation
//! at BW time. The Plan's selector pegs the indices dtype to `i32`
//! across every kernel SKU in this family.

pub mod argsort;
pub mod bincount;
pub mod histogram;
pub mod histogramdd;
pub mod kthvalue;
pub mod kthvalue_backward;
pub mod msort;
pub mod searchsorted;
pub mod sort;
pub mod sort_backward;
pub mod topk;
pub mod topk_backward;
pub mod unique;
pub mod unique_consecutive;

pub use argsort::{ArgsortArgs, ArgsortDescriptor, ArgsortPlan};
pub use bincount::{BincountArgs, BincountDescriptor, BincountPlan};
pub use histogram::{HistogramArgs, HistogramDescriptor, HistogramPlan};
pub use histogramdd::{HistogramddArgs, HistogramddDescriptor, HistogramddPlan};
pub use kthvalue::{KthvalueArgs, KthvalueDescriptor, KthvaluePlan};
pub use kthvalue_backward::{
    KthvalueBackwardArgs, KthvalueBackwardDescriptor, KthvalueBackwardPlan,
};
pub use msort::{MsortArgs, MsortBackwardArgs, MsortBackwardDescriptor, MsortBackwardPlan,
    MsortDescriptor, MsortPlan};
pub use searchsorted::{SearchsortedArgs, SearchsortedDescriptor, SearchsortedPlan};
pub use sort::{SortArgs, SortDescriptor, SortPlan};
pub use sort_backward::{SortBackwardArgs, SortBackwardDescriptor, SortBackwardPlan};
pub use topk::{TopkArgs, TopkDescriptor, TopkPlan};
pub use topk_backward::{TopkBackwardArgs, TopkBackwardDescriptor, TopkBackwardPlan};
pub use unique::{UniqueArgs, UniqueDescriptor, UniquePlan};
pub use unique_consecutive::{
    UniqueConsecutiveArgs, UniqueConsecutiveDescriptor, UniqueConsecutivePlan,
};

use baracuda_cutlass::{Error, Result};

/// Maximum supported `row_len` in the block-bitonic trailblazer. Must
/// match `MAX_ROW` in `baracuda_sort.cuh`.
pub const SORT_MAX_ROW: i32 = 1024;
/// Maximum supported `k` in the block-bitonic topk trailblazer. Must
/// match `MAX_K` in `baracuda_topk.cuh`.
pub const TOPK_MAX_K: i32 = 64;

/// Shared status-code mapper for the sort family.
pub(crate) fn map_status(code: i32) -> Result<()> {
    match code {
        0 => Ok(()),
        1 => Err(Error::MisalignedOperand),
        2 => Err(Error::InvalidProblem(
            "baracuda-kernels-sys::sort reported invalid problem",
        )),
        3 => Err(Error::Unsupported(
            "baracuda-kernels-sys::sort reported unsupported configuration \
             (e.g. row_len > 1024 or k > 64)",
        )),
        4 => Err(Error::WorkspaceTooSmall { needed: 0, got: 0 }),
        n => Err(Error::CutlassInternal(n)),
    }
}
