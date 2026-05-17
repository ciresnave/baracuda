//! Segment / scatter-reduce op family — Category S.
//!
//! Phase 7 Milestone 7.6 of the baracuda-kernels comprehensive plan.
//! TF / JAX style: input is a rank-2 tensor `[N, D]`, `segment_ids` is
//! a `[N]` i32 array of segment IDs in `[0, num_segments)`, output is
//! `[num_segments, D]` after reducing rows with the same segment ID.
//!
//! Two families:
//!
//! - **Sorted** — `segment_ids` is monotonically non-decreasing. The
//!   kernel uses one thread per output `(s, d)` cell, binary-searches
//!   the seg-ids array for the range `[lo, hi)` covering segment `s`,
//!   and sweeps `input[i, d]` for `i ∈ [lo, hi)` with the chosen
//!   reduction. Ops: `segment_sum`, `segment_mean`, `segment_max`,
//!   `segment_min`, `segment_prod`.
//!
//! - **Unsorted** — `segment_ids` in any order. The kernel uses one
//!   thread per input `(n, d)` cell and emits an atomic into
//!   `output[seg_ids[n], d]`. Ops: `unsorted_segment_sum`,
//!   `unsorted_segment_mean`, `unsorted_segment_max`,
//!   `unsorted_segment_min`. `unsorted_segment_prod` is deferred
//!   (no native FP atomicMul; would need an `atomicCAS` retry loop).
//!
//! Backward coverage:
//!
//! - `sum` BW: `d_input[n, d] = d_output[seg[n], d]` — pure gather
//!   along seg-ids. Sorted and unsorted share the same kernel symbol.
//! - `mean` BW: `d_input[n, d] = d_output[seg[n], d] / count[seg[n]]`.
//!   Workspace = `num_segments * sizeof(i32)` for the count buffer.
//! - `max` / `min` / `prod` BW: **deferred** — Max / Min BW needs
//!   argmax / argmin tracking from FW; Prod BW needs numerically
//!   stable `prod / x_n` divisions.
//!
//! Trailblazer dtype coverage: `f32, f64` only — the kernels use
//! `atomicAdd` / `atomicMax`-via-`atomicCAS` / `atomicMin`-via-`atomicCAS`
//! which are restricted to native-FP-atomic types. f16 / bf16 deferred.
//!
//! Out-of-range segment IDs (`< 0` or `>= num_segments`) are silently
//! dropped (matches TF / JAX behavior — those frameworks call this
//! undefined and we choose the "skip" semantic).

pub mod segment_max;
pub mod segment_mean;
pub mod segment_mean_backward;
pub mod segment_min;
pub mod segment_prod;
pub mod segment_sum;
pub mod segment_sum_backward;
pub mod unsorted_segment_max;
pub mod unsorted_segment_mean;
pub mod unsorted_segment_mean_backward;
pub mod unsorted_segment_min;
pub mod unsorted_segment_sum;
pub mod unsorted_segment_sum_backward;

pub use segment_max::{SegmentMaxArgs, SegmentMaxDescriptor, SegmentMaxPlan};
pub use segment_mean::{SegmentMeanArgs, SegmentMeanDescriptor, SegmentMeanPlan};
pub use segment_mean_backward::{
    SegmentMeanBackwardArgs, SegmentMeanBackwardDescriptor, SegmentMeanBackwardPlan,
};
pub use segment_min::{SegmentMinArgs, SegmentMinDescriptor, SegmentMinPlan};
pub use segment_prod::{SegmentProdArgs, SegmentProdDescriptor, SegmentProdPlan};
pub use segment_sum::{SegmentSumArgs, SegmentSumDescriptor, SegmentSumPlan};
pub use segment_sum_backward::{
    SegmentSumBackwardArgs, SegmentSumBackwardDescriptor, SegmentSumBackwardPlan,
};
pub use unsorted_segment_max::{
    UnsortedSegmentMaxArgs, UnsortedSegmentMaxDescriptor, UnsortedSegmentMaxPlan,
};
pub use unsorted_segment_mean::{
    UnsortedSegmentMeanArgs, UnsortedSegmentMeanDescriptor, UnsortedSegmentMeanPlan,
};
pub use unsorted_segment_mean_backward::{
    UnsortedSegmentMeanBackwardArgs, UnsortedSegmentMeanBackwardDescriptor,
    UnsortedSegmentMeanBackwardPlan,
};
pub use unsorted_segment_min::{
    UnsortedSegmentMinArgs, UnsortedSegmentMinDescriptor, UnsortedSegmentMinPlan,
};
pub use unsorted_segment_sum::{
    UnsortedSegmentSumArgs, UnsortedSegmentSumDescriptor, UnsortedSegmentSumPlan,
};
pub use unsorted_segment_sum_backward::{
    UnsortedSegmentSumBackwardArgs, UnsortedSegmentSumBackwardDescriptor,
    UnsortedSegmentSumBackwardPlan,
};

use baracuda_cutlass::{Error, Result};

/// Shared status-code mapper (mirrors `indexing::gather::map_status`).
pub(crate) fn map_status(code: i32) -> Result<()> {
    match code {
        0 => Ok(()),
        1 => Err(Error::MisalignedOperand),
        2 => Err(Error::InvalidProblem(
            "baracuda-kernels-sys reported invalid problem",
        )),
        3 => Err(Error::Unsupported(
            "baracuda-kernels-sys reported unsupported configuration",
        )),
        4 => Err(Error::WorkspaceTooSmall { needed: 0, got: 0 }),
        n => Err(Error::CutlassInternal(n)),
    }
}
