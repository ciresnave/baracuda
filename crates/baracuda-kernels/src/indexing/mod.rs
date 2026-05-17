//! Indexing / scatter / gather op family — Category L.
//!
//! Phase 7 Milestone 7.3 of the baracuda-kernels comprehensive plan.
//! Plan-per-op because each op's descriptor / args shape differs (the
//! gather / scatter ops carry a `dim` parameter and a separate `index`
//! tensor; `masked_fill` carries a scalar fill value; `one_hot` carries
//! a `num_classes` extent for the appended output axis; `nonzero`
//! carries an upper-bound `max_nz` for the output coordinate table).
//!
//! Trailblazer dtype coverage:
//! - [`GatherPlan`], [`ScatterAddPlan`], [`IndexSelectPlan`]: `f32, f64,
//!   i32` (FW). Their BWs (`GatherBackwardPlan` / `IndexSelectBackwardPlan`)
//!   are `f32, f64` only — they use `atomicAdd` which is native-FP.
//! - [`MaskedFillPlan`] FW + BW: `f32, f64, i32, bool` — pure
//!   element-select; no arithmetic, no atomics, dtype-agnostic.
//! - [`OneHotPlan`]: input is always `i32` class indices; output dtype
//!   selected from `f32, f64, i32, bool`.
//! - [`NonzeroPlan`]: input dtype family `f32, f64, i32, bool`; output
//!   is always `i32` coordinates.
//!
//! Index dtype is `i32` only (i64 deferred). Out-of-bounds indices are
//! skipped (no write); negative indices are treated as out-of-bounds
//! (no PyTorch-style wrap).
//!
//! `ScatterAdd` has no separate BW plan — the backward of `scatter_add`
//! is `gather`, which callers route through [`GatherPlan`].

pub mod gather;
pub mod gather_backward;
pub mod index_select;
pub mod index_select_backward;
pub mod masked_fill;
pub mod masked_fill_backward;
pub mod nonzero;
pub mod one_hot;
pub mod scatter_add;

pub use gather::{GatherArgs, GatherDescriptor, GatherPlan};
pub use gather_backward::{GatherBackwardArgs, GatherBackwardDescriptor, GatherBackwardPlan};
pub use index_select::{IndexSelectArgs, IndexSelectDescriptor, IndexSelectPlan};
pub use index_select_backward::{
    IndexSelectBackwardArgs, IndexSelectBackwardDescriptor, IndexSelectBackwardPlan,
};
pub use masked_fill::{MaskedFillArgs, MaskedFillDescriptor, MaskedFillPlan};
pub use masked_fill_backward::{
    MaskedFillBackwardArgs, MaskedFillBackwardDescriptor, MaskedFillBackwardPlan,
};
pub use nonzero::{NonzeroArgs, NonzeroDescriptor, NonzeroPlan};
pub use one_hot::{OneHotArgs, OneHotDescriptor, OneHotPlan};
pub use scatter_add::{ScatterAddArgs, ScatterAddDescriptor, ScatterAddPlan};
