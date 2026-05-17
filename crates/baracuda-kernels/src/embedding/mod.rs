//! Embedding op family — Category M.
//!
//! Phase 7 Milestone 7.5 of the baracuda-kernels comprehensive plan.
//! Plan-per-op because the FW / BW shapes share structure with
//! `index_select` but carry an optional `padding_idx` (and, for
//! `embedding_bag`, a bag-reduction mode + per-bag offset table) that
//! makes composing the indexing plans awkward.
//!
//! Ops shipped:
//! - [`EmbeddingPlan`] FW: `out[i, :] = weight[indices[i], :]` with
//!   optional `padding_idx` zeroing matching rows.
//! - [`EmbeddingBackwardPlan`] BW: `dweight[indices[i], :] += dout[i, :]`
//!   (atomicAdd), skipping the padding row.
//! - [`EmbeddingBagPlan`] FW: per-bag reduction over the index range
//!   `offsets[b]..offsets[b+1]`. Modes: `Sum` / `Mean`. Max-mode is
//!   deferred (needs argmax tracking for BW).
//! - [`EmbeddingBagBackwardPlan`] BW: atomicAdd of `dout[b, :] / divisor`
//!   into `dweight[indices[k], :]` for each k in the bag.
//!
//! Trailblazer dtype coverage:
//! - FW (`Embedding`, `EmbeddingBag`): `f32, f64, f16, bf16` (pure
//!   copy / accumulator-typed sum).
//! - BW: `f32, f64` only — atomicAdd is native-FP.
//!
//! Index dtype is `i32` only. Negative or out-of-range indices are
//! treated as "skip" (no PyTorch-style wrap-around). The
//! padding-disabled sentinel is `i32::MIN` (mapped from `Option::None`).

pub mod embedding;
pub mod embedding_backward;
pub mod embedding_bag;
pub mod embedding_bag_backward;

pub use embedding::{EmbeddingArgs, EmbeddingDescriptor, EmbeddingPlan};
pub use embedding_backward::{
    EmbeddingBackwardArgs, EmbeddingBackwardDescriptor, EmbeddingBackwardPlan,
};
pub use embedding_bag::{
    EmbeddingBagArgs, EmbeddingBagDescriptor, EmbeddingBagMode, EmbeddingBagPlan,
};
pub use embedding_bag_backward::{
    EmbeddingBagBackwardArgs, EmbeddingBagBackwardDescriptor, EmbeddingBagBackwardPlan,
};

/// Sentinel passed to the kernel when the caller does not supply a
/// `padding_idx`. Matches `kPaddingDisabled` in
/// `kernels/include/baracuda_embedding.cuh`.
pub(crate) const PADDING_DISABLED: i32 = i32::MIN;
