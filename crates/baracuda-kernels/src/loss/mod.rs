//! Loss op family — Phase 5 Category R.
//!
//! Today wired (FW + BW × 4 FP dtypes: f32, f16, bf16, f64):
//!
//! - **MSE** — `y = mean((pred - target)²)` (or sum / per-cell).
//!   BW: `dpred = 2·(pred - target) · scale`. `dtarget` is symmetric
//!   (`-dpred`); caller handles the negation if both gradients are needed.
//!
//! - **NLLLoss** — `y = -mean(input[target_idx[i]])` along the feature axis.
//!   Heterogeneous-dtype: input is `T`, target is `i64` class indices.
//!   BW: `dinput[i, c] = -dy/N if c == target[i] else 0`.
//!
//! - **CrossEntropyLoss** — `y = NLLLoss(LogSoftmax(input), target)`.
//!   Fused into a single per-row two-pass kernel for numerical stability
//!   (max subtraction, sum-of-exp, then `-log_softmax[target]`). Class-
//!   index target only (`i64`); soft-target CE is reserved for a future
//!   fanout. BW: `dinput[i, c] = (softmax(input)[i, c] - 1{c == target[i]}) · scale`.
//!
//! - **BCELoss** — `y = -mean(target·log(pred) + (1-target)·log(1-pred))`.
//!   Caller ensures `pred ∈ (0, 1)`. BW: `dpred = (pred - target) /
//!   (pred·(1-pred)) · scale`.
//!
//! - **KLDivLoss** — `y = mean(target·(log(target) - input))`. PyTorch's
//!   "input is log-prob" convention. Cells with `target == 0` contribute 0
//!   (avoids `log(0)`). BW: `dinput = -target · scale`.
//!
//! ## Reduction modes ([`LossReduction`])
//!
//! - `None`: output is per-cell — same shape as the loss surface
//!   (`[numel]` for MSE / BCE / KLDiv; `[n_rows]` for NLL / CrossEntropy).
//! - `Mean`: output is a scalar — `[1]` element of `T`. Per-cell terms
//!   are summed then divided by the cell count.
//! - `Sum`: output is a scalar — `[1]` element of `T`. Per-cell terms
//!   are summed (no divide).
//!
//! ## Design — deterministic, no atomic adds
//!
//! Per-cell (MSE / BCE / KLDiv) or per-row (NLL / CrossEntropy) kernel
//! computes loss terms into a workspace buffer of `T` (size = numel ·
//! sizeof(T) or n_rows · sizeof(T)) for Mean / Sum modes; a single-block
//! tree reduction kernel collapses the buffer to the final scalar
//! (divide-by-N applied in the finalizer for Mean mode). For None mode
//! the per-cell kernel writes directly to the output buffer and the
//! reduction step is skipped. No atomicAdd — fully deterministic.
//!
//! f16 / bf16 always accumulate in f32; f64 keeps everything in double.
//!
//! ## Future fanout
//!
//! `L1`, `SmoothL1`, `HingeEmbedding`, `MarginRanking`, `TripletMargin`,
//! `Ctc`, `PoissonNll` are reserved discriminants. Soft-target
//! `CrossEntropy` (target as a probability tensor instead of class
//! indices) and `GroupNorm` / `BatchNorm` (which dispatch to cuDNN in a
//! later milestone) are also deferred.

pub(crate) mod common;
pub mod bce;
pub mod bce_with_logits;
pub mod cosine_embedding;
pub mod cross_entropy;
pub mod ctc;
pub mod fused_linear_cross_entropy;
#[cfg(feature = "cudnn")]
pub mod ctc_loss_cudnn;
pub mod gaussian_nll;
pub mod hinge_embedding;
pub mod huber;
pub mod kl_div;
pub mod l1;
pub mod margin_ranking;
pub mod mse;
pub mod multi_margin;
pub mod multilabel_margin;
pub mod multilabel_soft_margin;
pub mod nll;
pub mod poisson_nll;
pub mod smooth_l1;
pub mod triplet_margin;

pub use bce::{
    BceLossArgs, BceLossBackwardArgs, BceLossBackwardDescriptor, BceLossBackwardPlan,
    BceLossDescriptor, BceLossPlan,
};
pub use bce_with_logits::{
    BceWithLogitsLossArgs, BceWithLogitsLossBackwardArgs, BceWithLogitsLossBackwardDescriptor,
    BceWithLogitsLossBackwardPlan, BceWithLogitsLossDescriptor, BceWithLogitsLossPlan,
};
pub use cross_entropy::{
    CrossEntropyLossArgs, CrossEntropyLossBackwardArgs, CrossEntropyLossBackwardDescriptor,
    CrossEntropyLossBackwardPlan, CrossEntropyLossDescriptor, CrossEntropyLossPlan,
};
pub use fused_linear_cross_entropy::{
    FusedLinearCrossEntropyArgs, FusedLinearCrossEntropyBackwardArgs,
    FusedLinearCrossEntropyBackwardDescriptor, FusedLinearCrossEntropyBackwardPlan,
    FusedLinearCrossEntropyDescriptor, FusedLinearCrossEntropyPlan, FLCE_DEFAULT_IGNORE_INDEX,
};
pub use gaussian_nll::{
    GaussianNllLossArgs, GaussianNllLossBackwardArgs, GaussianNllLossBackwardDescriptor,
    GaussianNllLossBackwardPlan, GaussianNllLossDescriptor, GaussianNllLossPlan,
};
pub use huber::{
    HuberLossArgs, HuberLossBackwardArgs, HuberLossBackwardDescriptor, HuberLossBackwardPlan,
    HuberLossDescriptor, HuberLossPlan,
};
pub use kl_div::{
    KlDivLossArgs, KlDivLossBackwardArgs, KlDivLossBackwardDescriptor, KlDivLossBackwardPlan,
    KlDivLossDescriptor, KlDivLossPlan,
};
pub use l1::{
    L1LossArgs, L1LossBackwardArgs, L1LossBackwardDescriptor, L1LossBackwardPlan,
    L1LossDescriptor, L1LossPlan,
};
pub use mse::{
    MseLossArgs, MseLossBackwardArgs, MseLossBackwardDescriptor, MseLossBackwardPlan,
    MseLossDescriptor, MseLossPlan,
};
pub use nll::{
    NllLossArgs, NllLossBackwardArgs, NllLossBackwardDescriptor, NllLossBackwardPlan,
    NllLossDescriptor, NllLossPlan,
};
pub use poisson_nll::{
    PoissonNllLossArgs, PoissonNllLossBackwardArgs, PoissonNllLossBackwardDescriptor,
    PoissonNllLossBackwardPlan, PoissonNllLossDescriptor, PoissonNllLossPlan,
};
pub use smooth_l1::{
    SmoothL1LossArgs, SmoothL1LossBackwardArgs, SmoothL1LossBackwardDescriptor,
    SmoothL1LossBackwardPlan, SmoothL1LossDescriptor, SmoothL1LossPlan,
};
pub use cosine_embedding::{
    CosineEmbeddingLossArgs, CosineEmbeddingLossBackwardArgs,
    CosineEmbeddingLossBackwardDescriptor, CosineEmbeddingLossBackwardPlan,
    CosineEmbeddingLossDescriptor, CosineEmbeddingLossPlan,
};
pub use hinge_embedding::{
    HingeEmbeddingLossArgs, HingeEmbeddingLossBackwardArgs,
    HingeEmbeddingLossBackwardDescriptor, HingeEmbeddingLossBackwardPlan,
    HingeEmbeddingLossDescriptor, HingeEmbeddingLossPlan,
};
pub use margin_ranking::{
    MarginRankingLossArgs, MarginRankingLossBackwardArgs,
    MarginRankingLossBackwardDescriptor, MarginRankingLossBackwardPlan,
    MarginRankingLossDescriptor, MarginRankingLossPlan,
};
pub use multi_margin::{
    MultiMarginLossArgs, MultiMarginLossBackwardArgs, MultiMarginLossBackwardDescriptor,
    MultiMarginLossBackwardPlan, MultiMarginLossDescriptor, MultiMarginLossPlan,
};
pub use multilabel_margin::{
    MultilabelMarginLossArgs, MultilabelMarginLossBackwardArgs,
    MultilabelMarginLossBackwardDescriptor, MultilabelMarginLossBackwardPlan,
    MultilabelMarginLossDescriptor, MultilabelMarginLossPlan,
};
pub use multilabel_soft_margin::{
    MultilabelSoftMarginLossArgs, MultilabelSoftMarginLossBackwardArgs,
    MultilabelSoftMarginLossBackwardDescriptor, MultilabelSoftMarginLossBackwardPlan,
    MultilabelSoftMarginLossDescriptor, MultilabelSoftMarginLossPlan,
};
pub use triplet_margin::{
    TripletMarginLossArgs, TripletMarginLossBackwardArgs,
    TripletMarginLossBackwardDescriptor, TripletMarginLossBackwardPlan,
    TripletMarginLossDescriptor, TripletMarginLossPlan,
};
pub use ctc::{
    CtcLossArgs, CtcLossBackwardArgs, CtcLossBackwardDescriptor, CtcLossBackwardPlan,
    CtcLossDescriptor, CtcLossPlan,
};
#[cfg(feature = "cudnn")]
pub use ctc_loss_cudnn::{CtcLossCudnnArgs, CtcLossCudnnDescriptor, CtcLossCudnnPlan};
