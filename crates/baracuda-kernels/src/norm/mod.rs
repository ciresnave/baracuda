//! Normalization op family — Phase 5 Category G.
//!
//! Length-preserving per-row normalization. Output shape equals input
//! shape across all variants. Today wired:
//!
//! - **RMSNorm** (FW + BW) — `y = x / sqrt(mean(x², over norm_axes) + eps) * gamma`.
//!   Llama / Mistral / Gemma block-pre-norm. Multi-axis via `norm_axes_mask`
//!   bitmask (PyTorch `normalized_shape` convention: must be a suffix of
//!   the input shape).
//!
//! - **LayerNorm** (FW + BW) — `y = (x - mean) / sqrt(var + eps) * gamma + beta`.
//!   Same multi-axis spec as RMSNorm.
//!
//! - **BatchNorm** (FW + BW) — per-channel normalization across
//!   `(N, *spatial*)`. Training mode only (inference mode using running
//!   statistics is deferred).
//!
//! - **GroupNorm** (FW + BW) — splits channel axis into `num_groups`
//!   groups, normalizes per `(sample, group, *spatial*)`.
//!
//! - **InstanceNorm** (FW + BW) — thin wrapper over GroupNorm with
//!   `num_groups == num_channels` (same kernel symbols).
//!
//! ## Deferred
//!
//! - `WeightNorm` — a parameterization, not a plain op.
//! - `LocalResponseNorm` — rarely used.
//! - BatchNorm inference mode (running statistics → per-channel affine
//!   multiply).
//!
//! ## Design notes
//!
//! - **No atomic adds.** Affine-grad accumulators (`dgamma`, `dbeta`)
//!   and group-stats reductions use one-block-per-feature kernels with
//!   warp shuffles + smem — fully deterministic, no half / bf16
//!   atomicAdd arch quirks.
//!
//! - **f16 / bf16 accumulate in f32** (mandatory — variance in half
//!   precision is catastrophic). f64 uses double throughout. For
//!   BatchNorm BW workspace partials we keep f32 even at f64 (acceptable
//!   precision loss on the partial-sum workspace for the trailblazer).
//!
//! - **Per-output-cell two-pass per-row.** Same naive O(extent²) total
//!   work per row as the softmax kernel for RMSNorm / LayerNorm; the
//!   BN/GN three-stage scheme amortizes the per-group reduction.

pub mod rms_norm;
pub mod rms_norm_backward;
pub mod layer_norm;
pub mod layer_norm_backward;
pub mod batch_norm;
pub mod batch_norm_backward;
pub mod group_norm;
pub mod group_norm_backward;
pub mod instance_norm;
pub mod instance_norm_backward;

pub use rms_norm::{RMSNormArgs, RMSNormDescriptor, RMSNormPlan};
pub use rms_norm_backward::{RMSNormBackwardArgs, RMSNormBackwardDescriptor, RMSNormBackwardPlan};
pub use layer_norm::{LayerNormArgs, LayerNormDescriptor, LayerNormPlan};
pub use layer_norm_backward::{
    LayerNormBackwardArgs, LayerNormBackwardDescriptor, LayerNormBackwardPlan,
};
pub use batch_norm::{BatchNormArgs, BatchNormDescriptor, BatchNormPlan};
pub use batch_norm_backward::{
    BatchNormBackwardArgs, BatchNormBackwardDescriptor, BatchNormBackwardPlan,
};
pub use group_norm::{GroupNormArgs, GroupNormDescriptor, GroupNormPlan};
pub use group_norm_backward::{
    GroupNormBackwardArgs, GroupNormBackwardDescriptor, GroupNormBackwardPlan,
};
pub use instance_norm::{InstanceNormArgs, InstanceNormDescriptor, InstanceNormPlan};
pub use instance_norm_backward::{
    InstanceNormBackwardArgs, InstanceNormBackwardDescriptor, InstanceNormBackwardPlan,
};
