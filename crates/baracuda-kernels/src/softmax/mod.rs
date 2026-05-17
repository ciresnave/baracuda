//! Softmax op family — Phase 5 Category H.
//!
//! Length-preserving transform along a single axis. Output shape ==
//! input shape (in contrast to reductions; same shape relation as
//! scans). Today wired:
//!
//! - **Softmax** (FW + BW) — `y[k] = exp(x[k] - max(x)) / Σ_j exp(x[j]
//!   - max(x))`. Numerically stable via max subtraction. BW formula:
//!   `dx[k] = y[k] · (dy[k] - Σ_j y[j] · dy[j])`. Needs saved
//!   forward output `y`.
//!
//! - **LogSoftmax** (FW + BW) — `y[k] = x[k] - logsumexp(x)`. BW:
//!   `dx[k] = dy[k] - exp(y[k]) · Σ_j dy[j]`. Needs saved `y`.
//!
//! - **GumbelSoftmax** (FW + BW) — `y = softmax((x + g) / τ)` with
//!   `g = -log(-log(u))`, `u ~ Uniform(0, 1)` drawn from cuRAND.
//!   Optional `hard` mode emits a one-hot at the row's noisy argmax
//!   (straight-through gradient). BW pipes through the Softmax BW
//!   kernel using the saved soft output.
//!
//! - **Sparsemax** (FW + BW) — `y = ProjSimplex(x)` via sort-then-
//!   threshold. Truly sparse output (active set determined by τ).
//!   BW formula: `dx[i] = dy[i] - mean(dy[active])` for active
//!   positions, `0` elsewhere. Row extent limited to 64 in the
//!   trailblazer (per-thread serial sort in local memory).

pub mod axis;
pub mod axis_backward;
pub mod gumbel;
pub mod gumbel_backward;
pub mod sparsemax;
pub mod sparsemax_backward;

pub use axis::{SoftmaxArgs, SoftmaxDescriptor, SoftmaxPlan};
pub use axis_backward::{SoftmaxBackwardArgs, SoftmaxBackwardDescriptor, SoftmaxBackwardPlan};
pub use gumbel::{GumbelSoftmaxArgs, GumbelSoftmaxDescriptor, GumbelSoftmaxPlan};
pub use gumbel_backward::{
    GumbelSoftmaxBackwardArgs, GumbelSoftmaxBackwardDescriptor, GumbelSoftmaxBackwardPlan,
};
pub use sparsemax::{SparsemaxArgs, SparsemaxDescriptor, SparsemaxPlan, SPARSEMAX_MAX_EXTENT};
pub use sparsemax_backward::{
    SparsemaxBackwardArgs, SparsemaxBackwardDescriptor, SparsemaxBackwardPlan,
};
