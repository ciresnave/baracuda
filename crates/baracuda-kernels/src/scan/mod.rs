//! Scan (associative prefix) op family — Phase 4 Category F.
//!
//! Length-preserving prefix operators along a single axis. Output
//! shape == input shape (no collapse — distinct from reductions).
//! Inclusive scans by default (PyTorch convention: `y[i] = op(x[0], …,
//! x[i])`); `reverse` flag in the descriptor flips the direction.
//!
//! Today wired (FW + BW × all 4 FP dtypes: f32, f16, bf16, f64):
//!
//! - **Cumsum** — `y[i] = Σ_{j≤i} x[j]`. BW reuses the FW kernel with
//!   the direction flipped (gradient of `cumsum` is the reverse cumsum
//!   of `dy`).
//! - **Cumprod** — `y[i] = Π_{j≤i} x[j]`. BW needs saved `x` + `y`.
//! - **Cummax** / **Cummin** — running max / min. BW routes gradient to
//!   the first-occurrence argmax/argmin position; needs saved `x`.
//! - **LogCumsumExp** — `y[i] = log(Σ_{j≤i} exp(x[j]))`. Numerically
//!   stable via per-cell running max. BW needs saved `x` + `y`.

pub mod axis;
pub mod axis_backward;

pub use axis::{ScanArgs, ScanDescriptor, ScanPlan};
pub use axis_backward::{ScanBackwardArgs, ScanBackwardDescriptor, ScanBackwardPlan};
