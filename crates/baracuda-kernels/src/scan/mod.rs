//! Scan (associative prefix) op family — Phase 4 Category F.
//!
//! Length-preserving prefix operators along a single axis. Output
//! shape == input shape (no collapse — distinct from reductions).
//! Inclusive scans by default (PyTorch convention: `y[i] = op(x[0], …,
//! x[i])`); `reverse` flag in the descriptor flips the direction.
//!
//! Today wired: `{Cumsum} × {f32, f16, bf16, f64}` — FW + BW. The
//! BW reuses the FW kernel with the direction flipped (gradient of
//! `cumsum` is the reverse cumsum of `dy`). Cumprod / Cummax /
//! Cummin / LogCumsumExp follow in fanout sessions.

pub mod axis;
pub mod axis_backward;

pub use axis::{ScanArgs, ScanDescriptor, ScanPlan};
pub use axis_backward::{ScanBackwardArgs, ScanBackwardDescriptor, ScanBackwardPlan};
