//! Pooling op family — Phase 7 Milestone 7.2 (Category Pooling / J).
//!
//! Wraps cuDNN's legacy descriptor-based pooling API. The trailblazer is
//! 2-D NCHW pooling — both max-pool ([`MaxPool2dPlan`]) and average-pool
//! ([`AvgPool2dPlan`]). 1-D / 3-D pooling, adaptive pooling, LP-pool,
//! and fractional-max-pool follow in fanout milestones.
//!
//! ## Plan layout
//!
//! - [`MaxPool2dPlan`] / [`AvgPool2dPlan`] — each owns one cuDNN handle
//!   plus three lazy descriptors (`x_desc`, `y_desc`, `pool_desc`)
//!   created on first `run_fw` and reused across launches. No workspace
//!   caches — pooling is **workspace-free** in cuDNN's legacy API.
//!
//! ## Average-pool padding modes
//!
//! cuDNN exposes two avg-pool flavors:
//!
//! - `AVERAGE_COUNT_INCLUDE_PADDING` — divide by `window_h * window_w`
//!   (zero-padded cells count toward the denominator). Matches
//!   TensorFlow's default.
//! - `AVERAGE_COUNT_EXCLUDE_PADDING` — divide only by the count of
//!   *valid* (non-padded) cells in each window. **PyTorch's
//!   `nn.AvgPool2d` default** (`count_include_pad=False`).
//!
//! [`AvgPool2dPlan`] dispatches on the [`PoolMode`] field of the
//! descriptor; the trailblazer defaults the convenience constructor to
//! `AvgExcludePad` for PyTorch parity.
//!
//! ## Backward pass
//!
//! [`Pool2dBwArgs`] carries **both** `y` (saved FW output) and `x`
//! (saved FW input) because cuDNN's pooling-BW API requires both — for
//! max-pool it uses them to recover the per-window argmax (no separate
//! indices tensor is materialized by the legacy API); for avg-pool the
//! gradient depends only on `x` but cuDNN still demands `y` for API
//! uniformity. Callers must retain `y` and `x` from the FW launch.
//!
//! ## Handle ownership
//!
//! Each plan lazily owns one `cudnnHandle_t` in a `Cell<>` (created on
//! first `run`; bound to the caller's stream on every launch so the
//! plan is reusable across streams). cuDNN handles are **not** thread-
//! safe — the plan is `!Sync` / `!Send` by virtue of the `Cell<>` it
//! holds. The handle and all descriptors are released in `Drop`.
//!
//! ## Workspace
//!
//! [`Workspace::None`] suffices — cuDNN's pooling kernel allocates its
//! small internal scratch itself. The `run_*` methods accept any
//! `Workspace<'_>` for caller convenience but never read from it.
//!
//! ## Output spatial extents
//!
//! Computed by both plans as
//! `H_out = floor((H_in + 2·pad_h - window_h) / stride_h) + 1`,
//! and similarly for `W_out`. This matches PyTorch / cuDNN convention
//! (no `ceil_mode` knob in the trailblazer — that's a fanout extension).
//!
//! ## Dtype coverage
//!
//! `f32`, `f64`, `f16`, `bf16` — the four cuDNN-supported FP types for
//! pooling. The cuDNN alpha/beta scalar dtype is `f32` for `f32` /
//! `f16` / `bf16` operands and `f64` for `f64` operands.

pub mod avg_pool2d;
pub mod max_pool2d;

pub use avg_pool2d::AvgPool2dPlan;
pub use max_pool2d::MaxPool2dPlan;

// Shared descriptor / args / mode types live in the max_pool2d module
// (which gets compiled first) and are re-exported here so callers can
// reach for `pool::Pool2dDescriptor` regardless of which plan they pick.
pub use max_pool2d::{Pool2dBwArgs, Pool2dDescriptor, Pool2dFwArgs, PoolMode};
