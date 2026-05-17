//! Convolution op family — Phase 7 (Category Convolution / I).
//!
//! Wraps cuDNN's legacy descriptor-based convolution API for the
//! forward pass plus the two backward passes (data gradient + filter
//! gradient). The trailblazer is 2-D NCHW convolution; Conv1d /
//! Conv3d / NHWC / transposed-conv land in fanout milestones.
//!
//! ## Plan layout
//!
//! - [`Conv2dPlan`] — owns one cuDNN handle plus four lazy
//!   descriptors (`x_desc`, `w_desc`, `y_desc`, `conv_desc`) created
//!   on first `run` and reused across launches. The plan also owns
//!   workspace-size caches (one per direction — FW / BW data / BW
//!   filter), populated lazily on first call to the corresponding
//!   `query_*_workspace_size` accessor.
//!
//! ## Algorithm selection
//!
//! The trailblazer pins:
//!
//! - Forward: `IMPLICIT_GEMM` (algo `0`) — universally supported, no
//!   shape restrictions.
//! - Backward data: `ALGO_1` (`IMPLICIT_PRECOMP_GEMM`) — universal
//!   coverage baseline.
//! - Backward filter: `ALGO_1` (`IMPLICIT_PRECOMP_GEMM`) — universal
//!   coverage baseline.
//!
//! Heuristic algorithm search via
//! `cudnnGetConvolutionForwardAlgorithm_v7` is a perf-tuning
//! follow-up — the fixed-algo path covers correctness today.
//!
//! ## Handle ownership
//!
//! Each plan lazily owns one `cudnnHandle_t` in a `Cell<>` (created
//! on first `run`; bound to the caller's stream on every launch so
//! the plan is reusable across streams). cuDNN handles are
//! **not** thread-safe — the plan is `!Sync` / `!Send` by virtue
//! of the `Cell<cudnnHandle_t>` it holds. The handle and all
//! descriptors are released in `Drop`.
//!
//! ## Workspace
//!
//! Workspace is **caller-provided** (`Workspace::Borrowed`). The
//! plan reports the required byte count through
//! `*_workspace_size()` accessors, which reflect the cached size
//! after a `query_*_workspace_size(stream)` call. Each direction
//! has its own workspace requirement; callers running multiple
//! directions over the same plan should request the max across
//! directions when sizing the workspace buffer.
//!
//! ## PyTorch semantics
//!
//! PyTorch's `torch.nn.Conv2d` is mathematically **cross-correlation**
//! (kernel applied directly, not flipped). The plan layer always
//! sets cuDNN's mode to `CUDNN_CROSS_CORRELATION` to match.
//!
//! ## Dtype coverage
//!
//! `f32`, `f64`, `f16`, `bf16` — the four cuDNN-supported FP types
//! for convolution. The compute (accumulator) type is `f32` for
//! `f32` / `f16` / `bf16` and `f64` for `f64`, matching cuDNN's
//! mixed-precision conventions.

pub mod conv2d;

pub use conv2d::{Conv2dArgs, Conv2dBwArgs, Conv2dDescriptor, Conv2dDwArgs, Conv2dPlan};
