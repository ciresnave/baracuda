//! Convolution op family — Phase 7 trailblazer + Phase 11.7 fanout
//! (Category Convolution / I).
//!
//! Wraps cuDNN's legacy descriptor-based convolution API for the
//! forward pass plus the two backward passes (data gradient + filter
//! gradient). The trailblazer was 2-D NCHW dense convolution; the
//! Phase 11.7 sweep adds 1-D / 3-D / transposed (1d/2d/3d) and lifts
//! the `groups == 1` restriction on [`Conv2dPlan`] so the same plan
//! covers grouped / depthwise / dense conv via cuDNN's
//! `cudnnSetConvolutionGroupCount`.
//!
//! ## Plan layout
//!
//! Each plan owns one cuDNN handle plus four lazy descriptors (`x`,
//! `w`, `y`, `conv`) created on first `run` and reused across launches.
//! Each plan also owns workspace-size caches — one per direction (FW /
//! BW data / BW filter), populated lazily on first call to the
//! corresponding `query_*_workspace_size` accessor.
//!
//! Plans:
//!
//! - [`Conv1dPlan`] / [`Conv2dPlan`] / [`Conv3dPlan`] — dense forward
//!   conv, FW + BW data + BW filter. Conv2d covers depthwise via the
//!   `groups` field on the descriptor (`groups == c_in`).
//! - [`ConvTranspose1dPlan`] / [`ConvTranspose2dPlan`] /
//!   [`ConvTranspose3dPlan`] — transposed ("deconv") conv, FW + BW
//!   data + BW filter. cuDNN has no direct transpose entry point;
//!   these plans dispatch through `cudnnConvolutionBackwardData` /
//!   `cudnnConvolutionForward` with the input/output roles swapped,
//!   as documented in each plan's rustdoc.
//!
//! ## Algorithm selection
//!
//! All plans pin the universal-coverage baselines:
//!
//! - Forward: `IMPLICIT_GEMM` (algo `0`).
//! - Backward data: `ALGO_1` (`IMPLICIT_PRECOMP_GEMM`).
//! - Backward filter: `ALGO_1`.
//!
//! Heuristic algorithm search via
//! `cudnnGetConvolutionForwardAlgorithm_v7` is a perf-tuning follow-up.
//!
//! ## Handle ownership
//!
//! Each plan lazily owns one `cudnnHandle_t` in a `Cell<>` (created
//! on first `run`; bound to the caller's stream on every launch).
//! cuDNN handles are **not** thread-safe — every plan is `!Sync` /
//! `!Send` by virtue of the `Cell<cudnnHandle_t>` it holds. The handle
//! and all descriptors are released in `Drop`.
//!
//! ## Workspace
//!
//! Workspace is **caller-provided** (`Workspace::Borrowed`). Each
//! plan reports the required byte count through `*_workspace_size()`
//! accessors, which reflect the cached size after a
//! `query_*_workspace_size(stream)` call. Each direction has its own
//! workspace requirement; callers running multiple directions over
//! the same plan should request the max across directions.
//!
//! ## PyTorch semantics
//!
//! PyTorch's `torch.nn.Conv{1,2,3}d` is mathematically
//! **cross-correlation** (kernel applied directly, not flipped). All
//! plans set cuDNN's mode to `CUDNN_CROSS_CORRELATION`. Tensor /
//! filter shapes follow PyTorch:
//!
//! - Dense conv input: `[N, C_in, ...]`. Filter:
//!   `[C_out, C_in / groups, ...]`.
//! - Transposed conv input: `[N, C_in, ...]`. Filter:
//!   `[C_in, C_out / groups, ...]` (note: c_in is leading for
//!   transpose, opposite of dense conv).
//!
//! ## Dtype coverage
//!
//! `f32`, `f64`, `f16`, `bf16` — the four cuDNN-supported FP types
//! for convolution. Accumulator is `f32` for `f32` / `f16` / `bf16`
//! and `f64` for `f64`.

pub mod conv1d;
pub mod conv2d;
pub mod conv3d;
pub mod conv_transpose1d;
pub mod conv_transpose2d;
pub mod conv_transpose3d;

// Phase 19.3 — bespoke im2col / im2col1d / col2im1d (sliding-window
// unfold + 1-D inverse). No cuDNN dependency; building blocks for
// Fuel's conv-via-im2col-and-GEMM lowering.
pub mod col2im1d;
pub mod im2col;
pub mod im2col1d;

pub use conv1d::{Conv1dArgs, Conv1dBwArgs, Conv1dDescriptor, Conv1dDwArgs, Conv1dPlan};
pub use conv2d::{Conv2dArgs, Conv2dBwArgs, Conv2dDescriptor, Conv2dDwArgs, Conv2dPlan};
pub use conv3d::{Conv3dArgs, Conv3dBwArgs, Conv3dDescriptor, Conv3dDwArgs, Conv3dPlan};
pub use conv_transpose1d::{
    ConvTranspose1dArgs, ConvTranspose1dBwArgs, ConvTranspose1dDescriptor,
    ConvTranspose1dDwArgs, ConvTranspose1dPlan,
};
pub use conv_transpose2d::{
    ConvTranspose2dArgs, ConvTranspose2dBwArgs, ConvTranspose2dDescriptor,
    ConvTranspose2dDwArgs, ConvTranspose2dPlan,
};
pub use conv_transpose3d::{
    ConvTranspose3dArgs, ConvTranspose3dBwArgs, ConvTranspose3dDescriptor,
    ConvTranspose3dDwArgs, ConvTranspose3dPlan,
};

pub use col2im1d::{Col2Im1dArgs, Col2Im1dDescriptor, Col2Im1dPlan};
pub use im2col::{Im2ColArgs, Im2ColDescriptor, Im2ColPlan};
pub use im2col1d::{Im2Col1dArgs, Im2Col1dDescriptor, Im2Col1dPlan};
