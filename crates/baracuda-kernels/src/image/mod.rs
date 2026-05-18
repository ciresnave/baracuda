//! Image / spatial-transform op family — Category T.
//!
//! Phase 9 of the baracuda-kernels comprehensive plan. Bespoke CUDA
//! kernels for the canonical vision-domain ops:
//!
//! - [`InterpolatePlan`] / [`InterpolateBackwardPlan`] — bilinear-2D
//!   spatial up/downsample (trailblazer). Other modes (nearest,
//!   bicubic, trilinear, linear, area) reserved on the [`ImageKind`]
//!   enum and return `Unsupported`.
//! - [`GridSamplePlan`] / [`GridSampleBackwardPlan`] — sample input
//!   at arbitrary normalized (x, y) coords. PyTorch default config
//!   (`bilinear`, `zeros` pad, `align_corners=false`).
//! - [`AffineGridPlan`] — generate a normalized sampling grid from a
//!   2x3 affine matrix.
//! - [`PixelShufflePlan`] / [`PixelUnshufflePlan`] — pure layout
//!   permutation. Each is the other's backward (memory-bound,
//!   dtype-agnostic; covers f32 / f64 / f16 / bf16).
//! - [`RoiAlignPlan`] / [`RoiAlignBackwardPlan`] — bilinear RoI feature
//!   extraction (PyTorch convention: `sampling_ratio=0` adaptive,
//!   `aligned=false`).
//! - [`RoiPoolPlan`] / [`RoiPoolBackwardPlan`] — max-pool RoI variant.
//!   FW emits an `argmax` buffer that BW reads to route gradient.
//! - [`NmsPlan`] — non-max suppression. Returns a boolean keep mask +
//!   count scalar. No BW (set-valued op).
//!
//! Layout convention: NCHW (matches conv / pool plans).
//!
//! Dtype coverage:
//! - Math-bearing ops (interpolate, grid_sample, affine_grid,
//!   roi_align, roi_pool, nms): `f32, f64`.
//! - Pure-layout ops (pixel_shuffle, pixel_unshuffle): `f32, f64,
//!   f16, bf16`.
//!
//! BW ops that scatter via atomicAdd carry
//! `deterministic == false` / `bit_stable_on_same_hardware == false`
//! on their precision_guarantee.

pub mod affine_grid;
pub mod grid_sample;
pub mod grid_sample_backward;
pub mod interpolate;
pub mod interpolate_backward;
pub mod nms;
pub mod pixel_shuffle;
pub mod pixel_unshuffle;
pub mod roi_align;
pub mod roi_align_backward;
pub mod roi_pool;
pub mod roi_pool_backward;

pub use affine_grid::{AffineGridArgs, AffineGridDescriptor, AffineGridPlan};
pub use grid_sample::{GridSampleArgs, GridSampleDescriptor, GridSamplePlan};
pub use grid_sample_backward::{
    GridSampleBackwardArgs, GridSampleBackwardDescriptor, GridSampleBackwardPlan,
};
pub use interpolate::{
    InterpolateArgs, InterpolateDescriptor, InterpolateMode, InterpolatePlan,
};
pub use interpolate_backward::{
    InterpolateBackwardArgs, InterpolateBackwardDescriptor, InterpolateBackwardPlan,
};
pub use nms::{NmsArgs, NmsDescriptor, NmsPlan};
pub use pixel_shuffle::{PixelShuffleArgs, PixelShuffleDescriptor, PixelShufflePlan};
pub use pixel_unshuffle::{
    PixelUnshuffleArgs, PixelUnshuffleDescriptor, PixelUnshufflePlan,
};
pub use roi_align::{RoiAlignArgs, RoiAlignDescriptor, RoiAlignPlan};
pub use roi_align_backward::{
    RoiAlignBackwardArgs, RoiAlignBackwardDescriptor, RoiAlignBackwardPlan,
};
pub use roi_pool::{RoiPoolArgs, RoiPoolDescriptor, RoiPoolPlan};
pub use roi_pool_backward::{
    RoiPoolBackwardArgs, RoiPoolBackwardDescriptor, RoiPoolBackwardPlan,
};

/// Map an `i32` status from the FFI launcher to a typed `Result`.
/// Shared across the image family (mirrors the convention in
/// `crate::indexing::gather::map_status`).
pub(crate) fn map_status(code: i32) -> baracuda_cutlass::Result<()> {
    use baracuda_cutlass::Error;
    match code {
        0 => Ok(()),
        1 => Err(Error::MisalignedOperand),
        2 => Err(Error::InvalidProblem(
            "baracuda-kernels-sys reported invalid problem",
        )),
        3 => Err(Error::Unsupported(
            "baracuda-kernels-sys reported unsupported configuration",
        )),
        4 => Err(Error::WorkspaceTooSmall { needed: 0, got: 0 }),
        n => Err(Error::CutlassInternal(n)),
    }
}
