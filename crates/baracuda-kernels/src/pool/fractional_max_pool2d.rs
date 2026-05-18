//! FractionalMaxPool2d — stub plan (bespoke kernel required).
//!
//! PyTorch's `nn.FractionalMaxPool2d` uses pseudorandom per-output-cell
//! sampling to decide which input window to pool from. cuDNN does not
//! ship an equivalent operator and the kernel can't be approximated by
//! a uniform-window cuDNN pool (the whole point is *non-uniform*
//! sampling).
//!
//! This plan currently rejects `select()` with [`Error::Unsupported`].
//! A bespoke `.cu` kernel that mirrors PyTorch's sampling is required
//! to wire this up; file a feature request if needed for production
//! work.

use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{Element, ElementKind, PlanPreference};

/// Descriptor for `FractionalMaxPool2d` — kept around so callers can
/// stage the plan-builder code in advance of the bespoke kernel landing.
///
/// **Currently unused** — `select()` returns `Error::Unsupported` for
/// every descriptor.
#[derive(Copy, Clone, Debug)]
pub struct FractionalMaxPool2dDescriptor {
    /// Batch `N`.
    pub batch: i32,
    /// Channels `C`.
    pub channels: i32,
    /// Input height.
    pub h_in: i32,
    /// Input width.
    pub w_in: i32,
    /// Window height.
    pub window_h: i32,
    /// Window width.
    pub window_w: i32,
    /// Desired output height.
    pub h_out: i32,
    /// Desired output width.
    pub w_out: i32,
    /// PRNG seed for the per-cell sampling.
    pub seed: u64,
    /// Element dtype.
    pub element: ElementKind,
}

/// FractionalMaxPool2d plan (not implemented).
pub struct FractionalMaxPool2dPlan<T: Element> {
    _never: core::convert::Infallible,
    _marker: PhantomData<T>,
}

impl<T: Element> FractionalMaxPool2dPlan<T> {
    /// Always returns `Error::Unsupported` — see module docs.
    pub fn select(
        _stream: &Stream,
        _desc: &FractionalMaxPool2dDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        // TODO(phase-12): implement a bespoke kernel mirroring PyTorch's
        // pseudorandom per-output-cell sampling. Currently the cuDNN
        // pool API has no equivalent. Tracking issue: Fuel feedback #9.
        Err(Error::Unsupported(
            "baracuda-kernels::FractionalMaxPool2dPlan: bespoke kernel not yet \
             implemented — cuDNN has no equivalent, and the uniform-window cuDNN \
             approximation used for adaptive pools doesn't apply to the \
             fractional-sampling case. File a feature request to prioritize \
             this op.",
        ))
    }
}
