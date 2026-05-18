//! FractionalMaxPool3d — stub plan (bespoke kernel required).
//!
//! See [`super::fractional_max_pool2d`] for the rationale. Same gap;
//! `select()` always returns [`Error::Unsupported`].

use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{Element, ElementKind, PlanPreference};

/// Descriptor for `FractionalMaxPool3d` — staging-only; the plan
/// rejects every descriptor today.
#[derive(Copy, Clone, Debug)]
pub struct FractionalMaxPool3dDescriptor {
    /// Batch `N`.
    pub batch: i32,
    /// Channels `C`.
    pub channels: i32,
    /// Input depth.
    pub d_in: i32,
    /// Input height.
    pub h_in: i32,
    /// Input width.
    pub w_in: i32,
    /// Window depth.
    pub window_d: i32,
    /// Window height.
    pub window_h: i32,
    /// Window width.
    pub window_w: i32,
    /// Desired output depth.
    pub d_out: i32,
    /// Desired output height.
    pub h_out: i32,
    /// Desired output width.
    pub w_out: i32,
    /// PRNG seed.
    pub seed: u64,
    /// Element dtype.
    pub element: ElementKind,
}

/// FractionalMaxPool3d plan (not implemented).
pub struct FractionalMaxPool3dPlan<T: Element> {
    _never: core::convert::Infallible,
    _marker: PhantomData<T>,
}

impl<T: Element> FractionalMaxPool3dPlan<T> {
    /// Always returns `Error::Unsupported` — see module docs.
    pub fn select(
        _stream: &Stream,
        _desc: &FractionalMaxPool3dDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        // TODO(phase-12): bespoke 3-D sampler kernel. See
        // `super::fractional_max_pool2d` for the rationale.
        Err(Error::Unsupported(
            "baracuda-kernels::FractionalMaxPool3dPlan: bespoke kernel not yet \
             implemented — see super::fractional_max_pool2d for details.",
        ))
    }
}
