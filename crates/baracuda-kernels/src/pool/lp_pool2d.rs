//! LPPool2d — stub plan (composite kernel deferred).
//!
//! See [`super::lp_pool1d`] for the rationale. PyTorch
//! `nn.LPPool2d(p, kernel)` = `(avg_pool2d(|x|^p))^(1/p)`. Deferred.

use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{Element, ElementKind, PlanPreference};

/// Descriptor for `LPPool2d` — staging-only.
#[derive(Copy, Clone, Debug)]
pub struct LpPool2dDescriptor {
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
    /// Stride along the height axis.
    pub stride_h: i32,
    /// Stride along the width axis.
    pub stride_w: i32,
    /// Norm exponent `p`.
    pub p: f32,
    /// Element dtype.
    pub element: ElementKind,
}

/// LPPool2d plan (not implemented).
pub struct LpPool2dPlan<T: Element> {
    _never: core::convert::Infallible,
    _marker: PhantomData<T>,
}

impl<T: Element> LpPool2dPlan<T> {
    /// Always returns `Error::Unsupported` — see module docs.
    pub fn select(
        _stream: &Stream,
        _desc: &LpPool2dDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        // TODO(phase-12): wire the composite `pow → avg_pool2d → pow`.
        Err(Error::Unsupported(
            "baracuda-kernels::LpPool2dPlan: composite kernel deferred — see \
             super::lp_pool1d for details.",
        ))
    }
}
