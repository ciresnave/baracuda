//! LPPool1d — stub plan (composite kernel deferred).
//!
//! PyTorch `nn.LPPool1d(p, kernel)` computes
//! `y = (avg_pool(|x|^p))^(1/p)`. Implementable as a composite of
//! existing primitives (`Pow` → `AvgPool1d` → `Pow`), but it requires a
//! parameterized-pow elementwise plan (currently only `Lerp` is wired
//! through `BinaryParamPlan`), plus device scratch for the intermediate
//! `|x|^p` tensor.
//!
//! Deferred to a future milestone — `select()` returns
//! [`Error::Unsupported`]. File a feature request if needed.

use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{Element, ElementKind, PlanPreference};

/// Descriptor for `LPPool1d` — staging-only.
#[derive(Copy, Clone, Debug)]
pub struct LpPool1dDescriptor {
    /// Batch `N`.
    pub batch: i32,
    /// Channels `C`.
    pub channels: i32,
    /// Input length.
    pub l_in: i32,
    /// Pool window length.
    pub window: i32,
    /// Stride.
    pub stride: i32,
    /// Norm exponent `p` (typically 2 for L2-pool).
    pub p: f32,
    /// Element dtype.
    pub element: ElementKind,
}

/// LPPool1d plan (not implemented).
pub struct LpPool1dPlan<T: Element> {
    _never: core::convert::Infallible,
    _marker: PhantomData<T>,
}

impl<T: Element> LpPool1dPlan<T> {
    /// Always returns `Error::Unsupported` — see module docs.
    pub fn select(
        _stream: &Stream,
        _desc: &LpPool1dDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        // TODO(phase-12): wire the composite `pow → avg_pool1d → pow`
        // path once a `UnaryParamPlan::Pow(p)` flavor exists. See Fuel
        // feedback #9.
        Err(Error::Unsupported(
            "baracuda-kernels::LpPool1dPlan: composite kernel deferred — needs a \
             parameterized `Pow(p)` unary plan that's not yet wired. Use \
             `UnaryPlan` (Pow via BinaryPlan with stride-0 scalar broadcast) + \
             `AvgPool1dPlan` manually in the meantime.",
        ))
    }
}
