//! `histogramdd` plan — N-D histogram. Reserved for follow-up.
//!
//! The 1-D path lives in [`crate::sort::HistogramPlan`]. This file
//! exists as the public API shape (descriptor / args / plan structs)
//! to keep the surface stable; `select` returns `Unsupported` for the
//! N > 1 case in the trailblazer.

use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    Element, ElementKind, KernelSku, PlanPreference, PrecisionGuarantee, SortKind, TensorMut,
    TensorRef, Workspace,
};

use super::histogram::build_atomic_sku;

/// Descriptor for a `histogramdd` op (reserved).
#[derive(Copy, Clone, Debug)]
pub struct HistogramddDescriptor {
    /// Number of input samples.
    pub numel: i64,
    /// Number of dimensions.
    pub ndim: i32,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a `histogramdd` launch.
pub struct HistogramddArgs<'a, T: Element> {
    /// Input `[numel, ndim]`.
    pub input: TensorRef<'a, T, 2>,
    /// Output `[product(num_bins_per_dim)]` (i32).
    pub output: TensorMut<'a, i32, 1>,
}

/// `histogramdd` plan (reserved — returns `Unsupported`).
///
/// **Status**: API stub. `select()` always returns `Unsupported`
/// in the trailblazer; use [`HistogramPlan`](crate::HistogramPlan)
/// for 1-D histograms today. This file pins the public surface
/// (`Descriptor` / `Args` / `Plan` struct names) so callers can
/// type-check against the eventual N-D path without churn.
///
/// **When the real kernel lands**: PyTorch `torch.histogramdd`
/// shape — input `[numel, ndim]`, output flat
/// `[prod(num_bins_per_dim)]`.
pub struct HistogramddPlan<T: Element> {
    _desc: HistogramddDescriptor,
    _sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> HistogramddPlan<T> {
    /// Pick a kernel for `desc` — returns `Unsupported` in trailblazer.
    pub fn select(
        _stream: &Stream,
        desc: &HistogramddDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::HistogramddPlan: descriptor element != type parameter T",
            ));
        }
        if desc.ndim != 1 {
            return Err(Error::Unsupported(
                "baracuda-kernels::HistogramddPlan: ndim > 1 not supported in the trailblazer \
                 (use HistogramPlan for the 1-D path)",
            ));
        }
        Err(Error::Unsupported(
            "baracuda-kernels::HistogramddPlan: reserved API surface — use HistogramPlan for \
             the 1-D case",
        ))
    }

    /// Workspace size in bytes.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Identity of the kernel this plan picked.
    #[inline]
    pub fn sku(&self) -> KernelSku {
        self._sku
    }

    /// Numerical guarantees for this plan's kernel.
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self._sku.precision_guarantee
    }

    /// Validate args — always returns `Unsupported`.
    pub fn can_implement(&self, _args: &HistogramddArgs<'_, T>) -> Result<()> {
        Err(Error::Unsupported(
            "baracuda-kernels::HistogramddPlan: reserved API surface",
        ))
    }

    /// Launch — always returns `Unsupported`.
    pub fn run(
        &self,
        _stream: &Stream,
        _workspace: Workspace<'_>,
        _args: HistogramddArgs<'_, T>,
    ) -> Result<()> {
        Err(Error::Unsupported(
            "baracuda-kernels::HistogramddPlan: reserved API surface",
        ))
    }
}

// Anchor `build_atomic_sku` for the future N-D path so the import is
// kept warm. (Drop this once the real implementation lands.)
#[allow(dead_code)]
fn _anchor<T: Element>() -> KernelSku {
    build_atomic_sku::<T>(SortKind::Histogramdd)
}
