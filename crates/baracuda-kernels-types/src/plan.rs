//! Device-coupled plan handle: the borrowed [`Workspace`] scratch buffer.
//!
//! The pure-data plan *descriptors* ([`PlanPreference`], [`PrecisionGuarantee`])
//! are driver-free and live in [`unpopped_vocab::plan`]; they are
//! re-exported here so the `baracuda_kernels_types::plan::{PlanPreference,
//! PrecisionGuarantee}` paths are unchanged. Only [`Workspace`] — which borrows
//! a device [`DeviceSliceMut`](baracuda_driver::DeviceSliceMut) — needs the
//! driver, so it stays in this crate.

use baracuda_driver::DeviceSliceMut;

pub use unpopped_vocab::{PlanPreference, PrecisionGuarantee};

/// Caller-supplied workspace for a launch.
///
/// Plans never own device memory in baracuda — pass scratch in at
/// `run` time. Pass [`Workspace::None`] for plans whose
/// workspace size is zero.
///
/// **Intentionally NOT `#[non_exhaustive]`** — the two-variant
/// `None` / `Borrowed` split is hot-path-matched by every plan's
/// `run` method, and the API has been stable through 27 alphas. If
/// a third variant (pool-backed, per-stream-cached) ever lands it
/// will be a deliberate breaking change with a major-version bump.
#[derive(Debug)]
pub enum Workspace<'a> {
    /// No workspace (only valid when the plan reports zero bytes needed).
    None,
    /// Borrowed device scratch. Length must be at least the plan's
    /// reported workspace size.
    Borrowed(DeviceSliceMut<'a, u8>),
}
