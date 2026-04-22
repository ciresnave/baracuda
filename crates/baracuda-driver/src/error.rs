//! Error type for `baracuda-driver`.

use baracuda_cuda_sys::CUresult;

/// A driver-API error: either a non-success `CUresult`, a loader failure, or
/// a feature-not-supported-on-this-driver error.
pub type Error = baracuda_core::Error<CUresult>;

/// Convenient `Result` alias.
pub type Result<T, E = Error> = core::result::Result<T, E>;

/// Turn a raw `CUresult` into `Result<()>`.
#[inline]
pub(crate) fn check(status: CUresult) -> Result<()> {
    Error::check(status)
}
