//! Error type for `baracuda-runtime`.

use baracuda_cuda_sys::runtime::cudaError_t;

/// A runtime-API error: a non-success `cudaError_t`, a loader failure, or a
/// feature-not-supported-on-this-driver error.
pub type Error = baracuda_core::Error<cudaError_t>;

/// Convenient `Result` alias.
pub type Result<T, E = Error> = core::result::Result<T, E>;

/// Turn a raw `cudaError_t` into `Result<()>`.
#[inline]
pub(crate) fn check(status: cudaError_t) -> Result<()> {
    Error::check(status)
}
