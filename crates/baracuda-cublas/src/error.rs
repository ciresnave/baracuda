//! Error type for `baracuda-cublas`.

use baracuda_cublas_sys::cublasStatus_t;

/// A cuBLAS error: a non-success `cublasStatus_t`, a loader failure, or a
/// feature-not-supported error.
pub type Error = baracuda_core::Error<cublasStatus_t>;

/// Result alias.
pub type Result<T, E = Error> = core::result::Result<T, E>;

/// Turn a raw `cublasStatus_t` into `Result<()>`.
#[inline]
pub(crate) fn check(status: cublasStatus_t) -> Result<()> {
    Error::check(status)
}
