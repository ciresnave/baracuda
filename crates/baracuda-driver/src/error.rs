//! Error type for `baracuda-driver`.

use baracuda_cuda_sys::{driver, CUresult};

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

/// Look up the symbolic name of a `CUresult` (e.g. `"CUDA_ERROR_OUT_OF_MEMORY"`).
/// Returns `"CUDA_UNKNOWN_ERROR"` if the driver doesn't recognize the code.
pub fn error_name(status: CUresult) -> Result<&'static str> {
    let d = driver()?;
    let cu = d.cu_get_error_name()?;
    let mut p: *const core::ffi::c_char = core::ptr::null();
    check(unsafe { cu(status, &mut p) })?;
    if p.is_null() {
        return Ok("CUDA_UNKNOWN_ERROR");
    }
    Ok(unsafe { core::ffi::CStr::from_ptr(p) }
        .to_str()
        .unwrap_or("CUDA_UNKNOWN_ERROR"))
}

/// Look up the human-readable description of a `CUresult`.
pub fn error_string(status: CUresult) -> Result<&'static str> {
    let d = driver()?;
    let cu = d.cu_get_error_string()?;
    let mut p: *const core::ffi::c_char = core::ptr::null();
    check(unsafe { cu(status, &mut p) })?;
    if p.is_null() {
        return Ok("unknown error");
    }
    Ok(unsafe { core::ffi::CStr::from_ptr(p) }
        .to_str()
        .unwrap_or("unknown error"))
}
