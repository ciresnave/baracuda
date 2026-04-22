//! Runtime-API initialization helpers.
//!
//! The CUDA Runtime API initializes lazily on first use (typically when you
//! call `cudaSetDevice`), so there's no explicit `init()` you _must_ call.
//! The helpers here exist for fail-fast setup: version queries, `cudaInitDevice`
//! for pre-warming the primary context without making it current, etc.

use baracuda_cuda_sys::runtime::runtime;
use baracuda_types::CudaVersion;

use crate::error::{check, Result};

/// CUDA Runtime version linked via `libcudart`.
pub fn runtime_version() -> Result<CudaVersion> {
    let r = runtime()?;
    let cu = r.cuda_runtime_get_version()?;
    let mut raw: core::ffi::c_int = 0;
    check(unsafe { cu(&mut raw) })?;
    Ok(CudaVersion::from_raw(raw as u32))
}

/// CUDA driver version (latest supported by the installed `libcuda`).
pub fn driver_version() -> Result<CudaVersion> {
    let r = runtime()?;
    let cu = r.cuda_driver_get_version()?;
    let mut raw: core::ffi::c_int = 0;
    check(unsafe { cu(&mut raw) })?;
    Ok(CudaVersion::from_raw(raw as u32))
}

/// Block the calling host thread until all work on the current device has
/// completed. Equivalent to `cudaDeviceSynchronize`.
pub fn device_synchronize() -> Result<()> {
    let r = runtime()?;
    let cu = r.cuda_device_synchronize()?;
    check(unsafe { cu() })
}

/// Retrieve and clear the per-thread "sticky" error from the runtime.
///
/// This is how the Runtime API reports async kernel failures — failed
/// launches are latched into a thread-local slot that persists across
/// unrelated calls until this function (or `cudaPeekAtLastError`) reads it.
pub fn last_error() -> Result<()> {
    let r = runtime()?;
    let cu = r.cuda_get_last_error()?;
    check(unsafe { cu() })
}

/// As [`last_error`] but doesn't clear the sticky slot.
pub fn peek_last_error() -> Result<()> {
    let r = runtime()?;
    let cu = r.cuda_peek_at_last_error()?;
    check(unsafe { cu() })
}

/// Set the process's device-level scheduling/map flags. Typically called
/// before the first CUDA call on the current thread — the flags bind
/// when the primary context is created. Passes are flags from
/// [`baracuda_cuda_sys::runtime::types::cudaDeviceScheduleFlags`].
pub fn set_device_flags(flags: u32) -> Result<()> {
    let r = runtime()?;
    let cu = r.cuda_set_device_flags()?;
    check(unsafe { cu(flags) })
}

/// Query current device-scheduling flags.
pub fn get_device_flags() -> Result<u32> {
    let r = runtime()?;
    let cu = r.cuda_get_device_flags()?;
    let mut flags: core::ffi::c_uint = 0;
    check(unsafe { cu(&mut flags) })?;
    Ok(flags)
}
