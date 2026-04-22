//! CUDA GPU core-dump configuration (CUDA 12.1+).
//!
//! Control whether the driver dumps GPU state to disk when a kernel
//! faults. The attribute selectors live in
//! [`baracuda_cuda_sys::types::CUcoredumpSettings`].
//!
//! Each attribute's payload type varies — most are `bool` (represented
//! by a 1-byte `u8` or 4-byte `i32`), a few are strings (path/pipe). We
//! expose raw byte-level accessors and typed shorthands for the common
//! boolean ones.
//!
//! Two sets of functions exist: *context* attributes (per-context
//! settings, set before core dump) and *global* attributes (persist
//! across contexts).

use baracuda_cuda_sys::driver;
use baracuda_cuda_sys::types::CUcoredumpSettings;

use crate::error::{check, Result};

/// Per-context coredump attribute — set a raw byte payload.
///
/// # Safety
///
/// `value` must match the attribute's expected payload (see NVIDIA's
/// core-dump attribute table). Passing the wrong type yields undefined
/// behavior.
pub unsafe fn set_attribute_raw(attr: i32, value: &mut [u8]) -> Result<()> {
    let d = driver()?;
    let cu = d.cu_coredump_set_attribute()?;
    let mut size = value.len();
    check(cu(
        attr,
        value.as_mut_ptr() as *mut core::ffi::c_void,
        &mut size,
    ))
}

/// Per-context coredump attribute — read a raw byte payload.
///
/// # Safety
///
/// `buf` must be sized for the attribute's value type.
pub unsafe fn get_attribute_raw(attr: i32, buf: &mut [u8]) -> Result<usize> {
    let d = driver()?;
    let cu = d.cu_coredump_get_attribute()?;
    let mut size = buf.len();
    check(cu(
        attr,
        buf.as_mut_ptr() as *mut core::ffi::c_void,
        &mut size,
    ))?;
    Ok(size)
}

/// Globally enable/disable core-dump generation on exception.
pub fn set_enable_on_exception(enabled: bool) -> Result<()> {
    let mut v = [if enabled { 1u8 } else { 0u8 }];
    // SAFETY: the attribute expects a 1-byte bool.
    unsafe { set_attribute_raw(CUcoredumpSettings::ENABLE_ON_EXCEPTION, &mut v) }
}

/// Query whether core dumps are currently enabled on exception.
pub fn enable_on_exception() -> Result<bool> {
    let mut v = [0u8; 1];
    unsafe {
        get_attribute_raw(CUcoredumpSettings::ENABLE_ON_EXCEPTION, &mut v)?;
    }
    Ok(v[0] != 0)
}

/// Global (process-wide) variant of [`set_attribute_raw`].
///
/// # Safety
///
/// Same discipline as [`set_attribute_raw`].
pub unsafe fn set_attribute_global_raw(attr: i32, value: &mut [u8]) -> Result<()> {
    let d = driver()?;
    let cu = d.cu_coredump_set_attribute_global()?;
    let mut size = value.len();
    check(cu(
        attr,
        value.as_mut_ptr() as *mut core::ffi::c_void,
        &mut size,
    ))
}

/// Global (process-wide) variant of [`get_attribute_raw`].
///
/// # Safety
///
/// `buf` must be sized for the attribute's value type.
pub unsafe fn get_attribute_global_raw(attr: i32, buf: &mut [u8]) -> Result<usize> {
    let d = driver()?;
    let cu = d.cu_coredump_get_attribute_global()?;
    let mut size = buf.len();
    check(cu(
        attr,
        buf.as_mut_ptr() as *mut core::ffi::c_void,
        &mut size,
    ))?;
    Ok(size)
}

/// `cuFlushGPUDirectRDMAWrites` — flush any outstanding GPUDirect-RDMA
/// writes to the given scope.
pub fn flush_gpudirect_rdma_writes(target: i32, scope: i32) -> Result<()> {
    let d = driver()?;
    let cu = d.cu_flush_gpudirect_rdma_writes()?;
    check(unsafe { cu(target, scope) })
}
