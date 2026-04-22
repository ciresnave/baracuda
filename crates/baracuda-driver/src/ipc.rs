//! Inter-Process Communication for CUDA events and allocations.
//!
//! Two producers / two consumers of cross-process state:
//!
//! - Device memory: [`mem_get_handle`] → transmit bytes to another process
//!   → [`mem_open_handle`] gives you a device pointer aliasing the same
//!   physical memory. Close with [`mem_close_handle`].
//! - Events: [`event_get_handle`] / [`event_open_handle`] for cross-process
//!   synchronization.
//!
//! Availability: Linux only in practice. Windows drivers return
//! `CUDA_ERROR_NOT_SUPPORTED`. Use external-memory / semaphore interop
//! for Windows IPC.

use baracuda_cuda_sys::types::{CUipcEventHandle, CUipcMemHandle};
use baracuda_cuda_sys::{driver, CUdeviceptr, CUevent};

use crate::error::{check, Result};
use crate::event::Event;

/// Export a CUDA event for sharing with another process.
pub fn event_get_handle(event: &Event) -> Result<CUipcEventHandle> {
    let d = driver()?;
    let cu = d.cu_ipc_get_event_handle()?;
    let mut h = CUipcEventHandle::default();
    check(unsafe { cu(&mut h, event.as_raw()) })?;
    Ok(h)
}

/// Open a peer-exported event handle. Returns a raw `CUevent`; wrap it
/// with [`Event::from_raw`] if needed.
///
/// Note the handle is passed by value (CUDA's ABI), not by pointer.
pub fn event_open_handle(handle: CUipcEventHandle) -> Result<CUevent> {
    let d = driver()?;
    let cu = d.cu_ipc_open_event_handle()?;
    let mut event: CUevent = core::ptr::null_mut();
    check(unsafe { cu(&mut event, handle) })?;
    Ok(event)
}

/// Export a device allocation for sharing with another process.
pub fn mem_get_handle(dptr: CUdeviceptr) -> Result<CUipcMemHandle> {
    let d = driver()?;
    let cu = d.cu_ipc_get_mem_handle()?;
    let mut h = CUipcMemHandle::default();
    check(unsafe { cu(&mut h, dptr) })?;
    Ok(h)
}

/// Open a peer-exported device-memory handle. Returns a device pointer
/// valid in the *current* context.
pub fn mem_open_handle(handle: CUipcMemHandle, flags: u32) -> Result<CUdeviceptr> {
    let d = driver()?;
    let cu = d.cu_ipc_open_mem_handle()?;
    let mut dptr = CUdeviceptr(0);
    check(unsafe { cu(&mut dptr, handle, flags) })?;
    Ok(dptr)
}

/// Release an imported device pointer previously opened via
/// [`mem_open_handle`].
pub fn mem_close_handle(dptr: CUdeviceptr) -> Result<()> {
    let d = driver()?;
    let cu = d.cu_ipc_close_mem_handle()?;
    check(unsafe { cu(dptr) })
}
