//! Runtime-API IPC — share events and device allocations between
//! processes. Linux-primary; Windows returns `NOT_SUPPORTED` or
//! similar on most paths.

use core::ffi::c_void;

use baracuda_cuda_sys::runtime::{cudaEvent_t, runtime};
use baracuda_cuda_sys::types::{CUipcEventHandle, CUipcMemHandle};

use crate::error::{check, Result};
use crate::event::Event;

/// Export a CUDA event for sharing with another process.
pub fn event_get_handle(event: &Event) -> Result<CUipcEventHandle> {
    let r = runtime()?;
    let cu = r.cuda_ipc_get_event_handle()?;
    let mut h = CUipcEventHandle::default();
    check(unsafe { cu(&mut h, event.as_raw()) })?;
    Ok(h)
}

/// Open a peer-exported event handle into a raw `cudaEvent_t`.
pub fn event_open_handle(handle: CUipcEventHandle) -> Result<cudaEvent_t> {
    let r = runtime()?;
    let cu = r.cuda_ipc_open_event_handle()?;
    let mut event: cudaEvent_t = core::ptr::null_mut();
    check(unsafe { cu(&mut event, handle) })?;
    Ok(event)
}

/// Export a device allocation's pointer for peer-process import.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn mem_get_handle(dev_ptr: *mut c_void) -> Result<CUipcMemHandle> {
    let r = runtime()?;
    let cu = r.cuda_ipc_get_mem_handle()?;
    let mut h = CUipcMemHandle::default();
    check(unsafe { cu(&mut h, dev_ptr) })?;
    Ok(h)
}

/// Open a peer-exported memory handle. The returned pointer is valid
/// in the *current* process's default device context.
pub fn mem_open_handle(handle: CUipcMemHandle, flags: u32) -> Result<*mut c_void> {
    let r = runtime()?;
    let cu = r.cuda_ipc_open_mem_handle()?;
    let mut ptr: *mut c_void = core::ptr::null_mut();
    check(unsafe { cu(&mut ptr, handle, flags) })?;
    Ok(ptr)
}

/// Release a peer-imported memory handle.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn mem_close_handle(ptr: *mut c_void) -> Result<()> {
    let r = runtime()?;
    let cu = r.cuda_ipc_close_mem_handle()?;
    check(unsafe { cu(ptr) })
}
