//! External memory / semaphore interop via the Runtime API.
//!
//! Mirrors [`baracuda_driver::external`]. Because `cudaExternalMemory_t`
//! and `CUexternalMemory` are typedef-compatible (same underlying C
//! pointer), the two wrappers are interchangeable at the handle level —
//! this module exists so Runtime-API users don't have to pull in the
//! Driver crate just for external-resource import.
//!
//! Struct layouts (`CUDA_EXTERNAL_MEMORY_HANDLE_DESC` etc.) are shared
//! with the Driver API — populate using the same typed builders in
//! [`baracuda_cuda_sys::types`].

use std::sync::Arc;

use baracuda_cuda_sys::runtime::{cudaExternalMemory_t, cudaExternalSemaphore_t, runtime};
use baracuda_cuda_sys::types::{
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC, CUDA_EXTERNAL_MEMORY_HANDLE_DESC,
    CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC, CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS,
    CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS,
};

use crate::error::{check, Result};
use crate::stream::Stream;

/// An imported external-memory handle (Vulkan `VkDeviceMemory`, D3D12
/// heap / resource, NvSciBuf, DMA-buf FD, ...). Destroyed on drop.
#[derive(Clone)]
pub struct ExternalMemory {
    inner: Arc<ExternalMemoryInner>,
}

struct ExternalMemoryInner {
    handle: cudaExternalMemory_t,
}

unsafe impl Send for ExternalMemoryInner {}
unsafe impl Sync for ExternalMemoryInner {}

impl core::fmt::Debug for ExternalMemoryInner {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ExternalMemory")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

impl core::fmt::Debug for ExternalMemory {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.inner.fmt(f)
    }
}

impl ExternalMemory {
    /// Import an external-memory handle described by `desc`.
    ///
    /// # Safety
    ///
    /// `desc.handle` must describe a live OS object that the process can
    /// access (file descriptor, NT HANDLE, NvSciBufObj, ...). CUDA
    /// retains a reference to the underlying memory until this
    /// `ExternalMemory` drops.
    pub unsafe fn import(desc: &CUDA_EXTERNAL_MEMORY_HANDLE_DESC) -> Result<Self> {
        let r = runtime()?;
        let cu = r.cuda_import_external_memory()?;
        let mut handle: cudaExternalMemory_t = core::ptr::null_mut();
        check(cu(&mut handle, desc))?;
        Ok(Self {
            inner: Arc::new(ExternalMemoryInner { handle }),
        })
    }

    /// Expose a subregion of the imported memory as a device pointer
    /// valid in the *current* CUDA context.
    pub fn mapped_buffer(
        &self,
        offset: u64,
        size: u64,
        flags: u32,
    ) -> Result<*mut core::ffi::c_void> {
        let r = runtime()?;
        let cu = r.cuda_external_memory_get_mapped_buffer()?;
        let desc = CUDA_EXTERNAL_MEMORY_BUFFER_DESC {
            offset,
            size,
            flags,
            reserved: [0; 16],
        };
        let mut ptr: *mut core::ffi::c_void = core::ptr::null_mut();
        check(unsafe { cu(&mut ptr, self.inner.handle, &desc) })?;
        Ok(ptr)
    }

    /// Raw handle. Interchangeable with `CUexternalMemory` at the ABI
    /// level — cast via `as baracuda_cuda_sys::CUexternalMemory` if you
    /// need to interop with the Driver-side wrapper.
    #[inline]
    pub fn as_raw(&self) -> cudaExternalMemory_t {
        self.inner.handle
    }
}

impl Drop for ExternalMemoryInner {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        if let Ok(r) = runtime() {
            if let Ok(cu) = r.cuda_destroy_external_memory() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}

/// An imported external-semaphore handle. Destroyed on drop.
#[derive(Clone)]
pub struct ExternalSemaphore {
    inner: Arc<ExternalSemaphoreInner>,
}

struct ExternalSemaphoreInner {
    handle: cudaExternalSemaphore_t,
}

unsafe impl Send for ExternalSemaphoreInner {}
unsafe impl Sync for ExternalSemaphoreInner {}

impl core::fmt::Debug for ExternalSemaphoreInner {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ExternalSemaphore")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

impl core::fmt::Debug for ExternalSemaphore {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.inner.fmt(f)
    }
}

impl ExternalSemaphore {
    /// Import an external-semaphore handle.
    ///
    /// # Safety
    ///
    /// Same discipline as [`ExternalMemory::import`].
    pub unsafe fn import(desc: &CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC) -> Result<Self> {
        let r = runtime()?;
        let cu = r.cuda_import_external_semaphore()?;
        let mut handle: cudaExternalSemaphore_t = core::ptr::null_mut();
        check(cu(&mut handle, desc))?;
        Ok(Self {
            inner: Arc::new(ExternalSemaphoreInner { handle }),
        })
    }

    /// Enqueue a signal of fence value `value` on `stream`.
    pub fn signal_fence_async(&self, value: u64, stream: &Stream) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_signal_external_semaphores_async()?;
        let params = CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS::fence_value(value);
        check(unsafe { cu(&self.inner.handle, &params, 1, stream.as_raw()) })
    }

    /// Enqueue a wait for fence value `value` on `stream`.
    pub fn wait_fence_async(&self, value: u64, stream: &Stream) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_wait_external_semaphores_async()?;
        let params = CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS::fence_value(value);
        check(unsafe { cu(&self.inner.handle, &params, 1, stream.as_raw()) })
    }

    #[inline]
    pub fn as_raw(&self) -> cudaExternalSemaphore_t {
        self.inner.handle
    }
}

impl Drop for ExternalSemaphoreInner {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        if let Ok(r) = runtime() {
            if let Ok(cu) = r.cuda_destroy_external_semaphore() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}
