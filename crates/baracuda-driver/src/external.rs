//! External memory / semaphore interop — import buffers and sync
//! primitives from Vulkan, D3D11, D3D12, NvSci, and OpaqueFd sources.
//!
//! A typical pipeline is:
//!
//! 1. A graphics API (Vulkan, D3D12) exports a buffer or image and a
//!    timeline fence as OS-level handles (file descriptor on Linux, NT
//!    HANDLE on Windows).
//! 2. CUDA imports those handles via [`ExternalMemory::import`] and
//!    [`ExternalSemaphore::import`].
//! 3. CUDA obtains a device pointer into the shared buffer with
//!    [`ExternalMemory::mapped_buffer`].
//! 4. Each frame, CUDA [`ExternalSemaphore::wait_fence_async`]s on the
//!    graphics-API fence, does compute, then
//!    [`ExternalSemaphore::signal_fence_async`]s a fence value the
//!    graphics API is waiting on.
//!
//! **Testing note:** the baracuda crate ships these APIs but cannot
//! end-to-end test them without a live Vulkan/D3D12 context. Layout and
//! symbol-resolution are verified via unit tests in this module; a
//! matching external-memory/-semaphore example belongs in an
//! examples/external_interop crate (not yet present).

use std::sync::Arc;

use baracuda_cuda_sys::types::{
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC, CUDA_EXTERNAL_MEMORY_HANDLE_DESC,
    CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC, CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS,
    CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS,
};
use baracuda_cuda_sys::{driver, CUdeviceptr, CUexternalMemory, CUexternalSemaphore};

use crate::context::Context;
use crate::error::{check, Result};
use crate::stream::Stream;

/// An imported external-memory handle (Vulkan VkDeviceMemory, D3D12 heap,
/// NvSciBuf, ...). Destroyed on drop.
#[derive(Clone)]
pub struct ExternalMemory {
    inner: Arc<ExternalMemoryInner>,
}

struct ExternalMemoryInner {
    handle: CUexternalMemory,
    #[allow(dead_code)]
    context: Context,
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
    /// `desc` must describe a live OS object that the calling process has
    /// permission to access — a Vulkan-exported file descriptor, a
    /// D3D12-exported NT HANDLE, etc. CUDA retains a reference to the
    /// underlying object until this `ExternalMemory` drops.
    pub unsafe fn import(
        context: &Context,
        desc: &CUDA_EXTERNAL_MEMORY_HANDLE_DESC,
    ) -> Result<Self> {
        context.set_current()?;
        let d = driver()?;
        let cu = d.cu_import_external_memory()?;
        let mut handle: CUexternalMemory = core::ptr::null_mut();
        check(cu(&mut handle, desc))?;
        Ok(Self {
            inner: Arc::new(ExternalMemoryInner {
                handle,
                context: context.clone(),
            }),
        })
    }

    /// Expose a subregion of the external memory as a device pointer.
    /// The returned pointer is valid until this `ExternalMemory` drops.
    pub fn mapped_buffer(&self, offset: u64, size: u64, flags: u32) -> Result<CUdeviceptr> {
        let d = driver()?;
        let cu = d.cu_external_memory_get_mapped_buffer()?;
        let desc = CUDA_EXTERNAL_MEMORY_BUFFER_DESC {
            offset,
            size,
            flags,
            reserved: [0; 16],
        };
        let mut ptr = CUdeviceptr(0);
        check(unsafe { cu(&mut ptr, self.inner.handle, &desc) })?;
        Ok(ptr)
    }

    #[inline]
    pub fn as_raw(&self) -> CUexternalMemory {
        self.inner.handle
    }
}

impl Drop for ExternalMemoryInner {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        if let Ok(d) = driver() {
            if let Ok(cu) = d.cu_destroy_external_memory() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}

/// An imported external-semaphore handle (Vulkan VkSemaphore / timeline,
/// D3D12 fence, NvSciSync, keyed mutex). Destroyed on drop.
#[derive(Clone)]
pub struct ExternalSemaphore {
    inner: Arc<ExternalSemaphoreInner>,
}

struct ExternalSemaphoreInner {
    handle: CUexternalSemaphore,
    #[allow(dead_code)]
    context: Context,
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
    /// Import an external-semaphore handle described by `desc`.
    ///
    /// # Safety
    ///
    /// Same discipline as [`ExternalMemory::import`]: `desc.handle` must
    /// be a live OS object this process may access.
    pub unsafe fn import(
        context: &Context,
        desc: &CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC,
    ) -> Result<Self> {
        context.set_current()?;
        let d = driver()?;
        let cu = d.cu_import_external_semaphore()?;
        let mut handle: CUexternalSemaphore = core::ptr::null_mut();
        check(cu(&mut handle, desc))?;
        Ok(Self {
            inner: Arc::new(ExternalSemaphoreInner {
                handle,
                context: context.clone(),
            }),
        })
    }

    /// Enqueue a signal of fence value `value` on `stream` for timeline /
    /// D3D12 fence semaphores.
    pub fn signal_fence_async(&self, value: u64, stream: &Stream) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_signal_external_semaphores_async()?;
        let params = CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS::fence_value(value);
        check(unsafe { cu(&self.inner.handle, &params, 1, stream.as_raw()) })
    }

    /// Enqueue a wait for fence value `value` on `stream`. The stream
    /// blocks (on-device) until the external fence reaches that value.
    pub fn wait_fence_async(&self, value: u64, stream: &Stream) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_wait_external_semaphores_async()?;
        let params = CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS::fence_value(value);
        check(unsafe { cu(&self.inner.handle, &params, 1, stream.as_raw()) })
    }

    #[inline]
    pub fn as_raw(&self) -> CUexternalSemaphore {
        self.inner.handle
    }
}

impl Drop for ExternalSemaphoreInner {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        if let Ok(d) = driver() {
            if let Ok(cu) = d.cu_destroy_external_semaphore() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use baracuda_cuda_sys::types::CUexternalMemoryHandleType;

    #[test]
    fn struct_sizes_match_cuda_abi() {
        // Catches accidental layout drift — these constants come from the
        // CUDA 13.0 header. If they ever change, this test fires before
        // anyone uses the FFI.
        use core::mem::size_of;
        assert_eq!(size_of::<CUDA_EXTERNAL_MEMORY_HANDLE_DESC>(), 104);
        assert_eq!(size_of::<CUDA_EXTERNAL_MEMORY_BUFFER_DESC>(), 88);
        assert_eq!(size_of::<CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC>(), 96);
        // params(72) + flags(4) + reserved[16](64) = 140, padded to 144 for 8-byte alignment.
        assert_eq!(size_of::<CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS>(), 144);
        assert_eq!(size_of::<CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS>(), 144);
    }

    #[test]
    fn handle_desc_builders_encode_fd_and_win32() {
        let d = CUDA_EXTERNAL_MEMORY_HANDLE_DESC::from_fd(42, 1024);
        assert_eq!(d.type_, CUexternalMemoryHandleType::OPAQUE_FD);
        assert_eq!(d.size, 1024);
        // fd sits in the low 4 bytes of handle[0].
        assert_eq!(d.handle[0] as i32, 42);

        let h: *mut core::ffi::c_void = 0xDEAD_BEEF_1234_5678u64 as *mut _;
        let n: *const core::ffi::c_void = 0xAAAA_BBBB_CCCC_DDDDu64 as *const _;
        let d = unsafe {
            CUDA_EXTERNAL_MEMORY_HANDLE_DESC::from_win32_handle(
                CUexternalMemoryHandleType::OPAQUE_WIN32,
                h,
                n,
                2048,
            )
        };
        assert_eq!(d.handle[0], 0xDEAD_BEEF_1234_5678);
        assert_eq!(d.handle[1], 0xAAAA_BBBB_CCCC_DDDD);
        assert_eq!(d.size, 2048);
    }
}
