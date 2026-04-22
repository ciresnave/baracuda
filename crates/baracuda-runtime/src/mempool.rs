//! Stream-ordered memory pools (Runtime API, CUDA 11.2+).
//!
//! Mirrors [`baracuda_driver::mempool`] — a pool is a device-backed
//! allocator with a configurable release threshold, accessed via
//! `cudaMallocFromPoolAsync` and returned with `cudaFreeAsync`.
//! Each device exposes a default pool via [`default_pool`].

use std::sync::Arc;

use baracuda_cuda_sys::runtime::runtime;
use baracuda_cuda_sys::runtime::types::{
    cudaMemAccessDesc, cudaMemAllocationHandleType, cudaMemAllocationType, cudaMemLocation,
    cudaMemLocationType, cudaMemPoolAttr, cudaMemPoolProps, cudaMemPoolPtrExportData,
    cudaMemPool_t,
};

use crate::device::Device;
use crate::error::{check, Result};
use crate::stream::Stream;

/// Access rights granted to a device for a pool's allocations.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum AccessFlags {
    None,
    Read,
    ReadWrite,
}

impl AccessFlags {
    #[inline]
    fn raw(self) -> core::ffi::c_int {
        use baracuda_cuda_sys::runtime::types::cudaMemAccessFlags;
        match self {
            AccessFlags::None => cudaMemAccessFlags::NONE,
            AccessFlags::Read => cudaMemAccessFlags::READ,
            AccessFlags::ReadWrite => cudaMemAccessFlags::READ_WRITE,
        }
    }

    #[inline]
    fn from_raw(raw: core::ffi::c_int) -> Self {
        use baracuda_cuda_sys::runtime::types::cudaMemAccessFlags;
        match raw {
            x if x == cudaMemAccessFlags::READ => AccessFlags::Read,
            x if x == cudaMemAccessFlags::READ_WRITE => AccessFlags::ReadWrite,
            _ => AccessFlags::None,
        }
    }
}

/// A memory pool. Owned pools are destroyed on last-clone drop; borrowed
/// pools (returned by [`default_pool`] / [`current_pool`]) are not.
#[derive(Clone)]
pub struct MemoryPool {
    inner: Arc<MemoryPoolInner>,
}

struct MemoryPoolInner {
    handle: cudaMemPool_t,
    owned: bool,
}

unsafe impl Send for MemoryPoolInner {}
unsafe impl Sync for MemoryPoolInner {}

impl core::fmt::Debug for MemoryPoolInner {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MemoryPool")
            .field("handle", &self.handle)
            .field("owned", &self.owned)
            .finish()
    }
}

impl core::fmt::Debug for MemoryPool {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.inner.fmt(f)
    }
}

impl MemoryPool {
    /// Create a fresh pool backed on `device`.
    pub fn new(device: &Device) -> Result<Self> {
        let r = runtime()?;
        let cu = r.cuda_mem_pool_create()?;
        let props = cudaMemPoolProps {
            alloc_type: cudaMemAllocationType::PINNED,
            handle_types: cudaMemAllocationHandleType::NONE,
            location: cudaMemLocation {
                type_: cudaMemLocationType::DEVICE,
                id: device.ordinal(),
            },
            ..Default::default()
        };
        let mut handle: cudaMemPool_t = core::ptr::null_mut();
        check(unsafe { cu(&mut handle, &props) })?;
        Ok(Self {
            inner: Arc::new(MemoryPoolInner {
                handle,
                owned: true,
            }),
        })
    }

    /// Wrap a raw pool handle without taking ownership.
    ///
    /// # Safety
    ///
    /// `handle` must outlive this wrapper.
    pub unsafe fn from_borrowed(handle: cudaMemPool_t) -> Self {
        Self {
            inner: Arc::new(MemoryPoolInner {
                handle,
                owned: false,
            }),
        }
    }

    #[inline]
    pub fn as_raw(&self) -> cudaMemPool_t {
        self.inner.handle
    }

    /// Set the release threshold (bytes retained before the pool starts
    /// returning memory to the OS). Default is 0.
    pub fn set_release_threshold(&self, bytes: u64) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_mem_pool_set_attribute()?;
        let mut v = bytes;
        check(unsafe {
            cu(
                self.inner.handle,
                cudaMemPoolAttr::RELEASE_THRESHOLD,
                &mut v as *mut u64 as *mut core::ffi::c_void,
            )
        })
    }

    pub fn release_threshold(&self) -> Result<u64> {
        self.get_u64_attr(cudaMemPoolAttr::RELEASE_THRESHOLD)
    }

    /// Current bytes handed out to allocations.
    pub fn used_bytes(&self) -> Result<u64> {
        self.get_u64_attr(cudaMemPoolAttr::USED_MEM_CURRENT)
    }

    /// Current bytes reserved for the pool (used + kept-free).
    pub fn reserved_bytes(&self) -> Result<u64> {
        self.get_u64_attr(cudaMemPoolAttr::RESERVED_MEM_CURRENT)
    }

    fn get_u64_attr(&self, attr: i32) -> Result<u64> {
        let r = runtime()?;
        let cu = r.cuda_mem_pool_get_attribute()?;
        let mut v: u64 = 0;
        check(unsafe {
            cu(
                self.inner.handle,
                attr,
                &mut v as *mut u64 as *mut core::ffi::c_void,
            )
        })?;
        Ok(v)
    }

    /// Release memory down to `min_bytes_to_keep`.
    pub fn trim_to(&self, min_bytes_to_keep: usize) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_mem_pool_trim_to()?;
        check(unsafe { cu(self.inner.handle, min_bytes_to_keep) })
    }

    /// Grant `device` the specified access to allocations from this pool.
    pub fn set_access(&self, device: &Device, flags: AccessFlags) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_mem_pool_set_access()?;
        let desc = cudaMemAccessDesc {
            location: cudaMemLocation {
                type_: cudaMemLocationType::DEVICE,
                id: device.ordinal(),
            },
            flags: flags.raw(),
        };
        check(unsafe { cu(self.inner.handle, &desc, 1) })
    }

    /// Query `device`'s access flags for this pool.
    pub fn access(&self, device: &Device) -> Result<AccessFlags> {
        let r = runtime()?;
        let cu = r.cuda_mem_pool_get_access()?;
        let mut loc = cudaMemLocation {
            type_: cudaMemLocationType::DEVICE,
            id: device.ordinal(),
        };
        let mut flags: core::ffi::c_int = 0;
        check(unsafe { cu(&mut flags, self.inner.handle, &mut loc) })?;
        Ok(AccessFlags::from_raw(flags))
    }

    /// Allocate `bytes` bytes of device memory from this pool, ordered on
    /// `stream`. Returns a raw device pointer — free via
    /// [`crate::DeviceBuffer::free_async`] or by calling
    /// [`Self::free_async`] on the raw pointer.
    pub fn alloc_async(&self, bytes: usize, stream: &Stream) -> Result<*mut core::ffi::c_void> {
        let r = runtime()?;
        let cu = r.cuda_malloc_from_pool_async()?;
        let mut ptr: *mut core::ffi::c_void = core::ptr::null_mut();
        check(unsafe { cu(&mut ptr, bytes, self.inner.handle, stream.as_raw()) })?;
        Ok(ptr)
    }

    /// Free a device pointer previously returned by
    /// [`Self::alloc_async`] (routes through `cudaFreeAsync`).
    ///
    /// # Safety
    ///
    /// `ptr` must be a live allocation from this (or another) pool.
    pub unsafe fn free_async(&self, ptr: *mut core::ffi::c_void, stream: &Stream) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_free_async()?;
        check(cu(ptr, stream.as_raw()))
    }

    /// Export a pointer in this pool for sharing with a peer process.
    ///
    /// # Safety
    ///
    /// `ptr` must be a live allocation from this pool.
    pub unsafe fn export_pointer(
        &self,
        ptr: *mut core::ffi::c_void,
    ) -> Result<cudaMemPoolPtrExportData> {
        let r = runtime()?;
        let cu = r.cuda_mem_pool_export_pointer()?;
        let mut data = cudaMemPoolPtrExportData::default();
        check(cu(&mut data, ptr))?;
        Ok(data)
    }

    /// Import an exported pointer into this pool.
    pub fn import_pointer(
        &self,
        mut data: cudaMemPoolPtrExportData,
    ) -> Result<*mut core::ffi::c_void> {
        let r = runtime()?;
        let cu = r.cuda_mem_pool_import_pointer()?;
        let mut ptr: *mut core::ffi::c_void = core::ptr::null_mut();
        check(unsafe { cu(&mut ptr, self.inner.handle, &mut data) })?;
        Ok(ptr)
    }
}

impl Drop for MemoryPoolInner {
    fn drop(&mut self) {
        if !self.owned || self.handle.is_null() {
            return;
        }
        if let Ok(r) = runtime() {
            if let Ok(cu) = r.cuda_mem_pool_destroy() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}

/// Return the device's default memory pool (borrowed — not destroyed on drop).
pub fn default_pool(device: &Device) -> Result<MemoryPool> {
    let r = runtime()?;
    let cu = r.cuda_device_get_default_mem_pool()?;
    let mut handle: cudaMemPool_t = core::ptr::null_mut();
    check(unsafe { cu(&mut handle, device.ordinal()) })?;
    // SAFETY: the runtime owns the default pool; we wrap non-owning.
    Ok(unsafe { MemoryPool::from_borrowed(handle) })
}

/// Return the pool currently used by `cudaMallocAsync` on `device`.
pub fn current_pool(device: &Device) -> Result<MemoryPool> {
    let r = runtime()?;
    let cu = r.cuda_device_get_mem_pool()?;
    let mut handle: cudaMemPool_t = core::ptr::null_mut();
    check(unsafe { cu(&mut handle, device.ordinal()) })?;
    Ok(unsafe { MemoryPool::from_borrowed(handle) })
}

/// Replace the pool used by `cudaMallocAsync` on `device`.
pub fn set_current_pool(device: &Device, pool: &MemoryPool) -> Result<()> {
    let r = runtime()?;
    let cu = r.cuda_device_set_mem_pool()?;
    check(unsafe { cu(device.ordinal(), pool.as_raw()) })
}
