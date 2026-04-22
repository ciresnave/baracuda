//! Stream-ordered memory pools (CUDA 11.2+).
//!
//! A [`MemoryPool`] is a GPU-backed allocator with a release threshold:
//! `cuMemAllocFromPoolAsync` allocates out of the pool on a specific stream,
//! `cuMemFreeAsync` returns memory to the pool, and the pool holds on to
//! returned blocks up to a user-configured "keep" threshold so that the
//! next same-size allocation is cheap. This is the backbone of framework
//! memory allocators (PyTorch caching allocator, JAX, TensorFlow).
//!
//! Every device has a default pool accessible via [`default_pool`]. You
//! can also create independent pools via [`MemoryPool::new`].

use core::ffi::c_void;
use std::sync::Arc;

use baracuda_cuda_sys::types::{
    CUmemAccessDesc, CUmemAllocationHandleType, CUmemAllocationType, CUmemLocation,
    CUmemLocationType, CUmemPoolProps, CUmemPoolPtrExportData, CUmemPool_attribute,
};
use baracuda_cuda_sys::{driver, CUdeviceptr, CUmemoryPool};

use crate::context::Context;
use crate::device::Device;
use crate::error::{check, Result};
use crate::stream::Stream;
use crate::vmm::AccessFlags;

/// A CUDA memory pool. Dropping the handle calls `cuMemPoolDestroy` (the
/// default per-device pool is *not* owned by this type — see [`default_pool`]
/// vs [`MemoryPool::new`]).
#[derive(Clone)]
pub struct MemoryPool {
    inner: Arc<MemoryPoolInner>,
}

struct MemoryPoolInner {
    handle: CUmemoryPool,
    owned: bool,
    #[allow(dead_code)]
    context: Context,
}

unsafe impl Send for MemoryPoolInner {}
unsafe impl Sync for MemoryPoolInner {}

impl core::fmt::Debug for MemoryPoolInner {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MemoryPool")
            .field("handle", &self.handle)
            .field("owned", &self.owned)
            .finish_non_exhaustive()
    }
}

impl core::fmt::Debug for MemoryPool {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.inner.fmt(f)
    }
}

impl MemoryPool {
    /// Create a fresh pool whose backing memory lives on `device`. The pool
    /// is destroyed when the last `MemoryPool` clone drops.
    pub fn new(context: &Context, device: &Device) -> Result<Self> {
        context.set_current()?;
        let d = driver()?;
        let cu = d.cu_mem_pool_create()?;
        let props = CUmemPoolProps {
            alloc_type: CUmemAllocationType::PINNED,
            handle_types: CUmemAllocationHandleType::NONE,
            location: CUmemLocation {
                type_: CUmemLocationType::DEVICE,
                id: device.as_raw().0,
            },
            ..Default::default()
        };
        let mut handle: CUmemoryPool = core::ptr::null_mut();
        check(unsafe { cu(&mut handle, &props) })?;
        Ok(Self {
            inner: Arc::new(MemoryPoolInner {
                handle,
                owned: true,
                context: context.clone(),
            }),
        })
    }

    /// Wrap a raw pool handle without taking ownership. Drop is a no-op.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `CUmemoryPool`. The caller guarantees it
    /// outlives this wrapper.
    pub unsafe fn from_borrowed(context: &Context, handle: CUmemoryPool) -> Self {
        Self {
            inner: Arc::new(MemoryPoolInner {
                handle,
                owned: false,
                context: context.clone(),
            }),
        }
    }

    /// Raw `CUmemoryPool`. Use with care.
    #[inline]
    pub fn as_raw(&self) -> CUmemoryPool {
        self.inner.handle
    }

    /// `u64` release threshold — bytes above which the pool starts returning
    /// memory to the OS. Default is 0 (aggressive release).
    pub fn set_release_threshold(&self, bytes: u64) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_mem_pool_set_attribute()?;
        let mut v = bytes;
        check(unsafe {
            cu(
                self.inner.handle,
                CUmemPool_attribute::RELEASE_THRESHOLD,
                &mut v as *mut u64 as *mut c_void,
            )
        })
    }

    pub fn release_threshold(&self) -> Result<u64> {
        let d = driver()?;
        let cu = d.cu_mem_pool_get_attribute()?;
        let mut v: u64 = 0;
        check(unsafe {
            cu(
                self.inner.handle,
                CUmemPool_attribute::RELEASE_THRESHOLD,
                &mut v as *mut u64 as *mut c_void,
            )
        })?;
        Ok(v)
    }

    /// Current bytes handed out to allocations.
    pub fn used_bytes(&self) -> Result<u64> {
        let d = driver()?;
        let cu = d.cu_mem_pool_get_attribute()?;
        let mut v: u64 = 0;
        check(unsafe {
            cu(
                self.inner.handle,
                CUmemPool_attribute::USED_MEM_CURRENT,
                &mut v as *mut u64 as *mut c_void,
            )
        })?;
        Ok(v)
    }

    /// Current bytes reserved for the pool (used + free-but-kept).
    pub fn reserved_bytes(&self) -> Result<u64> {
        let d = driver()?;
        let cu = d.cu_mem_pool_get_attribute()?;
        let mut v: u64 = 0;
        check(unsafe {
            cu(
                self.inner.handle,
                CUmemPool_attribute::RESERVED_MEM_CURRENT,
                &mut v as *mut u64 as *mut c_void,
            )
        })?;
        Ok(v)
    }

    /// Release memory down to `min_bytes_to_keep` bytes.
    pub fn trim_to(&self, min_bytes_to_keep: usize) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_mem_pool_trim_to()?;
        check(unsafe { cu(self.inner.handle, min_bytes_to_keep) })
    }

    /// Grant `device` the specified access to allocations from this pool.
    /// Required for peer-access patterns.
    pub fn set_access(&self, device: &Device, flags: AccessFlags) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_mem_pool_set_access()?;
        let desc = CUmemAccessDesc {
            location: CUmemLocation {
                type_: CUmemLocationType::DEVICE,
                id: device.as_raw().0,
            },
            flags: flags.raw(),
        };
        check(unsafe { cu(self.inner.handle, &desc, 1) })
    }

    /// Query `device`'s access flags for this pool.
    pub fn access(&self, device: &Device) -> Result<AccessFlags> {
        let d = driver()?;
        let cu = d.cu_mem_pool_get_access()?;
        let mut loc = CUmemLocation {
            type_: CUmemLocationType::DEVICE,
            id: device.as_raw().0,
        };
        let mut flags: core::ffi::c_int = 0;
        check(unsafe { cu(&mut flags, self.inner.handle, &mut loc) })?;
        Ok(AccessFlags::from_raw(flags))
    }

    /// Allocate `bytes` bytes of device memory from this pool, ordered on
    /// `stream`. Free via [`crate::DeviceBuffer::free_async`] or by letting
    /// the returned buffer drop (sync free).
    pub fn alloc_async(&self, bytes: usize, stream: &Stream) -> Result<CUdeviceptr> {
        let d = driver()?;
        let cu = d.cu_mem_alloc_from_pool_async()?;
        let mut ptr = CUdeviceptr(0);
        check(unsafe { cu(&mut ptr, bytes, self.inner.handle, stream.as_raw()) })?;
        Ok(ptr)
    }

    /// Export an opaque blob that a peer process can import via
    /// [`MemoryPool::import_pointer`]. Both ends must share the same pool
    /// via its shareable-handle mechanism first (see
    /// [`MemoryPool::export_to_shareable_handle`]).
    pub fn export_pointer(&self, ptr: CUdeviceptr) -> Result<CUmemPoolPtrExportData> {
        let d = driver()?;
        let cu = d.cu_mem_pool_export_pointer()?;
        let mut data = CUmemPoolPtrExportData::default();
        check(unsafe { cu(&mut data, ptr) })?;
        Ok(data)
    }

    /// Inverse of [`MemoryPool::export_pointer`]: resolve the exported blob
    /// to a device pointer valid in the importing process.
    pub fn import_pointer(&self, mut data: CUmemPoolPtrExportData) -> Result<CUdeviceptr> {
        let d = driver()?;
        let cu = d.cu_mem_pool_import_pointer()?;
        let mut ptr = CUdeviceptr(0);
        check(unsafe { cu(&mut ptr, self.inner.handle, &mut data) })?;
        Ok(ptr)
    }
}

impl AccessFlags {
    #[inline]
    fn from_raw(raw: core::ffi::c_int) -> Self {
        use baracuda_cuda_sys::types::CUmemAccess_flags;
        match raw {
            x if x == CUmemAccess_flags::READ => AccessFlags::Read,
            x if x == CUmemAccess_flags::READWRITE => AccessFlags::ReadWrite,
            _ => AccessFlags::None,
        }
    }
}

impl Drop for MemoryPoolInner {
    fn drop(&mut self) {
        if !self.owned || self.handle.is_null() {
            return;
        }
        if let Ok(d) = driver() {
            if let Ok(cu) = d.cu_mem_pool_destroy() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}

/// Return the device's default memory pool — shared across the process,
/// not owned by the caller.
pub fn default_pool(context: &Context, device: &Device) -> Result<MemoryPool> {
    context.set_current()?;
    let d = driver()?;
    let cu = d.cu_device_get_default_mem_pool()?;
    let mut handle: CUmemoryPool = core::ptr::null_mut();
    check(unsafe { cu(&mut handle, device.as_raw()) })?;
    // SAFETY: returned handle is owned by the driver, not us — mark
    // non-owning so our Drop doesn't try to destroy it.
    Ok(unsafe { MemoryPool::from_borrowed(context, handle) })
}

/// Return the current memory pool that `cuMemAllocAsync` uses on `device`
/// (defaults to the default pool unless changed via [`set_current_pool`]).
pub fn current_pool(context: &Context, device: &Device) -> Result<MemoryPool> {
    context.set_current()?;
    let d = driver()?;
    let cu = d.cu_device_get_mem_pool()?;
    let mut handle: CUmemoryPool = core::ptr::null_mut();
    check(unsafe { cu(&mut handle, device.as_raw()) })?;
    Ok(unsafe { MemoryPool::from_borrowed(context, handle) })
}

/// Replace the pool used by `cuMemAllocAsync` on `device`. Pool must
/// outlive all async allocations it services.
pub fn set_current_pool(device: &Device, pool: &MemoryPool) -> Result<()> {
    let d = driver()?;
    let cu = d.cu_device_set_mem_pool()?;
    check(unsafe { cu(device.as_raw(), pool.as_raw()) })
}
