//! Multicast objects (CUDA 12.0+, NVSwitch systems only).
//!
//! A multicast object aliases one virtual-memory range across a group of
//! peer devices so that a write from any one of them fans out to all of
//! them. Requires NVSwitch fabric (HGX, DGX H100) — on a single-GPU
//! system most of these calls will return `CUDA_ERROR_NOT_SUPPORTED`.
//!
//! Typical flow:
//!
//! 1. [`multicast_granularity`] → round your allocation size to a valid
//!    multicast chunk.
//! 2. [`MulticastObject::new`] → create the object.
//! 3. Call [`MulticastObject::add_device`] for each participating device.
//! 4. Bind VMM allocations with [`MulticastObject::bind_addr`] (or
//!    [`bind_mem`](MulticastObject::bind_mem)).
//! 5. Kernels on any device write into the multicast range and every
//!    bound allocation receives the update.

use std::sync::Arc;

use baracuda_cuda_sys::types::{
    CUmemAllocationHandleType, CUmulticastGranularity_flags, CUmulticastObjectProp,
};
use baracuda_cuda_sys::{driver, CUdeviceptr, CUmemGenericAllocationHandle};

use crate::device::Device;
use crate::error::{check, Result};

/// Query the minimum or recommended multicast granularity (bytes) for the
/// given number of peer devices and `size_bytes`.
pub fn multicast_granularity(
    num_devices: u32,
    size_bytes: usize,
    recommended: bool,
) -> Result<usize> {
    let d = driver()?;
    let cu = d.cu_multicast_get_granularity()?;
    let prop = CUmulticastObjectProp {
        num_devices,
        size: size_bytes,
        handle_types: CUmemAllocationHandleType::NONE as u64,
        flags: 0,
    };
    let mut g: usize = 0;
    let option = if recommended {
        CUmulticastGranularity_flags::RECOMMENDED
    } else {
        CUmulticastGranularity_flags::MINIMUM
    };
    check(unsafe { cu(&mut g, &prop, option) })?;
    Ok(g)
}

/// A multicast object. Drop releases the underlying
/// `CUmemGenericAllocationHandle` via `cuMemRelease`.
#[derive(Clone)]
pub struct MulticastObject {
    inner: Arc<MulticastObjectInner>,
}

struct MulticastObjectInner {
    handle: CUmemGenericAllocationHandle,
}

unsafe impl Send for MulticastObjectInner {}
unsafe impl Sync for MulticastObjectInner {}

impl core::fmt::Debug for MulticastObjectInner {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MulticastObject")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

impl core::fmt::Debug for MulticastObject {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.inner.fmt(f)
    }
}

impl MulticastObject {
    pub fn new(num_devices: u32, size_bytes: usize) -> Result<Self> {
        let d = driver()?;
        let cu = d.cu_multicast_create()?;
        let prop = CUmulticastObjectProp {
            num_devices,
            size: size_bytes,
            handle_types: CUmemAllocationHandleType::NONE as u64,
            flags: 0,
        };
        let mut handle: CUmemGenericAllocationHandle = 0;
        check(unsafe { cu(&mut handle, &prop) })?;
        Ok(Self {
            inner: Arc::new(MulticastObjectInner { handle }),
        })
    }

    pub fn add_device(&self, device: &Device) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_multicast_add_device()?;
        check(unsafe { cu(self.inner.handle, device.as_raw()) })
    }

    /// Bind a VMM allocation handle at `mc_offset` → `(mem_handle, mem_offset, size)`.
    pub fn bind_mem(
        &self,
        mc_offset: usize,
        mem_handle: CUmemGenericAllocationHandle,
        mem_offset: usize,
        size: usize,
    ) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_multicast_bind_mem()?;
        check(unsafe {
            cu(
                self.inner.handle,
                mc_offset,
                mem_handle,
                mem_offset,
                size,
                0,
            )
        })
    }

    /// Bind an already-mapped device pointer into the multicast object.
    pub fn bind_addr(&self, mc_offset: usize, ptr: CUdeviceptr, size: usize) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_multicast_bind_addr()?;
        check(unsafe { cu(self.inner.handle, mc_offset, ptr, size, 0) })
    }

    pub fn unbind(&self, device: &Device, mc_offset: usize, size: usize) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_multicast_unbind()?;
        check(unsafe { cu(self.inner.handle, device.as_raw(), mc_offset, size) })
    }

    #[inline]
    pub fn as_raw(&self) -> CUmemGenericAllocationHandle {
        self.inner.handle
    }
}

impl Drop for MulticastObjectInner {
    fn drop(&mut self) {
        if self.handle == 0 {
            return;
        }
        if let Ok(d) = driver() {
            if let Ok(cu) = d.cu_mem_release() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}
