//! Multicast objects (CUDA 12.0+).
//!
//! A multicast object is a single VMM handle bound to multiple devices;
//! writes through the handle are implicitly replicated across them.
//! Useful for NVLink-connected GPUs (A100/H100) on all-reduce-like
//! workloads when you don't want to go through NCCL.
//!
//! Not supported on older drivers — returns
//! [`crate::Error::FeatureNotSupported`].

use core::ffi::c_void;

use baracuda_cuda_sys::runtime::runtime;
use baracuda_cuda_sys::runtime::types::cudaMemGenericAllocationHandle_t;
use baracuda_types::{supports, Feature};

use crate::device::Device;
use crate::error::{check, Error, Result};

/// Properties passed to [`MulticastObject::new`]. Layout matches
/// `cudaMulticastObjectProp`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
#[allow(non_camel_case_types)]
pub struct MulticastProp {
    pub num_devices: core::ffi::c_uint,
    pub size: usize,
    pub handle_types: core::ffi::c_int,
    pub flags: u64,
}

fn require_multicast() -> Result<()> {
    let installed = crate::init::driver_version()?;
    if supports(installed, Feature::MulticastObjects) {
        Ok(())
    } else {
        Err(Error::FeatureNotSupported {
            api: "cudaMulticast*",
            since: Feature::MulticastObjects.required_version(),
        })
    }
}

/// A multicast object. Drop releases it via `cudaMemRelease`.
#[derive(Debug)]
pub struct MulticastObject {
    handle: cudaMemGenericAllocationHandle_t,
}

impl MulticastObject {
    /// Create a multicast object with the given props.
    pub fn new(prop: &MulticastProp) -> Result<Self> {
        require_multicast()?;
        let r = runtime()?;
        let cu = r.cuda_multicast_create()?;
        let mut h: cudaMemGenericAllocationHandle_t = 0;
        check(unsafe { cu(&mut h, prop as *const MulticastProp as *const c_void) })?;
        Ok(Self { handle: h })
    }

    #[inline]
    pub fn as_raw(&self) -> cudaMemGenericAllocationHandle_t {
        self.handle
    }

    /// Add a participating device to this object.
    pub fn add_device(&self, device: &Device) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_multicast_add_device()?;
        check(unsafe { cu(self.handle, device.ordinal()) })
    }

    /// Bind a physical-memory handle (from [`crate::vmm::MemHandle`]) at
    /// `mc_offset` within this object, `size` bytes.
    ///
    /// # Safety
    ///
    /// `mem_handle` must be a live VMM allocation on a device that was
    /// already added via [`Self::add_device`].
    pub unsafe fn bind_mem(
        &self,
        mc_offset: usize,
        mem_handle: cudaMemGenericAllocationHandle_t,
        mem_offset: usize,
        size: usize,
        flags: u64,
    ) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_multicast_bind_mem()?;
        check(cu(
            self.handle,
            mc_offset,
            mem_handle,
            mem_offset,
            size,
            flags,
        ))
    }

    /// Bind a device address (instead of a handle).
    ///
    /// # Safety
    ///
    /// `mem_ptr` must be a mapped VMM address on a registered device.
    pub unsafe fn bind_addr(
        &self,
        mc_offset: usize,
        mem_ptr: *mut c_void,
        size: usize,
        flags: u64,
    ) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_multicast_bind_addr()?;
        check(cu(self.handle, mc_offset, mem_ptr, size, flags))
    }

    /// Unbind the region `[mc_offset, mc_offset + size)` from `device`.
    pub fn unbind(&self, device: &Device, mc_offset: usize, size: usize) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_multicast_unbind()?;
        check(unsafe { cu(self.handle, device.ordinal(), mc_offset, size) })
    }
}

impl Drop for MulticastObject {
    fn drop(&mut self) {
        if let Ok(r) = runtime() {
            if let Ok(cu) = r.cuda_mem_release() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}

/// Report the granularity (min alignment / min size) for multicast
/// objects with the given props. `option`: 0 = minimum, 1 = recommended.
pub fn multicast_granularity(prop: &MulticastProp, option: i32) -> Result<usize> {
    require_multicast()?;
    let r = runtime()?;
    let cu = r.cuda_multicast_get_granularity()?;
    let mut g: usize = 0;
    check(unsafe {
        cu(
            &mut g,
            prop as *const MulticastProp as *const c_void,
            option,
        )
    })?;
    Ok(g)
}
