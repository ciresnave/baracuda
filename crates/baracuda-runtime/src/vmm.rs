//! Virtual memory management (Runtime API).
//!
//! Mirrors [`baracuda_driver::vmm`]. Workflow:
//!
//! 1. Reserve a VA range: [`address_reserve`].
//! 2. Create a physical allocation: [`MemHandle::new`].
//! 3. Map the allocation onto the VA range: [`map`].
//! 4. Set access rights: [`set_access`].
//! 5. Use the pointer like any other device buffer.
//! 6. [`unmap`], drop the [`MemHandle`], [`address_free`].

use core::ffi::c_void;

use baracuda_cuda_sys::runtime::runtime;
use baracuda_cuda_sys::runtime::types::{
    cudaMemAccessDesc, cudaMemAllocationHandleType, cudaMemAllocationProp, cudaMemAllocationType,
    cudaMemGenericAllocationHandle_t, cudaMemLocation, cudaMemLocationType,
};

use crate::device::Device;
use crate::error::{check, Result};
use crate::mempool::AccessFlags;

/// Reserve a contiguous virtual-address range. Returns a device pointer.
///
/// `size` must be a multiple of the granularity reported by
/// [`allocation_granularity`]. `alignment = 0` lets CUDA pick.
pub fn address_reserve(size: usize, alignment: usize, flags: u64) -> Result<*mut c_void> {
    let r = runtime()?;
    let cu = r.cuda_mem_address_reserve()?;
    let mut ptr: *mut c_void = core::ptr::null_mut();
    check(unsafe { cu(&mut ptr, size, alignment, core::ptr::null_mut(), flags) })?;
    Ok(ptr)
}

/// Release a reserved VA range.
///
/// # Safety
///
/// `ptr` / `size` must match a prior [`address_reserve`].
pub unsafe fn address_free(ptr: *mut c_void, size: usize) -> Result<()> {
    let r = runtime()?;
    let cu = r.cuda_mem_address_free()?;
    check(cu(ptr, size))
}

/// Build the allocation-properties struct for a fresh device-backed
/// VMM allocation.
pub fn device_alloc_prop(device: &Device) -> cudaMemAllocationProp {
    cudaMemAllocationProp {
        alloc_type: cudaMemAllocationType::PINNED,
        requested_handle_types: cudaMemAllocationHandleType::NONE,
        location: cudaMemLocation {
            type_: cudaMemLocationType::DEVICE,
            id: device.ordinal(),
        },
        win32_handle_meta_data: core::ptr::null_mut(),
        allocation_flags: [0; 32],
    }
}

/// Minimum allocation size and VA-alignment granularity for `prop`.
/// `option`: 0 = minimum, 1 = recommended.
pub fn allocation_granularity(prop: &cudaMemAllocationProp, option: i32) -> Result<usize> {
    let r = runtime()?;
    let cu = r.cuda_mem_get_allocation_granularity()?;
    let mut g: usize = 0;
    check(unsafe {
        cu(
            &mut g,
            prop as *const cudaMemAllocationProp as *const c_void,
            option,
        )
    })?;
    Ok(g)
}

/// A physical VMM allocation handle. Drop releases it.
#[derive(Debug)]
pub struct MemHandle {
    handle: cudaMemGenericAllocationHandle_t,
}

impl MemHandle {
    /// Allocate `size` bytes of physical memory with the given props.
    pub fn new(size: usize, prop: &cudaMemAllocationProp, flags: u64) -> Result<Self> {
        let r = runtime()?;
        let cu = r.cuda_mem_create()?;
        let mut h: cudaMemGenericAllocationHandle_t = 0;
        check(unsafe {
            cu(
                &mut h,
                size,
                prop as *const cudaMemAllocationProp as *const c_void,
                flags,
            )
        })?;
        Ok(Self { handle: h })
    }

    /// Wrap an already-created handle.
    ///
    /// # Safety
    ///
    /// `handle` must be live and will be released when this wrapper
    /// drops.
    pub unsafe fn from_raw(handle: cudaMemGenericAllocationHandle_t) -> Self {
        Self { handle }
    }

    #[inline]
    pub fn as_raw(&self) -> cudaMemGenericAllocationHandle_t {
        self.handle
    }

    /// Retain a second reference to the allocation backing `addr`.
    ///
    /// # Safety
    ///
    /// `addr` must be inside a mapped VMM region.
    pub unsafe fn retain(addr: *mut c_void) -> Result<Self> {
        let r = runtime()?;
        let cu = r.cuda_mem_retain_allocation_handle()?;
        let mut h: cudaMemGenericAllocationHandle_t = 0;
        check(cu(&mut h, addr))?;
        Ok(Self { handle: h })
    }

    /// Query the properties of the underlying allocation.
    pub fn properties(&self) -> Result<cudaMemAllocationProp> {
        let r = runtime()?;
        let cu = r.cuda_mem_get_allocation_properties_from_handle()?;
        let mut prop = cudaMemAllocationProp::default();
        check(unsafe {
            cu(
                &mut prop as *mut cudaMemAllocationProp as *mut c_void,
                self.handle,
            )
        })?;
        Ok(prop)
    }
}

impl Drop for MemHandle {
    fn drop(&mut self) {
        if let Ok(r) = runtime() {
            if let Ok(cu) = r.cuda_mem_release() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}

/// Map the physical allocation `handle` onto `[ptr + offset, ptr + offset + size)`.
///
/// # Safety
///
/// `ptr` must come from [`address_reserve`]; `size` + `offset` must fit
/// the reservation and the underlying allocation.
pub unsafe fn map(
    ptr: *mut c_void,
    size: usize,
    offset: usize,
    handle: &MemHandle,
    flags: u64,
) -> Result<()> {
    let r = runtime()?;
    let cu = r.cuda_mem_map()?;
    check(cu(ptr, size, offset, handle.as_raw(), flags))
}

/// Unmap `[ptr, ptr + size)`.
///
/// # Safety
///
/// Must match a prior [`map`] call.
pub unsafe fn unmap(ptr: *mut c_void, size: usize) -> Result<()> {
    let r = runtime()?;
    let cu = r.cuda_mem_unmap()?;
    check(cu(ptr, size))
}

/// Grant the given access rights to a device for a mapped region.
///
/// # Safety
///
/// `ptr` / `size` must cover a mapped region.
pub unsafe fn set_access(
    ptr: *mut c_void,
    size: usize,
    device: &Device,
    flags: AccessFlags,
) -> Result<()> {
    let r = runtime()?;
    let cu = r.cuda_mem_set_access()?;
    // Reuse `AccessFlags`'s internal mapping — matches cudaMemAccessFlags.
    let flags_raw = match flags {
        AccessFlags::None => 0,
        AccessFlags::Read => 1,
        AccessFlags::ReadWrite => 3,
    };
    let desc = cudaMemAccessDesc {
        location: cudaMemLocation {
            type_: cudaMemLocationType::DEVICE,
            id: device.ordinal(),
        },
        flags: flags_raw,
    };
    check(cu(
        ptr,
        size,
        &desc as *const cudaMemAccessDesc as *const c_void,
        1,
    ))
}

/// Query access flags for `device` at `ptr`.
///
/// # Safety
///
/// `ptr` must be inside a mapped region.
pub unsafe fn get_access(ptr: *mut c_void, device: &Device) -> Result<u64> {
    let r = runtime()?;
    let cu = r.cuda_mem_get_access()?;
    let loc = cudaMemLocation {
        type_: cudaMemLocationType::DEVICE,
        id: device.ordinal(),
    };
    let mut flags: u64 = 0;
    check(cu(
        &mut flags,
        &loc as *const cudaMemLocation as *const c_void,
        ptr,
    ))?;
    Ok(flags)
}
