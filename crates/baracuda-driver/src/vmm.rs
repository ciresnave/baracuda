//! CUDA Virtual Memory Management (VMM) — the fine-grained alternative to
//! `cuMemAlloc`.
//!
//! The Driver-API malloc (`DeviceBuffer`) hides allocation + virtual
//! address mapping behind one call. The VMM API splits them:
//!
//! 1. **Reserve** a virtual address range ([`AddressRange`]).
//! 2. **Create** a physical allocation ([`PhysicalMemory`]).
//! 3. **Map** the physical memory into the reserved address range.
//! 4. **Grant access** to one or more devices.
//! 5. ... use the memory like any device pointer ...
//! 6. On drop: unmap, release physical, free virtual range.
//!
//! This buys three things:
//!
//! - Safe remapping (resize-in-place for tensor libraries).
//! - Explicit peer-access control (per-device `READ` / `READWRITE`).
//! - IPC / external-resource export via `CUmemAllocationHandleType` (future).
//!
//! Availability: CUDA 10.2+, Linux + Windows with WDDM 2.0 driver model
//! (which is all modern NVIDIA Windows setups).

use std::sync::Arc;

use baracuda_cuda_sys::types::{
    CUmemAccessDesc, CUmemAccess_flags, CUmemAllocationGranularity_flags, CUmemAllocationProp,
    CUmemAllocationPropFlags, CUmemAllocationType, CUmemLocation, CUmemLocationType,
};
use baracuda_cuda_sys::{driver, CUdevice, CUdeviceptr, CUmemGenericAllocationHandle};

use crate::context::Context;
use crate::device::Device;
use crate::error::{check, Result};

/// Query the minimum or recommended allocation granularity for a device.
/// VMM allocations must be sized (and address ranges aligned) to this value.
pub fn allocation_granularity(device: &Device, recommended: bool) -> Result<usize> {
    let d = driver()?;
    let cu = d.cu_mem_get_allocation_granularity()?;
    let prop = device_prop(device.as_raw());
    let mut gran: usize = 0;
    let option = if recommended {
        CUmemAllocationGranularity_flags::RECOMMENDED
    } else {
        CUmemAllocationGranularity_flags::MINIMUM
    };
    check(unsafe { cu(&mut gran, &prop, option) })?;
    Ok(gran)
}

fn device_prop(dev: CUdevice) -> CUmemAllocationProp {
    CUmemAllocationProp {
        type_: CUmemAllocationType::PINNED,
        requested_handle_types: 0, // no IPC export
        location: CUmemLocation {
            type_: CUmemLocationType::DEVICE,
            id: dev.0,
        },
        win32_handle_meta_data: core::ptr::null_mut(),
        alloc_flags: CUmemAllocationPropFlags::default(),
    }
}

/// A reserved virtual address range (not yet backed by physical memory).
/// Drops release the VA range via `cuMemAddressFree`.
pub struct AddressRange {
    inner: Arc<AddressRangeInner>,
}

struct AddressRangeInner {
    ptr: CUdeviceptr,
    size: usize,
    #[allow(dead_code)]
    context: Context,
}

unsafe impl Send for AddressRangeInner {}
unsafe impl Sync for AddressRangeInner {}

impl core::fmt::Debug for AddressRangeInner {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AddressRange")
            .field("ptr", &format_args!("{:#x}", self.ptr.0))
            .field("size", &self.size)
            .finish()
    }
}

impl core::fmt::Debug for AddressRange {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.inner.fmt(f)
    }
}

impl Clone for AddressRange {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl AddressRange {
    /// Reserve `size` bytes of device virtual address space.
    pub fn reserve(context: &Context, size: usize, alignment: usize) -> Result<Self> {
        context.set_current()?;
        let d = driver()?;
        let cu = d.cu_mem_address_reserve()?;
        let mut ptr = CUdeviceptr(0);
        check(unsafe { cu(&mut ptr, size, alignment, CUdeviceptr(0), 0) })?;
        Ok(Self {
            inner: Arc::new(AddressRangeInner {
                ptr,
                size,
                context: context.clone(),
            }),
        })
    }

    #[inline]
    pub fn as_raw(&self) -> CUdeviceptr {
        self.inner.ptr
    }
    #[inline]
    pub fn size(&self) -> usize {
        self.inner.size
    }
}

impl Drop for AddressRangeInner {
    fn drop(&mut self) {
        if self.ptr.0 == 0 {
            return;
        }
        if let Ok(d) = driver() {
            if let Ok(cu) = d.cu_mem_address_free() {
                let _ = unsafe { cu(self.ptr, self.size) };
            }
        }
    }
}

/// A physical device-memory allocation. Not usable until mapped into an
/// [`AddressRange`] via [`MappedRange::new`].
pub struct PhysicalMemory {
    inner: Arc<PhysicalMemoryInner>,
}

struct PhysicalMemoryInner {
    handle: CUmemGenericAllocationHandle,
    size: usize,
    device: CUdevice,
    #[allow(dead_code)]
    context: Context,
}

unsafe impl Send for PhysicalMemoryInner {}
unsafe impl Sync for PhysicalMemoryInner {}

impl core::fmt::Debug for PhysicalMemoryInner {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PhysicalMemory")
            .field("handle", &self.handle)
            .field("size", &self.size)
            .field("device", &self.device.0)
            .finish()
    }
}

impl core::fmt::Debug for PhysicalMemory {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.inner.fmt(f)
    }
}

impl Clone for PhysicalMemory {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl PhysicalMemory {
    /// Create a physical allocation of `size` bytes on `device`. `size` must
    /// be a multiple of [`allocation_granularity`].
    pub fn create(context: &Context, device: &Device, size: usize) -> Result<Self> {
        context.set_current()?;
        let d = driver()?;
        let cu = d.cu_mem_create()?;
        let prop = device_prop(device.as_raw());
        let mut handle: CUmemGenericAllocationHandle = 0;
        check(unsafe { cu(&mut handle, size, &prop, 0) })?;
        Ok(Self {
            inner: Arc::new(PhysicalMemoryInner {
                handle,
                size,
                device: device.as_raw(),
                context: context.clone(),
            }),
        })
    }

    #[inline]
    pub fn as_raw(&self) -> CUmemGenericAllocationHandle {
        self.inner.handle
    }
    #[inline]
    pub fn size(&self) -> usize {
        self.inner.size
    }
}

impl Drop for PhysicalMemoryInner {
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

/// Access rights granted to a device by [`MappedRange::set_access`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum AccessFlags {
    None,
    Read,
    ReadWrite,
}

impl AccessFlags {
    #[doc(hidden)]
    #[inline]
    pub fn raw(self) -> core::ffi::c_int {
        match self {
            AccessFlags::None => CUmemAccess_flags::NONE,
            AccessFlags::Read => CUmemAccess_flags::READ,
            AccessFlags::ReadWrite => CUmemAccess_flags::READWRITE,
        }
    }
}

/// An address range with physical backing mapped in. `cuMemUnmap` is called
/// on drop; the underlying [`AddressRange`] and [`PhysicalMemory`] remain
/// live through their own refcounts.
pub struct MappedRange {
    range: AddressRange,
    physical: PhysicalMemory,
    offset: usize,
    size: usize,
}

impl core::fmt::Debug for MappedRange {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MappedRange")
            .field("ptr", &format_args!("{:#x}", self.range.as_raw().0))
            .field("offset", &self.offset)
            .field("size", &self.size)
            .finish()
    }
}

impl MappedRange {
    /// Map `physical` into `range` at `offset` within the range.
    /// `offset + physical.size() <= range.size()`.
    pub fn new(range: &AddressRange, physical: &PhysicalMemory, offset: usize) -> Result<Self> {
        assert!(
            offset + physical.size() <= range.size(),
            "physical size + offset ({} + {}) overflows the reserved range ({})",
            offset,
            physical.size(),
            range.size()
        );
        let d = driver()?;
        let cu = d.cu_mem_map()?;
        let target = CUdeviceptr(range.as_raw().0 + offset as u64);
        check(unsafe { cu(target, physical.size(), 0, physical.as_raw(), 0) })?;
        Ok(Self {
            range: range.clone(),
            physical: physical.clone(),
            offset,
            size: physical.size(),
        })
    }

    /// Grant `flags` access to `device` for this mapping. Must be called at
    /// least once (typically as [`AccessFlags::ReadWrite`]) before the
    /// mapping becomes usable — a fresh `cuMemMap` defaults to no access.
    pub fn set_access(&self, device: &Device, flags: AccessFlags) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_mem_set_access()?;
        let desc = CUmemAccessDesc {
            location: CUmemLocation {
                type_: CUmemLocationType::DEVICE,
                id: device.as_raw().0,
            },
            flags: flags.raw(),
        };
        check(unsafe { cu(self.as_raw(), self.size, &desc, 1) })
    }

    /// The device pointer at which the physical memory is now accessible.
    #[inline]
    pub fn as_raw(&self) -> CUdeviceptr {
        CUdeviceptr(self.range.as_raw().0 + self.offset as u64)
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for MappedRange {
    fn drop(&mut self) {
        if self.range.as_raw().0 == 0 {
            return;
        }
        if let Ok(d) = driver() {
            if let Ok(cu) = d.cu_mem_unmap() {
                let _ = unsafe { cu(self.as_raw(), self.size) };
            }
        }
        // keep `physical` alive here so the above unmap precedes release
        let _ = &self.physical;
    }
}
