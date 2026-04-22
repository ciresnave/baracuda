//! Pointer attribute queries (`cuPointerGetAttribute`).

use baracuda_cuda_sys::driver;
use baracuda_cuda_sys::types::CUpointer_attribute;
use baracuda_cuda_sys::CUdeviceptr;

use crate::error::{check, Result};

/// Raw pointer-attribute query. `attribute` is one of
/// [`baracuda_cuda_sys::types::CUpointer_attribute`]; the caller must
/// provide a writable `out` of the correct size for that attribute.
///
/// # Safety
///
/// `out` must point to a buffer large enough to receive the attribute's
/// value type (typically 4 or 8 bytes). See NVIDIA's driver API reference
/// for per-attribute size.
pub unsafe fn raw_attribute(
    attribute: i32,
    ptr: CUdeviceptr,
    out: *mut core::ffi::c_void,
) -> Result<()> {
    let d = driver()?;
    let cu = d.cu_pointer_get_attribute()?;
    check(cu(out, attribute, ptr))
}

/// Memory "kind" returned by `CUpointer_attribute::MEMORY_TYPE`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MemoryType {
    Host,
    Device,
    Array,
    Unified,
    /// Unrecognized value; includes the raw code for forward compatibility.
    Unknown(u32),
}

impl MemoryType {
    #[inline]
    fn from_raw(raw: u32) -> Self {
        use baracuda_cuda_sys::types::CUmemorytype;
        match raw {
            CUmemorytype::HOST => MemoryType::Host,
            CUmemorytype::DEVICE => MemoryType::Device,
            CUmemorytype::ARRAY => MemoryType::Array,
            CUmemorytype::UNIFIED => MemoryType::Unified,
            other => MemoryType::Unknown(other),
        }
    }
}

/// Query the memory type of a device pointer.
pub fn memory_type(ptr: CUdeviceptr) -> Result<MemoryType> {
    let mut raw: u32 = 0;
    // SAFETY: `raw` is a `u32` (4 bytes) which matches the attribute's size.
    unsafe {
        raw_attribute(
            CUpointer_attribute::MEMORY_TYPE,
            ptr,
            &mut raw as *mut u32 as *mut core::ffi::c_void,
        )?;
    }
    Ok(MemoryType::from_raw(raw))
}

/// Query whether a pointer refers to managed (unified) memory.
pub fn is_managed(ptr: CUdeviceptr) -> Result<bool> {
    let mut raw: u32 = 0;
    unsafe {
        raw_attribute(
            CUpointer_attribute::IS_MANAGED,
            ptr,
            &mut raw as *mut u32 as *mut core::ffi::c_void,
        )?;
    }
    Ok(raw != 0)
}

/// Query the device ordinal this allocation was created on.
pub fn device_ordinal(ptr: CUdeviceptr) -> Result<i32> {
    let mut raw: i32 = 0;
    unsafe {
        raw_attribute(
            CUpointer_attribute::DEVICE_ORDINAL,
            ptr,
            &mut raw as *mut i32 as *mut core::ffi::c_void,
        )?;
    }
    Ok(raw)
}

/// Query the size (bytes) of the range this pointer sits inside.
pub fn range_size(ptr: CUdeviceptr) -> Result<usize> {
    let mut raw: usize = 0;
    unsafe {
        raw_attribute(
            CUpointer_attribute::RANGE_SIZE,
            ptr,
            &mut raw as *mut usize as *mut core::ffi::c_void,
        )?;
    }
    Ok(raw)
}

/// Batched pointer-attribute query. For each `attribute` in `attributes`
/// there must be a matching writable slot in `data` sized for that
/// attribute's value type.
///
/// # Safety
///
/// Each `data[i]` must point to a buffer large enough to receive
/// `attributes[i]`'s value type.
pub unsafe fn raw_attributes_batched(
    attributes: &mut [i32],
    data: &mut [*mut core::ffi::c_void],
    ptr: CUdeviceptr,
) -> Result<()> {
    assert_eq!(
        attributes.len(),
        data.len(),
        "attributes / data length mismatch"
    );
    let d = driver()?;
    let cu = d.cu_pointer_get_attributes()?;
    check(cu(
        attributes.len() as core::ffi::c_uint,
        attributes.as_mut_ptr(),
        data.as_mut_ptr(),
        ptr,
    ))
}

/// Query a single attribute on a managed-memory range.
///
/// # Safety
///
/// `out` must point to a buffer of `data_size` bytes sized for the
/// attribute — consult the NVIDIA docs for
/// [`CUmem_range_attribute`](baracuda_cuda_sys::types).
pub unsafe fn range_attribute_raw(
    attribute: i32,
    ptr: CUdeviceptr,
    count: usize,
    out: *mut core::ffi::c_void,
    data_size: usize,
) -> Result<()> {
    let d = driver()?;
    let cu = d.cu_mem_range_get_attribute()?;
    check(cu(out, data_size, attribute, ptr, count))
}

/// Batched range-attribute query. `data[i]` has size `data_sizes[i]`.
///
/// # Safety
///
/// Caller guarantees slot sizes match the attribute types.
pub unsafe fn range_attributes_batched(
    attributes: &mut [i32],
    data: &mut [*mut core::ffi::c_void],
    data_sizes: &mut [usize],
    ptr: CUdeviceptr,
    count: usize,
) -> Result<()> {
    assert_eq!(attributes.len(), data.len());
    assert_eq!(attributes.len(), data_sizes.len());
    let d = driver()?;
    let cu = d.cu_mem_range_get_attributes()?;
    check(cu(
        data.as_mut_ptr(),
        data_sizes.as_mut_ptr(),
        attributes.as_mut_ptr(),
        attributes.len(),
        ptr,
        count,
    ))
}

/// Set a single pointer attribute (typically `SYNC_MEMOPS`).
///
/// # Safety
///
/// `value` must point to a valid payload matching the attribute.
pub unsafe fn set_attribute_raw(
    value: *const core::ffi::c_void,
    attribute: i32,
    ptr: CUdeviceptr,
) -> Result<()> {
    let d = driver()?;
    let cu = d.cu_pointer_set_attribute()?;
    check(cu(value, attribute, ptr))
}
