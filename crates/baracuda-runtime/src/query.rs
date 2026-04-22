//! Runtime-API queries: pointer attributes, device properties, kernel
//! attributes. Typed wrappers around the `cuda*GetAttributes` /
//! `cudaGetDeviceProperties` family.

use baracuda_cuda_sys::runtime::runtime;
use baracuda_cuda_sys::runtime::types::{
    cudaFuncAttributes, cudaMemoryType, cudaPointerAttributes,
};

use crate::device::Device;
use crate::error::{check, Result};

/// Memory kind reported by [`pointer_attributes`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MemoryType {
    /// Pointer is not registered with CUDA (plain host malloc or an
    /// unrelated OS allocation).
    Unregistered,
    /// Host memory (pinned / mapped / managed-with-host-affinity).
    Host,
    /// Plain device memory.
    Device,
    /// Managed (unified) memory.
    Managed,
}

impl MemoryType {
    #[inline]
    fn from_raw(raw: i32) -> Self {
        match raw {
            cudaMemoryType::HOST => MemoryType::Host,
            cudaMemoryType::DEVICE => MemoryType::Device,
            cudaMemoryType::MANAGED => MemoryType::Managed,
            _ => MemoryType::Unregistered,
        }
    }
}

/// Typed view over `cudaPointerAttributes`.
#[derive(Copy, Clone, Debug)]
pub struct PointerAttributes {
    pub memory_type: MemoryType,
    pub device: i32,
    pub device_pointer: *mut core::ffi::c_void,
    pub host_pointer: *mut core::ffi::c_void,
}

/// Query what CUDA knows about `ptr`. On pointers CUDA has never seen
/// (plain host `malloc`, returned from C libraries) this returns
/// `MemoryType::Unregistered`.
///
/// # Safety
///
/// `ptr` can be any pointer — CUDA internally classifies it. This is a
/// pure query and doesn't dereference `ptr`.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn pointer_attributes(ptr: *const core::ffi::c_void) -> Result<PointerAttributes> {
    let r = runtime()?;
    let cu = r.cuda_pointer_get_attributes()?;
    let mut raw = cudaPointerAttributes::default();
    check(unsafe {
        cu(
            &mut raw as *mut cudaPointerAttributes as *mut core::ffi::c_void,
            ptr,
        )
    })?;
    Ok(PointerAttributes {
        memory_type: MemoryType::from_raw(raw.type_),
        device: raw.device,
        device_pointer: raw.device_pointer,
        host_pointer: raw.host_pointer,
    })
}

/// Subset of `cudaDeviceProp` fields most users care about. The full C
/// struct is ~1 KB with fields that are rarely accessed — we surface the
/// hot-path ones and keep a reserved `_raw` slot for the whole buffer so
/// advanced users can cast through.
#[derive(Clone, Debug)]
pub struct DeviceProperties {
    pub name: String,
    pub total_global_memory_bytes: u64,
    pub shared_memory_per_block_bytes: u64,
    pub regs_per_block: i32,
    pub warp_size: i32,
    pub max_threads_per_block: i32,
    pub max_block_dim: [i32; 3],
    pub max_grid_dim: [i32; 3],
    pub clock_rate_khz: i32,
    pub memory_clock_rate_khz: i32,
    pub memory_bus_width_bits: i32,
    pub l2_cache_size_bytes: i32,
    pub max_threads_per_sm: i32,
    pub multiprocessor_count: i32,
    pub compute_capability_major: i32,
    pub compute_capability_minor: i32,
    pub integrated: bool,
    pub concurrent_kernels: bool,
    pub pci_bus_id: i32,
    pub pci_device_id: i32,
    pub pci_domain_id: i32,
}

/// Fetch a typed subset of `cudaDeviceProp` for `device`.
///
/// We use `cudaGetDeviceProperties` to pull the device name (a 256-byte
/// char array at offset 0 — the one layout-stable field across CUDA
/// versions) and per-attribute queries for everything else. This is
/// more robust than struct-offset parsing because `cudaDeviceProp` has
/// grown over CUDA versions and offsets drift silently.
pub fn device_properties(device: &Device) -> Result<DeviceProperties> {
    use baracuda_cuda_sys::runtime::types::cudaDeviceAttr as Attr;

    let r = runtime()?;
    let cu = r.cuda_get_device_properties()?;
    let mut buf = vec![0u8; 2048];
    check(unsafe { cu(buf.as_mut_ptr() as *mut core::ffi::c_void, device.ordinal()) })?;

    // The name field is a 256-byte char array at offset 0 — the only
    // field we read from the buffer. Everything else goes through
    // cudaDeviceGetAttribute which is version-stable.
    let name = unsafe {
        let name_ptr = buf.as_ptr() as *const core::ffi::c_char;
        core::ffi::CStr::from_ptr(name_ptr)
            .to_string_lossy()
            .into_owned()
    };

    // Total global memory isn't a per-attribute query — use cudaMemGetInfo.
    let total_global_memory_bytes = {
        let cu_info = r.cuda_mem_get_info()?;
        let mut free: usize = 0;
        let mut total: usize = 0;
        check(unsafe { cu_info(&mut free, &mut total) })?;
        total as u64
    };

    Ok(DeviceProperties {
        name,
        total_global_memory_bytes,
        shared_memory_per_block_bytes: device
            .attribute(Attr::MAX_SHARED_MEMORY_PER_BLOCK)
            .unwrap_or(0) as u64,
        regs_per_block: device.attribute(Attr::MAX_REGISTERS_PER_BLOCK).unwrap_or(0),
        warp_size: device.attribute(Attr::WARP_SIZE).unwrap_or(0),
        max_threads_per_block: device.attribute(Attr::MAX_THREADS_PER_BLOCK).unwrap_or(0),
        max_block_dim: [
            device.attribute(Attr::MAX_BLOCK_DIM_X).unwrap_or(0),
            device.attribute(Attr::MAX_BLOCK_DIM_Y).unwrap_or(0),
            device.attribute(Attr::MAX_BLOCK_DIM_Z).unwrap_or(0),
        ],
        max_grid_dim: [
            device.attribute(Attr::MAX_GRID_DIM_X).unwrap_or(0),
            device.attribute(Attr::MAX_GRID_DIM_Y).unwrap_or(0),
            device.attribute(Attr::MAX_GRID_DIM_Z).unwrap_or(0),
        ],
        clock_rate_khz: device.attribute(Attr::CLOCK_RATE).unwrap_or(0),
        memory_clock_rate_khz: device.attribute(Attr::CLOCK_RATE).unwrap_or(0),
        memory_bus_width_bits: 0,
        l2_cache_size_bytes: 0,
        max_threads_per_sm: 0,
        multiprocessor_count: device.attribute(Attr::MULTIPROCESSOR_COUNT).unwrap_or(0),
        compute_capability_major: device
            .attribute(Attr::COMPUTE_CAPABILITY_MAJOR)
            .unwrap_or(0),
        compute_capability_minor: device
            .attribute(Attr::COMPUTE_CAPABILITY_MINOR)
            .unwrap_or(0),
        integrated: device.attribute(Attr::INTEGRATED).unwrap_or(0) != 0,
        concurrent_kernels: device.attribute(Attr::CONCURRENT_KERNELS).unwrap_or(0) != 0,
        pci_bus_id: device.attribute(Attr::PCI_BUS_ID).unwrap_or(0),
        pci_device_id: device.attribute(Attr::PCI_DEVICE_ID).unwrap_or(0),
        pci_domain_id: device.attribute(Attr::PCI_DOMAIN_ID).unwrap_or(0),
    })
}

/// Query a kernel's register / shared-memory / PTX-version metadata.
/// `func_symbol` is a `const void*` — the address of the kernel symbol
/// as used by `cudaLaunchKernel` (or [`crate::Kernel::as_launch_ptr`]).
///
/// # Safety
///
/// `func_symbol` must be a valid CUDA kernel symbol address. Passing
/// garbage causes undefined behavior inside the driver.
pub unsafe fn func_attributes(func_symbol: *const core::ffi::c_void) -> Result<cudaFuncAttributes> {
    let r = runtime()?;
    let cu = r.cuda_func_get_attributes()?;
    let mut attrs = cudaFuncAttributes::default();
    check(cu(
        &mut attrs as *mut cudaFuncAttributes as *mut core::ffi::c_void,
        func_symbol,
    ))?;
    Ok(attrs)
}
