//! Core handle types used by the CUDA Driver API.
//!
//! These mirror the C typedefs in `cuda.h`. Pointer-typed handles
//! (`CUcontext`, `CUstream`, ...) are opaque to the Rust side and are
//! simply raw pointers; integer-typed handles (`CUdevice`, `CUdeviceptr`)
//! are `#[repr(transparent)]` newtypes so they cannot be accidentally
//! confused with other integer parameters.

use core::ffi::c_void;

/// Ordinal of a CUDA device. `cuDeviceGet(&dev, ordinal)` yields one of these.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
#[repr(transparent)]
pub struct CUdevice(pub i32);

/// A device-side virtual address. 64-bit on every platform baracuda supports
/// (CUDA 4.0+).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
#[repr(transparent)]
pub struct CUdeviceptr(pub u64);

// SAFETY: `CUdevice` and `CUdeviceptr` are `#[repr(transparent)]` over plain
// integers, so their ABI-layout matches a kernel arg of the same width.
// `CUdeviceptr` in particular is what every device-pointer kernel parameter
// expects (`void*` / `T*` on the CUDA side is the same 64 bits).
unsafe impl baracuda_types::DeviceRepr for CUdevice {}
unsafe impl baracuda_types::DeviceRepr for CUdeviceptr {}

/// Opaque context handle.
pub type CUcontext = *mut c_void;

/// Opaque module handle (holds compiled kernels).
pub type CUmodule = *mut c_void;

/// Opaque function handle (a kernel entry point within a module).
pub type CUfunction = *mut c_void;

/// Opaque library handle (CUDA 12.0+ context-independent module).
pub type CUlibrary = *mut c_void;

/// Opaque kernel handle (CUDA 12.0+ library-based equivalent of `CUfunction`).
pub type CUkernel = *mut c_void;

/// Opaque stream handle.
pub type CUstream = *mut c_void;

/// Opaque event handle.
pub type CUevent = *mut c_void;

/// Opaque graph handle.
pub type CUgraph = *mut c_void;

/// Opaque graph-node handle.
pub type CUgraphNode = *mut c_void;

/// Opaque executable-graph handle.
pub type CUgraphExec = *mut c_void;

/// Opaque memory-pool handle (CUDA 11.2+).
pub type CUmemoryPool = *mut c_void;

/// Opaque CUDA array handle (backing storage for textures / surfaces).
pub type CUarray = *mut c_void;

/// 64-bit texture-object handle (created via `cuTexObjectCreate`).
pub type CUtexObject = u64;

/// 64-bit surface-object handle (created via `cuSurfObjectCreate`).
pub type CUsurfObject = u64;

/// Generic allocation handle for the CUDA VMM (Virtual Memory Management)
/// API — opaque 64-bit cookie returned by `cuMemCreate`.
pub type CUmemGenericAllocationHandle = u64;

/// Opaque external-memory handle.
pub type CUexternalMemory = *mut c_void;

/// Opaque external-semaphore handle.
pub type CUexternalSemaphore = *mut c_void;

/// The special "null stream" (legacy default stream).
pub const CU_STREAM_LEGACY: CUstream = 0x1 as CUstream;

/// The special "per-thread default stream" (CUDA 7.0+).
pub const CU_STREAM_PER_THREAD: CUstream = 0x2 as CUstream;

/// Event flags (OR-able into `cuEventCreate`).
#[allow(non_snake_case)]
pub mod CUevent_flags {
    pub const DEFAULT: u32 = 0x0;
    pub const BLOCKING_SYNC: u32 = 0x1;
    pub const DISABLE_TIMING: u32 = 0x2;
    pub const INTERPROCESS: u32 = 0x4;
}

/// Stream flags (OR-able into `cuStreamCreate`).
#[allow(non_snake_case)]
pub mod CUstream_flags {
    pub const DEFAULT: u32 = 0x0;
    pub const NON_BLOCKING: u32 = 0x1;
}

/// Context flags (OR-able into `cuCtxCreate`).
#[allow(non_snake_case)]
pub mod CUcontext_flags {
    pub const SCHED_AUTO: u32 = 0x0;
    pub const SCHED_SPIN: u32 = 0x1;
    pub const SCHED_YIELD: u32 = 0x2;
    pub const SCHED_BLOCKING_SYNC: u32 = 0x4;
    pub const MAP_HOST: u32 = 0x8;
    pub const LMEM_RESIZE_TO_MAX: u32 = 0x10;
}

/// `CUlimit` — selector for `cuCtxGetLimit` / `cuCtxSetLimit`.
#[allow(non_snake_case)]
pub mod CUlimit {
    pub const STACK_SIZE: u32 = 0x00;
    pub const PRINTF_FIFO_SIZE: u32 = 0x01;
    pub const MALLOC_HEAP_SIZE: u32 = 0x02;
    pub const DEV_RUNTIME_SYNC_DEPTH: u32 = 0x03;
    pub const DEV_RUNTIME_PENDING_LAUNCH_COUNT: u32 = 0x04;
    pub const MAX_L2_FETCH_GRANULARITY: u32 = 0x05;
    pub const PERSISTING_L2_CACHE_SIZE: u32 = 0x06;
}

/// `CUfunc_cache` — L1-vs-shared carveout preference.
#[allow(non_snake_case)]
pub mod CUfunc_cache {
    pub const PREFER_NONE: u32 = 0x00;
    pub const PREFER_SHARED: u32 = 0x01;
    pub const PREFER_L1: u32 = 0x02;
    pub const PREFER_EQUAL: u32 = 0x03;
}

/// Memory-attach flags for `cuMemAllocManaged`.
#[allow(non_snake_case)]
pub mod CUmemAttach_flags {
    /// Accessible from any stream on any device (default).
    pub const GLOBAL: u32 = 0x01;
    /// Accessible only from the host.
    pub const HOST: u32 = 0x02;
    /// Accessible only from the stream it was attached to.
    pub const SINGLE: u32 = 0x04;
}

/// `CUmem_advise` — hints for `cuMemAdvise`.
#[allow(non_snake_case)]
pub mod CUmem_advise {
    pub const SET_READ_MOSTLY: i32 = 1;
    pub const UNSET_READ_MOSTLY: i32 = 2;
    pub const SET_PREFERRED_LOCATION: i32 = 3;
    pub const UNSET_PREFERRED_LOCATION: i32 = 4;
    pub const SET_ACCESSED_BY: i32 = 5;
    pub const UNSET_ACCESSED_BY: i32 = 6;
}

/// `CUmemRangeHandleType` — for `cuMemGetHandleForAddressRange`.
#[allow(non_snake_case)]
pub mod CUmemRangeHandleType {
    pub const DMA_BUF_FD: i32 = 1;
}

/// `CUarraySparseSubresourceType` — tag for the `subresource` union inside
/// [`CUarrayMapInfo`].
#[allow(non_snake_case)]
pub mod CUarraySparseSubresourceType {
    /// Tile-indexed sparse level update.
    pub const SPARSE_LEVEL: i32 = 0;
    /// Mipmap-tail update.
    pub const MIPTAIL: i32 = 1;
}

/// `CUmemOperationType` — whether a [`CUarrayMapInfo`] describes a map
/// or an unmap operation.
#[allow(non_snake_case)]
pub mod CUmemOperationType {
    pub const MAP: i32 = 1;
    pub const UNMAP: i32 = 2;
}

/// `CUmemHandleType` — handle kind for the `memHandle` union inside
/// [`CUarrayMapInfo`].
#[allow(non_snake_case)]
pub mod CUmemHandleType {
    pub const GENERIC: i32 = 0;
}

/// `CUarrayMapInfo` — 96-byte descriptor for `cuMemMapArrayAsync`.
///
/// Three tagged unions in one struct (resource / subresource / memHandle).
/// The typed builder methods below populate them correctly; raw field
/// access is available for advanced users.
///
/// Layout (CUDA 13.x):
/// ```text
/// struct CUarrayMapInfo {
///     CUresourcetype resourceType;       // offset 0, 4 bytes
///     // 4 bytes pad
///     union { CUmipmappedArray; CUarray; } resource; // offset 8, 8 bytes
///     CUarraySparseSubresourceType subresourceType;  // offset 16, 4 bytes
///     // 4 bytes pad
///     union {                            // offset 24, 32 bytes
///         struct { u32 level, layer, ox, oy, oz, ew, eh, ed; } sparseLevel;
///         struct { u32 layer; u64 offset; u64 size; } miptail;
///     } subresource;
///     CUmemOperationType memOperationType; // offset 56, 4 bytes
///     CUmemHandleType memHandleType;       // offset 60, 4 bytes
///     union { CUmemGenericAllocationHandle; } memHandle; // offset 64, 8 bytes
///     u64 offset;                          // offset 72, 8 bytes
///     u32 deviceBitMask;                   // offset 80, 4 bytes
///     u32 flags;                           // offset 84, 4 bytes
///     u32 reserved[2];                     // offset 88, 8 bytes
/// };
/// ```
#[repr(C)]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
pub struct CUarrayMapInfo {
    pub resource_type: core::ffi::c_int,
    _pad0: u32,
    /// Union payload: `CUarray` or `CUmipmappedArray` (both pointer-sized).
    pub resource_raw: u64,
    pub subresource_type: core::ffi::c_int,
    _pad1: u32,
    /// Union payload for the subresource (32 bytes, enough for
    /// `sparseLevel`'s 8 u32s or `miptail`'s 3-field struct).
    pub subresource_raw: [u64; 4],
    pub mem_operation_type: core::ffi::c_int,
    pub mem_handle_type: core::ffi::c_int,
    pub mem_handle_raw: u64,
    pub offset: u64,
    pub device_bit_mask: core::ffi::c_uint,
    pub flags: core::ffi::c_uint,
    pub reserved: [core::ffi::c_uint; 2],
}

impl Default for CUarrayMapInfo {
    fn default() -> Self {
        Self {
            resource_type: 0,
            _pad0: 0,
            resource_raw: 0,
            subresource_type: 0,
            _pad1: 0,
            subresource_raw: [0; 4],
            mem_operation_type: 0,
            mem_handle_type: CUmemHandleType::GENERIC,
            mem_handle_raw: 0,
            offset: 0,
            device_bit_mask: 0,
            flags: 0,
            reserved: [0; 2],
        }
    }
}

impl core::fmt::Debug for CUarrayMapInfo {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CUarrayMapInfo")
            .field("resource_type", &self.resource_type)
            .field("subresource_type", &self.subresource_type)
            .field("mem_operation_type", &self.mem_operation_type)
            .field("offset", &self.offset)
            .finish_non_exhaustive()
    }
}

impl CUarrayMapInfo {
    /// Point the resource union at a [`CUarray`] handle.
    pub fn with_array(mut self, array: CUarray) -> Self {
        self.resource_type = CUresourcetype::ARRAY as core::ffi::c_int;
        self.resource_raw = array as usize as u64;
        self
    }

    /// Point the resource union at a [`CUmipmappedArray`] handle.
    pub fn with_mipmapped_array(mut self, mipmap: CUmipmappedArray) -> Self {
        self.resource_type = CUresourcetype::MIPMAPPED_ARRAY as core::ffi::c_int;
        self.resource_raw = mipmap as usize as u64;
        self
    }

    /// Set the subresource to a sparse-level tile update. `offset_*`
    /// and `extent_*` are in tiles (not bytes).
    #[allow(clippy::too_many_arguments)]
    pub fn with_sparse_level(
        mut self,
        level: u32,
        layer: u32,
        offset_x: u32,
        offset_y: u32,
        offset_z: u32,
        extent_width: u32,
        extent_height: u32,
        extent_depth: u32,
    ) -> Self {
        self.subresource_type = CUarraySparseSubresourceType::SPARSE_LEVEL;
        // Layout: 8 u32s packed into 32 bytes = 4 u64s.
        // SAFETY: subresource_raw is [u64; 4] = 32 bytes, 8-aligned; we
        // write an 8-u32 little-/native-endian struct through a pointer.
        let sl = [
            level,
            layer,
            offset_x,
            offset_y,
            offset_z,
            extent_width,
            extent_height,
            extent_depth,
        ];
        unsafe {
            let p = self.subresource_raw.as_mut_ptr() as *mut [u32; 8];
            p.write(sl);
        }
        self
    }

    /// Set the subresource to a mipmap-tail update.
    pub fn with_miptail(mut self, layer: u32, tail_offset: u64, tail_size: u64) -> Self {
        self.subresource_type = CUarraySparseSubresourceType::MIPTAIL;
        // Layout: { u32 layer; u64 offset; u64 size; } with 4-byte pad
        // after `layer` to align the u64s. Total 24 bytes.
        #[repr(C)]
        struct Miptail {
            layer: u32,
            _pad: u32,
            offset: u64,
            size: u64,
        }
        let m = Miptail {
            layer,
            _pad: 0,
            offset: tail_offset,
            size: tail_size,
        };
        unsafe {
            let p = self.subresource_raw.as_mut_ptr() as *mut Miptail;
            p.write(m);
        }
        self
    }

    /// Set the mem-handle union to a VMM generic allocation handle.
    pub fn with_mem_handle(mut self, handle: CUmemGenericAllocationHandle) -> Self {
        self.mem_handle_type = CUmemHandleType::GENERIC;
        self.mem_handle_raw = handle;
        self
    }

    /// Mark this entry as a map operation.
    pub fn as_map(mut self) -> Self {
        self.mem_operation_type = CUmemOperationType::MAP;
        self
    }

    /// Mark this entry as an unmap operation.
    pub fn as_unmap(mut self) -> Self {
        self.mem_operation_type = CUmemOperationType::UNMAP;
        self
    }

    /// Byte offset into the backing allocation handle.
    pub fn with_offset(mut self, offset: u64) -> Self {
        self.offset = offset;
        self
    }

    /// Bitmask of devices the mapping applies to (one bit per peer).
    pub fn with_device_bit_mask(mut self, mask: u32) -> Self {
        self.device_bit_mask = mask;
        self
    }
}

// ---- Wave 28: medium-value consolidated ---------------------------------

/// `CUexecAffinityType` — kind of per-context execution affinity.
#[allow(non_snake_case)]
pub mod CUexecAffinityType {
    pub const SM_COUNT: i32 = 0;
}

/// `CUdevice_P2PAttribute` — passed to `cuDeviceGetP2PAttribute`.
#[allow(non_snake_case)]
pub mod CUdevice_P2PAttribute {
    pub const PERFORMANCE_RANK: i32 = 1;
    pub const ACCESS_SUPPORTED: i32 = 2;
    pub const NATIVE_ATOMIC_SUPPORTED: i32 = 3;
    pub const CUDA_ARRAY_ACCESS_SUPPORTED: i32 = 4;
}

/// `CUflushGPUDirectRDMAWritesTarget`.
#[allow(non_snake_case)]
pub mod CUflushGPUDirectRDMAWritesTarget {
    pub const CURRENT_CTX: i32 = 0;
}

/// `CUflushGPUDirectRDMAWritesScope`.
#[allow(non_snake_case)]
pub mod CUflushGPUDirectRDMAWritesScope {
    pub const TO_OWNER: i32 = 100;
    pub const TO_ALL_DEVICES: i32 = 200;
}

/// `CUcoredumpSettings` — attribute selectors for `cuCoredumpGet/SetAttribute`.
#[allow(non_snake_case)]
pub mod CUcoredumpSettings {
    pub const ENABLE_ON_EXCEPTION: i32 = 1;
    pub const TRIGGER_HOST: i32 = 2;
    pub const LIGHTWEIGHT: i32 = 3;
    pub const ENABLE_USER_TRIGGER: i32 = 4;
    pub const FILE: i32 = 5;
    pub const PIPE: i32 = 6;
    pub const GENERATION_FLAGS: i32 = 7;
}

/// `CUDA_ARRAY_SPARSE_PROPERTIES` — per-array sparse / tiled layout info.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
#[allow(non_camel_case_types)]
pub struct CUDA_ARRAY_SPARSE_PROPERTIES {
    pub tile_extent_width: core::ffi::c_uint,
    pub tile_extent_height: core::ffi::c_uint,
    pub tile_extent_depth: core::ffi::c_uint,
    pub miptail_first_level: core::ffi::c_uint,
    pub miptail_size: u64,
    pub flags: core::ffi::c_uint,
    pub reserved: [core::ffi::c_uint; 4],
}

/// `CUDA_ARRAY_MEMORY_REQUIREMENTS` — size/alignment for an array's
/// backing VMM allocation.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
#[allow(non_camel_case_types)]
pub struct CUDA_ARRAY_MEMORY_REQUIREMENTS {
    pub size: usize,
    pub alignment: usize,
    pub reserved: [core::ffi::c_uint; 4],
}

// ---- Wave 29-31: graphics interop + Jetson NvSci ------------------------

/// `CUgraphicsMapResourceFlags` — map-time access hints.
#[allow(non_snake_case)]
pub mod CUgraphicsMapResourceFlags {
    pub const NONE: u32 = 0;
    pub const READ_ONLY: u32 = 1;
    pub const WRITE_DISCARD: u32 = 2;
}

/// `CUgraphicsRegisterFlags` — shared register-time flags across GL /
/// D3D / VDPAU / EGL.
#[allow(non_snake_case)]
pub mod CUgraphicsRegisterFlags {
    pub const NONE: u32 = 0;
    pub const READ_ONLY: u32 = 1;
    pub const WRITE_DISCARD: u32 = 2;
    pub const SURFACE_LDST: u32 = 4;
    pub const TEXTURE_GATHER: u32 = 8;
}

/// Selector for `cuGLGetDevices_v2` / `cu<API>GetDevices`.
#[allow(non_snake_case)]
pub mod CUGLDeviceList {
    pub const ALL: u32 = 0x01;
    pub const CURRENT_FRAME: u32 = 0x02;
    pub const NEXT_FRAME: u32 = 0x03;
}

/// Alias for D3D-family enum kinds (shared with `CUGLDeviceList`).
pub use CUGLDeviceList as CUd3dXDeviceList;

// --- OpenGL handle types ---
// GL types live in `gl.h`; we use minimum-compatible Rust types.
pub type GLuint = core::ffi::c_uint;
pub type GLenum = core::ffi::c_uint;

// --- Direct3D handle types ---
// All D3D* device / resource pointers are opaque from CUDA's POV.
pub type ID3DDevice = *mut c_void;
pub type ID3DResource = *mut c_void;

// --- VDPAU handle types ---
// `VdpDevice`, `VdpGetProcAddress`, `VdpVideoSurface`, `VdpOutputSurface`
// are all 32-bit unsigned handles in libvdpau.
pub type VdpDevice = core::ffi::c_uint;
pub type VdpGetProcAddress = *mut c_void;
pub type VdpVideoSurface = core::ffi::c_uint;
pub type VdpOutputSurface = core::ffi::c_uint;

// --- EGL handle types ---
pub type EGLImageKHR = *mut c_void;
pub type EGLStreamKHR = *mut c_void;
pub type EGLSyncKHR = *mut c_void;

/// `CUeglFrame` — YUV / RGB frame layout used by EGL stream interop
/// (Jetson video pipelines). 80 bytes in `cuda.h`; exposed as an opaque
/// blob so callers can populate from bindgen-generated layouts.
#[repr(C)]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
pub struct CUeglFrame {
    pub raw: [u64; 10],
}

impl Default for CUeglFrame {
    #[allow(clippy::derivable_impls)]
    fn default() -> Self {
        Self { raw: [0; 10] }
    }
}

impl core::fmt::Debug for CUeglFrame {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CUeglFrame").finish_non_exhaustive()
    }
}

// --- NvSci handle types (Jetson / DRIVE) ---
// `NvSciSyncAttrList` and `NvSciBufObj` are opaque pointers from
// NVIDIA's NvSci libraries. Users working with NvSci pass in pointers
// obtained from libnvsciSync / libnvsciBuf.
pub type NvSciSyncAttrList = *mut c_void;
pub type NvSciSyncObj = *mut c_void;
pub type NvSciSyncFence = *mut c_void;
pub type NvSciBufObj = *mut c_void;

/// `CUnvSciSyncAttr` — direction flags for
/// `cuDeviceGetNvSciSyncAttributes`.
#[allow(non_snake_case)]
pub mod CUnvSciSyncAttr {
    pub const SIGNAL: i32 = 1;
    pub const WAIT: i32 = 2;
}

/// `CUpointer_attribute` — selector for `cuPointerGetAttribute`.
#[allow(non_snake_case)]
pub mod CUpointer_attribute {
    pub const CONTEXT: i32 = 1;
    pub const MEMORY_TYPE: i32 = 2;
    pub const DEVICE_POINTER: i32 = 3;
    pub const HOST_POINTER: i32 = 4;
    pub const P2P_TOKENS: i32 = 5;
    pub const SYNC_MEMOPS: i32 = 6;
    pub const BUFFER_ID: i32 = 7;
    pub const IS_MANAGED: i32 = 8;
    pub const DEVICE_ORDINAL: i32 = 9;
    pub const IS_LEGACY_CUDA_IPC_CAPABLE: i32 = 10;
    pub const RANGE_START_ADDR: i32 = 11;
    pub const RANGE_SIZE: i32 = 12;
    pub const MAPPED: i32 = 13;
    pub const ALLOWED_HANDLE_TYPES: i32 = 14;
    pub const IS_GPU_DIRECT_RDMA_CAPABLE: i32 = 15;
    pub const ACCESS_FLAGS: i32 = 16;
    pub const MEMPOOL_HANDLE: i32 = 17;
    pub const MAPPING_SIZE: i32 = 18;
    pub const MAPPING_BASE_ADDR: i32 = 19;
    pub const MEMORY_BLOCK_ID: i32 = 20;
}

/// `CUmemorytype` — values returned via `CUpointer_attribute::MEMORY_TYPE`.
#[allow(non_snake_case)]
pub mod CUmemorytype {
    pub const HOST: u32 = 0x01;
    pub const DEVICE: u32 = 0x02;
    pub const ARRAY: u32 = 0x03;
    pub const UNIFIED: u32 = 0x04;
}

/// `CUlaunchAttributeID` — selector for entries in a `CUlaunchConfig`'s
/// attribute array (passed to `cuLaunchKernelEx`, CUDA 12.0+).
#[allow(non_snake_case)]
pub mod CUlaunchAttributeID {
    pub const IGNORE: u32 = 0;
    pub const ACCESS_POLICY_WINDOW: u32 = 1;
    pub const COOPERATIVE: u32 = 2;
    pub const SYNCHRONIZATION_POLICY: u32 = 3;
    pub const CLUSTER_DIMENSION: u32 = 4;
    pub const CLUSTER_SCHEDULING_POLICY_PREFERENCE: u32 = 5;
    pub const PROGRAMMATIC_STREAM_SERIALIZATION: u32 = 6;
    pub const PROGRAMMATIC_EVENT: u32 = 7;
    pub const PRIORITY: u32 = 8;
    pub const MEM_SYNC_DOMAIN_MAP: u32 = 9;
    pub const MEM_SYNC_DOMAIN: u32 = 10;
    pub const LAUNCH_COMPLETION_EVENT: u32 = 12;
    pub const DEVICE_UPDATABLE_KERNEL_NODE: u32 = 13;
}

/// `CUlaunchAttributeValue` — union of payloads for a launch attribute.
/// 64-byte fixed-size union in `cuda.h`; we expose it as an opaque byte
/// array so callers can bit-cast. Zero-initialized for "no payload".
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
pub struct CUlaunchAttributeValue(pub [u8; 64]);

impl Default for CUlaunchAttributeValue {
    fn default() -> Self {
        Self([0u8; 64])
    }
}

/// `CUaccessProperty` — hit/miss cache policy used inside a
/// [`CUaccessPolicyWindow`].
#[allow(non_snake_case)]
pub mod CUaccessProperty {
    pub const NORMAL: i32 = 0;
    pub const STREAMING: i32 = 1;
    pub const PERSISTING: i32 = 2;
}

/// `CUaccessPolicyWindow` — describes an L2-persistence hint attached
/// to a launch via the `ACCESS_POLICY_WINDOW` attribute.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
pub struct CUaccessPolicyWindow {
    pub base_ptr: *mut c_void,
    pub num_bytes: usize,
    pub hit_ratio: f32,
    pub hit_prop: core::ffi::c_int,
    pub miss_prop: core::ffi::c_int,
}

impl Default for CUaccessPolicyWindow {
    fn default() -> Self {
        Self {
            base_ptr: core::ptr::null_mut(),
            num_bytes: 0,
            hit_ratio: 0.0,
            hit_prop: CUaccessProperty::NORMAL,
            miss_prop: CUaccessProperty::NORMAL,
        }
    }
}

/// `CUlaunchAttribute` — one entry in a `CUlaunchConfig`'s attribute list.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
#[allow(non_camel_case_types)]
pub struct CUlaunchAttribute {
    pub id: core::ffi::c_uint,
    /// `cuda.h` inserts 4 bytes of padding before the union.
    pub pad: [u8; 4],
    pub value: CUlaunchAttributeValue,
}

/// `CUlaunchConfig` — the descriptor passed to `cuLaunchKernelEx`.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
pub struct CUlaunchConfig {
    pub grid_dim_x: core::ffi::c_uint,
    pub grid_dim_y: core::ffi::c_uint,
    pub grid_dim_z: core::ffi::c_uint,
    pub block_dim_x: core::ffi::c_uint,
    pub block_dim_y: core::ffi::c_uint,
    pub block_dim_z: core::ffi::c_uint,
    pub shared_mem_bytes: core::ffi::c_uint,
    pub stream: CUstream,
    pub attrs: *mut CUlaunchAttribute,
    pub num_attrs: core::ffi::c_uint,
}

// Null-initialized default for CUlaunchConfig so callers can `..Default::default()`.
impl Default for CUlaunchConfig {
    fn default() -> Self {
        Self {
            grid_dim_x: 1,
            grid_dim_y: 1,
            grid_dim_z: 1,
            block_dim_x: 1,
            block_dim_y: 1,
            block_dim_z: 1,
            shared_mem_bytes: 0,
            stream: core::ptr::null_mut(),
            attrs: core::ptr::null_mut(),
            num_attrs: 0,
        }
    }
}

/// `CUfunction_attribute` — selector for `cuFuncGetAttribute` / `cuFuncSetAttribute`.
#[allow(non_snake_case)]
pub mod CUfunction_attribute {
    pub const MAX_THREADS_PER_BLOCK: i32 = 0;
    pub const SHARED_SIZE_BYTES: i32 = 1;
    pub const CONST_SIZE_BYTES: i32 = 2;
    pub const LOCAL_SIZE_BYTES: i32 = 3;
    pub const NUM_REGS: i32 = 4;
    pub const PTX_VERSION: i32 = 5;
    pub const BINARY_VERSION: i32 = 6;
    pub const CACHE_MODE_CA: i32 = 7;
    pub const MAX_DYNAMIC_SHARED_SIZE_BYTES: i32 = 8;
    pub const PREFERRED_SHARED_MEMORY_CARVEOUT: i32 = 9;
    pub const CLUSTER_SIZE_MUST_BE_SET: i32 = 10;
    pub const REQUIRED_CLUSTER_WIDTH: i32 = 11;
    pub const REQUIRED_CLUSTER_HEIGHT: i32 = 12;
    pub const REQUIRED_CLUSTER_DEPTH: i32 = 13;
    pub const NON_PORTABLE_CLUSTER_SIZE_ALLOWED: i32 = 14;
    pub const CLUSTER_SCHEDULING_POLICY_PREFERENCE: i32 = 15;
}

/// `CUDA_MEMCPY2D` — descriptor for 2D memory copies between any combination
/// of host / device / array memory.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
pub struct CUDA_MEMCPY2D {
    pub src_x_in_bytes: usize,
    pub src_y: usize,
    pub src_memory_type: u32,
    pub src_host: *const c_void,
    pub src_device: CUdeviceptr,
    pub src_array: *mut c_void,
    pub src_pitch: usize,

    pub dst_x_in_bytes: usize,
    pub dst_y: usize,
    pub dst_memory_type: u32,
    pub dst_host: *mut c_void,
    pub dst_device: CUdeviceptr,
    pub dst_array: *mut c_void,
    pub dst_pitch: usize,

    pub width_in_bytes: usize,
    pub height: usize,
}

impl Default for CUDA_MEMCPY2D {
    fn default() -> Self {
        Self {
            src_x_in_bytes: 0,
            src_y: 0,
            src_memory_type: 0,
            src_host: core::ptr::null(),
            src_device: CUdeviceptr(0),
            src_array: core::ptr::null_mut(),
            src_pitch: 0,
            dst_x_in_bytes: 0,
            dst_y: 0,
            dst_memory_type: 0,
            dst_host: core::ptr::null_mut(),
            dst_device: CUdeviceptr(0),
            dst_array: core::ptr::null_mut(),
            dst_pitch: 0,
            width_in_bytes: 0,
            height: 0,
        }
    }
}

/// `CUDA_KERNEL_NODE_PARAMS` — shape passed to `cuGraphAddKernelNode` /
/// `cuGraphKernelNodeSetParams`.
///
/// The `kern` / `ctx` fields only exist in CUDA 12.0+. On older drivers
/// they're silently ignored, so writing zero for both is portable.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
pub struct CUDA_KERNEL_NODE_PARAMS {
    pub func: CUfunction,
    pub grid_dim_x: core::ffi::c_uint,
    pub grid_dim_y: core::ffi::c_uint,
    pub grid_dim_z: core::ffi::c_uint,
    pub block_dim_x: core::ffi::c_uint,
    pub block_dim_y: core::ffi::c_uint,
    pub block_dim_z: core::ffi::c_uint,
    pub shared_mem_bytes: core::ffi::c_uint,
    pub kernel_params: *mut *mut c_void,
    pub extra: *mut *mut c_void,
    pub kern: CUkernel,
    pub ctx: CUcontext,
}

impl Default for CUDA_KERNEL_NODE_PARAMS {
    fn default() -> Self {
        Self {
            func: core::ptr::null_mut(),
            grid_dim_x: 1,
            grid_dim_y: 1,
            grid_dim_z: 1,
            block_dim_x: 1,
            block_dim_y: 1,
            block_dim_z: 1,
            shared_mem_bytes: 0,
            kernel_params: core::ptr::null_mut(),
            extra: core::ptr::null_mut(),
            kern: core::ptr::null_mut(),
            ctx: core::ptr::null_mut(),
        }
    }
}

/// `CUDA_MEMSET_NODE_PARAMS` — shape passed to `cuGraphAddMemsetNode`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
#[allow(non_camel_case_types)]
pub struct CUDA_MEMSET_NODE_PARAMS {
    pub dst: CUdeviceptr,
    pub pitch: usize,
    pub value: core::ffi::c_uint,
    pub element_size: core::ffi::c_uint,
    pub width: usize,
    pub height: usize,
}

/// Host-function signature used by `cuGraphAddHostNode` / `cuLaunchHostFunc`.
pub type CUhostFnRaw = Option<unsafe extern "C" fn(user_data: *mut c_void)>;

/// `CUDA_HOST_NODE_PARAMS` — `{ fn, user_data }` for `cuGraphAddHostNode`.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
pub struct CUDA_HOST_NODE_PARAMS {
    pub fn_: CUhostFnRaw,
    pub user_data: *mut c_void,
}

impl Default for CUDA_HOST_NODE_PARAMS {
    fn default() -> Self {
        Self {
            fn_: None,
            user_data: core::ptr::null_mut(),
        }
    }
}

/// `CUtensorMap` — 128-byte opaque Hopper TMA descriptor. Created via
/// `cuTensorMapEncodeTiled` / `cuTensorMapEncodeIm2col`; consumed by TMA
/// instructions in SM 9.0+ kernels.
#[repr(C, align(64))]
#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
pub struct CUtensorMap {
    pub opaque: [u64; 16],
}

#[allow(clippy::derivable_impls)]
impl Default for CUtensorMap {
    fn default() -> Self {
        Self { opaque: [0; 16] }
    }
}

/// `CUtensorMapDataType` — element type encoding for TMA descriptors.
#[allow(non_snake_case)]
pub mod CUtensorMapDataType {
    pub const UINT8: i32 = 0;
    pub const UINT16: i32 = 1;
    pub const UINT32: i32 = 2;
    pub const INT32: i32 = 3;
    pub const UINT64: i32 = 4;
    pub const INT64: i32 = 5;
    pub const FLOAT16: i32 = 6;
    pub const FLOAT32: i32 = 7;
    pub const FLOAT64: i32 = 8;
    pub const BFLOAT16: i32 = 9;
    pub const FLOAT32_FTZ: i32 = 10;
    pub const TFLOAT32: i32 = 11;
    pub const TFLOAT32_FTZ: i32 = 12;
}

/// `CUtensorMapInterleave`.
#[allow(non_snake_case)]
pub mod CUtensorMapInterleave {
    pub const NONE: i32 = 0;
    pub const INTERLEAVE_16B: i32 = 1;
    pub const INTERLEAVE_32B: i32 = 2;
}

/// `CUtensorMapSwizzle`.
#[allow(non_snake_case)]
pub mod CUtensorMapSwizzle {
    pub const NONE: i32 = 0;
    pub const SWIZZLE_32B: i32 = 1;
    pub const SWIZZLE_64B: i32 = 2;
    pub const SWIZZLE_128B: i32 = 3;
}

/// `CUtensorMapL2promotion` — L2 prefetch hint.
#[allow(non_snake_case)]
pub mod CUtensorMapL2promotion {
    pub const NONE: i32 = 0;
    pub const L2_64B: i32 = 1;
    pub const L2_128B: i32 = 2;
    pub const L2_256B: i32 = 3;
}

/// `CUtensorMapFloatOOBfill` — out-of-bounds fill behavior.
#[allow(non_snake_case)]
pub mod CUtensorMapFloatOOBfill {
    pub const NONE: i32 = 0;
    pub const NAN_REQUEST_ZERO_FMA: i32 = 1;
}

// ---- Wave 20: IPC -------------------------------------------------------

/// `CUipcEventHandle` — 64-byte opaque cookie for sharing CUevents across
/// processes (Linux; Windows returns NOT_SUPPORTED).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
pub struct CUipcEventHandle {
    pub reserved: [core::ffi::c_char; 64],
}

impl Default for CUipcEventHandle {
    fn default() -> Self {
        Self { reserved: [0; 64] }
    }
}

/// `CUipcMemHandle` — 64-byte opaque cookie for sharing device
/// allocations across processes.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
pub struct CUipcMemHandle {
    pub reserved: [core::ffi::c_char; 64],
}

impl Default for CUipcMemHandle {
    fn default() -> Self {
        Self { reserved: [0; 64] }
    }
}

// ---- Wave 19: conditional + switch graph nodes --------------------------

/// 64-bit handle used by conditional graph nodes (CUDA 12.3+).
pub type CUgraphConditionalHandle = u64;

/// `CUgraphConditionalNodeType`.
#[allow(non_snake_case)]
pub mod CUgraphConditionalNodeType {
    pub const IF: i32 = 0;
    pub const WHILE: i32 = 1;
    pub const SWITCH: i32 = 2;
}

/// `CUDA_CONDITIONAL_NODE_PARAMS` — parameters for a conditional-node
/// variant inside [`CUgraphNodeParams`].
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
pub struct CUDA_CONDITIONAL_NODE_PARAMS {
    pub handle: CUgraphConditionalHandle,
    pub type_: core::ffi::c_int,
    pub size: core::ffi::c_uint,
    pub body_graph_out: *mut CUgraph,
    pub ctx: CUcontext,
}

impl Default for CUDA_CONDITIONAL_NODE_PARAMS {
    fn default() -> Self {
        Self {
            handle: 0,
            type_: CUgraphConditionalNodeType::IF,
            size: 1,
            body_graph_out: core::ptr::null_mut(),
            ctx: core::ptr::null_mut(),
        }
    }
}

/// `CUgraphNodeParams` — generic node-params tagged union. We model the
/// payload as an opaque `[u64; 30]` (large enough to hold any variant)
/// plus the discriminant and tail. Safe wrappers populate the payload via
/// typed helpers; raw users can cast through `payload.as_mut_ptr()`.
///
/// Layout (CUDA 13.x):
/// ```text
/// struct CUgraphNodeParams {
///     CUgraphNodeType type;     // 4
///     int reserved0[3];         // 12
///     union { ... } payload;    // 232 bytes (29 × c_longlong)
///     long long reserved2;      // 8
/// };
/// ```
/// Total 256 bytes, alignment 8.
#[repr(C)]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
pub struct CUgraphNodeParams {
    pub type_: core::ffi::c_int,
    pub reserved0: [core::ffi::c_int; 3],
    pub payload: [u64; 29],
    pub reserved2: core::ffi::c_longlong,
}

impl Default for CUgraphNodeParams {
    fn default() -> Self {
        Self {
            type_: CUgraphNodeType::EMPTY,
            reserved0: [0; 3],
            payload: [0; 29],
            reserved2: 0,
        }
    }
}

impl core::fmt::Debug for CUgraphNodeParams {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CUgraphNodeParams")
            .field("type", &self.type_)
            .finish_non_exhaustive()
    }
}

/// `CUgraphEdgeData` — optional edge metadata (used by the v2 add-node /
/// add-dependencies APIs). 8 bytes.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
#[allow(non_camel_case_types)]
pub struct CUgraphEdgeData {
    pub from_port: u8,
    pub to_port: u8,
    pub type_: u8,
    pub reserved: [u8; 5],
}

/// `CUmulticastObjectProp` — creation props for `cuMulticastCreate`.
/// CUDA 12.0+, NVSwitch-only.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
#[allow(non_camel_case_types)]
pub struct CUmulticastObjectProp {
    pub num_devices: core::ffi::c_uint,
    pub size: usize,
    pub handle_types: u64,
    pub flags: u64,
}

/// `CUmulticastGranularity_flags` — pass to `cuMulticastGetGranularity`.
#[allow(non_snake_case)]
pub mod CUmulticastGranularity_flags {
    pub const MINIMUM: i32 = 0;
    pub const RECOMMENDED: i32 = 1;
}

/// `CUdevResourceType` — green-context resource-kind enum (CUDA 12.4+).
#[allow(non_snake_case)]
pub mod CUdevResourceType {
    pub const INVALID: i32 = 0;
    pub const SM: i32 = 1;
}

/// `CUdevSmResource` — SM-count resource payload (12 bytes).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
#[allow(non_camel_case_types)]
pub struct CUdevSmResource {
    pub sm_count: core::ffi::c_uint,
    pub min_sm_partition_size: core::ffi::c_uint,
    pub sm_coscheduled_alignment: core::ffi::c_uint,
}

/// `CUdevResource` — 144-byte resource descriptor. Tagged by `type_`;
/// the 48-byte union holds the variant-specific payload (`CUdevSmResource`
/// for `SM`). We model the union as a fixed `[u64; 6]` blob and provide
/// helpers for the SM case.
#[repr(C)]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
pub struct CUdevResource {
    pub type_: core::ffi::c_int,
    pub internal_padding: [core::ffi::c_uchar; 92],
    pub res: [u64; 6], // 48-byte union
}

impl Default for CUdevResource {
    fn default() -> Self {
        Self {
            type_: CUdevResourceType::INVALID,
            internal_padding: [0u8; 92],
            res: [0u64; 6],
        }
    }
}

impl core::fmt::Debug for CUdevResource {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CUdevResource")
            .field("type", &self.type_)
            .finish_non_exhaustive()
    }
}

impl CUdevResource {
    /// View the SM-specific payload. Only meaningful when `type_ == SM`.
    #[inline]
    pub fn as_sm(&self) -> CUdevSmResource {
        // SAFETY: res is 48 bytes 8-byte aligned; CUdevSmResource is 12
        // bytes 4-byte aligned — reading the first 12 bytes is well-defined.
        unsafe { core::ptr::read(self.res.as_ptr() as *const CUdevSmResource) }
    }
}

/// `CUgraphNodeType` — returned by `cuGraphNodeGetType`.
#[allow(non_snake_case)]
pub mod CUgraphNodeType {
    pub const KERNEL: i32 = 0;
    pub const MEMCPY: i32 = 1;
    pub const MEMSET: i32 = 2;
    pub const HOST: i32 = 3;
    pub const GRAPH: i32 = 4;
    pub const EMPTY: i32 = 5;
    pub const WAIT_EVENT: i32 = 6;
    pub const EVENT_RECORD: i32 = 7;
    pub const EXT_SEMAS_SIGNAL: i32 = 8;
    pub const EXT_SEMAS_WAIT: i32 = 9;
    pub const MEM_ALLOC: i32 = 10;
    pub const MEM_FREE: i32 = 11;
    pub const BATCH_MEM_OP: i32 = 12;
    pub const CONDITIONAL: i32 = 13;
}

/// Stream-capture mode (passed to `cuStreamBeginCapture`).
#[allow(non_snake_case)]
pub mod CUstreamCaptureMode {
    /// Operations on any stream in the process are captured while this
    /// thread's chosen stream is capturing. Discouraged in modern code.
    pub const GLOBAL: u32 = 0;
    /// Only operations on streams whose capture was initiated from the
    /// current thread are captured. Recommended.
    pub const THREAD_LOCAL: u32 = 1;
    /// Permissive mode — allows unsynchronized cross-stream activity.
    pub const RELAXED: u32 = 2;
}

/// Stream-capture status (returned by `cuStreamIsCapturing`).
#[allow(non_snake_case)]
pub mod CUstreamCaptureStatus {
    pub const NONE: u32 = 0;
    pub const ACTIVE: u32 = 1;
    pub const INVALIDATED: u32 = 2;
}

/// Flags for `cuGraphInstantiateWithFlags`.
#[allow(non_snake_case)]
pub mod CUgraphInstantiate_flags {
    /// Automatically free allocations created in the graph after launch completes.
    pub const AUTO_FREE_ON_LAUNCH: u64 = 1;
    /// Upload the executable graph to the device immediately on instantiate.
    pub const UPLOAD: u64 = 2;
    /// Use node priorities when scheduling.
    pub const USE_NODE_PRIORITY: u64 = 8;
}

/// Device attribute selector (subset of `CUdevice_attribute`).
#[allow(non_snake_case)]
pub mod CUdevice_attribute {
    pub const MAX_THREADS_PER_BLOCK: i32 = 1;
    pub const MAX_BLOCK_DIM_X: i32 = 2;
    pub const MAX_BLOCK_DIM_Y: i32 = 3;
    pub const MAX_BLOCK_DIM_Z: i32 = 4;
    pub const MAX_GRID_DIM_X: i32 = 5;
    pub const MAX_GRID_DIM_Y: i32 = 6;
    pub const MAX_GRID_DIM_Z: i32 = 7;
    pub const MAX_SHARED_MEMORY_PER_BLOCK: i32 = 8;
    pub const TOTAL_CONSTANT_MEMORY: i32 = 9;
    pub const WARP_SIZE: i32 = 10;
    pub const MAX_PITCH: i32 = 11;
    pub const MAX_REGISTERS_PER_BLOCK: i32 = 12;
    pub const CLOCK_RATE: i32 = 13;
    pub const TEXTURE_ALIGNMENT: i32 = 14;
    pub const MULTIPROCESSOR_COUNT: i32 = 16;
    pub const INTEGRATED: i32 = 18;
    pub const COMPUTE_CAPABILITY_MAJOR: i32 = 75;
    pub const COMPUTE_CAPABILITY_MINOR: i32 = 76;
    pub const PCI_BUS_ID: i32 = 33;
    pub const PCI_DEVICE_ID: i32 = 34;
    pub const PCI_DOMAIN_ID: i32 = 50;
    pub const CONCURRENT_KERNELS: i32 = 31;
    pub const ECC_ENABLED: i32 = 32;
}

// ---- Wave 6: arrays, textures, surfaces ----------------------------------

/// `CUarray_format` — scalar format of an array's texels.
#[allow(non_snake_case)]
pub mod CUarray_format {
    pub const UNSIGNED_INT8: u32 = 0x01;
    pub const UNSIGNED_INT16: u32 = 0x02;
    pub const UNSIGNED_INT32: u32 = 0x03;
    pub const SIGNED_INT8: u32 = 0x08;
    pub const SIGNED_INT16: u32 = 0x09;
    pub const SIGNED_INT32: u32 = 0x0a;
    pub const HALF: u32 = 0x10;
    pub const FLOAT: u32 = 0x20;
}

/// `CUaddress_mode` — out-of-bounds behavior for texture sampling.
#[allow(non_snake_case)]
pub mod CUaddress_mode {
    pub const WRAP: u32 = 0;
    pub const CLAMP: u32 = 1;
    pub const MIRROR: u32 = 2;
    pub const BORDER: u32 = 3;
}

/// `CUfilter_mode` — point vs. linear filtering.
#[allow(non_snake_case)]
pub mod CUfilter_mode {
    pub const POINT: u32 = 0;
    pub const LINEAR: u32 = 1;
}

/// `CUresourcetype` — tag for the variant inside a [`CUDA_RESOURCE_DESC`].
#[allow(non_snake_case)]
pub mod CUresourcetype {
    pub const ARRAY: u32 = 0;
    pub const MIPMAPPED_ARRAY: u32 = 1;
    pub const LINEAR: u32 = 2;
    pub const PITCH2D: u32 = 3;
}

/// `CUDA_ARRAY_DESCRIPTOR` — shape passed to `cuArrayCreate_v2`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
#[allow(non_camel_case_types)]
pub struct CUDA_ARRAY_DESCRIPTOR {
    pub width: usize,
    pub height: usize,
    pub format: u32,
    pub num_channels: core::ffi::c_uint,
}

/// `CUDA_RESOURCE_DESC` — untagged union of resource-type variants with a
/// leading discriminant.
///
/// The C layout is:
/// ```text
/// struct CUDA_RESOURCE_DESC {
///     CUresourcetype resType;    // c_int (4 bytes)
///     // 4 bytes padding (union is 8-byte aligned due to CUdeviceptr / pointers inside)
///     union { ... } res;         // 128 bytes, 8-byte aligned
///     unsigned int flags;        // 4 bytes
///     // 4 bytes tail padding to keep the struct 8-byte aligned overall
/// };
/// ```
/// Total size: 144 bytes. We model the union as `[u64; 16]` (128 bytes,
/// align 8) which reproduces the correct layout for free.
#[repr(C)]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
pub struct CUDA_RESOURCE_DESC {
    pub res_type: core::ffi::c_int,
    _pad0: u32,
    /// Variant-specific payload (128 bytes, `int reserved[32]` in `cuda.h`).
    pub res: [u64; 16],
    pub flags: core::ffi::c_uint,
    _pad1: u32,
}

impl Default for CUDA_RESOURCE_DESC {
    fn default() -> Self {
        Self {
            res_type: CUresourcetype::ARRAY as core::ffi::c_int,
            _pad0: 0,
            res: [0u64; 16],
            flags: 0,
            _pad1: 0,
        }
    }
}

impl core::fmt::Debug for CUDA_RESOURCE_DESC {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CUDA_RESOURCE_DESC")
            .field("res_type", &self.res_type)
            .field("flags", &self.flags)
            .finish_non_exhaustive()
    }
}

impl CUDA_RESOURCE_DESC {
    /// Point this descriptor at a [`CUarray`] (resource-type `ARRAY`). The
    /// pointer is only *stored* in the union — never dereferenced by us —
    /// so we do not require an `unsafe` boundary here.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn from_array(array: CUarray) -> Self {
        let mut d = Self::default();
        // The `array` variant of the union is `struct { CUarray hArray; }`,
        // placed at offset 0 of the union (which is offset 8 of the outer
        // struct, after `res_type` + padding). [u64; 16] is 8-byte aligned,
        // so writing a pointer at `res[0]` is well-defined.
        //
        // SAFETY: `res` is 128 bytes and 8-byte aligned; `CUarray` is a
        // pointer (8 bytes), fits at offset 0.
        unsafe {
            let p = d.res.as_mut_ptr() as *mut CUarray;
            p.write(array);
        }
        d
    }
}

/// `CUDA_TEXTURE_DESC` — texture-sampling parameters.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
pub struct CUDA_TEXTURE_DESC {
    pub address_mode: [u32; 3],
    pub filter_mode: u32,
    pub flags: core::ffi::c_uint,
    pub max_anisotropy: core::ffi::c_uint,
    pub mipmap_filter_mode: u32,
    pub mipmap_level_bias: f32,
    pub min_mipmap_level_clamp: f32,
    pub max_mipmap_level_clamp: f32,
    pub border_color: [f32; 4],
    pub reserved: [core::ffi::c_int; 12],
}

impl Default for CUDA_TEXTURE_DESC {
    fn default() -> Self {
        Self {
            address_mode: [CUaddress_mode::CLAMP; 3],
            filter_mode: CUfilter_mode::POINT,
            flags: 0,
            max_anisotropy: 0,
            mipmap_filter_mode: CUfilter_mode::POINT,
            mipmap_level_bias: 0.0,
            min_mipmap_level_clamp: 0.0,
            max_mipmap_level_clamp: 0.0,
            border_color: [0.0; 4],
            reserved: [0; 12],
        }
    }
}

// ---- Wave 7: virtual memory management (VMM) ----------------------------

/// `CUmemAllocationType` — what kind of physical backing to create.
#[allow(non_snake_case)]
pub mod CUmemAllocationType {
    pub const INVALID: i32 = 0;
    pub const PINNED: i32 = 1;
}

/// `CUmemLocationType` — identifies what device/host the backing lives on.
#[allow(non_snake_case)]
pub mod CUmemLocationType {
    pub const INVALID: i32 = 0;
    pub const DEVICE: i32 = 1;
    pub const HOST: i32 = 2;
    pub const HOST_NUMA: i32 = 3;
    pub const HOST_NUMA_CURRENT: i32 = 4;
}

/// `CUmemAllocationHandleType` — OS-level handle shape for IPC sharing.
#[allow(non_snake_case)]
pub mod CUmemAllocationHandleType {
    pub const NONE: i32 = 0;
    pub const POSIX_FILE_DESCRIPTOR: i32 = 1;
    pub const WIN32: i32 = 2;
    pub const WIN32_KMT: i32 = 4;
    pub const FABRIC: i32 = 8;
}

/// `CUmemAccess_flags` — access rights granted by `cuMemSetAccess`.
#[allow(non_snake_case)]
pub mod CUmemAccess_flags {
    pub const NONE: i32 = 0;
    pub const READ: i32 = 1;
    pub const READWRITE: i32 = 3;
}

/// `CUmemAllocationGranularity_flags` — pass to
/// `cuMemGetAllocationGranularity`.
#[allow(non_snake_case)]
pub mod CUmemAllocationGranularity_flags {
    pub const MINIMUM: i32 = 0;
    pub const RECOMMENDED: i32 = 1;
}

/// `CUmemLocation` — `(type, id)` pair identifying a device or NUMA node.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
#[allow(non_camel_case_types)]
pub struct CUmemLocation {
    pub type_: core::ffi::c_int,
    pub id: core::ffi::c_int,
}

/// Inline flag block inside [`CUmemAllocationProp`] (8 bytes).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
#[allow(non_camel_case_types)]
pub struct CUmemAllocationPropFlags {
    pub compression_type: core::ffi::c_uchar,
    pub gpu_direct_rdma_capable: core::ffi::c_uchar,
    pub usage: core::ffi::c_ushort,
    pub reserved: [core::ffi::c_uchar; 4],
}

/// `CUmemAllocationProp` — passed to `cuMemCreate` to describe what kind
/// of backing (type, location, IPC handle shape) to produce.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
#[allow(non_camel_case_types)]
pub struct CUmemAllocationProp {
    pub type_: core::ffi::c_int,
    pub requested_handle_types: core::ffi::c_int,
    pub location: CUmemLocation,
    pub win32_handle_meta_data: *mut c_void,
    pub alloc_flags: CUmemAllocationPropFlags,
}

/// `CUmemAccessDesc` — passed to `cuMemSetAccess` to grant a device
/// `flags` access to a virtual-memory range.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
#[allow(non_camel_case_types)]
pub struct CUmemAccessDesc {
    pub location: CUmemLocation,
    pub flags: core::ffi::c_int,
}

// ---- Wave 8: memory pools -----------------------------------------------

/// `CUmemPool_attribute` — pass to `cuMemPoolSetAttribute` / `GetAttribute`.
#[allow(non_snake_case)]
pub mod CUmemPool_attribute {
    pub const REUSE_FOLLOW_EVENT_DEPENDENCIES: i32 = 1;
    pub const REUSE_ALLOW_OPPORTUNISTIC: i32 = 2;
    pub const REUSE_ALLOW_INTERNAL_DEPENDENCIES: i32 = 3;
    pub const RELEASE_THRESHOLD: i32 = 4;
    pub const RESERVED_MEM_CURRENT: i32 = 5;
    pub const RESERVED_MEM_HIGH: i32 = 6;
    pub const USED_MEM_CURRENT: i32 = 7;
    pub const USED_MEM_HIGH: i32 = 8;
}

/// `CUmemPoolProps` — creation props for `cuMemPoolCreate`. 88 bytes in C.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
pub struct CUmemPoolProps {
    pub alloc_type: core::ffi::c_int,
    pub handle_types: core::ffi::c_int,
    pub location: CUmemLocation,
    pub win32_security_attributes: *mut c_void,
    pub max_size: usize,
    pub usage: core::ffi::c_ushort,
    pub reserved: [core::ffi::c_uchar; 54],
}

impl Default for CUmemPoolProps {
    fn default() -> Self {
        Self {
            alloc_type: CUmemAllocationType::PINNED,
            handle_types: CUmemAllocationHandleType::NONE,
            location: CUmemLocation::default(),
            win32_security_attributes: core::ptr::null_mut(),
            max_size: 0,
            usage: 0,
            reserved: [0u8; 54],
        }
    }
}

/// `CUmemPoolPtrExportData` — opaque 64-byte blob returned by
/// `cuMemPoolExportPointer`.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
pub struct CUmemPoolPtrExportData {
    pub reserved: [core::ffi::c_uchar; 64],
}

impl Default for CUmemPoolPtrExportData {
    fn default() -> Self {
        Self {
            reserved: [0u8; 64],
        }
    }
}

// ---- Wave 9: external memory / semaphore interop ------------------------

/// `CUexternalMemoryHandleType` — which OS handle shape you're importing.
#[allow(non_snake_case)]
pub mod CUexternalMemoryHandleType {
    pub const OPAQUE_FD: i32 = 1;
    pub const OPAQUE_WIN32: i32 = 2;
    pub const OPAQUE_WIN32_KMT: i32 = 3;
    pub const D3D12_HEAP: i32 = 4;
    pub const D3D12_RESOURCE: i32 = 5;
    pub const D3D11_RESOURCE: i32 = 6;
    pub const D3D11_RESOURCE_KMT: i32 = 7;
    pub const NVSCIBUF: i32 = 8;
}

/// `CUexternalSemaphoreHandleType` — OS handle shape for imported sem.
#[allow(non_snake_case)]
pub mod CUexternalSemaphoreHandleType {
    pub const OPAQUE_FD: i32 = 1;
    pub const OPAQUE_WIN32: i32 = 2;
    pub const OPAQUE_WIN32_KMT: i32 = 3;
    pub const D3D12_FENCE: i32 = 4;
    pub const D3D11_FENCE: i32 = 5;
    pub const NVSCISYNC: i32 = 6;
    pub const KEYED_MUTEX: i32 = 7;
    pub const KEYED_MUTEX_KMT: i32 = 8;
    pub const TIMELINE_SEMAPHORE_FD: i32 = 9;
    pub const TIMELINE_SEMAPHORE_WIN32: i32 = 10;
}

// `CUexternalMemory` and `CUexternalSemaphore` are declared near the top
// of this module alongside the other opaque handles.

/// Opaque mipmapped CUDA array handle (CUDA 5+).
pub type CUmipmappedArray = *mut c_void;

/// Opaque user-object handle (CUDA 12.0+) — refcounted RAII slot for
/// attaching external resources to CUDA graphs.
pub type CUuserObject = *mut c_void;

/// Opaque graphics-resource handle (registered GL buffer, D3D resource,
/// VDPAU surface, EGL image, ...). See the `cuGraphics*` API family.
pub type CUgraphicsResource = *mut c_void;

/// Opaque CUDA Logs callback registration (CUDA 12.9+).
pub type CUlogsCallbackHandle = *mut c_void;

/// `CUlogIterator` — 32-bit cursor into the driver's in-memory log ring.
pub type CUlogIterator = core::ffi::c_uint;

/// `CUlogLevel` returned to [`CUlogsCallback`].
#[allow(non_snake_case)]
pub mod CUlogLevel {
    pub const ERROR: i32 = 0;
    pub const WARNING: i32 = 1;
    pub const INFO: i32 = 2;
    pub const TRACE: i32 = 3;
}

/// `CUlogsCallback` — the function pointer passed to `cuLogsRegisterCallback`.
pub type CUlogsCallback = Option<
    unsafe extern "C" fn(
        data: *mut c_void,
        log_level: core::ffi::c_int,
        message: *const core::ffi::c_char,
        len: core::ffi::c_uint,
    ),
>;

/// `CUmoduleLoadingMode` — reported by `cuModuleGetLoadingMode`.
#[allow(non_snake_case)]
pub mod CUmoduleLoadingMode {
    pub const EAGER_LOADING: i32 = 0x1;
    pub const LAZY_LOADING: i32 = 0x2;
}

// ---- Wave 24: graph memory nodes + graph-exec update --------------------

/// `CUgraphExecUpdateResult` — outcome code returned from
/// `cuGraphExecUpdate_v2` via [`CUgraphExecUpdateResultInfo::result`].
#[allow(non_snake_case)]
pub mod CUgraphExecUpdateResult {
    pub const SUCCESS: i32 = 0;
    pub const ERROR: i32 = 1;
    pub const ERROR_TOPOLOGY_CHANGED: i32 = 2;
    pub const ERROR_NODE_TYPE_CHANGED: i32 = 3;
    pub const ERROR_FUNCTION_CHANGED: i32 = 4;
    pub const ERROR_PARAMETERS_CHANGED: i32 = 5;
    pub const ERROR_NOT_SUPPORTED: i32 = 6;
    pub const ERROR_UNSUPPORTED_FUNCTION_CHANGE: i32 = 7;
    pub const ERROR_ATTRIBUTES_CHANGED: i32 = 8;
}

/// `CUgraphExecUpdateResultInfo` — filled by `cuGraphExecUpdate_v2`
/// on (partial) failure to identify which node diverged.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
pub struct CUgraphExecUpdateResultInfo {
    pub result: core::ffi::c_int,
    pub error_node: CUgraphNode,
    pub error_from_node: CUgraphNode,
}

impl Default for CUgraphExecUpdateResultInfo {
    fn default() -> Self {
        Self {
            result: CUgraphExecUpdateResult::SUCCESS,
            error_node: core::ptr::null_mut(),
            error_from_node: core::ptr::null_mut(),
        }
    }
}

/// `CUgraphMem_attribute` — selector for per-device graph-mem limits.
#[allow(non_snake_case)]
pub mod CUgraphMem_attribute {
    pub const USED_MEM_CURRENT: i32 = 0;
    pub const USED_MEM_HIGH: i32 = 1;
    pub const RESERVED_MEM_CURRENT: i32 = 2;
    pub const RESERVED_MEM_HIGH: i32 = 3;
}

/// `CUDA_MEM_ALLOC_NODE_PARAMS` — description passed to
/// `cuGraphAddMemAllocNode`. `dptr` is written by CUDA on successful add
/// (it's the address the node will allocate when the graph runs).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
pub struct CUDA_MEM_ALLOC_NODE_PARAMS {
    pub pool_props: CUmemPoolProps,
    pub access_descs: *const CUmemAccessDesc,
    pub access_desc_count: usize,
    pub bytesize: usize,
    pub dptr: CUdeviceptr,
}

impl Default for CUDA_MEM_ALLOC_NODE_PARAMS {
    fn default() -> Self {
        Self {
            pool_props: CUmemPoolProps::default(),
            access_descs: core::ptr::null(),
            access_desc_count: 0,
            bytesize: 0,
            dptr: CUdeviceptr(0),
        }
    }
}

/// `CUstreamBatchMemOpType` — operation code inside a batch-memop entry.
#[allow(non_snake_case)]
pub mod CUstreamBatchMemOpType {
    pub const WAIT_VALUE_32: u32 = 1;
    pub const WRITE_VALUE_32: u32 = 2;
    pub const WAIT_VALUE_64: u32 = 4;
    pub const WRITE_VALUE_64: u32 = 5;
    pub const BARRIER: u32 = 6;
    pub const FLUSH_REMOTE_WRITES: u32 = 3;
}

/// `CUstreamWriteValue_flags` / `CUstreamWaitValue_flags` — bitmask for
/// the individual stream-value ops and their batch-memop equivalents.
#[allow(non_snake_case)]
pub mod CUstreamWaitValue_flags {
    pub const GEQ: u32 = 0x0;
    pub const EQ: u32 = 0x1;
    pub const AND: u32 = 0x2;
    pub const NOR: u32 = 0x3;
    pub const FLUSH: u32 = 1 << 30;
}

#[allow(non_snake_case)]
pub mod CUstreamWriteValue_flags {
    pub const DEFAULT: u32 = 0x0;
    pub const NO_MEMORY_BARRIER: u32 = 0x1;
}

/// `CUstreamBatchMemOpParams` — 48-byte tagged-union entry in a batched
/// stream memory-op array. We model it as `[u64; 6]` and provide typed
/// builders (`wait_value_32`, `write_value_64`, ...).
#[repr(C)]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
pub struct CUstreamBatchMemOpParams {
    pub raw: [u64; 6],
}

#[allow(clippy::derivable_impls)]
impl Default for CUstreamBatchMemOpParams {
    fn default() -> Self {
        Self { raw: [0; 6] }
    }
}

impl core::fmt::Debug for CUstreamBatchMemOpParams {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CUstreamBatchMemOpParams")
            .field("op", &(self.raw[0] as u32))
            .finish_non_exhaustive()
    }
}

/// Fixed layout of a wait-value or write-value entry (32 bytes, with
/// padding the rest of the 48-byte union).
///
/// Field offsets:
/// - [0..4]   operation (u32)
/// - [4..8]   reserved / pad
/// - [8..16]  address (CUdeviceptr)
/// - [16..24] value / value64 (u32 or u64)
/// - [24..28] flags (u32)
/// - [28..32] pad
/// - [32..40] alias (CUdeviceptr)
/// - [40..48] pad
impl CUstreamBatchMemOpParams {
    /// Build a `WaitValue32` entry.
    pub fn wait_value_32(address: CUdeviceptr, value: u32, flags: u32) -> Self {
        let mut s = Self::default();
        unsafe {
            let p = s.raw.as_mut_ptr() as *mut u8;
            (p as *mut u32).write(CUstreamBatchMemOpType::WAIT_VALUE_32);
            (p.add(8) as *mut u64).write(address.0);
            (p.add(16) as *mut u32).write(value);
            (p.add(24) as *mut u32).write(flags);
        }
        s
    }

    pub fn wait_value_64(address: CUdeviceptr, value: u64, flags: u32) -> Self {
        let mut s = Self::default();
        unsafe {
            let p = s.raw.as_mut_ptr() as *mut u8;
            (p as *mut u32).write(CUstreamBatchMemOpType::WAIT_VALUE_64);
            (p.add(8) as *mut u64).write(address.0);
            (p.add(16) as *mut u64).write(value);
            (p.add(24) as *mut u32).write(flags);
        }
        s
    }

    pub fn write_value_32(address: CUdeviceptr, value: u32, flags: u32) -> Self {
        let mut s = Self::default();
        unsafe {
            let p = s.raw.as_mut_ptr() as *mut u8;
            (p as *mut u32).write(CUstreamBatchMemOpType::WRITE_VALUE_32);
            (p.add(8) as *mut u64).write(address.0);
            (p.add(16) as *mut u32).write(value);
            (p.add(24) as *mut u32).write(flags);
        }
        s
    }

    pub fn write_value_64(address: CUdeviceptr, value: u64, flags: u32) -> Self {
        let mut s = Self::default();
        unsafe {
            let p = s.raw.as_mut_ptr() as *mut u8;
            (p as *mut u32).write(CUstreamBatchMemOpType::WRITE_VALUE_64);
            (p.add(8) as *mut u64).write(address.0);
            (p.add(16) as *mut u64).write(value);
            (p.add(24) as *mut u32).write(flags);
        }
        s
    }
}

/// `CUDA_BATCH_MEM_OP_NODE_PARAMS` — the fields passed to
/// `cuGraphAddBatchMemOpNode`.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
pub struct CUDA_BATCH_MEM_OP_NODE_PARAMS {
    pub ctx: CUcontext,
    pub count: core::ffi::c_uint,
    pub param_array: *mut CUstreamBatchMemOpParams,
    pub flags: core::ffi::c_uint,
}

impl Default for CUDA_BATCH_MEM_OP_NODE_PARAMS {
    fn default() -> Self {
        Self {
            ctx: core::ptr::null_mut(),
            count: 0,
            param_array: core::ptr::null_mut(),
            flags: 0,
        }
    }
}

/// Opaque green-context handle (CUDA 12.4+).
pub type CUgreenCtx = *mut c_void;

/// Opaque device-resource descriptor handle (produced by
/// `cuDevResourceGenerateDesc`, consumed by `cuGreenCtxCreate`).
pub type CUdevResourceDesc = *mut c_void;

/// `CUDA_EXTERNAL_MEMORY_HANDLE_DESC` — union-bearing import descriptor.
/// The 16-byte `handle` slot holds either an `int fd`, a pair of
/// `(HANDLE, LPCWSTR)` pointers, or an nvSciBuf object pointer.
/// See [`CUDA_EXTERNAL_MEMORY_HANDLE_DESC::from_win32_handle`] and
/// [`from_fd`](CUDA_EXTERNAL_MEMORY_HANDLE_DESC::from_fd) for the common cases.
#[repr(C)]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
pub struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
    pub type_: core::ffi::c_int,
    _pad0: u32,
    /// Union payload: max member is `{ HANDLE, LPCWSTR }` = 16 bytes, align 8.
    pub handle: [u64; 2],
    pub size: u64,
    pub flags: core::ffi::c_uint,
    pub reserved: [core::ffi::c_uint; 16],
}

#[allow(clippy::derivable_impls)]
impl Default for CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
    fn default() -> Self {
        Self {
            type_: 0,
            _pad0: 0,
            handle: [0; 2],
            size: 0,
            flags: 0,
            reserved: [0; 16],
        }
    }
}

impl core::fmt::Debug for CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CUDA_EXTERNAL_MEMORY_HANDLE_DESC")
            .field("type", &self.type_)
            .field("size", &self.size)
            .field("flags", &self.flags)
            .finish_non_exhaustive()
    }
}

impl CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
    /// Import from a POSIX file descriptor (Linux / NvSci).
    pub fn from_fd(fd: core::ffi::c_int, size: u64) -> Self {
        let mut d = Self {
            type_: CUexternalMemoryHandleType::OPAQUE_FD,
            size,
            ..Default::default()
        };
        // handle.fd lives in the first 4 bytes of the union.
        let slot = d.handle.as_mut_ptr() as *mut core::ffi::c_int;
        unsafe { slot.write(fd) };
        d
    }

    /// Import from a Windows NT HANDLE (or optional Unicode name for a
    /// named object). Leaves `name` as null when unused.
    ///
    /// # Safety
    ///
    /// `handle` and (if non-null) `name` must be live OS objects for the
    /// duration of the resulting `cuImportExternalMemory` call.
    pub unsafe fn from_win32_handle(
        type_: core::ffi::c_int,
        handle: *mut c_void,
        name: *const c_void,
        size: u64,
    ) -> Self {
        let mut d = Self {
            type_,
            size,
            ..Default::default()
        };
        // handle.win32 = { HANDLE, LPCWSTR } at offset 0 of the union.
        let p = d.handle.as_mut_ptr() as *mut [*mut c_void; 2];
        unsafe { p.write([handle, name as *mut c_void]) };
        d
    }
}

/// `CUDA_EXTERNAL_MEMORY_BUFFER_DESC` — offset + size subregion of an
/// imported external memory to expose as a device pointer.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
#[allow(non_camel_case_types)]
pub struct CUDA_EXTERNAL_MEMORY_BUFFER_DESC {
    pub offset: u64,
    pub size: u64,
    pub flags: core::ffi::c_uint,
    pub reserved: [core::ffi::c_uint; 16],
}

/// `CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC` — same shape as the memory
/// handle desc but without the trailing `size`.
#[repr(C)]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
pub struct CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC {
    pub type_: core::ffi::c_int,
    _pad0: u32,
    pub handle: [u64; 2],
    pub flags: core::ffi::c_uint,
    pub reserved: [core::ffi::c_uint; 16],
}

#[allow(clippy::derivable_impls)]
impl Default for CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC {
    fn default() -> Self {
        Self {
            type_: 0,
            _pad0: 0,
            handle: [0; 2],
            flags: 0,
            reserved: [0; 16],
        }
    }
}

impl core::fmt::Debug for CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC")
            .field("type", &self.type_)
            .field("flags", &self.flags)
            .finish_non_exhaustive()
    }
}

impl CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC {
    pub fn from_fd(fd: core::ffi::c_int, type_: core::ffi::c_int) -> Self {
        let mut d = Self {
            type_,
            ..Default::default()
        };
        let slot = d.handle.as_mut_ptr() as *mut core::ffi::c_int;
        unsafe { slot.write(fd) };
        d
    }

    /// # Safety
    ///
    /// `handle` and (if non-null) `name` must be live OS objects.
    pub unsafe fn from_win32_handle(
        type_: core::ffi::c_int,
        handle: *mut c_void,
        name: *const c_void,
    ) -> Self {
        let mut d = Self {
            type_,
            ..Default::default()
        };
        let p = d.handle.as_mut_ptr() as *mut [*mut c_void; 2];
        unsafe { p.write([handle, name as *mut c_void]) };
        d
    }
}

/// `CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS` — which value to signal.
/// The `params` union has three members (fence / nvSciSync / keyedMutex)
/// plus reserved, totalling 72 bytes (8 for value + 8 nvSci + 8 key +
/// 48 reserved).
#[repr(C)]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
pub struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS {
    /// Layout:
    /// - [0..8]   fence.value (u64) — the fence value to signal.
    /// - [8..16]  nvSciSync.{fence|reserved} (pointer or u64).
    /// - [16..24] keyedMutex.key (u64).
    /// - [24..72] reserved[12] u32.
    pub params: [u64; 9],
    pub flags: core::ffi::c_uint,
    pub reserved: [core::ffi::c_uint; 16],
}

#[allow(clippy::derivable_impls)]
impl Default for CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS {
    fn default() -> Self {
        Self {
            params: [0; 9],
            flags: 0,
            reserved: [0; 16],
        }
    }
}

impl core::fmt::Debug for CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS")
            .field("fence_value", &self.params[0])
            .field("flags", &self.flags)
            .finish_non_exhaustive()
    }
}

impl CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS {
    /// Signal-fence-value helper for D3D12/Vulkan timeline semaphores.
    pub fn fence_value(value: u64) -> Self {
        let mut s = Self::default();
        s.params[0] = value;
        s
    }
}

/// `CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS` — same layout but `keyedMutex`
/// has an extra `timeoutMs: u32` (so the overall params size is the same
/// 72 bytes, just different reserved count).
#[repr(C)]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
pub struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS {
    pub params: [u64; 9],
    pub flags: core::ffi::c_uint,
    pub reserved: [core::ffi::c_uint; 16],
}

#[allow(clippy::derivable_impls)]
impl Default for CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS {
    fn default() -> Self {
        Self {
            params: [0; 9],
            flags: 0,
            reserved: [0; 16],
        }
    }
}

impl core::fmt::Debug for CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS")
            .field("fence_value", &self.params[0])
            .field("flags", &self.flags)
            .finish_non_exhaustive()
    }
}

impl CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS {
    pub fn fence_value(value: u64) -> Self {
        let mut s = Self::default();
        s.params[0] = value;
        s
    }
}

// ---- Wave 10: 3D memcpy + 3D arrays + mipmapped arrays ------------------

/// `CUDA_ARRAY3D_DESCRIPTOR` — shape passed to `cuArray3DCreate_v2` and
/// `cuMipmappedArrayCreate`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
#[allow(non_camel_case_types)]
pub struct CUDA_ARRAY3D_DESCRIPTOR {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub format: u32,
    pub num_channels: core::ffi::c_uint,
    pub flags: core::ffi::c_uint,
}

/// `CUarray3D_flags` — creation-time flags for 3D / mipmapped arrays.
#[allow(non_snake_case)]
pub mod CUarray3D_flags {
    pub const LAYERED: u32 = 0x01;
    pub const SURFACE_LDST: u32 = 0x02;
    pub const CUBEMAP: u32 = 0x04;
    pub const TEXTURE_GATHER: u32 = 0x08;
    pub const DEPTH_TEXTURE: u32 = 0x10;
    pub const COLOR_ATTACHMENT: u32 = 0x20;
    pub const SPARSE: u32 = 0x40;
    pub const DEFERRED_MAPPING: u32 = 0x80;
}

/// `CUDA_MEMCPY3D` — 3-D memcpy descriptor. 200 bytes.
///
/// Populate `src_*` and `dst_*` fields according to each side's
/// [`CUmemorytype`]: `HOST` uses the host-pointer fields, `DEVICE` uses
/// the device-pointer + pitch, `ARRAY` uses the array-handle fields.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
pub struct CUDA_MEMCPY3D {
    pub src_x_in_bytes: usize,
    pub src_y: usize,
    pub src_z: usize,
    pub src_lod: usize,
    pub src_memory_type: u32,
    _pad0: u32,
    pub src_host: *const c_void,
    pub src_device: CUdeviceptr,
    pub src_array: *mut c_void,
    pub reserved0: *mut c_void,
    pub src_pitch: usize,
    pub src_height: usize,

    pub dst_x_in_bytes: usize,
    pub dst_y: usize,
    pub dst_z: usize,
    pub dst_lod: usize,
    pub dst_memory_type: u32,
    _pad1: u32,
    pub dst_host: *mut c_void,
    pub dst_device: CUdeviceptr,
    pub dst_array: *mut c_void,
    pub reserved1: *mut c_void,
    pub dst_pitch: usize,
    pub dst_height: usize,

    pub width_in_bytes: usize,
    pub height: usize,
    pub depth: usize,
}

impl Default for CUDA_MEMCPY3D {
    fn default() -> Self {
        Self {
            src_x_in_bytes: 0,
            src_y: 0,
            src_z: 0,
            src_lod: 0,
            src_memory_type: 0,
            _pad0: 0,
            src_host: core::ptr::null(),
            src_device: CUdeviceptr(0),
            src_array: core::ptr::null_mut(),
            reserved0: core::ptr::null_mut(),
            src_pitch: 0,
            src_height: 0,
            dst_x_in_bytes: 0,
            dst_y: 0,
            dst_z: 0,
            dst_lod: 0,
            dst_memory_type: 0,
            _pad1: 0,
            dst_host: core::ptr::null_mut(),
            dst_device: CUdeviceptr(0),
            dst_array: core::ptr::null_mut(),
            reserved1: core::ptr::null_mut(),
            dst_pitch: 0,
            dst_height: 0,
            width_in_bytes: 0,
            height: 0,
            depth: 0,
        }
    }
}
