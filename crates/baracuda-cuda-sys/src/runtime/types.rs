//! Core handle types used by the CUDA Runtime API.
//!
//! Most of these are typedef-compatible with the Driver API handles
//! (`cudaStream_t == CUstream`, `cudaEvent_t == CUevent`, ...) so
//! Driver↔Runtime conversions are zero-cost.

use core::ffi::c_void;

/// Opaque CUDA stream (typedef-compatible with [`crate::CUstream`]).
pub type cudaStream_t = *mut c_void;

/// Opaque CUDA event (typedef-compatible with [`crate::CUevent`]).
pub type cudaEvent_t = *mut c_void;

/// Opaque CUDA graph.
pub type cudaGraph_t = *mut c_void;

/// Opaque executable graph.
pub type cudaGraphExec_t = *mut c_void;

/// Opaque graph node.
pub type cudaGraphNode_t = *mut c_void;

/// Opaque CUDA memory pool.
pub type cudaMemPool_t = *mut c_void;

/// Opaque CUDA array.
pub type cudaArray_t = *mut c_void;

/// Opaque CUDA mipmapped array (CUDA 5+).
pub type cudaMipmappedArray_t = *mut c_void;

/// `cudaTextureObject_t` — 64-bit opaque texture object handle.
pub type cudaTextureObject_t = u64;

/// `cudaSurfaceObject_t` — 64-bit opaque surface object handle.
pub type cudaSurfaceObject_t = u64;

/// `cudaMemGenericAllocationHandle_t` — opaque VMM allocation handle.
pub type cudaMemGenericAllocationHandle_t = u64;

/// `cudaGreenCtx_t` — green-context handle (CUDA 13.1+).
pub type cudaGreenCtx_t = *mut c_void;

/// `cudaChannelFormatKind` — texel format family.
#[allow(non_snake_case)]
pub mod cudaChannelFormatKind {
    pub const SIGNED: i32 = 0;
    pub const UNSIGNED: i32 = 1;
    pub const FLOAT: i32 = 2;
    pub const NONE: i32 = 3;
}

/// `cudaChannelFormatDesc` — 20-byte descriptor (4×c_int + c_int kind).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
#[allow(non_camel_case_types)]
pub struct cudaChannelFormatDesc {
    pub x: core::ffi::c_int,
    pub y: core::ffi::c_int,
    pub z: core::ffi::c_int,
    pub w: core::ffi::c_int,
    pub kind: core::ffi::c_int,
}

/// `cudaExtent` — 3D size in elements (width) + texels (height/depth).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
#[allow(non_camel_case_types)]
pub struct cudaExtent {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
}

/// `cudaPos` — 3D starting offset inside an array.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
#[allow(non_camel_case_types)]
pub struct cudaPos {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

/// `cudaPitchedPtr` — pitched device pointer (from `cudaMalloc3D`).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
pub struct cudaPitchedPtr {
    pub ptr: *mut c_void,
    pub pitch: usize,
    pub xsize: usize,
    pub ysize: usize,
}

impl Default for cudaPitchedPtr {
    fn default() -> Self {
        Self {
            ptr: core::ptr::null_mut(),
            pitch: 0,
            xsize: 0,
            ysize: 0,
        }
    }
}

/// `cudaMemcpy3DParms` — full 3D memcpy descriptor.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
pub struct cudaMemcpy3DParms {
    pub src_array: cudaArray_t,
    pub src_pos: cudaPos,
    pub src_ptr: cudaPitchedPtr,
    pub dst_array: cudaArray_t,
    pub dst_pos: cudaPos,
    pub dst_ptr: cudaPitchedPtr,
    pub extent: cudaExtent,
    pub kind: cudaMemcpyKind,
}

impl Default for cudaMemcpy3DParms {
    fn default() -> Self {
        Self {
            src_array: core::ptr::null_mut(),
            src_pos: cudaPos::default(),
            src_ptr: cudaPitchedPtr::default(),
            dst_array: core::ptr::null_mut(),
            dst_pos: cudaPos::default(),
            dst_ptr: cudaPitchedPtr::default(),
            extent: cudaExtent::default(),
            kind: cudaMemcpyKind::Default,
        }
    }
}

/// `cudaResourceType` — what a `cudaResourceDesc` points at.
#[allow(non_snake_case)]
pub mod cudaResourceType {
    pub const ARRAY: i32 = 0;
    pub const MIPMAPPED_ARRAY: i32 = 1;
    pub const LINEAR: i32 = 2;
    pub const PITCH_2D: i32 = 3;
}

/// `cudaResourceDesc` — tagged union describing a texture/surface source.
///
/// Layout (verified against CUDA 12 runtime headers, x86_64):
/// - offset 0:  `resType: i32` (4 bytes; `enum cudaResourceType`)
/// - offset 4:  pad (4 bytes) to align the union to 8
/// - offset 8:  `payload: [u8; 56]` — the widest union arm:
///   ```text
///   pitch2D = devPtr(8) + desc(20) + pad(4) + width(8) + height(8) + pitch(8) = 56
///   ```
/// - total: 64 bytes; struct alignment: 8 (because the union contains
///   `void*` and `size_t`).
///
/// **Two prior bugs corrected here**:
///
/// 1. **Size**: an earlier revision used a 32-byte payload + 8-byte
///    trailing pad (48 bytes total). 16 bytes too small —
///    `cudaGetSurfaceObjectResourceDesc` overran by 16 bytes →
///    stack corruption → STATUS_ACCESS_VIOLATION at the next CUDA call.
/// 2. **Alignment**: with only `c_int` fields, Rust gave the struct
///    4-byte alignment. The C union's `void*` and `size_t` fields
///    require 8-byte alignment of the struct itself. On the stack,
///    a 4-aligned struct landed at a non-8-aligned address, and the
///    optimizer's pointer math then aliased the size_t fields onto
///    misaligned addresses → release-only ACCESS_VIOLATION (debug
///    build tolerated it). Fixed via `#[repr(C, align(8))]`.
#[repr(C, align(8))]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
pub struct cudaResourceDesc {
    pub res_type: core::ffi::c_int,
    _pad0: u32,
    /// 120 bytes — over-allocated for safety. The pitch2D arm needs 56
    /// bytes; padding to 120 absorbs any future CUDA-12.x extensions to
    /// the cudaResourceDesc union (`mipmap` extensions, `sparse` arms,
    /// etc.) without ABI breakage.
    pub payload: [u8; 120],
}

impl Default for cudaResourceDesc {
    fn default() -> Self {
        Self {
            res_type: cudaResourceType::ARRAY,
            _pad0: 0,
            payload: [0; 120],
        }
    }
}

impl core::fmt::Debug for cudaResourceDesc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("cudaResourceDesc")
            .field("type", &self.res_type)
            .finish_non_exhaustive()
    }
}

impl cudaResourceDesc {
    /// Build an `ARRAY`-type descriptor.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn from_array(array: cudaArray_t) -> Self {
        let mut s = Self {
            res_type: cudaResourceType::ARRAY,
            ..Default::default()
        };
        unsafe {
            let p = s.payload.as_mut_ptr();
            (p as *mut cudaArray_t).write_unaligned(array);
        }
        s
    }

    /// Build a `MIPMAPPED_ARRAY`-type descriptor.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn from_mipmapped(mipmap: cudaMipmappedArray_t) -> Self {
        let mut s = Self {
            res_type: cudaResourceType::MIPMAPPED_ARRAY,
            ..Default::default()
        };
        unsafe {
            let p = s.payload.as_mut_ptr();
            (p as *mut cudaMipmappedArray_t).write_unaligned(mipmap);
        }
        s
    }

    /// Build a `LINEAR`-type descriptor: bytewise view of a device pointer.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn from_linear(
        dev_ptr: *mut c_void,
        desc: cudaChannelFormatDesc,
        size_in_bytes: usize,
    ) -> Self {
        let mut s = Self {
            res_type: cudaResourceType::LINEAR,
            ..Default::default()
        };
        unsafe {
            let p = s.payload.as_mut_ptr();
            (p as *mut *mut c_void).write_unaligned(dev_ptr);
            (p.add(8) as *mut cudaChannelFormatDesc).write_unaligned(desc);
            (p.add(28) as *mut usize).write_unaligned(size_in_bytes); // offset after 8+20=28
        }
        s
    }
}

/// `cudaTextureAddressMode` — out-of-bounds sampler behavior.
#[allow(non_snake_case)]
pub mod cudaTextureAddressMode {
    pub const WRAP: i32 = 0;
    pub const CLAMP: i32 = 1;
    pub const MIRROR: i32 = 2;
    pub const BORDER: i32 = 3;
}

/// `cudaTextureFilterMode` — filter kernel on fetch.
#[allow(non_snake_case)]
pub mod cudaTextureFilterMode {
    pub const POINT: i32 = 0;
    pub const LINEAR: i32 = 1;
}

/// `cudaTextureReadMode` — element-type reinterpretation on fetch.
#[allow(non_snake_case)]
pub mod cudaTextureReadMode {
    pub const ELEMENT_TYPE: i32 = 0;
    pub const NORMALIZED_FLOAT: i32 = 1;
}

/// `cudaTextureDesc` — sampler state (filter/address/normalize/mipmap).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
pub struct cudaTextureDesc {
    pub address_mode: [core::ffi::c_int; 3],
    pub filter_mode: core::ffi::c_int,
    pub read_mode: core::ffi::c_int,
    pub srgb: core::ffi::c_int,
    pub border_color: [f32; 4],
    pub normalized_coords: core::ffi::c_int,
    pub max_anisotropy: core::ffi::c_uint,
    pub mipmap_filter_mode: core::ffi::c_int,
    pub mipmap_level_bias: f32,
    pub min_mipmap_level_clamp: f32,
    pub max_mipmap_level_clamp: f32,
    pub disable_trilinear_optimization: core::ffi::c_int,
    pub seamless_cubemap: core::ffi::c_int,
}

impl Default for cudaTextureDesc {
    fn default() -> Self {
        Self {
            address_mode: [cudaTextureAddressMode::CLAMP; 3],
            filter_mode: cudaTextureFilterMode::POINT,
            read_mode: cudaTextureReadMode::ELEMENT_TYPE,
            srgb: 0,
            border_color: [0.0; 4],
            normalized_coords: 0,
            max_anisotropy: 0,
            mipmap_filter_mode: cudaTextureFilterMode::POINT,
            mipmap_level_bias: 0.0,
            min_mipmap_level_clamp: 0.0,
            max_mipmap_level_clamp: 0.0,
            disable_trilinear_optimization: 0,
            seamless_cubemap: 0,
        }
    }
}

/// `cudaResourceViewDesc` — optional view transformation applied on fetch.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
#[allow(non_camel_case_types)]
pub struct cudaResourceViewDesc {
    pub format: core::ffi::c_int,
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub first_mipmap_level: core::ffi::c_uint,
    pub last_mipmap_level: core::ffi::c_uint,
    pub first_layer: core::ffi::c_uint,
    pub last_layer: core::ffi::c_uint,
}

/// `cudaLaunchAttributeID` — selectors for extended launch attributes.
#[allow(non_snake_case)]
pub mod cudaLaunchAttributeID {
    pub const IGNORE: i32 = 0;
    pub const ACCESS_POLICY_WINDOW: i32 = 1;
    pub const COOPERATIVE: i32 = 2;
    pub const SYNC_POLICY: i32 = 3;
    pub const CLUSTER_DIMENSION: i32 = 4;
    pub const CLUSTER_SCHEDULING_POLICY_PREFERENCE: i32 = 5;
    pub const PROGRAMMATIC_STREAM_SERIALIZATION: i32 = 6;
    pub const PROGRAMMATIC_EVENT: i32 = 7;
    pub const PRIORITY: i32 = 8;
    pub const MEM_SYNC_DOMAIN_MAP: i32 = 9;
    pub const MEM_SYNC_DOMAIN: i32 = 10;
    pub const LAUNCH_COMPLETION_EVENT: i32 = 12;
    pub const DEVICE_UPDATABLE_KERNEL_NODE: i32 = 13;
}

/// `cudaLaunchAttributeValue` — 64-byte union payload.
#[repr(C, align(8))]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
pub struct cudaLaunchAttributeValue {
    pub raw: [u8; 64],
}

impl Default for cudaLaunchAttributeValue {
    fn default() -> Self {
        Self { raw: [0; 64] }
    }
}

impl core::fmt::Debug for cudaLaunchAttributeValue {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("cudaLaunchAttributeValue")
            .finish_non_exhaustive()
    }
}

impl cudaLaunchAttributeValue {
    /// Cluster-dimension payload (3 × u32, rest zero).
    pub fn cluster_dimension(x: u32, y: u32, z: u32) -> Self {
        let mut v = Self::default();
        unsafe {
            let p = v.raw.as_mut_ptr() as *mut u32;
            p.write(x);
            p.add(1).write(y);
            p.add(2).write(z);
        }
        v
    }

    /// COOPERATIVE = 1 (i.e., enable cooperative launch).
    pub fn cooperative(enable: bool) -> Self {
        let mut v = Self::default();
        unsafe {
            let p = v.raw.as_mut_ptr() as *mut core::ffi::c_int;
            p.write(if enable { 1 } else { 0 });
        }
        v
    }

    /// Priority payload (signed 32-bit).
    pub fn priority(prio: i32) -> Self {
        let mut v = Self::default();
        unsafe {
            let p = v.raw.as_mut_ptr() as *mut i32;
            p.write(prio);
        }
        v
    }
}

/// `cudaLaunchAttribute` — paired ID + value entry.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
pub struct cudaLaunchAttribute {
    pub id: core::ffi::c_int,
    pub _pad: core::ffi::c_int,
    pub val: cudaLaunchAttributeValue,
}

/// `cudaLaunchConfig_t` — the config object consumed by `cudaLaunchKernelEx`.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
pub struct cudaLaunchConfig_t {
    pub grid_dim: dim3,
    pub block_dim: dim3,
    pub dynamic_smem_bytes: usize,
    pub stream: cudaStream_t,
    pub attrs: *mut cudaLaunchAttribute,
    pub num_attrs: core::ffi::c_uint,
}

impl Default for cudaLaunchConfig_t {
    fn default() -> Self {
        Self {
            grid_dim: dim3::default(),
            block_dim: dim3::default(),
            dynamic_smem_bytes: 0,
            stream: core::ptr::null_mut(),
            attrs: core::ptr::null_mut(),
            num_attrs: 0,
        }
    }
}

/// `cudaMemAllocationProp` — properties for VMM `cudaMemCreate`.
/// Shared ABI with the Driver API's `CUmemAllocationProp`.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
pub struct cudaMemAllocationProp {
    pub alloc_type: core::ffi::c_int, // cudaMemAllocationType
    pub requested_handle_types: core::ffi::c_int, // cudaMemAllocationHandleType
    pub location: super::types::cudaMemLocation,
    pub win32_handle_meta_data: *mut c_void,
    pub allocation_flags: [u8; 32], // rdma + compression + reserved
}

impl Default for cudaMemAllocationProp {
    fn default() -> Self {
        Self {
            alloc_type: 0,
            requested_handle_types: 0,
            location: super::types::cudaMemLocation { type_: 0, id: 0 },
            win32_handle_meta_data: core::ptr::null_mut(),
            allocation_flags: [0; 32],
        }
    }
}

/// Opaque CUDA library handle (CUDA 12.0+).
pub type cudaLibrary_t = *mut c_void;

/// Opaque CUDA kernel handle (CUDA 12.0+).
pub type cudaKernel_t = *mut c_void;

/// Opaque user-object handle (CUDA 12.0+).
pub type cudaUserObject_t = *mut c_void;

/// `cudaDeviceScheduleFlags` — bits for `cudaSetDeviceFlags`.
#[allow(non_snake_case)]
pub mod cudaDeviceScheduleFlags {
    pub const AUTO: u32 = 0x00;
    pub const SPIN: u32 = 0x01;
    pub const YIELD: u32 = 0x02;
    pub const BLOCKING_SYNC: u32 = 0x04;
    pub const MAP_HOST: u32 = 0x08;
    pub const LMEM_RESIZE_TO_MAX: u32 = 0x10;
}

/// External memory object (imported Vulkan / D3D12 / NVIDIA Buffer).
/// Typedef-compatible with [`crate::CUexternalMemory`] — pointer-level
/// swaps between Driver and Runtime wrappers are zero-cost.
pub type cudaExternalMemory_t = *mut c_void;

/// External semaphore object.
/// Typedef-compatible with [`crate::CUexternalSemaphore`].
pub type cudaExternalSemaphore_t = *mut c_void;

/// Direction-of-copy selector for `cudaMemcpy`.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudaMemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
    /// Let the runtime infer direction from the pointer attributes (UVA).
    Default = 4,
}

/// Three-dimensional `dim3` used for grid/block sizes.
#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct dim3 {
    pub x: core::ffi::c_uint,
    pub y: core::ffi::c_uint,
    pub z: core::ffi::c_uint,
}

impl dim3 {
    #[inline]
    pub const fn new(x: core::ffi::c_uint, y: core::ffi::c_uint, z: core::ffi::c_uint) -> Self {
        Self { x, y, z }
    }
}

impl Default for dim3 {
    fn default() -> Self {
        Self { x: 1, y: 1, z: 1 }
    }
}

/// Stream creation flags.
#[allow(non_snake_case)]
pub mod cudaStreamFlags {
    pub const DEFAULT: u32 = 0x0;
    pub const NON_BLOCKING: u32 = 0x1;
}

/// Event creation flags.
#[allow(non_snake_case)]
pub mod cudaEventFlags {
    pub const DEFAULT: u32 = 0x0;
    pub const BLOCKING_SYNC: u32 = 0x1;
    pub const DISABLE_TIMING: u32 = 0x2;
    pub const INTERPROCESS: u32 = 0x4;
}

/// `cudaMemoryAdvise` — values accepted by `cudaMemAdvise`.
#[allow(non_snake_case)]
pub mod cudaMemoryAdvise {
    pub const SET_READ_MOSTLY: i32 = 1;
    pub const UNSET_READ_MOSTLY: i32 = 2;
    pub const SET_PREFERRED_LOCATION: i32 = 3;
    pub const UNSET_PREFERRED_LOCATION: i32 = 4;
    pub const SET_ACCESSED_BY: i32 = 5;
    pub const UNSET_ACCESSED_BY: i32 = 6;
}

/// `cudaMemAttach*` — flags for `cudaMallocManaged` / `cudaStreamAttachMemAsync`.
#[allow(non_snake_case)]
pub mod cudaMemAttach {
    pub const GLOBAL: u32 = 0x01;
    pub const HOST: u32 = 0x02;
    pub const SINGLE: u32 = 0x04;
}

/// `cudaHostAllocFlags` — flags for `cudaHostAlloc`.
#[allow(non_snake_case)]
pub mod cudaHostAllocFlags {
    pub const DEFAULT: u32 = 0x00;
    pub const PORTABLE: u32 = 0x01;
    pub const MAPPED: u32 = 0x02;
    pub const WRITE_COMBINED: u32 = 0x04;
}

/// `cudaStreamCaptureMode` — argument to `cudaStreamBeginCapture`.
#[allow(non_snake_case)]
pub mod cudaStreamCaptureMode {
    pub const GLOBAL: i32 = 0;
    pub const THREAD_LOCAL: i32 = 1;
    pub const RELAXED: i32 = 2;
}

/// `cudaStreamCaptureStatus` — returned by `cudaStreamIsCapturing`.
#[allow(non_snake_case)]
pub mod cudaStreamCaptureStatus {
    pub const NONE: i32 = 0;
    pub const ACTIVE: i32 = 1;
    pub const INVALIDATED: i32 = 2;
}

/// Host-function trampoline type (parallel to `CUhostFn`).
pub type cudaHostFn_t = Option<unsafe extern "C" fn(user_data: *mut c_void)>;

// ---- Memory-pool types ---------------------------------------------------

/// `cudaMemAllocationType` — same values as the Driver side.
#[allow(non_snake_case)]
pub mod cudaMemAllocationType {
    pub const INVALID: i32 = 0;
    pub const PINNED: i32 = 1;
}

/// `cudaMemLocationType` — same values as the Driver side.
#[allow(non_snake_case)]
pub mod cudaMemLocationType {
    pub const INVALID: i32 = 0;
    pub const DEVICE: i32 = 1;
    pub const HOST: i32 = 2;
    pub const HOST_NUMA: i32 = 3;
    pub const HOST_NUMA_CURRENT: i32 = 4;
}

/// `cudaMemAccessFlags`.
#[allow(non_snake_case)]
pub mod cudaMemAccessFlags {
    pub const NONE: i32 = 0;
    pub const READ: i32 = 1;
    pub const READ_WRITE: i32 = 3;
}

/// `cudaMemAllocationHandleType` — IPC export type.
#[allow(non_snake_case)]
pub mod cudaMemAllocationHandleType {
    pub const NONE: i32 = 0;
    pub const POSIX_FILE_DESCRIPTOR: i32 = 1;
    pub const WIN32: i32 = 2;
    pub const WIN32_KMT: i32 = 4;
    pub const FABRIC: i32 = 8;
}

/// `cudaMemPoolAttr` — selector for `cudaMemPoolSetAttribute` / `GetAttribute`.
#[allow(non_snake_case)]
pub mod cudaMemPoolAttr {
    pub const REUSE_FOLLOW_EVENT_DEPENDENCIES: i32 = 1;
    pub const REUSE_ALLOW_OPPORTUNISTIC: i32 = 2;
    pub const REUSE_ALLOW_INTERNAL_DEPENDENCIES: i32 = 3;
    pub const RELEASE_THRESHOLD: i32 = 4;
    pub const RESERVED_MEM_CURRENT: i32 = 5;
    pub const RESERVED_MEM_HIGH: i32 = 6;
    pub const USED_MEM_CURRENT: i32 = 7;
    pub const USED_MEM_HIGH: i32 = 8;
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
#[allow(non_camel_case_types)]
pub struct cudaMemLocation {
    pub type_: core::ffi::c_int,
    pub id: core::ffi::c_int,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
#[allow(non_camel_case_types)]
pub struct cudaMemAccessDesc {
    pub location: cudaMemLocation,
    pub flags: core::ffi::c_int,
}

/// `cudaMemPoolProps` — 88 bytes in C. Matches the Driver-side
/// `CUmemPoolProps` layout byte-for-byte.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
pub struct cudaMemPoolProps {
    pub alloc_type: core::ffi::c_int,
    pub handle_types: core::ffi::c_int,
    pub location: cudaMemLocation,
    pub win32_security_attributes: *mut c_void,
    pub max_size: usize,
    pub usage: core::ffi::c_ushort,
    pub reserved: [core::ffi::c_uchar; 54],
}

impl Default for cudaMemPoolProps {
    fn default() -> Self {
        Self {
            alloc_type: cudaMemAllocationType::PINNED,
            handle_types: cudaMemAllocationHandleType::NONE,
            location: cudaMemLocation::default(),
            win32_security_attributes: core::ptr::null_mut(),
            max_size: 0,
            usage: 0,
            reserved: [0u8; 54],
        }
    }
}

/// `cudaMemPoolPtrExportData` — 64-byte opaque blob.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
pub struct cudaMemPoolPtrExportData {
    pub reserved: [core::ffi::c_uchar; 64],
}

impl Default for cudaMemPoolPtrExportData {
    fn default() -> Self {
        Self {
            reserved: [0u8; 64],
        }
    }
}

// ---- Kernel launch parameters for graph nodes ----------------------------

/// `cudaKernelNodeParams` — parameters for `cudaGraphAddKernelNode`.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
pub struct cudaKernelNodeParams {
    pub func: *mut c_void,
    pub grid_dim: dim3,
    pub block_dim: dim3,
    pub shared_mem_bytes: core::ffi::c_uint,
    pub kernel_params: *mut *mut c_void,
    pub extra: *mut *mut c_void,
}

impl Default for cudaKernelNodeParams {
    fn default() -> Self {
        Self {
            func: core::ptr::null_mut(),
            grid_dim: dim3::default(),
            block_dim: dim3::default(),
            shared_mem_bytes: 0,
            kernel_params: core::ptr::null_mut(),
            extra: core::ptr::null_mut(),
        }
    }
}

/// `cudaMemsetParams` — parameters for `cudaGraphAddMemsetNode`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
#[allow(non_camel_case_types)]
pub struct cudaMemsetParams {
    pub dst: *mut c_void,
    pub pitch: usize,
    pub value: core::ffi::c_uint,
    pub element_size: core::ffi::c_uint,
    pub width: usize,
    pub height: usize,
}

/// `cudaHostNodeParams` — `{ fn, user_data }` for `cudaGraphAddHostNode`.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
pub struct cudaHostNodeParams {
    pub fn_: cudaHostFn_t,
    pub user_data: *mut c_void,
}

impl Default for cudaHostNodeParams {
    fn default() -> Self {
        Self {
            fn_: None,
            user_data: core::ptr::null_mut(),
        }
    }
}

/// `cudaMemAllocNodeParams` — parameters for `cudaGraphAddMemAllocNode`.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
pub struct cudaMemAllocNodeParams {
    pub pool_props: cudaMemPoolProps,
    pub access_descs: *const cudaMemAccessDesc,
    pub access_desc_count: usize,
    pub bytesize: usize,
    pub dptr: *mut c_void,
}

impl Default for cudaMemAllocNodeParams {
    fn default() -> Self {
        Self {
            pool_props: cudaMemPoolProps::default(),
            access_descs: core::ptr::null(),
            access_desc_count: 0,
            bytesize: 0,
            dptr: core::ptr::null_mut(),
        }
    }
}

/// `cudaGraphExecUpdateResult` — outcome of `cudaGraphExecUpdate`.
#[allow(non_snake_case)]
pub mod cudaGraphExecUpdateResult {
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

// ---- Pointer / function attributes ---------------------------------------

/// `cudaPointerAttributes` — returned by `cudaPointerGetAttributes`.
///
/// The C struct is 48 bytes. We model the fields directly.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
pub struct cudaPointerAttributes {
    /// `cudaMemoryType` — 0=unregistered, 1=host, 2=device, 3=managed.
    pub type_: core::ffi::c_int,
    /// Device ordinal the allocation lives on.
    pub device: core::ffi::c_int,
    /// Device-addressable pointer (null for plain host memory).
    pub device_pointer: *mut c_void,
    /// Host-addressable pointer (null for plain device memory).
    pub host_pointer: *mut c_void,
}

impl Default for cudaPointerAttributes {
    fn default() -> Self {
        Self {
            type_: 0,
            device: 0,
            device_pointer: core::ptr::null_mut(),
            host_pointer: core::ptr::null_mut(),
        }
    }
}

/// `cudaMemoryType` — `cudaPointerAttributes::type_` values.
#[allow(non_snake_case)]
pub mod cudaMemoryType {
    pub const UNREGISTERED: i32 = 0;
    pub const HOST: i32 = 1;
    pub const DEVICE: i32 = 2;
    pub const MANAGED: i32 = 3;
}

/// `cudaFuncAttributes` — kernel metadata (36 bytes in C).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
#[allow(non_camel_case_types)]
pub struct cudaFuncAttributes {
    pub shared_size_bytes: usize,
    pub const_size_bytes: usize,
    pub local_size_bytes: usize,
    pub max_threads_per_block: core::ffi::c_int,
    pub num_regs: core::ffi::c_int,
    pub ptx_version: core::ffi::c_int,
    pub binary_version: core::ffi::c_int,
    pub cache_mode_ca: core::ffi::c_int,
    pub max_dynamic_shared_size_bytes: core::ffi::c_int,
    pub preferred_shmem_carveout: core::ffi::c_int,
    pub cluster_dim_must_be_set: core::ffi::c_int,
    pub required_cluster_width: core::ffi::c_int,
    pub required_cluster_height: core::ffi::c_int,
    pub required_cluster_depth: core::ffi::c_int,
    pub cluster_scheduling_policy_preference: core::ffi::c_int,
    pub non_portable_cluster_size_allowed: core::ffi::c_int,
    pub reserved: [core::ffi::c_int; 16],
}

/// `cudaStreamWriteValueFlags`.
#[allow(non_snake_case)]
pub mod cudaStreamWriteValueFlags {
    pub const DEFAULT: u32 = 0x0;
    pub const NO_MEMORY_BARRIER: u32 = 0x1;
}

/// `cudaStreamWaitValueFlags`.
#[allow(non_snake_case)]
pub mod cudaStreamWaitValueFlags {
    pub const GEQ: u32 = 0x0;
    pub const EQ: u32 = 0x1;
    pub const AND: u32 = 0x2;
    pub const NOR: u32 = 0x3;
    pub const FLUSH: u32 = 1 << 30;
}

/// Device-attribute selector (matches `cudaDeviceAttr` values).
#[allow(non_snake_case)]
pub mod cudaDeviceAttr {
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
    pub const MULTIPROCESSOR_COUNT: i32 = 16;
    pub const COMPUTE_CAPABILITY_MAJOR: i32 = 75;
    pub const COMPUTE_CAPABILITY_MINOR: i32 = 76;
    pub const CONCURRENT_KERNELS: i32 = 31;
    pub const ECC_ENABLED: i32 = 32;
    pub const PCI_BUS_ID: i32 = 33;
    pub const PCI_DEVICE_ID: i32 = 34;
    pub const PCI_DOMAIN_ID: i32 = 50;
    pub const INTEGRATED: i32 = 18;
}
