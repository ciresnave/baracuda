//! Core handle types used by the CUDA Runtime API.
//!
//! Most of these are typedef-compatible with the Driver API handles
//! (`cudaStream_t == CUstream`, `cudaEvent_t == CUevent`, ...) so
//! Driver↔Runtime conversions are zero-cost.

#![allow(
    non_camel_case_types,
    non_snake_case,
    reason = "names mirror the vendor C API (FFI bindings)"
)]

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
pub mod cudaChannelFormatKind {
    /// `cudaChannelFormatKind::SIGNED` — signed.
    pub const SIGNED: i32 = 0;
    /// `cudaChannelFormatKind::UNSIGNED` — unsigned.
    pub const UNSIGNED: i32 = 1;
    /// `cudaChannelFormatKind::FLOAT` — float.
    pub const FLOAT: i32 = 2;
    /// `cudaChannelFormatKind::NONE` — none.
    pub const NONE: i32 = 3;
}

/// `cudaChannelFormatDesc` — 20-byte descriptor (4×c_int + c_int kind).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct cudaChannelFormatDesc {
    /// `x` field.
    pub x: core::ffi::c_int,
    /// `y` field.
    pub y: core::ffi::c_int,
    /// `z` field.
    pub z: core::ffi::c_int,
    /// `w` field.
    pub w: core::ffi::c_int,
    /// `kind` field.
    pub kind: core::ffi::c_int,
}

/// `cudaExtent` — 3D size in elements (width) + texels (height/depth).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct cudaExtent {
    /// `width` field.
    pub width: usize,
    /// `height` field.
    pub height: usize,
    /// `depth` field.
    pub depth: usize,
}

/// `cudaPos` — 3D starting offset inside an array.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct cudaPos {
    /// `x` field.
    pub x: usize,
    /// `y` field.
    pub y: usize,
    /// `z` field.
    pub z: usize,
}

/// `cudaPitchedPtr` — pitched device pointer (from `cudaMalloc3D`).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct cudaPitchedPtr {
    /// `ptr` field.
    pub ptr: *mut c_void,
    /// `pitch` field.
    pub pitch: usize,
    /// `xsize` field.
    pub xsize: usize,
    /// `ysize` field.
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
pub struct cudaMemcpy3DParms {
    /// `src_array` field.
    pub src_array: cudaArray_t,
    /// `src_pos` field.
    pub src_pos: cudaPos,
    /// `src_ptr` field.
    pub src_ptr: cudaPitchedPtr,
    /// `dst_array` field.
    pub dst_array: cudaArray_t,
    /// `dst_pos` field.
    pub dst_pos: cudaPos,
    /// `dst_ptr` field.
    pub dst_ptr: cudaPitchedPtr,
    /// `extent` field.
    pub extent: cudaExtent,
    /// `kind` field.
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
pub mod cudaResourceType {
    /// `cudaResourceType::ARRAY` — array.
    pub const ARRAY: i32 = 0;
    /// `cudaResourceType::MIPMAPPED_ARRAY` — mipmapped array.
    pub const MIPMAPPED_ARRAY: i32 = 1;
    /// `cudaResourceType::LINEAR` — linear.
    pub const LINEAR: i32 = 2;
    /// `cudaResourceType::PITCH_2D` — pitch 2 d.
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
pub struct cudaResourceDesc {
    /// `res_type` field.
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
    #[allow(
        clippy::not_unsafe_ptr_arg_deref,
        reason = "the pointer is stored or forwarded to the driver, never dereferenced in Rust"
    )]
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
    #[allow(
        clippy::not_unsafe_ptr_arg_deref,
        reason = "the pointer is stored or forwarded to the driver, never dereferenced in Rust"
    )]
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
    #[allow(
        clippy::not_unsafe_ptr_arg_deref,
        reason = "the pointer is stored or forwarded to the driver, never dereferenced in Rust"
    )]
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
pub mod cudaTextureAddressMode {
    /// `cudaTextureAddressMode::WRAP` — wrap.
    pub const WRAP: i32 = 0;
    /// `cudaTextureAddressMode::CLAMP` — clamp.
    pub const CLAMP: i32 = 1;
    /// `cudaTextureAddressMode::MIRROR` — mirror.
    pub const MIRROR: i32 = 2;
    /// `cudaTextureAddressMode::BORDER` — border.
    pub const BORDER: i32 = 3;
}

/// `cudaTextureFilterMode` — filter kernel on fetch.
pub mod cudaTextureFilterMode {
    /// `cudaTextureFilterMode::POINT` — point.
    pub const POINT: i32 = 0;
    /// `cudaTextureFilterMode::LINEAR` — linear.
    pub const LINEAR: i32 = 1;
}

/// `cudaTextureReadMode` — element-type reinterpretation on fetch.
pub mod cudaTextureReadMode {
    /// `cudaTextureReadMode::ELEMENT_TYPE` — element type.
    pub const ELEMENT_TYPE: i32 = 0;
    /// `cudaTextureReadMode::NORMALIZED_FLOAT` — normalized float.
    pub const NORMALIZED_FLOAT: i32 = 1;
}

/// `cudaTextureDesc` — sampler state (filter/address/normalize/mipmap).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct cudaTextureDesc {
    /// `address_mode` field.
    pub address_mode: [core::ffi::c_int; 3],
    /// `filter_mode` field.
    pub filter_mode: core::ffi::c_int,
    /// `read_mode` field.
    pub read_mode: core::ffi::c_int,
    /// `srgb` field.
    pub srgb: core::ffi::c_int,
    /// `border_color` field.
    pub border_color: [f32; 4],
    /// `normalized_coords` field.
    pub normalized_coords: core::ffi::c_int,
    /// `max_anisotropy` field.
    pub max_anisotropy: core::ffi::c_uint,
    /// `mipmap_filter_mode` field.
    pub mipmap_filter_mode: core::ffi::c_int,
    /// `mipmap_level_bias` field.
    pub mipmap_level_bias: f32,
    /// `min_mipmap_level_clamp` field.
    pub min_mipmap_level_clamp: f32,
    /// `max_mipmap_level_clamp` field.
    pub max_mipmap_level_clamp: f32,
    /// `disable_trilinear_optimization` field.
    pub disable_trilinear_optimization: core::ffi::c_int,
    /// `seamless_cubemap` field.
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
pub struct cudaResourceViewDesc {
    /// `format` field.
    pub format: core::ffi::c_int,
    /// `width` field.
    pub width: usize,
    /// `height` field.
    pub height: usize,
    /// `depth` field.
    pub depth: usize,
    /// `first_mipmap_level` field.
    pub first_mipmap_level: core::ffi::c_uint,
    /// `last_mipmap_level` field.
    pub last_mipmap_level: core::ffi::c_uint,
    /// `first_layer` field.
    pub first_layer: core::ffi::c_uint,
    /// `last_layer` field.
    pub last_layer: core::ffi::c_uint,
    /// `reserved[16]` — "must be zero" in `driver_types.h`. Present so the
    /// struct is the 112 bytes the runtime reads through `pResViewDesc`.
    pub reserved: [core::ffi::c_uint; 16],
}

/// `cudaLaunchAttributeID` — selectors for extended launch attributes.
pub mod cudaLaunchAttributeID {
    /// `cudaLaunchAttributeID::IGNORE` — ignore.
    pub const IGNORE: i32 = 0;
    /// `cudaLaunchAttributeID::ACCESS_POLICY_WINDOW` — access policy window.
    pub const ACCESS_POLICY_WINDOW: i32 = 1;
    /// `cudaLaunchAttributeID::COOPERATIVE` — cooperative.
    pub const COOPERATIVE: i32 = 2;
    /// `cudaLaunchAttributeID::SYNC_POLICY` — sync policy.
    pub const SYNC_POLICY: i32 = 3;
    /// `cudaLaunchAttributeID::CLUSTER_DIMENSION` — cluster dimension.
    pub const CLUSTER_DIMENSION: i32 = 4;
    /// `cudaLaunchAttributeID::CLUSTER_SCHEDULING_POLICY_PREFERENCE` — cluster scheduling policy preference.
    pub const CLUSTER_SCHEDULING_POLICY_PREFERENCE: i32 = 5;
    /// `cudaLaunchAttributeID::PROGRAMMATIC_STREAM_SERIALIZATION` — programmatic stream serialization.
    pub const PROGRAMMATIC_STREAM_SERIALIZATION: i32 = 6;
    /// `cudaLaunchAttributeID::PROGRAMMATIC_EVENT` — programmatic event.
    pub const PROGRAMMATIC_EVENT: i32 = 7;
    /// `cudaLaunchAttributeID::PRIORITY` — priority.
    pub const PRIORITY: i32 = 8;
    /// `cudaLaunchAttributeID::MEM_SYNC_DOMAIN_MAP` — mem sync domain map.
    pub const MEM_SYNC_DOMAIN_MAP: i32 = 9;
    /// `cudaLaunchAttributeID::MEM_SYNC_DOMAIN` — mem sync domain.
    pub const MEM_SYNC_DOMAIN: i32 = 10;
    /// `cudaLaunchAttributeID::LAUNCH_COMPLETION_EVENT` — launch completion event.
    pub const LAUNCH_COMPLETION_EVENT: i32 = 12;
    /// `cudaLaunchAttributeID::DEVICE_UPDATABLE_KERNEL_NODE` — device updatable kernel node.
    pub const DEVICE_UPDATABLE_KERNEL_NODE: i32 = 13;
}

/// `cudaLaunchAttributeValue` — 64-byte union payload.
#[repr(C, align(8))]
#[derive(Copy, Clone)]
pub struct cudaLaunchAttributeValue {
    /// `raw` field.
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
pub struct cudaLaunchAttribute {
    /// `id` field.
    pub id: core::ffi::c_int,
    /// `_pad` field.
    pub _pad: core::ffi::c_int,
    /// `val` field.
    pub val: cudaLaunchAttributeValue,
}

/// `cudaLaunchConfig_t` — the config object consumed by `cudaLaunchKernelEx`.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct cudaLaunchConfig_t {
    /// `grid_dim` field.
    pub grid_dim: dim3,
    /// `block_dim` field.
    pub block_dim: dim3,
    /// `dynamic_smem_bytes` field.
    pub dynamic_smem_bytes: usize,
    /// `stream` field.
    pub stream: cudaStream_t,
    /// `attrs` field.
    pub attrs: *mut cudaLaunchAttribute,
    /// `num_attrs` field.
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
pub struct cudaMemAllocationProp {
    /// `alloc_type` field.
    pub alloc_type: core::ffi::c_int, // cudaMemAllocationType
    /// `requested_handle_types` field.
    pub requested_handle_types: core::ffi::c_int, // cudaMemAllocationHandleType
    /// `location` field.
    pub location: super::types::cudaMemLocation,
    /// `win32_handle_meta_data` field.
    pub win32_handle_meta_data: *mut c_void,
    /// `allocation_flags` field.
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
pub mod cudaDeviceScheduleFlags {
    /// `cudaDeviceScheduleFlags::AUTO` — auto.
    pub const AUTO: u32 = 0x00;
    /// `cudaDeviceScheduleFlags::SPIN` — spin.
    pub const SPIN: u32 = 0x01;
    /// `cudaDeviceScheduleFlags::YIELD` — yield.
    pub const YIELD: u32 = 0x02;
    /// `cudaDeviceScheduleFlags::BLOCKING_SYNC` — blocking sync.
    pub const BLOCKING_SYNC: u32 = 0x04;
    /// `cudaDeviceScheduleFlags::MAP_HOST` — map host.
    pub const MAP_HOST: u32 = 0x08;
    /// `cudaDeviceScheduleFlags::LMEM_RESIZE_TO_MAX` — lmem resize to max.
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
    /// `HostToHost` — host to host.
    HostToHost = 0,
    /// `HostToDevice` — host to device.
    HostToDevice = 1,
    /// `DeviceToHost` — device to host.
    DeviceToHost = 2,
    /// `DeviceToDevice` — device to device.
    DeviceToDevice = 3,
    /// Let the runtime infer direction from the pointer attributes (UVA).
    Default = 4,
}

/// Three-dimensional `dim3` used for grid/block sizes.
#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct dim3 {
    /// `x` field.
    pub x: core::ffi::c_uint,
    /// `y` field.
    pub y: core::ffi::c_uint,
    /// `z` field.
    pub z: core::ffi::c_uint,
}

impl dim3 {
    #[inline]
    /// `new` — new.
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
pub mod cudaStreamFlags {
    /// `cudaStreamFlags::DEFAULT` — default.
    pub const DEFAULT: u32 = 0x0;
    /// `cudaStreamFlags::NON_BLOCKING` — non blocking.
    pub const NON_BLOCKING: u32 = 0x1;
}

/// Event creation flags.
pub mod cudaEventFlags {
    /// `cudaEventFlags::DEFAULT` — default.
    pub const DEFAULT: u32 = 0x0;
    /// `cudaEventFlags::BLOCKING_SYNC` — blocking sync.
    pub const BLOCKING_SYNC: u32 = 0x1;
    /// `cudaEventFlags::DISABLE_TIMING` — disable timing.
    pub const DISABLE_TIMING: u32 = 0x2;
    /// `cudaEventFlags::INTERPROCESS` — interprocess.
    pub const INTERPROCESS: u32 = 0x4;
}

/// `cudaMemoryAdvise` — values accepted by `cudaMemAdvise`.
pub mod cudaMemoryAdvise {
    /// `cudaMemoryAdvise::SET_READ_MOSTLY` — set read mostly.
    pub const SET_READ_MOSTLY: i32 = 1;
    /// `cudaMemoryAdvise::UNSET_READ_MOSTLY` — unset read mostly.
    pub const UNSET_READ_MOSTLY: i32 = 2;
    /// `cudaMemoryAdvise::SET_PREFERRED_LOCATION` — set preferred location.
    pub const SET_PREFERRED_LOCATION: i32 = 3;
    /// `cudaMemoryAdvise::UNSET_PREFERRED_LOCATION` — unset preferred location.
    pub const UNSET_PREFERRED_LOCATION: i32 = 4;
    /// `cudaMemoryAdvise::SET_ACCESSED_BY` — set accessed by.
    pub const SET_ACCESSED_BY: i32 = 5;
    /// `cudaMemoryAdvise::UNSET_ACCESSED_BY` — unset accessed by.
    pub const UNSET_ACCESSED_BY: i32 = 6;
}

/// `cudaMemAttach*` — flags for `cudaMallocManaged` / `cudaStreamAttachMemAsync`.
pub mod cudaMemAttach {
    /// `cudaMemAttach::GLOBAL` — global.
    pub const GLOBAL: u32 = 0x01;
    /// `cudaMemAttach::HOST` — host.
    pub const HOST: u32 = 0x02;
    /// `cudaMemAttach::SINGLE` — single.
    pub const SINGLE: u32 = 0x04;
}

/// `cudaHostAllocFlags` — flags for `cudaHostAlloc`.
pub mod cudaHostAllocFlags {
    /// `cudaHostAllocFlags::DEFAULT` — default.
    pub const DEFAULT: u32 = 0x00;
    /// `cudaHostAllocFlags::PORTABLE` — portable.
    pub const PORTABLE: u32 = 0x01;
    /// `cudaHostAllocFlags::MAPPED` — mapped.
    pub const MAPPED: u32 = 0x02;
    /// `cudaHostAllocFlags::WRITE_COMBINED` — write combined.
    pub const WRITE_COMBINED: u32 = 0x04;
}

/// `cudaStreamCaptureMode` — argument to `cudaStreamBeginCapture`.
pub mod cudaStreamCaptureMode {
    /// `cudaStreamCaptureMode::GLOBAL` — global.
    pub const GLOBAL: i32 = 0;
    /// `cudaStreamCaptureMode::THREAD_LOCAL` — thread local.
    pub const THREAD_LOCAL: i32 = 1;
    /// `cudaStreamCaptureMode::RELAXED` — relaxed.
    pub const RELAXED: i32 = 2;
}

/// `cudaStreamCaptureStatus` — returned by `cudaStreamIsCapturing`.
pub mod cudaStreamCaptureStatus {
    /// `cudaStreamCaptureStatus::NONE` — none.
    pub const NONE: i32 = 0;
    /// `cudaStreamCaptureStatus::ACTIVE` — active.
    pub const ACTIVE: i32 = 1;
    /// `cudaStreamCaptureStatus::INVALIDATED` — invalidated.
    pub const INVALIDATED: i32 = 2;
}

/// Host-function trampoline type (parallel to `CUhostFn`).
pub type cudaHostFn_t = Option<unsafe extern "C" fn(user_data: *mut c_void)>;

// ---- Memory-pool types ---------------------------------------------------

/// `cudaMemAllocationType` — same values as the Driver side.
pub mod cudaMemAllocationType {
    /// `cudaMemAllocationType::INVALID` — invalid.
    pub const INVALID: i32 = 0;
    /// `cudaMemAllocationType::PINNED` — pinned.
    pub const PINNED: i32 = 1;
}

/// `cudaMemLocationType` — same values as the Driver side.
pub mod cudaMemLocationType {
    /// `cudaMemLocationType::INVALID` — invalid.
    pub const INVALID: i32 = 0;
    /// `cudaMemLocationType::DEVICE` — device.
    pub const DEVICE: i32 = 1;
    /// `cudaMemLocationType::HOST` — host.
    pub const HOST: i32 = 2;
    /// `cudaMemLocationType::HOST_NUMA` — host numa.
    pub const HOST_NUMA: i32 = 3;
    /// `cudaMemLocationType::HOST_NUMA_CURRENT` — host numa current.
    pub const HOST_NUMA_CURRENT: i32 = 4;
}

/// `cudaMemAccessFlags`.
pub mod cudaMemAccessFlags {
    /// `cudaMemAccessFlags::NONE` — none.
    pub const NONE: i32 = 0;
    /// `cudaMemAccessFlags::READ` — read.
    pub const READ: i32 = 1;
    /// `cudaMemAccessFlags::READ_WRITE` — read write.
    pub const READ_WRITE: i32 = 3;
}

/// `cudaMemAllocationHandleType` — IPC export type.
pub mod cudaMemAllocationHandleType {
    /// `cudaMemAllocationHandleType::NONE` — none.
    pub const NONE: i32 = 0;
    /// `cudaMemAllocationHandleType::POSIX_FILE_DESCRIPTOR` — posix file descriptor.
    pub const POSIX_FILE_DESCRIPTOR: i32 = 1;
    /// `cudaMemAllocationHandleType::WIN32` — win32.
    pub const WIN32: i32 = 2;
    /// `cudaMemAllocationHandleType::WIN32_KMT` — win32 kmt.
    pub const WIN32_KMT: i32 = 4;
    /// `cudaMemAllocationHandleType::FABRIC` — fabric.
    pub const FABRIC: i32 = 8;
}

/// `cudaMemPoolAttr` — selector for `cudaMemPoolSetAttribute` / `GetAttribute`.
pub mod cudaMemPoolAttr {
    /// `cudaMemPoolAttr::REUSE_FOLLOW_EVENT_DEPENDENCIES` — reuse follow event dependencies.
    pub const REUSE_FOLLOW_EVENT_DEPENDENCIES: i32 = 1;
    /// `cudaMemPoolAttr::REUSE_ALLOW_OPPORTUNISTIC` — reuse allow opportunistic.
    pub const REUSE_ALLOW_OPPORTUNISTIC: i32 = 2;
    /// `cudaMemPoolAttr::REUSE_ALLOW_INTERNAL_DEPENDENCIES` — reuse allow internal dependencies.
    pub const REUSE_ALLOW_INTERNAL_DEPENDENCIES: i32 = 3;
    /// `cudaMemPoolAttr::RELEASE_THRESHOLD` — release threshold.
    pub const RELEASE_THRESHOLD: i32 = 4;
    /// `cudaMemPoolAttr::RESERVED_MEM_CURRENT` — reserved mem current.
    pub const RESERVED_MEM_CURRENT: i32 = 5;
    /// `cudaMemPoolAttr::RESERVED_MEM_HIGH` — reserved mem high.
    pub const RESERVED_MEM_HIGH: i32 = 6;
    /// `cudaMemPoolAttr::USED_MEM_CURRENT` — used mem current.
    pub const USED_MEM_CURRENT: i32 = 7;
    /// `cudaMemPoolAttr::USED_MEM_HIGH` — used mem high.
    pub const USED_MEM_HIGH: i32 = 8;
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
/// `cudaMemLocation` — CUDA descriptor / record.
pub struct cudaMemLocation {
    /// `type_` field.
    pub type_: core::ffi::c_int,
    /// `id` field.
    pub id: core::ffi::c_int,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
/// `cudaMemAccessDesc` — CUDA descriptor / record.
pub struct cudaMemAccessDesc {
    /// `location` field.
    pub location: cudaMemLocation,
    /// `flags` field.
    pub flags: core::ffi::c_int,
}

/// `cudaMemPoolProps` — 88 bytes in C. Matches the Driver-side
/// `CUmemPoolProps` layout byte-for-byte.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct cudaMemPoolProps {
    /// `alloc_type` field.
    pub alloc_type: core::ffi::c_int,
    /// `handle_types` field.
    pub handle_types: core::ffi::c_int,
    /// `location` field.
    pub location: cudaMemLocation,
    /// `win32_security_attributes` field.
    pub win32_security_attributes: *mut c_void,
    /// `max_size` field.
    pub max_size: usize,
    /// `usage` field.
    pub usage: core::ffi::c_ushort,
    /// `reserved` field.
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
pub struct cudaMemPoolPtrExportData {
    /// `reserved` field.
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
pub struct cudaKernelNodeParams {
    /// `func` field.
    pub func: *mut c_void,
    /// `grid_dim` field.
    pub grid_dim: dim3,
    /// `block_dim` field.
    pub block_dim: dim3,
    /// `shared_mem_bytes` field.
    pub shared_mem_bytes: core::ffi::c_uint,
    /// `kernel_params` field.
    pub kernel_params: *mut *mut c_void,
    /// `extra` field.
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
pub struct cudaMemsetParams {
    /// `dst` field.
    pub dst: *mut c_void,
    /// `pitch` field.
    pub pitch: usize,
    /// `value` field.
    pub value: core::ffi::c_uint,
    /// `element_size` field.
    pub element_size: core::ffi::c_uint,
    /// `width` field.
    pub width: usize,
    /// `height` field.
    pub height: usize,
}

/// `cudaHostNodeParams` — `{ fn, user_data }` for `cudaGraphAddHostNode`.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct cudaHostNodeParams {
    /// `fn_` field.
    pub fn_: cudaHostFn_t,
    /// `user_data` field.
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
pub struct cudaMemAllocNodeParams {
    /// `pool_props` field.
    pub pool_props: cudaMemPoolProps,
    /// `access_descs` field.
    pub access_descs: *const cudaMemAccessDesc,
    /// `access_desc_count` field.
    pub access_desc_count: usize,
    /// `bytesize` field.
    pub bytesize: usize,
    /// `dptr` field.
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
pub mod cudaGraphExecUpdateResult {
    /// `cudaGraphExecUpdateResult::SUCCESS` — success.
    pub const SUCCESS: i32 = 0;
    /// `cudaGraphExecUpdateResult::ERROR` — error.
    pub const ERROR: i32 = 1;
    /// `cudaGraphExecUpdateResult::ERROR_TOPOLOGY_CHANGED` — error topology changed.
    pub const ERROR_TOPOLOGY_CHANGED: i32 = 2;
    /// `cudaGraphExecUpdateResult::ERROR_NODE_TYPE_CHANGED` — error node type changed.
    pub const ERROR_NODE_TYPE_CHANGED: i32 = 3;
    /// `cudaGraphExecUpdateResult::ERROR_FUNCTION_CHANGED` — error function changed.
    pub const ERROR_FUNCTION_CHANGED: i32 = 4;
    /// `cudaGraphExecUpdateResult::ERROR_PARAMETERS_CHANGED` — error parameters changed.
    pub const ERROR_PARAMETERS_CHANGED: i32 = 5;
    /// `cudaGraphExecUpdateResult::ERROR_NOT_SUPPORTED` — error not supported.
    pub const ERROR_NOT_SUPPORTED: i32 = 6;
    /// `cudaGraphExecUpdateResult::ERROR_UNSUPPORTED_FUNCTION_CHANGE` — error unsupported function change.
    pub const ERROR_UNSUPPORTED_FUNCTION_CHANGE: i32 = 7;
    /// `cudaGraphExecUpdateResult::ERROR_ATTRIBUTES_CHANGED` — error attributes changed.
    pub const ERROR_ATTRIBUTES_CHANGED: i32 = 8;
}

// ---- Pointer / function attributes ---------------------------------------

/// `cudaPointerAttributes` — returned by `cudaPointerGetAttributes`.
///
/// The C struct (CUDA 13.3 `driver_types.h`) ends in `long reserved[8]`, so it
/// is 56 bytes on Windows (`long` = 4) and 88 on LP64 Linux. The runtime
/// writes into the reserved tail, so the Rust mirror must be that big.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct cudaPointerAttributes {
    /// `cudaMemoryType` — 0=unregistered, 1=host, 2=device, 3=managed.
    pub type_: core::ffi::c_int,
    /// Device ordinal the allocation lives on.
    pub device: core::ffi::c_int,
    /// Device-addressable pointer (null for plain host memory).
    pub device_pointer: *mut c_void,
    /// Host-addressable pointer (null for plain device memory).
    pub host_pointer: *mut c_void,
    /// `long reserved[8]` — written by the runtime; callers should ignore it.
    pub reserved: [core::ffi::c_long; 8],
}

impl Default for cudaPointerAttributes {
    fn default() -> Self {
        Self {
            type_: 0,
            device: 0,
            device_pointer: core::ptr::null_mut(),
            host_pointer: core::ptr::null_mut(),
            reserved: [0; 8],
        }
    }
}

/// `cudaMemoryType` — `cudaPointerAttributes::type_` values.
pub mod cudaMemoryType {
    /// `cudaMemoryType::UNREGISTERED` — unregistered.
    pub const UNREGISTERED: i32 = 0;
    /// `cudaMemoryType::HOST` — host.
    pub const HOST: i32 = 1;
    /// `cudaMemoryType::DEVICE` — device.
    pub const DEVICE: i32 = 2;
    /// `cudaMemoryType::MANAGED` — managed.
    pub const MANAGED: i32 = 3;
}

/// `cudaFuncAttributes` — kernel metadata (36 bytes in C).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct cudaFuncAttributes {
    /// `shared_size_bytes` field.
    pub shared_size_bytes: usize,
    /// `const_size_bytes` field.
    pub const_size_bytes: usize,
    /// `local_size_bytes` field.
    pub local_size_bytes: usize,
    /// `max_threads_per_block` field.
    pub max_threads_per_block: core::ffi::c_int,
    /// `num_regs` field.
    pub num_regs: core::ffi::c_int,
    /// `ptx_version` field.
    pub ptx_version: core::ffi::c_int,
    /// `binary_version` field.
    pub binary_version: core::ffi::c_int,
    /// `cache_mode_ca` field.
    pub cache_mode_ca: core::ffi::c_int,
    /// `max_dynamic_shared_size_bytes` field.
    pub max_dynamic_shared_size_bytes: core::ffi::c_int,
    /// `preferred_shmem_carveout` field.
    pub preferred_shmem_carveout: core::ffi::c_int,
    /// `cluster_dim_must_be_set` field.
    pub cluster_dim_must_be_set: core::ffi::c_int,
    /// `required_cluster_width` field.
    pub required_cluster_width: core::ffi::c_int,
    /// `required_cluster_height` field.
    pub required_cluster_height: core::ffi::c_int,
    /// `required_cluster_depth` field.
    pub required_cluster_depth: core::ffi::c_int,
    /// `cluster_scheduling_policy_preference` field.
    pub cluster_scheduling_policy_preference: core::ffi::c_int,
    /// `non_portable_cluster_size_allowed` field.
    pub non_portable_cluster_size_allowed: core::ffi::c_int,
    /// `reserved` field.
    pub reserved: [core::ffi::c_int; 16],
}

/// `cudaStreamWriteValueFlags`.
pub mod cudaStreamWriteValueFlags {
    /// `cudaStreamWriteValueFlags::DEFAULT` — default.
    pub const DEFAULT: u32 = 0x0;
    /// `cudaStreamWriteValueFlags::NO_MEMORY_BARRIER` — no memory barrier.
    pub const NO_MEMORY_BARRIER: u32 = 0x1;
}

/// `cudaStreamWaitValueFlags`.
pub mod cudaStreamWaitValueFlags {
    /// `cudaStreamWaitValueFlags::GEQ` — geq.
    pub const GEQ: u32 = 0x0;
    /// `cudaStreamWaitValueFlags::EQ` — eq.
    pub const EQ: u32 = 0x1;
    /// `cudaStreamWaitValueFlags::AND` — and.
    pub const AND: u32 = 0x2;
    /// `cudaStreamWaitValueFlags::NOR` — nor.
    pub const NOR: u32 = 0x3;
    /// `cudaStreamWaitValueFlags::FLUSH` — flush.
    pub const FLUSH: u32 = 1 << 30;
}

/// Device-attribute selector (matches `cudaDeviceAttr` values).
pub mod cudaDeviceAttr {
    /// `cudaDeviceAttr::MAX_THREADS_PER_BLOCK` — max threads per block.
    pub const MAX_THREADS_PER_BLOCK: i32 = 1;
    /// `cudaDeviceAttr::MAX_BLOCK_DIM_X` — max block dim x.
    pub const MAX_BLOCK_DIM_X: i32 = 2;
    /// `cudaDeviceAttr::MAX_BLOCK_DIM_Y` — max block dim y.
    pub const MAX_BLOCK_DIM_Y: i32 = 3;
    /// `cudaDeviceAttr::MAX_BLOCK_DIM_Z` — max block dim z.
    pub const MAX_BLOCK_DIM_Z: i32 = 4;
    /// `cudaDeviceAttr::MAX_GRID_DIM_X` — max grid dim x.
    pub const MAX_GRID_DIM_X: i32 = 5;
    /// `cudaDeviceAttr::MAX_GRID_DIM_Y` — max grid dim y.
    pub const MAX_GRID_DIM_Y: i32 = 6;
    /// `cudaDeviceAttr::MAX_GRID_DIM_Z` — max grid dim z.
    pub const MAX_GRID_DIM_Z: i32 = 7;
    /// `cudaDeviceAttr::MAX_SHARED_MEMORY_PER_BLOCK` — max shared memory per block.
    pub const MAX_SHARED_MEMORY_PER_BLOCK: i32 = 8;
    /// `cudaDeviceAttr::TOTAL_CONSTANT_MEMORY` — total constant memory.
    pub const TOTAL_CONSTANT_MEMORY: i32 = 9;
    /// `cudaDeviceAttr::WARP_SIZE` — warp size.
    pub const WARP_SIZE: i32 = 10;
    /// `cudaDeviceAttr::MAX_PITCH` — max pitch.
    pub const MAX_PITCH: i32 = 11;
    /// `cudaDeviceAttr::MAX_REGISTERS_PER_BLOCK` — max registers per block.
    pub const MAX_REGISTERS_PER_BLOCK: i32 = 12;
    /// `cudaDeviceAttr::CLOCK_RATE` — clock rate.
    pub const CLOCK_RATE: i32 = 13;
    /// `cudaDeviceAttr::MULTIPROCESSOR_COUNT` — multiprocessor count.
    pub const MULTIPROCESSOR_COUNT: i32 = 16;
    /// `cudaDeviceAttr::COMPUTE_CAPABILITY_MAJOR` — compute capability major.
    pub const COMPUTE_CAPABILITY_MAJOR: i32 = 75;
    /// `cudaDeviceAttr::COMPUTE_CAPABILITY_MINOR` — compute capability minor.
    pub const COMPUTE_CAPABILITY_MINOR: i32 = 76;
    /// `cudaDeviceAttr::CONCURRENT_KERNELS` — concurrent kernels.
    pub const CONCURRENT_KERNELS: i32 = 31;
    /// `cudaDeviceAttr::ECC_ENABLED` — ecc enabled.
    pub const ECC_ENABLED: i32 = 32;
    /// `cudaDeviceAttr::PCI_BUS_ID` — pci bus id.
    pub const PCI_BUS_ID: i32 = 33;
    /// `cudaDeviceAttr::PCI_DEVICE_ID` — pci device id.
    pub const PCI_DEVICE_ID: i32 = 34;
    /// `cudaDeviceAttr::PCI_DOMAIN_ID` — pci domain id.
    pub const PCI_DOMAIN_ID: i32 = 50;
    /// `cudaDeviceAttr::INTEGRATED` — integrated.
    pub const INTEGRATED: i32 = 18;
}
