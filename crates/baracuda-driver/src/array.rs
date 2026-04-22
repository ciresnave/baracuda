//! CUDA arrays + texture / surface objects.
//!
//! A [`Array`] is an opaque on-device layout optimized for texture fetches
//! and surface loads/stores. Host↔Array copies go through the 2-D memcpy
//! path (see [`crate::memcpy2d`]), with `CUmemorytype::ARRAY` on whichever
//! side the array sits on.
//!
//! Texture and surface objects are created *from* an array (or a pitched
//! device pointer, for linear textures). This module exposes the modern
//! "object" API (CUDA 5+); the legacy reference-based API is not wrapped.

use core::ffi::c_void;
use core::mem::size_of;
use std::sync::Arc;

use baracuda_cuda_sys::types::{
    CUarray_format, CUDA_ARRAY_DESCRIPTOR, CUDA_RESOURCE_DESC, CUDA_TEXTURE_DESC,
};
use baracuda_cuda_sys::{driver, CUarray, CUsurfObject, CUtexObject};
use baracuda_types::DeviceRepr;

use crate::context::Context;
use crate::error::{check, Result};

/// A 2-D CUDA array. Element format is chosen at creation; channels are
/// typically 1, 2, or 4.
pub struct Array {
    inner: Arc<ArrayInner>,
}

struct ArrayInner {
    handle: CUarray,
    width: usize,
    height: usize,
    format: u32,
    num_channels: u32,
    #[allow(dead_code)]
    context: Context,
}

unsafe impl Send for ArrayInner {}
unsafe impl Sync for ArrayInner {}

impl core::fmt::Debug for ArrayInner {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Array")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("format", &self.format)
            .field("channels", &self.num_channels)
            .finish_non_exhaustive()
    }
}

impl core::fmt::Debug for Array {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.inner.fmt(f)
    }
}

impl Clone for Array {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

/// Element format shorthand for common single-channel arrays. Build a
/// multi-channel descriptor by passing a [`CUarray_format`] constant and a
/// channel count directly to [`Array::new`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ArrayFormat {
    U8,
    U16,
    U32,
    I8,
    I16,
    I32,
    F16,
    F32,
}

impl ArrayFormat {
    #[inline]
    fn raw(self) -> u32 {
        match self {
            ArrayFormat::U8 => CUarray_format::UNSIGNED_INT8,
            ArrayFormat::U16 => CUarray_format::UNSIGNED_INT16,
            ArrayFormat::U32 => CUarray_format::UNSIGNED_INT32,
            ArrayFormat::I8 => CUarray_format::SIGNED_INT8,
            ArrayFormat::I16 => CUarray_format::SIGNED_INT16,
            ArrayFormat::I32 => CUarray_format::SIGNED_INT32,
            ArrayFormat::F16 => CUarray_format::HALF,
            ArrayFormat::F32 => CUarray_format::FLOAT,
        }
    }

    /// Width of a single texel (one channel) in bytes.
    #[inline]
    pub fn bytes_per_channel(self) -> usize {
        match self {
            ArrayFormat::U8 | ArrayFormat::I8 => 1,
            ArrayFormat::U16 | ArrayFormat::I16 | ArrayFormat::F16 => 2,
            ArrayFormat::U32 | ArrayFormat::I32 | ArrayFormat::F32 => 4,
        }
    }
}

impl Array {
    /// Allocate a `width × height` 2-D array with `num_channels` elements of
    /// `format` per texel. Set `height = 0` for a 1-D array.
    pub fn new(
        context: &Context,
        width: usize,
        height: usize,
        format: ArrayFormat,
        num_channels: u32,
    ) -> Result<Self> {
        assert!(
            matches!(num_channels, 1 | 2 | 4),
            "CUDA arrays require 1, 2, or 4 channels (got {num_channels})",
        );
        context.set_current()?;
        let d = driver()?;
        let cu = d.cu_array_create()?;
        let desc = CUDA_ARRAY_DESCRIPTOR {
            width,
            height,
            format: format.raw(),
            num_channels,
        };
        let mut handle: CUarray = core::ptr::null_mut();
        check(unsafe { cu(&mut handle, &desc) })?;
        Ok(Self {
            inner: Arc::new(ArrayInner {
                handle,
                width,
                height,
                format: format.raw(),
                num_channels,
                context: context.clone(),
            }),
        })
    }

    /// Raw `CUarray`. Use with care.
    #[inline]
    pub fn as_raw(&self) -> CUarray {
        self.inner.handle
    }
    #[inline]
    pub fn width(&self) -> usize {
        self.inner.width
    }
    #[inline]
    pub fn height(&self) -> usize {
        self.inner.height
    }
    #[inline]
    pub fn num_channels(&self) -> u32 {
        self.inner.num_channels
    }
    /// Element width in bytes (channel size × channel count).
    pub fn bytes_per_element(&self) -> usize {
        let ch_size = match self.inner.format {
            CUarray_format::UNSIGNED_INT8 | CUarray_format::SIGNED_INT8 => 1,
            CUarray_format::UNSIGNED_INT16
            | CUarray_format::SIGNED_INT16
            | CUarray_format::HALF => 2,
            _ => 4,
        };
        ch_size * (self.inner.num_channels as usize)
    }

    /// Synchronous host→array 2-D copy. `host` must contain exactly
    /// `width × height` elements of type `T`; `T` size must match
    /// `self.bytes_per_element()`.
    pub fn copy_from_host<T: DeviceRepr>(&self, host: &[T]) -> Result<()> {
        assert_eq!(
            size_of::<T>(),
            self.bytes_per_element(),
            "host element type must match array texel size",
        );
        let h = self.inner.height.max(1);
        assert_eq!(host.len(), self.inner.width * h);
        let d = driver()?;
        let cu = d.cu_memcpy_2d()?;
        let p = baracuda_cuda_sys::types::CUDA_MEMCPY2D {
            src_memory_type: baracuda_cuda_sys::types::CUmemorytype::HOST,
            src_host: host.as_ptr() as *const c_void,
            src_pitch: self.inner.width * self.bytes_per_element(),
            dst_memory_type: baracuda_cuda_sys::types::CUmemorytype::ARRAY,
            dst_array: self.inner.handle,
            width_in_bytes: self.inner.width * self.bytes_per_element(),
            height: h,
            ..Default::default()
        };
        check(unsafe { cu(&p) })
    }

    /// Query this array's descriptor back from CUDA. Useful for arrays
    /// you received from an external source and don't have shape info for.
    pub fn descriptor(&self) -> Result<CUDA_ARRAY_DESCRIPTOR> {
        let d = driver()?;
        let cu = d.cu_array_get_descriptor()?;
        let mut desc = CUDA_ARRAY_DESCRIPTOR::default();
        check(unsafe { cu(&mut desc, self.inner.handle) })?;
        Ok(desc)
    }

    /// Query the array's memory-allocation size + alignment requirements
    /// on `device`. Useful when backing an array with a VMM allocation.
    pub fn memory_requirements(
        &self,
        device: &crate::Device,
    ) -> Result<baracuda_cuda_sys::types::CUDA_ARRAY_MEMORY_REQUIREMENTS> {
        let d = driver()?;
        let cu = d.cu_array_get_memory_requirements()?;
        let mut req = baracuda_cuda_sys::types::CUDA_ARRAY_MEMORY_REQUIREMENTS::default();
        check(unsafe { cu(&mut req, self.inner.handle, device.as_raw()) })?;
        Ok(req)
    }

    /// Query the array's sparse-tile properties. Meaningful on sparse /
    /// tiled arrays created with the `SPARSE` flag.
    pub fn sparse_properties(
        &self,
    ) -> Result<baracuda_cuda_sys::types::CUDA_ARRAY_SPARSE_PROPERTIES> {
        let d = driver()?;
        let cu = d.cu_array_get_sparse_properties()?;
        let mut sp = baracuda_cuda_sys::types::CUDA_ARRAY_SPARSE_PROPERTIES::default();
        check(unsafe { cu(&mut sp, self.inner.handle) })?;
        Ok(sp)
    }

    /// Return the raw `CUarray` handle of plane `plane_idx` of a
    /// multi-planar array (YUV / NV12). The plane is owned by `self` —
    /// the raw handle must NOT be passed to `cuArrayDestroy`. Use
    /// together with [`Array::descriptor_of_raw`] if you need shape info.
    pub fn plane_raw(&self, plane_idx: u32) -> Result<CUarray> {
        let d = driver()?;
        let cu = d.cu_array_get_plane()?;
        let mut out: CUarray = core::ptr::null_mut();
        check(unsafe { cu(&mut out, self.inner.handle, plane_idx) })?;
        Ok(out)
    }

    /// Helper: query the `CUDA_ARRAY_DESCRIPTOR` of a raw array handle
    /// (e.g. a plane returned by [`Array::plane_raw`]).
    ///
    /// # Safety
    ///
    /// `handle` must be a live `CUarray`.
    pub unsafe fn descriptor_of_raw(handle: CUarray) -> Result<CUDA_ARRAY_DESCRIPTOR> {
        let d = driver()?;
        let cu = d.cu_array_get_descriptor()?;
        let mut desc = CUDA_ARRAY_DESCRIPTOR::default();
        check(cu(&mut desc, handle))?;
        Ok(desc)
    }

    /// Synchronous array→host 2-D copy.
    pub fn copy_to_host<T: DeviceRepr>(&self, host: &mut [T]) -> Result<()> {
        assert_eq!(
            size_of::<T>(),
            self.bytes_per_element(),
            "host element type must match array texel size",
        );
        let h = self.inner.height.max(1);
        assert_eq!(host.len(), self.inner.width * h);
        let d = driver()?;
        let cu = d.cu_memcpy_2d()?;
        let p = baracuda_cuda_sys::types::CUDA_MEMCPY2D {
            src_memory_type: baracuda_cuda_sys::types::CUmemorytype::ARRAY,
            src_array: self.inner.handle,
            dst_memory_type: baracuda_cuda_sys::types::CUmemorytype::HOST,
            dst_host: host.as_mut_ptr() as *mut c_void,
            dst_pitch: self.inner.width * self.bytes_per_element(),
            width_in_bytes: self.inner.width * self.bytes_per_element(),
            height: h,
            ..Default::default()
        };
        check(unsafe { cu(&p) })
    }
}

impl Drop for ArrayInner {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        if let Ok(d) = driver() {
            if let Ok(cu) = d.cu_array_destroy() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}

/// A texture object — a read-only, filtered view onto a CUDA array.
pub struct TextureObject {
    handle: CUtexObject,
    _array: Array,
}

impl core::fmt::Debug for TextureObject {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TextureObject")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

unsafe impl Send for TextureObject {}
unsafe impl Sync for TextureObject {}

/// Configuration for [`TextureObject::new`].
#[derive(Copy, Clone, Debug)]
pub struct TextureDesc {
    pub address_mode: [TextureAddressMode; 3],
    pub filter_mode: TextureFilterMode,
    pub read_normalized: bool,
    pub normalized_coords: bool,
}

impl Default for TextureDesc {
    fn default() -> Self {
        Self {
            address_mode: [TextureAddressMode::Clamp; 3],
            filter_mode: TextureFilterMode::Point,
            read_normalized: false,
            normalized_coords: false,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TextureAddressMode {
    Wrap,
    Clamp,
    Mirror,
    Border,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TextureFilterMode {
    Point,
    Linear,
}

impl TextureAddressMode {
    fn raw(self) -> u32 {
        use baracuda_cuda_sys::types::CUaddress_mode;
        match self {
            TextureAddressMode::Wrap => CUaddress_mode::WRAP,
            TextureAddressMode::Clamp => CUaddress_mode::CLAMP,
            TextureAddressMode::Mirror => CUaddress_mode::MIRROR,
            TextureAddressMode::Border => CUaddress_mode::BORDER,
        }
    }
}

impl TextureFilterMode {
    fn raw(self) -> u32 {
        use baracuda_cuda_sys::types::CUfilter_mode;
        match self {
            TextureFilterMode::Point => CUfilter_mode::POINT,
            TextureFilterMode::Linear => CUfilter_mode::LINEAR,
        }
    }
}

impl TextureObject {
    /// Create a texture object that reads from `array`. Uses point filtering
    /// and clamp addressing by default; override with [`TextureObject::with_desc`].
    pub fn new(array: &Array) -> Result<Self> {
        Self::with_desc(array, TextureDesc::default())
    }

    pub fn with_desc(array: &Array, desc: TextureDesc) -> Result<Self> {
        let d = driver()?;
        let cu = d.cu_tex_object_create()?;
        let res_desc = CUDA_RESOURCE_DESC::from_array(array.as_raw());
        let mut flags: core::ffi::c_uint = 0;
        const CU_TRSF_READ_AS_INTEGER: core::ffi::c_uint = 0x01;
        const CU_TRSF_NORMALIZED_COORDINATES: core::ffi::c_uint = 0x02;
        if !desc.read_normalized {
            flags |= CU_TRSF_READ_AS_INTEGER;
        }
        if desc.normalized_coords {
            flags |= CU_TRSF_NORMALIZED_COORDINATES;
        }
        let tex_desc = CUDA_TEXTURE_DESC {
            address_mode: [
                desc.address_mode[0].raw(),
                desc.address_mode[1].raw(),
                desc.address_mode[2].raw(),
            ],
            filter_mode: desc.filter_mode.raw(),
            flags,
            ..Default::default()
        };
        let mut handle: CUtexObject = 0;
        check(unsafe { cu(&mut handle, &res_desc, &tex_desc, core::ptr::null()) })?;
        Ok(Self {
            handle,
            _array: array.clone(),
        })
    }

    #[inline]
    pub fn as_raw(&self) -> CUtexObject {
        self.handle
    }
}

impl Drop for TextureObject {
    fn drop(&mut self) {
        if self.handle == 0 {
            return;
        }
        if let Ok(d) = driver() {
            if let Ok(cu) = d.cu_tex_object_destroy() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}

/// A surface object — a read/write view onto a CUDA array.
pub struct SurfaceObject {
    handle: CUsurfObject,
    _array: Array,
}

impl core::fmt::Debug for SurfaceObject {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SurfaceObject")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

unsafe impl Send for SurfaceObject {}
unsafe impl Sync for SurfaceObject {}

impl SurfaceObject {
    pub fn new(array: &Array) -> Result<Self> {
        let d = driver()?;
        let cu = d.cu_surf_object_create()?;
        let res_desc = CUDA_RESOURCE_DESC::from_array(array.as_raw());
        let mut handle: CUsurfObject = 0;
        check(unsafe { cu(&mut handle, &res_desc) })?;
        Ok(Self {
            handle,
            _array: array.clone(),
        })
    }

    #[inline]
    pub fn as_raw(&self) -> CUsurfObject {
        self.handle
    }
}

impl Drop for SurfaceObject {
    fn drop(&mut self) {
        if self.handle == 0 {
            return;
        }
        if let Ok(d) = driver() {
            if let Ok(cu) = d.cu_surf_object_destroy() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}
