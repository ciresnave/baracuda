//! 3-D arrays + 3-D memcpy + mipmapped arrays.
//!
//! Volumetric data (MRI, CT, voxel grids) lives in `cuArray3DCreate`-backed
//! arrays; rectangular sub-region copies between host / device / array
//! go through `cuMemcpy3D`.
//!
//! [`Array3D`] handles plain 3-D arrays.
//! [`MipmappedArray`] holds a multi-level mipmap pyramid built from a
//! single `cuMipmappedArrayCreate`; individual levels are yielded as
//! [`Array3D`]s that *borrow* their parent's storage (don't free on drop).

use core::ffi::c_void;
use core::mem::size_of;
use std::sync::Arc;

use baracuda_cuda_sys::types::{
    CUarrayMapInfo, CUmemorytype, CUDA_ARRAY3D_DESCRIPTOR, CUDA_MEMCPY3D,
};
use baracuda_cuda_sys::{driver, CUarray, CUmipmappedArray};
use baracuda_types::DeviceRepr;

use crate::array::ArrayFormat;
use crate::context::Context;
use crate::error::{check, Result};
use crate::stream::Stream;

/// A 3-D CUDA array. Element format chosen at creation; channels are 1/2/4.
pub struct Array3D {
    inner: Arc<Array3DInner>,
}

struct Array3DInner {
    handle: CUarray,
    owned: bool,
    width: usize,
    height: usize,
    depth: usize,
    format: u32,
    num_channels: u32,
    #[allow(dead_code)]
    context: Context,
}

unsafe impl Send for Array3DInner {}
unsafe impl Sync for Array3DInner {}

impl core::fmt::Debug for Array3DInner {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Array3D")
            .field("w", &self.width)
            .field("h", &self.height)
            .field("d", &self.depth)
            .field("channels", &self.num_channels)
            .field("owned", &self.owned)
            .finish_non_exhaustive()
    }
}

impl core::fmt::Debug for Array3D {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.inner.fmt(f)
    }
}

impl Clone for Array3D {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl Array3D {
    /// Allocate a plain `width × height × depth` 3-D array.
    pub fn new(
        context: &Context,
        width: usize,
        height: usize,
        depth: usize,
        format: ArrayFormat,
        num_channels: u32,
    ) -> Result<Self> {
        Self::with_flags(context, width, height, depth, format, num_channels, 0)
    }

    /// Allocate with custom `flags` (layered, cubemap, surface-ldst, etc. —
    /// see [`baracuda_cuda_sys::types::CUarray3D_flags`]).
    pub fn with_flags(
        context: &Context,
        width: usize,
        height: usize,
        depth: usize,
        format: ArrayFormat,
        num_channels: u32,
        flags: u32,
    ) -> Result<Self> {
        assert!(
            matches!(num_channels, 1 | 2 | 4),
            "CUDA arrays require 1, 2, or 4 channels (got {num_channels})"
        );
        context.set_current()?;
        let d = driver()?;
        let cu = d.cu_array_3d_create()?;
        let desc = CUDA_ARRAY3D_DESCRIPTOR {
            width,
            height,
            depth,
            format: format_raw(format),
            num_channels,
            flags,
        };
        let mut handle: CUarray = core::ptr::null_mut();
        check(unsafe { cu(&mut handle, &desc) })?;
        Ok(Self {
            inner: Arc::new(Array3DInner {
                handle,
                owned: true,
                width,
                height,
                depth,
                format: format_raw(format),
                num_channels,
                context: context.clone(),
            }),
        })
    }

    /// Wrap an existing array handle (e.g. a mipmap level) without taking
    /// ownership. Drop is a no-op; the parent owns lifetime.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `CUarray` that outlives this wrapper.
    pub unsafe fn from_borrowed(
        context: &Context,
        handle: CUarray,
        width: usize,
        height: usize,
        depth: usize,
        format: ArrayFormat,
        num_channels: u32,
    ) -> Self {
        Self {
            inner: Arc::new(Array3DInner {
                handle,
                owned: false,
                width,
                height,
                depth,
                format: format_raw(format),
                num_channels,
                context: context.clone(),
            }),
        }
    }

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
    pub fn depth(&self) -> usize {
        self.inner.depth
    }
    /// Element width in bytes (channel size × channel count).
    pub fn bytes_per_element(&self) -> usize {
        let ch_size = match self.inner.format {
            baracuda_cuda_sys::types::CUarray_format::UNSIGNED_INT8
            | baracuda_cuda_sys::types::CUarray_format::SIGNED_INT8 => 1,
            baracuda_cuda_sys::types::CUarray_format::UNSIGNED_INT16
            | baracuda_cuda_sys::types::CUarray_format::SIGNED_INT16
            | baracuda_cuda_sys::types::CUarray_format::HALF => 2,
            _ => 4,
        };
        ch_size * (self.inner.num_channels as usize)
    }

    fn slice_count(&self) -> usize {
        self.inner.height.max(1) * self.inner.depth.max(1)
    }

    /// Synchronous host → 3-D array copy. `host.len()` must equal
    /// `width × max(height, 1) × max(depth, 1)`.
    pub fn copy_from_host<T: DeviceRepr>(&self, host: &[T]) -> Result<()> {
        assert_eq!(
            size_of::<T>(),
            self.bytes_per_element(),
            "host element type must match array texel size"
        );
        assert_eq!(host.len(), self.inner.width * self.slice_count());
        let d = driver()?;
        let cu = d.cu_memcpy_3d()?;
        let mut p = CUDA_MEMCPY3D::default();
        p.src_memory_type = CUmemorytype::HOST;
        p.src_host = host.as_ptr() as *const c_void;
        p.src_pitch = self.inner.width * self.bytes_per_element();
        p.src_height = self.inner.height.max(1);
        p.dst_memory_type = CUmemorytype::ARRAY;
        p.dst_array = self.inner.handle;
        p.width_in_bytes = self.inner.width * self.bytes_per_element();
        p.height = self.inner.height.max(1);
        p.depth = self.inner.depth.max(1);
        check(unsafe { cu(&p) })
    }

    /// Synchronous 3-D array → host copy.
    pub fn copy_to_host<T: DeviceRepr>(&self, host: &mut [T]) -> Result<()> {
        assert_eq!(
            size_of::<T>(),
            self.bytes_per_element(),
            "host element type must match array texel size"
        );
        assert_eq!(host.len(), self.inner.width * self.slice_count());
        let d = driver()?;
        let cu = d.cu_memcpy_3d()?;
        let mut p = CUDA_MEMCPY3D::default();
        p.src_memory_type = CUmemorytype::ARRAY;
        p.src_array = self.inner.handle;
        p.dst_memory_type = CUmemorytype::HOST;
        p.dst_host = host.as_mut_ptr() as *mut c_void;
        p.dst_pitch = self.inner.width * self.bytes_per_element();
        p.dst_height = self.inner.height.max(1);
        p.width_in_bytes = self.inner.width * self.bytes_per_element();
        p.height = self.inner.height.max(1);
        p.depth = self.inner.depth.max(1);
        check(unsafe { cu(&p) })
    }

    /// Async variant of [`Array3D::copy_from_host`] on `stream`.
    /// Note: CUDA requires the host buffer to be pinned for real async
    /// behavior — otherwise the driver falls back to a sync copy.
    pub fn copy_from_host_async<T: DeviceRepr>(&self, host: &[T], stream: &Stream) -> Result<()> {
        assert_eq!(size_of::<T>(), self.bytes_per_element());
        assert_eq!(host.len(), self.inner.width * self.slice_count());
        let d = driver()?;
        let cu = d.cu_memcpy_3d_async()?;
        let mut p = CUDA_MEMCPY3D::default();
        p.src_memory_type = CUmemorytype::HOST;
        p.src_host = host.as_ptr() as *const c_void;
        p.src_pitch = self.inner.width * self.bytes_per_element();
        p.src_height = self.inner.height.max(1);
        p.dst_memory_type = CUmemorytype::ARRAY;
        p.dst_array = self.inner.handle;
        p.width_in_bytes = self.inner.width * self.bytes_per_element();
        p.height = self.inner.height.max(1);
        p.depth = self.inner.depth.max(1);
        check(unsafe { cu(&p, stream.as_raw()) })
    }
}

impl Drop for Array3DInner {
    fn drop(&mut self) {
        if !self.owned || self.handle.is_null() {
            return;
        }
        if let Ok(d) = driver() {
            if let Ok(cu) = d.cu_array_destroy() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}

fn format_raw(format: ArrayFormat) -> u32 {
    use baracuda_cuda_sys::types::CUarray_format;
    match format {
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

/// A mipmap pyramid — `num_levels` 3-D arrays sharing one allocation.
pub struct MipmappedArray {
    handle: CUmipmappedArray,
    base_width: usize,
    base_height: usize,
    base_depth: usize,
    num_levels: u32,
    format: ArrayFormat,
    num_channels: u32,
    context: Context,
}

unsafe impl Send for MipmappedArray {}
unsafe impl Sync for MipmappedArray {}

impl core::fmt::Debug for MipmappedArray {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MipmappedArray")
            .field("levels", &self.num_levels)
            .field("base_w", &self.base_width)
            .field("base_h", &self.base_height)
            .field("base_d", &self.base_depth)
            .finish_non_exhaustive()
    }
}

impl MipmappedArray {
    /// Create a mipmap pyramid with `num_levels` levels. The base level is
    /// `(width, height, depth)`; each subsequent level is half-sized per
    /// axis (hardware-managed).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context: &Context,
        width: usize,
        height: usize,
        depth: usize,
        format: ArrayFormat,
        num_channels: u32,
        num_levels: u32,
        flags: u32,
    ) -> Result<Self> {
        assert!(
            matches!(num_channels, 1 | 2 | 4),
            "CUDA arrays require 1, 2, or 4 channels (got {num_channels})"
        );
        assert!(num_levels >= 1, "mipmap must have at least 1 level");
        context.set_current()?;
        let d = driver()?;
        let cu = d.cu_mipmapped_array_create()?;
        let desc = CUDA_ARRAY3D_DESCRIPTOR {
            width,
            height,
            depth,
            format: format_raw(format),
            num_channels,
            flags,
        };
        let mut handle: CUmipmappedArray = core::ptr::null_mut();
        check(unsafe { cu(&mut handle, &desc, num_levels) })?;
        Ok(Self {
            handle,
            base_width: width,
            base_height: height,
            base_depth: depth,
            num_levels,
            format,
            num_channels,
            context: context.clone(),
        })
    }

    /// Return level `level` as a borrowed [`Array3D`]. The returned view
    /// does not free its own storage — the parent `MipmappedArray` owns it.
    pub fn level(&self, level: u32) -> Result<Array3D> {
        assert!(
            level < self.num_levels,
            "mipmap level {level} out of range (0..{})",
            self.num_levels
        );
        let d = driver()?;
        let cu = d.cu_mipmapped_array_get_level()?;
        let mut arr: CUarray = core::ptr::null_mut();
        check(unsafe { cu(&mut arr, self.handle, level) })?;
        let shift = level as usize;
        let w = (self.base_width >> shift).max(1);
        let h = (self.base_height >> shift).max(1);
        let depth = (self.base_depth >> shift).max(self.base_depth.min(1));
        // SAFETY: `arr` is a valid CUarray owned by `self`; caller keeps
        // `self` alive while the Array3D is used (it's a non-owning clone).
        let view = unsafe {
            Array3D::from_borrowed(
                &self.context,
                arr,
                w,
                h,
                depth,
                self.format,
                self.num_channels,
            )
        };
        Ok(view)
    }

    #[inline]
    pub fn as_raw(&self) -> CUmipmappedArray {
        self.handle
    }
    #[inline]
    pub fn num_levels(&self) -> u32 {
        self.num_levels
    }
}

impl Drop for MipmappedArray {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        if let Ok(d) = driver() {
            if let Ok(cu) = d.cu_mipmapped_array_destroy() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}

/// Bulk map / unmap sparse-array tiles and mipmap tails on a stream.
///
/// Each [`CUarrayMapInfo`] entry describes one tile update — use the
/// builder methods on that type (`with_array` / `with_mipmapped_array`,
/// `with_sparse_level` / `with_miptail`, `with_mem_handle`, `as_map` /
/// `as_unmap`, `with_offset`) to construct entries, then pass the array
/// here. Mix map and unmap ops in a single call freely.
///
/// Availability: sparse-array support requires a device with sparse
/// residency (compute capability 7.0+ on desktop; Ampere+ for full
/// mipmap-tail support). Calls on non-supporting devices fail with
/// `CUDA_ERROR_NOT_SUPPORTED`.
pub fn map_array_async(info: &mut [CUarrayMapInfo], stream: &Stream) -> Result<()> {
    if info.is_empty() {
        return Ok(());
    }
    let d = driver()?;
    let cu = d.cu_mem_map_array_async()?;
    check(unsafe {
        cu(
            info.as_mut_ptr(),
            info.len() as core::ffi::c_uint,
            stream.as_raw(),
        )
    })
}
