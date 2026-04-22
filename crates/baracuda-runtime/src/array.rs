//! CUDA arrays + texture / surface objects (Runtime API).
//!
//! Mirrors [`baracuda_driver::array`]. An [`Array`] is an opaque
//! on-device layout optimized for texture fetches. [`TextureObject`] /
//! [`SurfaceObject`] wrap the CUDA 5+ object-based texture API; the
//! legacy reference-based API is intentionally not wrapped.

use core::ffi::c_void;
use std::sync::Arc;

use baracuda_cuda_sys::runtime::runtime;
use baracuda_cuda_sys::runtime::types::{
    cudaArray_t, cudaChannelFormatDesc, cudaChannelFormatKind, cudaExtent, cudaMipmappedArray_t,
    cudaResourceDesc, cudaResourceViewDesc, cudaSurfaceObject_t, cudaTextureDesc,
    cudaTextureObject_t,
};

use crate::error::{check, Result};

/// Construct a `cudaChannelFormatDesc` with 1/2/4 channels of `bits` bits
/// of the given `kind` (matches the `cudaCreateChannelDesc<T>()` helpers
/// in CUDA headers).
pub fn channel_desc(
    bits_x: i32,
    bits_y: i32,
    bits_z: i32,
    bits_w: i32,
    kind: i32,
) -> cudaChannelFormatDesc {
    cudaChannelFormatDesc {
        x: bits_x,
        y: bits_y,
        z: bits_z,
        w: bits_w,
        kind,
    }
}

/// `cudaCreateChannelDesc<u8>` — one 8-bit unsigned channel.
#[inline]
pub fn channel_desc_u8() -> cudaChannelFormatDesc {
    channel_desc(8, 0, 0, 0, cudaChannelFormatKind::UNSIGNED)
}

/// `cudaCreateChannelDesc<f32>` — one 32-bit float channel.
#[inline]
pub fn channel_desc_f32() -> cudaChannelFormatDesc {
    channel_desc(32, 0, 0, 0, cudaChannelFormatKind::FLOAT)
}

/// A 2-D / 3-D CUDA array handle.
#[derive(Clone)]
pub struct Array {
    inner: Arc<ArrayInner>,
}

struct ArrayInner {
    handle: cudaArray_t,
    width: usize,
    height: usize,
    depth: usize,
    desc: cudaChannelFormatDesc,
}

unsafe impl Send for ArrayInner {}
unsafe impl Sync for ArrayInner {}

impl core::fmt::Debug for ArrayInner {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Array")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("depth", &self.depth)
            .field("desc", &self.desc)
            .finish_non_exhaustive()
    }
}

impl core::fmt::Debug for Array {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.inner.fmt(f)
    }
}

impl Array {
    /// Allocate a 2-D array `width × height` with the given channel
    /// descriptor. `flags = 0` for the default layout.
    pub fn new_2d(
        desc: &cudaChannelFormatDesc,
        width: usize,
        height: usize,
        flags: u32,
    ) -> Result<Self> {
        let r = runtime()?;
        let cu = r.cuda_malloc_array()?;
        let mut arr: cudaArray_t = core::ptr::null_mut();
        check(unsafe {
            cu(
                &mut arr,
                desc as *const cudaChannelFormatDesc as *const c_void,
                width,
                height,
                flags,
            )
        })?;
        Ok(Self {
            inner: Arc::new(ArrayInner {
                handle: arr,
                width,
                height,
                depth: 0,
                desc: *desc,
            }),
        })
    }

    /// Allocate a 3-D array with the given extent + descriptor.
    /// `flags = 0` for the default.
    pub fn new_3d(desc: &cudaChannelFormatDesc, extent: cudaExtent, flags: u32) -> Result<Self> {
        let r = runtime()?;
        let cu = r.cuda_malloc_3d_array()?;
        let mut arr: cudaArray_t = core::ptr::null_mut();
        check(unsafe {
            cu(
                &mut arr,
                desc as *const cudaChannelFormatDesc as *const c_void,
                &extent as *const cudaExtent as *const c_void,
                flags,
            )
        })?;
        Ok(Self {
            inner: Arc::new(ArrayInner {
                handle: arr,
                width: extent.width,
                height: extent.height,
                depth: extent.depth,
                desc: *desc,
            }),
        })
    }

    /// Wrap an already-allocated `cudaArray_t`.
    ///
    /// # Safety
    ///
    /// `handle` must be a live CUDA array. The wrapper frees it on drop.
    pub unsafe fn from_raw(
        handle: cudaArray_t,
        desc: cudaChannelFormatDesc,
        width: usize,
        height: usize,
        depth: usize,
    ) -> Self {
        Self {
            inner: Arc::new(ArrayInner {
                handle,
                width,
                height,
                depth,
                desc,
            }),
        }
    }

    #[inline]
    pub fn as_raw(&self) -> cudaArray_t {
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
    #[inline]
    pub fn desc(&self) -> &cudaChannelFormatDesc {
        &self.inner.desc
    }

    /// Query the runtime-reported channel desc + extent.
    pub fn info(&self) -> Result<(cudaChannelFormatDesc, cudaExtent, u32)> {
        let r = runtime()?;
        let cu = r.cuda_array_get_info()?;
        let mut desc = cudaChannelFormatDesc::default();
        let mut extent = cudaExtent::default();
        let mut flags: core::ffi::c_uint = 0;
        check(unsafe {
            cu(
                &mut desc as *mut cudaChannelFormatDesc as *mut c_void,
                &mut extent as *mut cudaExtent as *mut c_void,
                &mut flags,
                self.inner.handle,
            )
        })?;
        Ok((desc, extent, flags))
    }
}

impl Drop for ArrayInner {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        if let Ok(r) = runtime() {
            if let Ok(cu) = r.cuda_free_array() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}

/// A mipmapped CUDA array.
#[derive(Clone)]
pub struct MipmappedArray {
    inner: Arc<MipmappedArrayInner>,
}

struct MipmappedArrayInner {
    handle: cudaMipmappedArray_t,
}

unsafe impl Send for MipmappedArrayInner {}
unsafe impl Sync for MipmappedArrayInner {}

impl core::fmt::Debug for MipmappedArray {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MipmappedArray")
            .field("handle", &self.inner.handle)
            .finish()
    }
}

impl MipmappedArray {
    /// Allocate a mipmapped array.
    pub fn new(
        desc: &cudaChannelFormatDesc,
        extent: cudaExtent,
        num_levels: u32,
        flags: u32,
    ) -> Result<Self> {
        let r = runtime()?;
        let cu = r.cuda_malloc_mipmapped_array()?;
        let mut h: cudaMipmappedArray_t = core::ptr::null_mut();
        check(unsafe {
            cu(
                &mut h,
                desc as *const cudaChannelFormatDesc as *const c_void,
                &extent as *const cudaExtent as *const c_void,
                num_levels,
                flags,
            )
        })?;
        Ok(Self {
            inner: Arc::new(MipmappedArrayInner { handle: h }),
        })
    }

    #[inline]
    pub fn as_raw(&self) -> cudaMipmappedArray_t {
        self.inner.handle
    }

    /// Fetch the `level`-th mipmap as a regular `cudaArray_t` (view; does
    /// NOT free on drop — the parent mipmapped array owns it).
    pub fn level(&self, level: u32) -> Result<cudaArray_t> {
        let r = runtime()?;
        let cu = r.cuda_get_mipmapped_array_level()?;
        let mut out: cudaArray_t = core::ptr::null_mut();
        check(unsafe { cu(&mut out, self.inner.handle, level) })?;
        Ok(out)
    }
}

impl Drop for MipmappedArrayInner {
    fn drop(&mut self) {
        if let Ok(r) = runtime() {
            if let Ok(cu) = r.cuda_free_mipmapped_array() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}

/// A texture object — a read-only sampler bound to an array (or linear
/// device memory). Pass `as_raw()` as a u64 kernel argument.
pub struct TextureObject {
    handle: cudaTextureObject_t,
    // Keep the backing array alive for the lifetime of the texture.
    _backing: Option<Array>,
}

impl core::fmt::Debug for TextureObject {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TextureObject")
            .field("handle", &self.handle)
            .finish()
    }
}

impl TextureObject {
    /// Create a texture object over an array with the given sampler + view.
    pub fn new(
        array: &Array,
        tex_desc: &cudaTextureDesc,
        view_desc: Option<&cudaResourceViewDesc>,
    ) -> Result<Self> {
        let res_desc = cudaResourceDesc::from_array(array.as_raw());
        let r = runtime()?;
        let cu = r.cuda_create_texture_object()?;
        let mut obj: cudaTextureObject_t = 0;
        let view_ptr = view_desc
            .map(|v| v as *const cudaResourceViewDesc as *const c_void)
            .unwrap_or(core::ptr::null());
        check(unsafe {
            cu(
                &mut obj,
                &res_desc as *const cudaResourceDesc as *const c_void,
                tex_desc as *const cudaTextureDesc as *const c_void,
                view_ptr,
            )
        })?;
        Ok(Self {
            handle: obj,
            _backing: Some(array.clone()),
        })
    }

    /// Create a texture over a raw `cudaResourceDesc` (e.g. a linear
    /// memory slab).
    ///
    /// # Safety
    ///
    /// `res_desc`'s backing memory must outlive the returned texture.
    pub unsafe fn from_resource(
        res_desc: &cudaResourceDesc,
        tex_desc: &cudaTextureDesc,
        view_desc: Option<&cudaResourceViewDesc>,
    ) -> Result<Self> {
        let r = runtime()?;
        let cu = r.cuda_create_texture_object()?;
        let mut obj: cudaTextureObject_t = 0;
        let view_ptr = view_desc
            .map(|v| v as *const cudaResourceViewDesc as *const c_void)
            .unwrap_or(core::ptr::null());
        check(cu(
            &mut obj,
            res_desc as *const cudaResourceDesc as *const c_void,
            tex_desc as *const cudaTextureDesc as *const c_void,
            view_ptr,
        ))?;
        Ok(Self {
            handle: obj,
            _backing: None,
        })
    }

    #[inline]
    pub fn as_raw(&self) -> cudaTextureObject_t {
        self.handle
    }

    /// Query the resource descriptor the texture was created with.
    pub fn resource_desc(&self) -> Result<cudaResourceDesc> {
        let r = runtime()?;
        let cu = r.cuda_get_texture_object_resource_desc()?;
        let mut d = cudaResourceDesc::default();
        check(unsafe { cu(&mut d as *mut cudaResourceDesc as *mut c_void, self.handle) })?;
        Ok(d)
    }

    /// Query the sampler (filter/address/normalize) state.
    pub fn texture_desc(&self) -> Result<cudaTextureDesc> {
        let r = runtime()?;
        let cu = r.cuda_get_texture_object_texture_desc()?;
        let mut d = cudaTextureDesc::default();
        check(unsafe { cu(&mut d as *mut cudaTextureDesc as *mut c_void, self.handle) })?;
        Ok(d)
    }
}

impl Drop for TextureObject {
    fn drop(&mut self) {
        if let Ok(r) = runtime() {
            if let Ok(cu) = r.cuda_destroy_texture_object() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}

/// A surface object — writable array access from kernels.
pub struct SurfaceObject {
    handle: cudaSurfaceObject_t,
    _backing: Option<Array>,
}

impl core::fmt::Debug for SurfaceObject {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SurfaceObject")
            .field("handle", &self.handle)
            .finish()
    }
}

impl SurfaceObject {
    /// Create a surface object over a CUDA array.
    pub fn new(array: &Array) -> Result<Self> {
        let res_desc = cudaResourceDesc::from_array(array.as_raw());
        let r = runtime()?;
        let cu = r.cuda_create_surface_object()?;
        let mut obj: cudaSurfaceObject_t = 0;
        check(unsafe {
            cu(
                &mut obj,
                &res_desc as *const cudaResourceDesc as *const c_void,
            )
        })?;
        Ok(Self {
            handle: obj,
            _backing: Some(array.clone()),
        })
    }

    #[inline]
    pub fn as_raw(&self) -> cudaSurfaceObject_t {
        self.handle
    }

    /// Query the resource descriptor the surface was created with.
    pub fn resource_desc(&self) -> Result<cudaResourceDesc> {
        let r = runtime()?;
        let cu = r.cuda_get_surface_object_resource_desc()?;
        let mut d = cudaResourceDesc::default();
        check(unsafe { cu(&mut d as *mut cudaResourceDesc as *mut c_void, self.handle) })?;
        Ok(d)
    }
}

impl Drop for SurfaceObject {
    fn drop(&mut self) {
        if let Ok(r) = runtime() {
            if let Ok(cu) = r.cuda_destroy_surface_object() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}
