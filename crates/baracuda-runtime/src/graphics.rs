//! Graphics-API interop (Runtime API).
//!
//! Mirrors [`baracuda_driver::graphics`](../../../baracuda_driver/graphics/index.html)
//! on the Runtime side. Runtime contexts are implicit, so `register_*`
//! doesn't take a `Context` — it uses the current device's primary
//! context.
//!
//! For modern Vulkan / D3D12 flows, prefer
//! [`crate::external::ExternalMemory`].
//!
//! # Workflow
//!
//! 1. Register the graphics resource (`gl::register_buffer`,
//!    `d3d11::register_resource`, ...) — returns a [`GraphicsResource`].
//! 2. Call [`GraphicsResource::map`] on a stream to hand it to CUDA.
//! 3. [`GraphicsResource::mapped_pointer`] or
//!    [`GraphicsResource::mapped_array`] → usable device memory.
//! 4. ... compute ...
//! 5. [`GraphicsResource::unmap`] releases it back to the graphics API.
//!
//! Drop unregisters the resource automatically.

use std::sync::Arc;

use baracuda_cuda_sys::runtime::runtime;
use baracuda_cuda_sys::CUgraphicsResource;

use crate::error::{check, Result};
use crate::stream::Stream;

pub use baracuda_cuda_sys::types::{
    CUgraphicsMapResourceFlags as MapResourceFlags, CUgraphicsRegisterFlags as RegisterFlags,
};

/// Runtime-side handle to a graphics-API resource. Drop unregisters it.
#[derive(Clone)]
pub struct GraphicsResource {
    inner: Arc<GraphicsResourceInner>,
}

struct GraphicsResourceInner {
    handle: CUgraphicsResource,
}

unsafe impl Send for GraphicsResourceInner {}
unsafe impl Sync for GraphicsResourceInner {}

impl core::fmt::Debug for GraphicsResourceInner {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GraphicsResource")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

impl core::fmt::Debug for GraphicsResource {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.inner.fmt(f)
    }
}

impl GraphicsResource {
    /// Wrap an already-registered resource.
    ///
    /// # Safety
    ///
    /// `handle` must be a live resource from a `cuda*Register*` call.
    /// baracuda unregisters it when the last clone drops.
    pub unsafe fn from_raw(handle: CUgraphicsResource) -> Self {
        Self {
            inner: Arc::new(GraphicsResourceInner { handle }),
        }
    }

    #[inline]
    pub fn as_raw(&self) -> CUgraphicsResource {
        self.inner.handle
    }

    /// `cudaGraphicsResourceSetMapFlags`.
    pub fn set_map_flags(&self, flags: u32) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_graphics_resource_set_map_flags()?;
        check(unsafe { cu(self.inner.handle, flags) })
    }

    /// Map the resource onto `stream`. Pair with [`Self::unmap`].
    pub fn map(&self, stream: &Stream) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_graphics_map_resources()?;
        let mut arr = [self.inner.handle];
        check(unsafe { cu(1, arr.as_mut_ptr(), stream.as_raw()) })
    }

    /// Unmap the resource on `stream`.
    pub fn unmap(&self, stream: &Stream) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_graphics_unmap_resources()?;
        let mut arr = [self.inner.handle];
        check(unsafe { cu(1, arr.as_mut_ptr(), stream.as_raw()) })
    }

    /// Fetch the mapped device pointer + size (buffer-type resources).
    pub fn mapped_pointer(&self) -> Result<(*mut core::ffi::c_void, usize)> {
        let r = runtime()?;
        let cu = r.cuda_graphics_resource_get_mapped_pointer()?;
        let mut dptr: *mut core::ffi::c_void = core::ptr::null_mut();
        let mut size: usize = 0;
        check(unsafe { cu(&mut dptr, &mut size, self.inner.handle) })?;
        Ok((dptr, size))
    }

    /// Fetch a subresource array (image-type resources).
    pub fn mapped_array(&self, array_index: u32, mip_level: u32) -> Result<*mut core::ffi::c_void> {
        let r = runtime()?;
        let cu = r.cuda_graphics_sub_resource_get_mapped_array()?;
        let mut arr: *mut core::ffi::c_void = core::ptr::null_mut();
        check(unsafe { cu(&mut arr, self.inner.handle, array_index, mip_level) })?;
        Ok(arr)
    }

    /// Fetch the mipmapped-array handle for a mapped mipmapped texture.
    pub fn mapped_mipmapped_array(&self) -> Result<*mut core::ffi::c_void> {
        let r = runtime()?;
        let cu = r.cuda_graphics_resource_get_mapped_mipmapped_array()?;
        let mut mip: *mut core::ffi::c_void = core::ptr::null_mut();
        check(unsafe { cu(&mut mip, self.inner.handle) })?;
        Ok(mip)
    }

    /// Bulk-map multiple resources in one call.
    pub fn map_all(resources: &[Self], stream: &Stream) -> Result<()> {
        if resources.is_empty() {
            return Ok(());
        }
        let r = runtime()?;
        let cu = r.cuda_graphics_map_resources()?;
        let mut raws: Vec<CUgraphicsResource> = resources.iter().map(|x| x.as_raw()).collect();
        check(unsafe {
            cu(
                raws.len() as core::ffi::c_int,
                raws.as_mut_ptr(),
                stream.as_raw(),
            )
        })
    }

    /// Bulk-unmap.
    pub fn unmap_all(resources: &[Self], stream: &Stream) -> Result<()> {
        if resources.is_empty() {
            return Ok(());
        }
        let r = runtime()?;
        let cu = r.cuda_graphics_unmap_resources()?;
        let mut raws: Vec<CUgraphicsResource> = resources.iter().map(|x| x.as_raw()).collect();
        check(unsafe {
            cu(
                raws.len() as core::ffi::c_int,
                raws.as_mut_ptr(),
                stream.as_raw(),
            )
        })
    }
}

impl Drop for GraphicsResourceInner {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        if let Ok(r) = runtime() {
            if let Ok(cu) = r.cuda_graphics_unregister_resource() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}

/// OpenGL interop.
pub mod gl {
    use super::*;
    use baracuda_cuda_sys::types::{GLenum, GLuint};

    /// List CUDA device ordinals associated with the current GL context.
    /// `device_list` selects which slice of the rendering pipeline
    /// (cudaGLDeviceList_all, _current_frame, _next_frame).
    pub fn get_devices(device_list: u32) -> Result<Vec<i32>> {
        let r = runtime()?;
        let cu = r.cuda_gl_get_devices()?;
        let mut count: core::ffi::c_uint = 0;
        let probe_rc = unsafe { cu(&mut count, core::ptr::null_mut(), 0, device_list) };
        // Treat "no GL context" as "no devices".
        if probe_rc != baracuda_cuda_sys::runtime::cudaError_t::Success {
            return Ok(Vec::new());
        }
        if count == 0 {
            return Ok(Vec::new());
        }
        let mut out = vec![0i32; count as usize];
        check(unsafe {
            cu(
                &mut count,
                out.as_mut_ptr(),
                out.len() as core::ffi::c_uint,
                device_list,
            )
        })?;
        out.truncate(count as usize);
        Ok(out)
    }

    /// Register a GL buffer object (VBO / SSBO / ...) for CUDA access.
    ///
    /// # Safety
    ///
    /// `buffer` must be a live GL buffer in a GL context current on the
    /// calling thread. The buffer must outlive the returned resource.
    pub unsafe fn register_buffer(buffer: GLuint, flags: u32) -> Result<GraphicsResource> {
        let r = runtime()?;
        let cu = r.cuda_graphics_gl_register_buffer()?;
        let mut res: CUgraphicsResource = core::ptr::null_mut();
        check(cu(&mut res, buffer, flags))?;
        Ok(GraphicsResource::from_raw(res))
    }

    /// Register a GL texture / renderbuffer. `target` is the GL binding
    /// target (`GL_TEXTURE_2D`, `GL_RENDERBUFFER`, ...).
    ///
    /// # Safety
    ///
    /// Same as [`register_buffer`].
    pub unsafe fn register_image(
        image: GLuint,
        target: GLenum,
        flags: u32,
    ) -> Result<GraphicsResource> {
        let r = runtime()?;
        let cu = r.cuda_graphics_gl_register_image()?;
        let mut res: CUgraphicsResource = core::ptr::null_mut();
        check(cu(&mut res, image, target, flags))?;
        Ok(GraphicsResource::from_raw(res))
    }
}

/// Direct3D 9 interop (Windows).
pub mod d3d9 {
    use super::*;

    /// CUDA device ordinal powering a given D3D9 adapter name.
    ///
    /// # Safety
    ///
    /// `adapter_name` must be a live NUL-terminated C string naming a
    /// DXGI adapter.
    pub unsafe fn get_device(adapter_name: *const core::ffi::c_char) -> Result<i32> {
        let r = runtime()?;
        let cu = r.cuda_d3d9_get_device()?;
        let mut dev: core::ffi::c_int = 0;
        check(cu(&mut dev, adapter_name))?;
        Ok(dev)
    }

    /// List CUDA devices associated with a D3D9 device.
    ///
    /// # Safety
    ///
    /// `d3d_device` must be a live `IDirect3DDevice9*`.
    pub unsafe fn get_devices(
        d3d_device: *mut core::ffi::c_void,
        device_list: u32,
    ) -> Result<Vec<i32>> {
        let r = runtime()?;
        let cu = r.cuda_d3d9_get_devices()?;
        let mut count: core::ffi::c_uint = 0;
        check(cu(
            &mut count,
            core::ptr::null_mut(),
            0,
            d3d_device,
            device_list,
        ))?;
        if count == 0 {
            return Ok(Vec::new());
        }
        let mut out = vec![0i32; count as usize];
        check(cu(
            &mut count,
            out.as_mut_ptr(),
            out.len() as core::ffi::c_uint,
            d3d_device,
            device_list,
        ))?;
        out.truncate(count as usize);
        Ok(out)
    }

    /// Register a D3D9 resource (`IDirect3DResource9*`).
    ///
    /// # Safety
    ///
    /// `resource` must outlive the returned [`GraphicsResource`].
    pub unsafe fn register_resource(
        resource: *mut core::ffi::c_void,
        flags: u32,
    ) -> Result<GraphicsResource> {
        let r = runtime()?;
        let cu = r.cuda_graphics_d3d9_register_resource()?;
        let mut res: CUgraphicsResource = core::ptr::null_mut();
        check(cu(&mut res, resource, flags))?;
        Ok(GraphicsResource::from_raw(res))
    }
}

/// Direct3D 10 interop (Windows).
pub mod d3d10 {
    use super::*;

    /// # Safety
    ///
    /// `adapter` must be a valid `IDXGIAdapter*`.
    pub unsafe fn get_device(adapter: *mut core::ffi::c_void) -> Result<i32> {
        let r = runtime()?;
        let cu = r.cuda_d3d10_get_device()?;
        let mut dev: core::ffi::c_int = 0;
        check(cu(&mut dev, adapter))?;
        Ok(dev)
    }

    /// # Safety
    ///
    /// `d3d_device` must be a valid `ID3D10Device*`.
    pub unsafe fn get_devices(
        d3d_device: *mut core::ffi::c_void,
        device_list: u32,
    ) -> Result<Vec<i32>> {
        let r = runtime()?;
        let cu = r.cuda_d3d10_get_devices()?;
        let mut count: core::ffi::c_uint = 0;
        check(cu(
            &mut count,
            core::ptr::null_mut(),
            0,
            d3d_device,
            device_list,
        ))?;
        if count == 0 {
            return Ok(Vec::new());
        }
        let mut out = vec![0i32; count as usize];
        check(cu(
            &mut count,
            out.as_mut_ptr(),
            out.len() as core::ffi::c_uint,
            d3d_device,
            device_list,
        ))?;
        out.truncate(count as usize);
        Ok(out)
    }

    /// # Safety
    ///
    /// `resource` must be a live `ID3D10Resource*`.
    pub unsafe fn register_resource(
        resource: *mut core::ffi::c_void,
        flags: u32,
    ) -> Result<GraphicsResource> {
        let r = runtime()?;
        let cu = r.cuda_graphics_d3d10_register_resource()?;
        let mut res: CUgraphicsResource = core::ptr::null_mut();
        check(cu(&mut res, resource, flags))?;
        Ok(GraphicsResource::from_raw(res))
    }
}

/// Direct3D 11 interop (Windows) — the D3D path most still relevant today.
pub mod d3d11 {
    use super::*;

    /// # Safety
    ///
    /// `adapter` must be a valid `IDXGIAdapter*`.
    pub unsafe fn get_device(adapter: *mut core::ffi::c_void) -> Result<i32> {
        let r = runtime()?;
        let cu = r.cuda_d3d11_get_device()?;
        let mut dev: core::ffi::c_int = 0;
        check(cu(&mut dev, adapter))?;
        Ok(dev)
    }

    /// # Safety
    ///
    /// `d3d_device` must be a valid `ID3D11Device*`.
    pub unsafe fn get_devices(
        d3d_device: *mut core::ffi::c_void,
        device_list: u32,
    ) -> Result<Vec<i32>> {
        let r = runtime()?;
        let cu = r.cuda_d3d11_get_devices()?;
        let mut count: core::ffi::c_uint = 0;
        check(cu(
            &mut count,
            core::ptr::null_mut(),
            0,
            d3d_device,
            device_list,
        ))?;
        if count == 0 {
            return Ok(Vec::new());
        }
        let mut out = vec![0i32; count as usize];
        check(cu(
            &mut count,
            out.as_mut_ptr(),
            out.len() as core::ffi::c_uint,
            d3d_device,
            device_list,
        ))?;
        out.truncate(count as usize);
        Ok(out)
    }

    /// # Safety
    ///
    /// `resource` must be a live `ID3D11Resource*`.
    pub unsafe fn register_resource(
        resource: *mut core::ffi::c_void,
        flags: u32,
    ) -> Result<GraphicsResource> {
        let r = runtime()?;
        let cu = r.cuda_graphics_d3d11_register_resource()?;
        let mut res: CUgraphicsResource = core::ptr::null_mut();
        check(cu(&mut res, resource, flags))?;
        Ok(GraphicsResource::from_raw(res))
    }
}

/// VDPAU interop (Linux video-decode pipelines).
pub mod vdpau {
    use super::*;

    /// # Safety
    ///
    /// `vdp_device` and `vdp_get_proc_address` must come from libvdpau's
    /// `VdpDeviceCreateX11` call on the current X display.
    pub unsafe fn get_device(
        vdp_device: *mut core::ffi::c_void,
        vdp_get_proc_address: *mut core::ffi::c_void,
    ) -> Result<i32> {
        let r = runtime()?;
        let cu = r.cuda_vdpau_get_device()?;
        let mut dev: core::ffi::c_int = 0;
        check(cu(&mut dev, vdp_device, vdp_get_proc_address))?;
        Ok(dev)
    }

    /// Register a VDPAU video surface (decoded frame).
    ///
    /// # Safety
    ///
    /// `vdp_surface` must live on the VDPAU device bound to the current
    /// CUDA context.
    pub unsafe fn register_video_surface(
        vdp_surface: *mut core::ffi::c_void,
        flags: u32,
    ) -> Result<GraphicsResource> {
        let r = runtime()?;
        let cu = r.cuda_graphics_vdpau_register_video_surface()?;
        let mut res: CUgraphicsResource = core::ptr::null_mut();
        check(cu(&mut res, vdp_surface, flags))?;
        Ok(GraphicsResource::from_raw(res))
    }

    /// Register a VDPAU output surface.
    ///
    /// # Safety
    ///
    /// Same as [`register_video_surface`].
    pub unsafe fn register_output_surface(
        vdp_surface: *mut core::ffi::c_void,
        flags: u32,
    ) -> Result<GraphicsResource> {
        let r = runtime()?;
        let cu = r.cuda_graphics_vdpau_register_output_surface()?;
        let mut res: CUgraphicsResource = core::ptr::null_mut();
        check(cu(&mut res, vdp_surface, flags))?;
        Ok(GraphicsResource::from_raw(res))
    }
}

/// EGL interop — Jetson camera / video pipelines via EGLImage + EGLStream.
pub mod egl {
    use super::*;
    use baracuda_cuda_sys::runtime::cudaEvent_t;
    use core::ffi::c_void;

    /// Register an `EGLImageKHR`.
    ///
    /// # Safety
    ///
    /// `image` must be a live `EGLImageKHR`.
    pub unsafe fn register_image(image: *mut c_void, flags: u32) -> Result<GraphicsResource> {
        let r = runtime()?;
        let cu = r.cuda_graphics_egl_register_image()?;
        let mut res: CUgraphicsResource = core::ptr::null_mut();
        check(cu(&mut res, image, flags))?;
        Ok(GraphicsResource::from_raw(res))
    }

    /// Fill `egl_frame_out` with the `cudaEglFrame` describing the
    /// mapped resource (YUV planes, RGB surface, etc).
    ///
    /// # Safety
    ///
    /// `egl_frame_out` must point at a writable `cudaEglFrame` slot.
    pub unsafe fn mapped_frame(
        resource: &GraphicsResource,
        egl_frame_out: *mut c_void,
        index: u32,
        mip_level: u32,
    ) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_graphics_resource_get_mapped_egl_frame()?;
        check(cu(egl_frame_out, resource.as_raw(), index, mip_level))
    }

    /// Wrap an `EGLSyncKHR` as a CUDA event.
    ///
    /// # Safety
    ///
    /// `egl_sync` must be a live `EGLSyncKHR`.
    pub unsafe fn event_from_sync(egl_sync: *mut c_void, flags: u32) -> Result<cudaEvent_t> {
        let r = runtime()?;
        let cu = r.cuda_event_create_from_egl_sync()?;
        let mut event: cudaEvent_t = core::ptr::null_mut();
        check(cu(&mut event, egl_sync, flags))?;
        Ok(event)
    }

    /// Connect CUDA as an EGLStream consumer.
    ///
    /// # Safety
    ///
    /// `connection` must point at a writable `cudaEglStreamConnection`
    /// slot; `egl_stream` must be a live `EGLStreamKHR`.
    pub unsafe fn stream_consumer_connect(
        connection: *mut c_void,
        egl_stream: *mut c_void,
    ) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_egl_stream_consumer_connect()?;
        check(cu(connection, egl_stream))
    }

    /// # Safety
    ///
    /// `connection` must be a connected consumer.
    pub unsafe fn stream_consumer_disconnect(connection: *mut c_void) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_egl_stream_consumer_disconnect()?;
        check(cu(connection))
    }

    /// # Safety
    ///
    /// `connection` must be a connected consumer. `stream_out` must
    /// point at a writable `cudaStream_t` slot.
    pub unsafe fn stream_consumer_acquire_frame(
        connection: *mut c_void,
        stream_out: *mut baracuda_cuda_sys::runtime::cudaStream_t,
        timeout: u32,
    ) -> Result<GraphicsResource> {
        let r = runtime()?;
        let cu = r.cuda_egl_stream_consumer_acquire_frame()?;
        let mut res: CUgraphicsResource = core::ptr::null_mut();
        check(cu(connection, &mut res, stream_out, timeout))?;
        Ok(GraphicsResource::from_raw(res))
    }

    /// # Safety
    ///
    /// `connection` must match the one used to acquire `resource`.
    pub unsafe fn stream_consumer_release_frame(
        connection: *mut c_void,
        resource: &GraphicsResource,
        stream_inout: *mut baracuda_cuda_sys::runtime::cudaStream_t,
    ) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_egl_stream_consumer_release_frame()?;
        check(cu(connection, resource.as_raw(), stream_inout))
    }

    /// Connect CUDA as an EGLStream producer.
    ///
    /// # Safety
    ///
    /// Same as [`stream_consumer_connect`].
    pub unsafe fn stream_producer_connect(
        connection: *mut c_void,
        egl_stream: *mut c_void,
        width: i32,
        height: i32,
    ) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_egl_stream_producer_connect()?;
        check(cu(connection, egl_stream, width, height))
    }

    /// # Safety
    ///
    /// `connection` must be a connected producer.
    pub unsafe fn stream_producer_disconnect(connection: *mut c_void) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_egl_stream_producer_disconnect()?;
        check(cu(connection))
    }

    /// Push a frame to the EGLStream.
    ///
    /// # Safety
    ///
    /// `egl_frame` must be a populated `cudaEglFrame`.
    pub unsafe fn stream_producer_present_frame(
        connection: *mut c_void,
        egl_frame: *mut c_void,
        stream_inout: *mut baracuda_cuda_sys::runtime::cudaStream_t,
    ) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_egl_stream_producer_present_frame()?;
        check(cu(connection, egl_frame, stream_inout))
    }

    /// Reclaim a previously-presented frame.
    ///
    /// # Safety
    ///
    /// `connection` must be a connected producer.
    pub unsafe fn stream_producer_return_frame(
        connection: *mut c_void,
        egl_frame_out: *mut c_void,
        stream_inout: *mut baracuda_cuda_sys::runtime::cudaStream_t,
    ) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_egl_stream_producer_return_frame()?;
        check(cu(connection, egl_frame_out, stream_inout))
    }
}

/// NvSci interop — query sync-object attributes CUDA expects from
/// NvSciSync signalers / waiters. Feed the resulting
/// `NvSciSyncAttrList` into libnvsciSync's reconcile + alloc, then
/// import the resulting object via
/// [`crate::external::ExternalSemaphore::import`] with the
/// `NvSciSync` handle type.
pub mod nvsci {
    use super::*;
    use crate::device::Device;

    /// `CUnvSciSyncAttr::SIGNAL`.
    pub const SIGNAL: i32 = 0;
    /// `CUnvSciSyncAttr::WAIT`.
    pub const WAIT: i32 = 1;

    /// `cudaDeviceGetNvSciSyncAttributes`.
    ///
    /// # Safety
    ///
    /// `attr_list` must be a live `NvSciSyncAttrList` created by
    /// libnvsciSync's `NvSciSyncAttrListCreate`.
    pub unsafe fn device_sync_attributes(
        attr_list: *mut core::ffi::c_void,
        device: &Device,
        direction: i32,
    ) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_device_get_nv_sci_sync_attributes()?;
        check(cu(attr_list, device.ordinal(), direction))
    }
}
