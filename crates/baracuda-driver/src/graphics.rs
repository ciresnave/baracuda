//! Graphics-API interop — register GL / D3D / VDPAU / EGL resources with
//! CUDA for zero-copy compute on their memory.
//!
//! **Note:** For modern flows on Vulkan and D3D12, prefer
//! [`crate::external::ExternalMemory`] — it's the forward-looking API
//! NVIDIA recommends. The `cuGraphics*` functions here remain useful for:
//!
//! - OpenGL interop on drivers without `GL_EXT_memory_object_fd/win32`.
//! - D3D9 / D3D10 / D3D11 legacy code paths.
//! - VDPAU video-surface import on Linux.
//! - EGL image / stream interop on Jetson.
//!
//! # Workflow
//!
//! 1. Register the graphics resource (`gl::register_buffer`,
//!    `d3d11::register_resource`, ...) — returns a [`GraphicsResource`].
//! 2. Call [`GraphicsResource::map`] on a stream to expose it to CUDA.
//! 3. Call [`GraphicsResource::mapped_pointer`] or
//!    [`GraphicsResource::mapped_array`] to get a usable handle.
//! 4. ... compute ...
//! 5. [`GraphicsResource::unmap`] to release it back to the graphics API.
//!
//! On drop, the resource is unregistered (and unmapped if still mapped,
//! best-effort).

use std::sync::Arc;

use baracuda_cuda_sys::types::CUmipmappedArray;
use baracuda_cuda_sys::{driver, CUarray, CUdeviceptr, CUgraphicsResource};

use crate::context::Context;
use crate::error::{check, Result};
use crate::stream::Stream;

pub use baracuda_cuda_sys::types::{
    CUGLDeviceList as GLDeviceList, CUgraphicsMapResourceFlags as MapResourceFlags,
    CUgraphicsRegisterFlags as RegisterFlags,
};

/// A CUDA-side handle to a graphics-API resource. Drops unregister the
/// resource via `cuGraphicsUnregisterResource`.
#[derive(Clone)]
pub struct GraphicsResource {
    inner: Arc<GraphicsResourceInner>,
}

struct GraphicsResourceInner {
    handle: CUgraphicsResource,
    #[allow(dead_code)]
    context: Context,
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
    /// Wrap a raw `CUgraphicsResource`. The resource is assumed to be
    /// registered already (e.g. from `cu<API>Register*`).
    ///
    /// # Safety
    ///
    /// `handle` must be a live, registered resource. baracuda unregisters
    /// it when this wrapper (and all clones) drop.
    pub unsafe fn from_raw(context: &Context, handle: CUgraphicsResource) -> Self {
        Self {
            inner: Arc::new(GraphicsResourceInner {
                handle,
                context: context.clone(),
            }),
        }
    }

    #[inline]
    pub fn as_raw(&self) -> CUgraphicsResource {
        self.inner.handle
    }

    /// Change the map flags before mapping (e.g. READ_ONLY for shader
    /// sampling, WRITE_DISCARD for compute output).
    pub fn set_map_flags(&self, flags: u32) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_graphics_resource_set_map_flags()?;
        check(unsafe { cu(self.inner.handle, flags) })
    }

    /// Map the resource on `stream` so CUDA can access it. Pair with
    /// [`Self::unmap`].
    pub fn map(&self, stream: &Stream) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_graphics_map_resources()?;
        let mut arr = [self.inner.handle];
        check(unsafe { cu(1, arr.as_mut_ptr(), stream.as_raw()) })
    }

    /// Unmap the resource on `stream`, releasing it back to the graphics API.
    pub fn unmap(&self, stream: &Stream) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_graphics_unmap_resources()?;
        let mut arr = [self.inner.handle];
        check(unsafe { cu(1, arr.as_mut_ptr(), stream.as_raw()) })
    }

    /// Retrieve the device pointer + size for a mapped buffer-type
    /// resource (typical of GL buffer objects, D3D buffers).
    pub fn mapped_pointer(&self) -> Result<(CUdeviceptr, usize)> {
        let d = driver()?;
        let cu = d.cu_graphics_resource_get_mapped_pointer()?;
        let mut dptr = CUdeviceptr(0);
        let mut size: usize = 0;
        check(unsafe { cu(&mut dptr, &mut size, self.inner.handle) })?;
        Ok((dptr, size))
    }

    /// Retrieve the subresource CUDA array for a mapped image / texture
    /// resource (GL textures, D3D textures, VDPAU surfaces).
    pub fn mapped_array(&self, array_index: u32, mip_level: u32) -> Result<CUarray> {
        let d = driver()?;
        let cu = d.cu_graphics_sub_resource_get_mapped_array()?;
        let mut arr: CUarray = core::ptr::null_mut();
        check(unsafe { cu(&mut arr, self.inner.handle, array_index, mip_level) })?;
        Ok(arr)
    }

    /// Retrieve the mipmapped-array handle for a mapped mipmapped texture.
    pub fn mapped_mipmapped_array(&self) -> Result<CUmipmappedArray> {
        let d = driver()?;
        let cu = d.cu_graphics_resource_get_mapped_mipmapped_array()?;
        let mut mip: CUmipmappedArray = core::ptr::null_mut();
        check(unsafe { cu(&mut mip, self.inner.handle) })?;
        Ok(mip)
    }

    /// Bulk-map multiple resources in one call. More efficient than
    /// calling [`Self::map`] per resource.
    pub fn map_all(resources: &[Self], stream: &Stream) -> Result<()> {
        if resources.is_empty() {
            return Ok(());
        }
        let d = driver()?;
        let cu = d.cu_graphics_map_resources()?;
        let mut raws: Vec<CUgraphicsResource> = resources.iter().map(|r| r.as_raw()).collect();
        check(unsafe {
            cu(
                raws.len() as core::ffi::c_uint,
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
        let d = driver()?;
        let cu = d.cu_graphics_unmap_resources()?;
        let mut raws: Vec<CUgraphicsResource> = resources.iter().map(|r| r.as_raw()).collect();
        check(unsafe {
            cu(
                raws.len() as core::ffi::c_uint,
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
        if let Ok(d) = driver() {
            if let Ok(cu) = d.cu_graphics_unregister_resource() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}

/// OpenGL-specific interop.
pub mod gl {
    use super::*;
    use crate::device::Device;
    use baracuda_cuda_sys::types::{GLenum, GLuint};

    /// Initialize the GL interop subsystem. Must be called from a thread
    /// with a current GL context. Deprecated but retained for legacy code.
    pub fn init() -> Result<()> {
        let d = driver()?;
        let cu = d.cu_gl_init()?;
        check(unsafe { cu() })
    }

    /// List CUDA devices associated with a given slice of the GL
    /// rendering pipeline (current frame, next frame, or all devices).
    pub fn get_devices(device_list: u32) -> Result<Vec<Device>> {
        let d = driver()?;
        let cu = d.cu_gl_get_devices()?;
        // Query count first by passing cuda_device_count_in = 0.
        let mut count: core::ffi::c_uint = 0;
        let probe = unsafe { cu(&mut count, core::ptr::null_mut(), 0, device_list) };
        // Some driver versions return SUCCESS with count set; others
        // return NOT_INITIALIZED if there's no GL context. Treat the
        // latter as "no GL devices" rather than error.
        if !probe.is_success() && probe != baracuda_cuda_sys::CUresult::ERROR_NOT_INITIALIZED {
            check(probe)?;
        }
        if count == 0 {
            return Ok(Vec::new());
        }
        let mut raw = vec![baracuda_cuda_sys::CUdevice(0); count as usize];
        check(unsafe {
            cu(
                &mut count,
                raw.as_mut_ptr(),
                raw.len() as core::ffi::c_uint,
                device_list,
            )
        })?;
        // SAFETY: `Device` is a `pub(crate)` newtype over `CUdevice`;
        // constructing from driver-returned ordinals is safe.
        Ok(raw.into_iter().map(Device).collect())
    }

    /// Register an OpenGL buffer object (VBO / SSBO / UBO / ...) for CUDA
    /// access. `buffer` is the GL buffer name from `glGenBuffers`.
    ///
    /// # Safety
    ///
    /// `buffer` must be a valid GL buffer in a GL context that is current
    /// on the calling thread. The buffer must outlive the returned
    /// [`GraphicsResource`].
    pub unsafe fn register_buffer(
        context: &Context,
        buffer: GLuint,
        flags: u32,
    ) -> Result<GraphicsResource> {
        context.set_current()?;
        let d = driver()?;
        let cu = d.cu_graphics_gl_register_buffer()?;
        let mut resource: CUgraphicsResource = core::ptr::null_mut();
        check(cu(&mut resource, buffer, flags))?;
        Ok(GraphicsResource::from_raw(context, resource))
    }

    /// Register an OpenGL texture / renderbuffer. `target` is the GL
    /// binding target (`GL_TEXTURE_2D`, `GL_RENDERBUFFER`, ...).
    ///
    /// # Safety
    ///
    /// Same discipline as [`register_buffer`]: `image` must be live in a
    /// current GL context, matching the declared `target`.
    pub unsafe fn register_image(
        context: &Context,
        image: GLuint,
        target: GLenum,
        flags: u32,
    ) -> Result<GraphicsResource> {
        context.set_current()?;
        let d = driver()?;
        let cu = d.cu_graphics_gl_register_image()?;
        let mut resource: CUgraphicsResource = core::ptr::null_mut();
        check(cu(&mut resource, image, target, flags))?;
        Ok(GraphicsResource::from_raw(context, resource))
    }
}

/// Direct3D 9 interop (Windows).
pub mod d3d9 {
    use super::*;
    use crate::device::Device;
    use baracuda_cuda_sys::types::{ID3DDevice, ID3DResource};

    /// Find the CUDA device powering a D3D9 adapter, by adapter name.
    ///
    /// # Safety
    ///
    /// `adapter_name` must be a NUL-terminated C string naming a live
    /// DXGI adapter.
    pub unsafe fn get_device(adapter_name: *const core::ffi::c_char) -> Result<Device> {
        let d = driver()?;
        let cu = d.cu_d3d9_get_device()?;
        let mut dev = baracuda_cuda_sys::CUdevice(0);
        check(cu(&mut dev, adapter_name))?;
        Ok(Device(dev))
    }

    /// List CUDA devices associated with a given D3D9 device.
    ///
    /// # Safety
    ///
    /// `d3d_device` must be a valid `IDirect3DDevice9*` pointer.
    pub unsafe fn get_devices(d3d_device: ID3DDevice, device_list: u32) -> Result<Vec<Device>> {
        let d = driver()?;
        let cu = d.cu_d3d9_get_devices()?;
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
        let mut raw = vec![baracuda_cuda_sys::CUdevice(0); count as usize];
        check(cu(
            &mut count,
            raw.as_mut_ptr(),
            raw.len() as core::ffi::c_uint,
            d3d_device,
            device_list,
        ))?;
        Ok(raw.into_iter().map(Device).collect())
    }

    /// Register a D3D9 resource (`IDirect3DResource9*`) for CUDA access.
    ///
    /// # Safety
    ///
    /// `resource` must be live for the duration of the returned
    /// [`GraphicsResource`].
    pub unsafe fn register_resource(
        context: &Context,
        resource: ID3DResource,
        flags: u32,
    ) -> Result<GraphicsResource> {
        context.set_current()?;
        let d = driver()?;
        let cu = d.cu_graphics_d3d9_register_resource()?;
        let mut res: CUgraphicsResource = core::ptr::null_mut();
        check(cu(&mut res, resource, flags))?;
        Ok(GraphicsResource::from_raw(context, res))
    }
}

/// Direct3D 10 interop (Windows).
pub mod d3d10 {
    use super::*;
    use crate::device::Device;
    use baracuda_cuda_sys::types::{ID3DDevice, ID3DResource};

    /// # Safety
    ///
    /// `adapter` must be a valid `IDXGIAdapter*`.
    pub unsafe fn get_device(adapter: ID3DDevice) -> Result<Device> {
        let d = driver()?;
        let cu = d.cu_d3d10_get_device()?;
        let mut dev = baracuda_cuda_sys::CUdevice(0);
        check(cu(&mut dev, adapter))?;
        Ok(Device(dev))
    }

    /// # Safety
    ///
    /// `d3d_device` must be a valid `ID3D10Device*`.
    pub unsafe fn get_devices(d3d_device: ID3DDevice, device_list: u32) -> Result<Vec<Device>> {
        let d = driver()?;
        let cu = d.cu_d3d10_get_devices()?;
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
        let mut raw = vec![baracuda_cuda_sys::CUdevice(0); count as usize];
        check(cu(
            &mut count,
            raw.as_mut_ptr(),
            raw.len() as core::ffi::c_uint,
            d3d_device,
            device_list,
        ))?;
        Ok(raw.into_iter().map(Device).collect())
    }

    /// # Safety
    ///
    /// `resource` must be a live `ID3D10Resource*`.
    pub unsafe fn register_resource(
        context: &Context,
        resource: ID3DResource,
        flags: u32,
    ) -> Result<GraphicsResource> {
        context.set_current()?;
        let d = driver()?;
        let cu = d.cu_graphics_d3d10_register_resource()?;
        let mut res: CUgraphicsResource = core::ptr::null_mut();
        check(cu(&mut res, resource, flags))?;
        Ok(GraphicsResource::from_raw(context, res))
    }
}

/// VDPAU interop (Linux). Register VDPAU video / output surfaces from
/// libvdpau into CUDA for zero-copy decode → compute pipelines.
pub mod vdpau {
    use super::*;
    use crate::device::Device;
    use baracuda_cuda_sys::types::{
        VdpDevice, VdpGetProcAddress, VdpOutputSurface, VdpVideoSurface,
    };

    /// Look up the CUDA device powering a given VDPAU device.
    ///
    /// # Safety
    ///
    /// `vdp_device` and `vdp_get_proc_address` must come from a live
    /// libvdpau `VdpDeviceCreateX11` call on the current X display.
    pub unsafe fn get_device(
        vdp_device: VdpDevice,
        vdp_get_proc_address: VdpGetProcAddress,
    ) -> Result<Device> {
        let d = driver()?;
        let cu = d.cu_vdpau_get_device()?;
        let mut dev = baracuda_cuda_sys::CUdevice(0);
        check(cu(&mut dev, vdp_device, vdp_get_proc_address))?;
        Ok(Device(dev))
    }

    /// Register a VDPAU video surface (e.g. a decoded frame output) for
    /// CUDA access.
    ///
    /// # Safety
    ///
    /// `vdp_surface` must be a live VDPAU video surface on the VDPAU
    /// device associated with the current CUDA context.
    pub unsafe fn register_video_surface(
        context: &Context,
        vdp_surface: VdpVideoSurface,
        flags: u32,
    ) -> Result<GraphicsResource> {
        context.set_current()?;
        let d = driver()?;
        let cu = d.cu_graphics_vdpau_register_video_surface()?;
        let mut res: CUgraphicsResource = core::ptr::null_mut();
        check(cu(&mut res, vdp_surface, flags))?;
        Ok(GraphicsResource::from_raw(context, res))
    }

    /// Register a VDPAU output surface.
    ///
    /// # Safety
    ///
    /// Same as [`register_video_surface`].
    pub unsafe fn register_output_surface(
        context: &Context,
        vdp_surface: VdpOutputSurface,
        flags: u32,
    ) -> Result<GraphicsResource> {
        context.set_current()?;
        let d = driver()?;
        let cu = d.cu_graphics_vdpau_register_output_surface()?;
        let mut res: CUgraphicsResource = core::ptr::null_mut();
        check(cu(&mut res, vdp_surface, flags))?;
        Ok(GraphicsResource::from_raw(context, res))
    }
}

/// EGL interop — primary path for Jetson video pipelines (NvMM streams,
/// camera capture). Supports both image-based and EGLStream-based flows.
pub mod egl {
    use super::*;
    use baracuda_cuda_sys::types::{CUeglFrame, EGLImageKHR, EGLStreamKHR, EGLSyncKHR};
    use baracuda_cuda_sys::CUevent;
    use core::ffi::c_void;

    /// Register an `EGLImageKHR` for CUDA access.
    ///
    /// # Safety
    ///
    /// `image` must be a live `EGLImageKHR` (typically created via
    /// `eglCreateImageKHR`). The image must outlive the returned resource.
    pub unsafe fn register_image(
        context: &Context,
        image: EGLImageKHR,
        flags: u32,
    ) -> Result<GraphicsResource> {
        context.set_current()?;
        let d = driver()?;
        let cu = d.cu_graphics_egl_register_image()?;
        let mut res: CUgraphicsResource = core::ptr::null_mut();
        check(cu(&mut res, image, flags))?;
        Ok(GraphicsResource::from_raw(context, res))
    }

    /// Fetch the `CUeglFrame` (YUV / RGB descriptor) for a mapped EGL
    /// resource. `index` and `mip_level` select the plane / mip.
    pub fn mapped_frame(
        resource: &GraphicsResource,
        index: u32,
        mip_level: u32,
    ) -> Result<CUeglFrame> {
        let d = driver()?;
        let cu = d.cu_graphics_resource_get_mapped_egl_frame()?;
        let mut frame = CUeglFrame::default();
        check(unsafe { cu(&mut frame, resource.as_raw(), index, mip_level) })?;
        Ok(frame)
    }

    /// Wrap an `EGLSyncKHR` as a CUDA event — lets CUDA wait on GPU work
    /// submitted via EGL's synchronization primitives.
    ///
    /// # Safety
    ///
    /// `egl_sync` must be a live `EGLSyncKHR`.
    pub unsafe fn event_from_sync(egl_sync: EGLSyncKHR, flags: u32) -> Result<CUevent> {
        let d = driver()?;
        let cu = d.cu_event_create_from_egl_sync()?;
        let mut event: CUevent = core::ptr::null_mut();
        check(cu(&mut event, egl_sync, flags))?;
        Ok(event)
    }

    /// EGLStream consumer-side connection — CUDA receives frames from an
    /// EGL producer (e.g. a camera or decoder).
    ///
    /// `connection` is an opaque pointer handed back by the driver;
    /// baracuda does not wrap it in a typed struct, so you'll need to
    /// keep the pointer alive yourself.
    ///
    /// # Safety
    ///
    /// `stream` must be a live `EGLStreamKHR`. `connection` must be a
    /// pointer to a single `CUeglStreamConnection` slot.
    pub unsafe fn stream_consumer_connect(
        connection: *mut c_void,
        stream: EGLStreamKHR,
    ) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_egl_stream_consumer_connect()?;
        check(cu(connection, stream))
    }

    /// # Safety
    ///
    /// `connection` must be a connection previously set up by
    /// [`stream_consumer_connect`].
    pub unsafe fn stream_consumer_disconnect(connection: *mut c_void) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_egl_stream_consumer_disconnect()?;
        check(cu(connection))
    }

    /// Acquire the next frame from an EGLStream consumer. `timeout` is
    /// in nanoseconds (0 = non-blocking).
    ///
    /// # Safety
    ///
    /// `connection` must be a connected consumer (see
    /// [`stream_consumer_connect`]).
    pub unsafe fn stream_consumer_acquire_frame(
        context: &Context,
        connection: *mut c_void,
        cu_stream_out: *mut baracuda_cuda_sys::CUstream,
        timeout: u32,
    ) -> Result<GraphicsResource> {
        let d = driver()?;
        let cu = d.cu_egl_stream_consumer_acquire_frame()?;
        let mut resource: CUgraphicsResource = core::ptr::null_mut();
        check(cu(connection, &mut resource, cu_stream_out, timeout))?;
        Ok(GraphicsResource::from_raw(context, resource))
    }

    /// # Safety
    ///
    /// `connection` must match the one used to acquire `resource`.
    pub unsafe fn stream_consumer_release_frame(
        connection: *mut c_void,
        resource: &GraphicsResource,
        cu_stream_inout: *mut baracuda_cuda_sys::CUstream,
    ) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_egl_stream_consumer_release_frame()?;
        check(cu(connection, resource.as_raw(), cu_stream_inout))
    }

    /// EGLStream producer-side connection — CUDA feeds frames into an
    /// EGL stream (typically to a compositor or encoder consumer).
    ///
    /// # Safety
    ///
    /// `stream` must be a live `EGLStreamKHR`. `connection` must be a
    /// pointer to a single `CUeglStreamConnection` slot.
    pub unsafe fn stream_producer_connect(
        connection: *mut c_void,
        stream: EGLStreamKHR,
        width: i32,
        height: i32,
    ) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_egl_stream_producer_connect()?;
        check(cu(connection, stream, width, height))
    }

    /// # Safety
    ///
    /// `connection` must be a connected producer.
    pub unsafe fn stream_producer_disconnect(connection: *mut c_void) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_egl_stream_producer_disconnect()?;
        check(cu(connection))
    }

    /// Push a frame to the EGL stream.
    ///
    /// # Safety
    ///
    /// `egl_frame` must be populated with live plane pointers pointing at
    /// memory the producer owns until the consumer returns it via
    /// [`stream_producer_return_frame`].
    pub unsafe fn stream_producer_present_frame(
        connection: *mut c_void,
        egl_frame: CUeglFrame,
        cu_stream_inout: *mut baracuda_cuda_sys::CUstream,
    ) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_egl_stream_producer_present_frame()?;
        check(cu(connection, egl_frame, cu_stream_inout))
    }

    /// Reclaim a previously-presented frame. `egl_frame` is overwritten
    /// with the plane descriptors of the returned frame.
    ///
    /// # Safety
    ///
    /// `connection` must be a connected producer.
    pub unsafe fn stream_producer_return_frame(
        connection: *mut c_void,
        egl_frame: *mut CUeglFrame,
        cu_stream_inout: *mut baracuda_cuda_sys::CUstream,
    ) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_egl_stream_producer_return_frame()?;
        check(cu(connection, egl_frame, cu_stream_inout))
    }
}

/// NvSci interop (Jetson / DRIVE platforms). Query attributes for
/// `NvSciSyncObj`s from a CUDA device, then feed the resulting objects
/// into [`crate::external::ExternalSemaphore::import`] with the
/// `NVSCISYNC` handle type.
pub mod nvsci {
    use super::*;
    use crate::device::Device;
    use baracuda_cuda_sys::types::NvSciSyncAttrList;

    /// `CUnvSciSyncAttr::SIGNAL` — see [`baracuda_cuda_sys::types::CUnvSciSyncAttr`].
    pub const SIGNAL: i32 = baracuda_cuda_sys::types::CUnvSciSyncAttr::SIGNAL;
    /// `CUnvSciSyncAttr::WAIT`.
    pub const WAIT: i32 = baracuda_cuda_sys::types::CUnvSciSyncAttr::WAIT;

    /// Fill `attr_list` with the NvSciSync attributes CUDA requires for
    /// `device` to participate as a signaler or waiter.
    ///
    /// # Safety
    ///
    /// `attr_list` must be a live `NvSciSyncAttrList` created via
    /// libnvsciSync's `NvSciSyncAttrListCreate`.
    pub unsafe fn device_sync_attributes(
        attr_list: NvSciSyncAttrList,
        device: &Device,
        direction: i32,
    ) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_device_get_nv_sci_sync_attributes()?;
        check(cu(attr_list, device.as_raw(), direction))
    }
}

/// Direct3D 11 interop (Windows) — the D3D path most still relevant today.
pub mod d3d11 {
    use super::*;
    use crate::device::Device;
    use baracuda_cuda_sys::types::{ID3DDevice, ID3DResource};

    /// # Safety
    ///
    /// `adapter` must be a valid `IDXGIAdapter*`.
    pub unsafe fn get_device(adapter: ID3DDevice) -> Result<Device> {
        let d = driver()?;
        let cu = d.cu_d3d11_get_device()?;
        let mut dev = baracuda_cuda_sys::CUdevice(0);
        check(cu(&mut dev, adapter))?;
        Ok(Device(dev))
    }

    /// # Safety
    ///
    /// `d3d_device` must be a valid `ID3D11Device*`.
    pub unsafe fn get_devices(d3d_device: ID3DDevice, device_list: u32) -> Result<Vec<Device>> {
        let d = driver()?;
        let cu = d.cu_d3d11_get_devices()?;
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
        let mut raw = vec![baracuda_cuda_sys::CUdevice(0); count as usize];
        check(cu(
            &mut count,
            raw.as_mut_ptr(),
            raw.len() as core::ffi::c_uint,
            d3d_device,
            device_list,
        ))?;
        Ok(raw.into_iter().map(Device).collect())
    }

    /// # Safety
    ///
    /// `resource` must be a live `ID3D11Resource*`.
    pub unsafe fn register_resource(
        context: &Context,
        resource: ID3DResource,
        flags: u32,
    ) -> Result<GraphicsResource> {
        context.set_current()?;
        let d = driver()?;
        let cu = d.cu_graphics_d3d11_register_resource()?;
        let mut res: CUgraphicsResource = core::ptr::null_mut();
        check(cu(&mut res, resource, flags))?;
        Ok(GraphicsResource::from_raw(context, res))
    }
}
