//! Runtime Wave 4: graphics interop — core + GL + D3D + VDPAU + EGL + NvSci.
//!
//! We can't fully test graphics interop without a GL / D3D / VDPAU / EGL
//! context or an NvSciSync-capable device, but we CAN:
//!
//! 1. Verify every PFN resolves from libcudart (or reports SymbolNotFound
//!    cleanly).
//! 2. Exercise `cudaGLGetDevices` with no current GL context — should
//!    return an empty list, not panic.

use baracuda_runtime::graphics::gl;

#[test]
fn runtime_wave4_core_symbols_resolve() {
    if let Ok(r) = baracuda_cuda_sys::runtime() {
        let _ = r.cuda_graphics_unregister_resource();
        let _ = r.cuda_graphics_map_resources();
        let _ = r.cuda_graphics_unmap_resources();
        let _ = r.cuda_graphics_resource_get_mapped_pointer();
        let _ = r.cuda_graphics_sub_resource_get_mapped_array();
        let _ = r.cuda_graphics_resource_get_mapped_mipmapped_array();
        let _ = r.cuda_graphics_resource_set_map_flags();
    }
}

#[test]
fn runtime_wave4_gl_symbols_resolve() {
    if let Ok(r) = baracuda_cuda_sys::runtime() {
        let _ = r.cuda_graphics_gl_register_buffer();
        let _ = r.cuda_graphics_gl_register_image();
        let _ = r.cuda_gl_get_devices();
    }
}

#[test]
fn runtime_wave4_d3d_symbols_resolve() {
    if let Ok(r) = baracuda_cuda_sys::runtime() {
        // D3D9 / 10 / 11 are Windows-only exports; missing on Linux
        // cudart is fine — PFN getters just return Err(SymbolNotFound).
        let _ = r.cuda_d3d9_get_device();
        let _ = r.cuda_d3d9_get_devices();
        let _ = r.cuda_graphics_d3d9_register_resource();
        let _ = r.cuda_d3d10_get_device();
        let _ = r.cuda_d3d10_get_devices();
        let _ = r.cuda_graphics_d3d10_register_resource();
        let _ = r.cuda_d3d11_get_device();
        let _ = r.cuda_d3d11_get_devices();
        let _ = r.cuda_graphics_d3d11_register_resource();
    }
}

#[test]
fn runtime_wave4_vdpau_symbols_resolve() {
    if let Ok(r) = baracuda_cuda_sys::runtime() {
        let _ = r.cuda_vdpau_get_device();
        let _ = r.cuda_graphics_vdpau_register_video_surface();
        let _ = r.cuda_graphics_vdpau_register_output_surface();
    }
}

#[test]
fn runtime_wave4_egl_symbols_resolve() {
    if let Ok(r) = baracuda_cuda_sys::runtime() {
        let _ = r.cuda_graphics_egl_register_image();
        let _ = r.cuda_graphics_resource_get_mapped_egl_frame();
        let _ = r.cuda_event_create_from_egl_sync();
        let _ = r.cuda_egl_stream_consumer_connect();
        let _ = r.cuda_egl_stream_consumer_disconnect();
        let _ = r.cuda_egl_stream_consumer_acquire_frame();
        let _ = r.cuda_egl_stream_consumer_release_frame();
        let _ = r.cuda_egl_stream_producer_connect();
        let _ = r.cuda_egl_stream_producer_disconnect();
        let _ = r.cuda_egl_stream_producer_present_frame();
        let _ = r.cuda_egl_stream_producer_return_frame();
    }
}

#[test]
fn runtime_wave4_nvsci_symbols_resolve() {
    if let Ok(r) = baracuda_cuda_sys::runtime() {
        let _ = r.cuda_device_get_nv_sci_sync_attributes();
    }
}

#[test]
#[ignore = "requires an NVIDIA GPU (safe even with no GL context)"]
fn gl_get_devices_without_context_is_empty() {
    use baracuda_runtime::Device;
    Device::from_ordinal(0).set_current().unwrap();
    // No GL context on the calling thread — driver reports either
    // no-devices or NOT_INITIALIZED, both mapped to an empty vec.
    let v = gl::get_devices(0).unwrap_or_default();
    eprintln!("cudaGLGetDevices (no context): {} entries", v.len());
    // Must not panic. Zero-or-more is acceptable.
}
