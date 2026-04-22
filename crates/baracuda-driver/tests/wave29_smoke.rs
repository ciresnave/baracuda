//! Integration test for Wave-29: graphics interop core + OpenGL.
//!
//! Without a live GL context we can't exercise full register / map /
//! unmap. We at least verify the symbols resolve and the "list GL
//! devices" call handles "no GL context" cleanly.

use baracuda_driver::graphics::gl;
use baracuda_driver::graphics::GLDeviceList;

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn gl_get_devices_without_gl_context_is_graceful() {
    baracuda_driver::init().unwrap();

    // No GL context current on this thread -> the driver typically
    // returns NOT_INITIALIZED; our wrapper maps that to an empty list.
    match gl::get_devices(GLDeviceList::ALL) {
        Ok(list) => eprintln!("cuGLGetDevices returned {} device(s)", list.len()),
        Err(e) => eprintln!("cuGLGetDevices (no GL ctx) returned: {e:?}"),
    }
}

#[test]
fn graphics_symbols_resolve() {
    // Pure host-side: confirm the sys-layer PFNs exist and resolve
    // without panicking (we don't need a GPU or GL for this).
    if let Ok(d) = baracuda_cuda_sys::driver() {
        // Don't unwrap — on a machine without libcuda the loader returns
        // Err, which is fine. We're only checking the no-panic path.
        let _ = d.cu_graphics_unregister_resource();
        let _ = d.cu_graphics_map_resources();
        let _ = d.cu_graphics_unmap_resources();
        let _ = d.cu_graphics_resource_get_mapped_pointer();
        let _ = d.cu_graphics_sub_resource_get_mapped_array();
        let _ = d.cu_graphics_gl_register_buffer();
        let _ = d.cu_graphics_gl_register_image();
        let _ = d.cu_gl_get_devices();
    }
}
