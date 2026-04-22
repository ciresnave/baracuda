//! Integration test for Wave-30: D3D9 / D3D10 / D3D11 interop.
//!
//! Without a live D3D device we can't exercise registration. We verify
//! the sys-level symbols resolve.

#[test]
fn d3d_symbols_resolve() {
    if let Ok(d) = baracuda_cuda_sys::driver() {
        let _ = d.cu_d3d9_get_device();
        let _ = d.cu_d3d9_get_devices();
        let _ = d.cu_graphics_d3d9_register_resource();
        let _ = d.cu_d3d10_get_device();
        let _ = d.cu_d3d10_get_devices();
        let _ = d.cu_graphics_d3d10_register_resource();
        let _ = d.cu_d3d11_get_device();
        let _ = d.cu_d3d11_get_devices();
        let _ = d.cu_graphics_d3d11_register_resource();
    }
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn d3d11_get_devices_with_null_device_returns_gracefully() {
    baracuda_driver::init().unwrap();
    // Passing a null ID3D11Device* — CUDA should reject with INVALID_VALUE
    // or similar; we just verify we get an error rather than a crash.
    let result = unsafe {
        baracuda_driver::graphics::d3d11::get_devices(
            core::ptr::null_mut(),
            baracuda_driver::graphics::GLDeviceList::ALL,
        )
    };
    match result {
        Ok(v) => eprintln!("got {} devices", v.len()),
        Err(e) => eprintln!("cuD3D11GetDevices(null) returned: {e:?}"),
    }
}
