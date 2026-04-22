//! Integration test for Wave-31: VDPAU + EGL + NvSci (Jetson).
//!
//! On the test host (Windows 11 + RTX 4070 Laptop) none of these APIs
//! are usable for end-to-end testing — VDPAU is Linux-only, EGL is
//! primarily Jetson, NvSci is Jetson/DRIVE-only. The symbols may resolve
//! (the driver still ships the stubs on Windows) but calls will return
//! NOT_SUPPORTED or NOT_INITIALIZED.
//!
//! The point of this test is to verify the FFI surface links and
//! baracuda's safe wrappers are reachable.

#[test]
fn jetson_and_vdpau_symbols_resolve() {
    if let Ok(d) = baracuda_cuda_sys::driver() {
        let _ = d.cu_vdpau_get_device();
        let _ = d.cu_vdpau_ctx_create();
        let _ = d.cu_graphics_vdpau_register_video_surface();
        let _ = d.cu_graphics_vdpau_register_output_surface();

        let _ = d.cu_graphics_egl_register_image();
        let _ = d.cu_graphics_resource_get_mapped_egl_frame();
        let _ = d.cu_event_create_from_egl_sync();
        let _ = d.cu_egl_stream_consumer_connect();
        let _ = d.cu_egl_stream_consumer_disconnect();
        let _ = d.cu_egl_stream_consumer_acquire_frame();
        let _ = d.cu_egl_stream_consumer_release_frame();
        let _ = d.cu_egl_stream_producer_connect();
        let _ = d.cu_egl_stream_producer_disconnect();
        let _ = d.cu_egl_stream_producer_present_frame();
        let _ = d.cu_egl_stream_producer_return_frame();

        let _ = d.cu_device_get_nv_sci_sync_attributes();
    }
}

#[test]
fn egl_frame_size_matches_cuda_abi() {
    // CUeglFrame is an 80-byte struct in cuda.h. Our opaque wrapper
    // reserves 10×u64 = 80 bytes.
    assert_eq!(
        core::mem::size_of::<baracuda_cuda_sys::types::CUeglFrame>(),
        80,
        "CUeglFrame size drift"
    );
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn nvsci_attrs_fail_gracefully_without_jetson() {
    baracuda_driver::init().unwrap();
    let device = baracuda_driver::Device::get(0).unwrap();
    // Pass a null NvSciSyncAttrList — the driver should reject cleanly.
    let rc = unsafe {
        baracuda_driver::graphics::nvsci::device_sync_attributes(
            core::ptr::null_mut(),
            &device,
            baracuda_driver::graphics::nvsci::SIGNAL,
        )
    };
    match rc {
        Ok(()) => eprintln!("nvsci attrs accepted (Jetson-like host?)"),
        Err(e) => eprintln!("nvsci attrs rejected (expected on desktop): {e:?}"),
    }
}
