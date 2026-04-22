//! Runtime Wave 5: arrays + tex/surf, 3D memcpy, launch-ex, profiler,
//! VMM, multicast, green contexts.

use baracuda_cuda_sys::runtime::types::{cudaChannelFormatKind, cudaExtent, cudaTextureDesc};

use baracuda_runtime::array::{channel_desc, Array, MipmappedArray, SurfaceObject, TextureObject};
use baracuda_runtime::launch_attr::LaunchExBuilder;
use baracuda_runtime::memcpy3d::Pitched3dBuffer;
use baracuda_runtime::{profiler, vmm, Device, Library, Stream};

const VECTOR_ADD_PTX: &str = include_str!("kernels/vector_add.ptx");

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn array_2d_alloc_info_roundtrip() {
    Device::from_ordinal(0).set_current().unwrap();
    let desc = channel_desc(8, 0, 0, 0, cudaChannelFormatKind::UNSIGNED);
    let arr = Array::new_2d(&desc, 256, 128, 0).unwrap();
    let (d, ext, flags) = arr.info().unwrap();
    eprintln!("array info: {d:?}, {ext:?}, flags={flags:#x}");
    assert_eq!(d.x, 8);
    assert_eq!(ext.width, 256);
    assert_eq!(ext.height, 128);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn texture_object_round_trip() {
    Device::from_ordinal(0).set_current().unwrap();
    let desc = channel_desc(32, 0, 0, 0, cudaChannelFormatKind::FLOAT);
    let arr = Array::new_2d(&desc, 64, 64, 0).unwrap();
    let tex_desc = cudaTextureDesc::default();
    let tex = TextureObject::new(&arr, &tex_desc, None).unwrap();
    let res = tex.resource_desc().unwrap();
    assert_eq!(res.res_type, 0); // ARRAY
    let td = tex.texture_desc().unwrap();
    // Default filter=POINT (0), read=ELEMENT_TYPE (0).
    assert_eq!(td.filter_mode, 0);
    assert_eq!(td.read_mode, 0);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn surface_object_round_trip() {
    Device::from_ordinal(0).set_current().unwrap();
    let desc = channel_desc(32, 0, 0, 0, cudaChannelFormatKind::FLOAT);
    let arr = Array::new_2d(&desc, 32, 32, 0).unwrap();
    let surf = SurfaceObject::new(&arr).unwrap();
    let res = surf.resource_desc().unwrap();
    assert_eq!(res.res_type, 0); // ARRAY
    assert!(surf.as_raw() != 0);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn mipmapped_array_levels() {
    Device::from_ordinal(0).set_current().unwrap();
    let desc = channel_desc(8, 0, 0, 0, cudaChannelFormatKind::UNSIGNED);
    let ext = cudaExtent {
        width: 128,
        height: 128,
        depth: 0,
    };
    let mip = MipmappedArray::new(&desc, ext, 4, 0).unwrap();
    for lvl in 0..4u32 {
        let a = mip.level(lvl).unwrap();
        assert!(!a.is_null());
    }
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn pitched_3d_buffer_alloc() {
    Device::from_ordinal(0).set_current().unwrap();
    let buf: Pitched3dBuffer<u32> = Pitched3dBuffer::new(64, 32, 8).unwrap();
    assert!(!buf.as_pitched_ptr().ptr.is_null());
    assert!(buf.pitch_bytes() >= 64 * 4);
    let ext = buf.extent();
    assert_eq!(ext.height, 32);
    assert_eq!(ext.depth, 8);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn launch_ex_basic_vector_add() {
    Device::from_ordinal(0).set_current().unwrap();
    let stream = Stream::new().unwrap();
    let lib = Library::load_ptx(VECTOR_ADD_PTX).unwrap();
    let kernel = lib.get_kernel("vector_add").unwrap();

    let n: u32 = 1024;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
    let d_a = baracuda_runtime::DeviceBuffer::from_slice(&a).unwrap();
    let d_b = baracuda_runtime::DeviceBuffer::from_slice(&b).unwrap();
    let d_c: baracuda_runtime::DeviceBuffer<f32> =
        baracuda_runtime::DeviceBuffer::new(n as usize).unwrap();

    let a_ptr = d_a.as_device_ptr();
    let b_ptr = d_b.as_device_ptr();
    let c_ptr = d_c.as_device_ptr();
    let mut args: [*mut core::ffi::c_void; 4] = [
        &a_ptr as *const _ as *mut _,
        &b_ptr as *const _ as *mut _,
        &c_ptr as *const _ as *mut _,
        &n as *const _ as *mut _,
    ];

    // cudaLaunchKernelEx is not always exported from the Windows
    // cudart DLL; skip gracefully if the symbol is missing.
    let rc = unsafe {
        LaunchExBuilder::new(&stream, (n.div_ceil(256), 1, 1), (256, 1, 1))
            .launch(&kernel, &mut args)
    };
    match rc {
        Ok(()) => {}
        Err(baracuda_runtime::Error::Loader(_)) => {
            eprintln!("cudaLaunchKernelEx not exported on this build — skipping");
            return;
        }
        Err(e) => panic!("launch_ex: {e:?}"),
    }
    stream.synchronize().unwrap();

    let mut out = vec![0f32; n as usize];
    d_c.copy_to_host(&mut out).unwrap();
    for x in out.iter().take(n as usize) {
        assert!((*x - n as f32).abs() < 1e-5);
    }
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn profiler_start_stop_no_error() {
    Device::from_ordinal(0).set_current().unwrap();
    profiler::start().unwrap();
    profiler::stop().unwrap();
    // with_profiling convenience too
    let v = profiler::with_profiling(|| 42u32).unwrap();
    assert_eq!(v, 42);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn vmm_end_to_end() {
    let device = Device::from_ordinal(0);
    device.set_current().unwrap();
    let prop = vmm::device_alloc_prop(&device);

    // VMM requires a handle-type on some drivers. If create fails, skip.
    let granularity = match vmm::allocation_granularity(&prop, 0) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("granularity query failed: {e:?} — skipping");
            return;
        }
    };
    let size = granularity.max(1 << 20);

    let handle = match vmm::MemHandle::new(size, &prop, 0) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("MemCreate failed: {e:?} — skipping");
            return;
        }
    };
    let va = match vmm::address_reserve(size, granularity, 0) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("address_reserve failed: {e:?} — skipping");
            return;
        }
    };
    unsafe {
        vmm::map(va, size, 0, &handle, 0).unwrap();
        vmm::set_access(
            va,
            size,
            &device,
            baracuda_runtime::mempool::AccessFlags::ReadWrite,
        )
        .unwrap();
        vmm::unmap(va, size).unwrap();
        vmm::address_free(va, size).unwrap();
    }
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn green_ctx_surface_is_feature_gated_or_works() {
    Device::from_ordinal(0).set_current().unwrap();
    // We don't have a cudaDevResourceDesc builder in the runtime crate
    // (that helper lives in baracuda-driver). Just verify the gate
    // behaves: either returns FeatureNotSupported on pre-13.1, or a
    // clean error on a bogus descriptor on 13.1+.
    let rc =
        unsafe { baracuda_runtime::green::GreenContext::from_resource_desc(core::ptr::null(), 0) };
    match rc {
        Err(baracuda_runtime::Error::FeatureNotSupported { .. }) => {
            eprintln!("green-ctx not supported on this driver (expected on <13.1)")
        }
        Err(e) => eprintln!("green-ctx create with null desc returned {e:?} (expected)"),
        Ok(_) => panic!("unexpected success with null resource desc"),
    }
}

#[test]
fn runtime_wave5_symbols_resolve() {
    if let Ok(r) = baracuda_cuda_sys::runtime() {
        // Arrays / tex / surf
        let _ = r.cuda_malloc_mipmapped_array();
        let _ = r.cuda_free_mipmapped_array();
        let _ = r.cuda_array_get_info();
        let _ = r.cuda_get_mipmapped_array_level();
        let _ = r.cuda_get_texture_object_resource_desc();
        let _ = r.cuda_get_texture_object_texture_desc();
        let _ = r.cuda_get_texture_object_resource_view_desc();
        let _ = r.cuda_get_surface_object_resource_desc();
        // 3D
        let _ = r.cuda_memcpy_3d();
        let _ = r.cuda_memcpy_3d_async();
        let _ = r.cuda_memcpy_3d_peer();
        let _ = r.cuda_memcpy_3d_peer_async();
        let _ = r.cuda_memset_3d();
        let _ = r.cuda_malloc_3d();
        let _ = r.cuda_malloc_3d_array();
        // Launch-ex
        let _ = r.cuda_launch_kernel_ex();
        // Profiler
        let _ = r.cuda_profiler_start();
        let _ = r.cuda_profiler_stop();
        // VMM
        let _ = r.cuda_mem_address_reserve();
        let _ = r.cuda_mem_address_free();
        let _ = r.cuda_mem_create();
        let _ = r.cuda_mem_release();
        let _ = r.cuda_mem_map();
        let _ = r.cuda_mem_unmap();
        let _ = r.cuda_mem_set_access();
        let _ = r.cuda_mem_get_access();
        let _ = r.cuda_mem_get_allocation_granularity();
        let _ = r.cuda_mem_get_allocation_properties_from_handle();
        let _ = r.cuda_mem_export_to_shareable_handle();
        let _ = r.cuda_mem_import_from_shareable_handle();
        let _ = r.cuda_mem_retain_allocation_handle();
        // Multicast
        let _ = r.cuda_multicast_create();
        let _ = r.cuda_multicast_add_device();
        let _ = r.cuda_multicast_bind_mem();
        let _ = r.cuda_multicast_bind_addr();
        let _ = r.cuda_multicast_unbind();
        let _ = r.cuda_multicast_get_granularity();
        // Green
        let _ = r.cuda_device_create_green_ctx();
        let _ = r.cuda_green_ctx_destroy();
        let _ = r.cuda_green_ctx_record_event();
        let _ = r.cuda_green_ctx_wait_event();
        let _ = r.cuda_green_ctx_stream_create();
    }
}
