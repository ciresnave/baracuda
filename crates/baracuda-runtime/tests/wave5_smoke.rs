//! Runtime Wave 5: arrays + tex/surf, 3D memcpy, launch-ex, profiler.
//!
//! VMM, multicast and green contexts are driver-API only (libcudart does not export them);
//! they live in `baracuda-driver` (ROADMAP SA-45).

#[cfg(feature = "device-tests")]
use baracuda_cuda_sys::runtime::types::{cudaChannelFormatKind, cudaExtent, cudaTextureDesc};

#[cfg(feature = "device-tests")]
use baracuda_runtime::array::{Array, MipmappedArray, SurfaceObject, TextureObject, channel_desc};
#[cfg(feature = "device-tests")]
use baracuda_runtime::launch_attr::LaunchExBuilder;
#[cfg(feature = "device-tests")]
use baracuda_runtime::memcpy3d::Pitched3dBuffer;
#[cfg(feature = "device-tests")]
use baracuda_runtime::{Device, Library, Stream, profiler};

#[cfg(feature = "device-tests")]
const VECTOR_ADD_PTX: &str = include_str!("kernels/vector_add.ptx");

#[test]
#[cfg(feature = "device-tests")]
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
#[cfg(feature = "device-tests")]
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
#[cfg(feature = "device-tests")]
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
#[cfg(feature = "device-tests")]
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
#[cfg(feature = "device-tests")]
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
#[cfg(feature = "device-tests")]
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

    // `cudaLaunchKernelEx` is a header template; the exported symbol is `cudaLaunchKernelExC`,
    // which the loader resolves. Any error is a real failure.
    unsafe {
        LaunchExBuilder::new(&stream, (n.div_ceil(256), 1, 1), (256, 1, 1))
            .launch(&kernel, &mut args)
            .expect("launch_ex via cudaLaunchKernelExC");
    }
    stream.synchronize().unwrap();

    let mut out = vec![0f32; n as usize];
    d_c.copy_to_host(&mut out).unwrap();
    for x in out.iter().take(n as usize) {
        assert!((*x - n as f32).abs() < 1e-5);
    }
}

#[test]
#[cfg(feature = "device-tests")]
fn profiler_start_stop_no_error() {
    Device::from_ordinal(0).set_current().unwrap();
    profiler::start().unwrap();
    profiler::stop().unwrap();
    // with_profiling convenience too
    let v = profiler::with_profiling(|| 42u32).unwrap();
    assert_eq!(v, 42);
}

/// Every Wave-5 runtime entry point must resolve against the installed `libcudart`. (Until alpha.40 this
/// test pinned 25 names that could never resolve; ROADMAP SA-45 removed them, and `cudaLaunchKernelEx`,
/// a header template, became the exported `cudaLaunchKernelExC`.)
/// Without a CUDA runtime on the machine there is nothing to resolve, so that case is
/// the only one that returns early; with `device-tests` it fails instead.
#[test]
fn runtime_wave5_symbols_resolve() {
    let Ok(r) = baracuda_cuda_sys::runtime() else {
        if cfg!(feature = "device-tests") {
            panic!("device-tests build but libcudart did not load");
        }
        return;
    };
    let mut missing: Vec<&str> = Vec::new();
    macro_rules! resolve {
        ($($f:ident),+ $(,)?) => {
            $(if r.$f().is_err() {
                missing.push(stringify!($f));
            })+
        };
    }
    resolve!(
        cuda_malloc_mipmapped_array,
        cuda_free_mipmapped_array,
        cuda_array_get_info,
        cuda_get_mipmapped_array_level,
        cuda_get_texture_object_resource_desc,
        cuda_get_texture_object_texture_desc,
        cuda_get_texture_object_resource_view_desc,
        cuda_get_surface_object_resource_desc,
        cuda_memcpy_3d,
        cuda_memcpy_3d_async,
        cuda_memcpy_3d_peer,
        cuda_memcpy_3d_peer_async,
        cuda_memset_3d,
        cuda_malloc_3d,
        cuda_malloc_3d_array,
        cuda_launch_kernel_ex_c,
        cuda_profiler_start,
        cuda_profiler_stop,
    );
    assert!(
        missing.is_empty(),
        "runtime symbols that did not resolve against libcudart: {missing:?}"
    );
}
