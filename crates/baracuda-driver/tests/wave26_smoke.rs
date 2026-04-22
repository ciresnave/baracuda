//! GPU-gated integration test for Wave-26 Driver-API additions:
//! typed launch-attribute builders applied via `cuLaunchKernelEx`.

use core::ffi::c_void;

use baracuda_driver::launch_attr::{self, LaunchAttr};
use baracuda_driver::{Context, Device, DeviceBuffer, Module, Stream};

const VECTOR_ADD_PTX: &str = include_str!("kernels/vector_add.ptx");

#[test]
#[ignore = "requires an NVIDIA GPU + CUDA 12.0+"]
fn launch_ex_with_typed_priority_attribute() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let stream = Stream::new(&ctx).unwrap();

    let n: u32 = 1024;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 2.5).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

    let d_a = DeviceBuffer::from_slice(&ctx, &a).unwrap();
    let d_b = DeviceBuffer::from_slice(&ctx, &b).unwrap();
    let d_c: DeviceBuffer<f32> = DeviceBuffer::new(&ctx, n as usize).unwrap();

    let module = Module::load_ptx(&ctx, VECTOR_ADD_PTX).unwrap();
    let kernel = module.get_function("vector_add").unwrap();

    // Use the least-priority value we're guaranteed (0 = default).
    let attrs_typed = [LaunchAttr::priority(0)];
    let mut attrs_raw = launch_attr::into_raw_vec(&attrs_typed);

    let a_ptr = d_a.as_raw();
    let b_ptr = d_b.as_raw();
    let c_ptr = d_c.as_raw();
    // SAFETY: kernel signature matches our arg types/order.
    unsafe {
        kernel
            .launch()
            .grid((n.div_ceil(256), 1, 1))
            .block((256u32, 1, 1))
            .stream(&stream)
            .arg(&a_ptr)
            .arg(&b_ptr)
            .arg(&c_ptr)
            .arg(&n)
            .launch_ex(&mut attrs_raw)
            .unwrap();
    }
    stream.synchronize().unwrap();

    let mut got = vec![0.0f32; n as usize];
    d_c.copy_to_host(&mut got).unwrap();
    assert_eq!(expected, got);
}

#[test]
fn launch_attr_builders_set_correct_id() {
    // Pure host-side test: the builders should record the right ID.
    use baracuda_cuda_sys::types::CUlaunchAttributeID;
    assert_eq!(
        LaunchAttr::priority(-3).as_raw().id,
        CUlaunchAttributeID::PRIORITY
    );
    assert_eq!(
        LaunchAttr::cluster_dim(2, 1, 1).as_raw().id,
        CUlaunchAttributeID::CLUSTER_DIMENSION
    );
    assert_eq!(
        LaunchAttr::cooperative().as_raw().id,
        CUlaunchAttributeID::COOPERATIVE
    );
    assert_eq!(
        LaunchAttr::programmatic_stream_serialization(true)
            .as_raw()
            .id,
        CUlaunchAttributeID::PROGRAMMATIC_STREAM_SERIALIZATION
    );
    // Sanity — silence unused warning from the `c_void` import if no
    // other test references it.
    let _: *mut c_void = core::ptr::null_mut();
}
