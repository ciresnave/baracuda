//! GPU-gated integration tests for `baracuda-runtime`.
//!
//! Run with `cargo test -p baracuda-runtime -- --ignored` on a machine
//! with an NVIDIA GPU.

use baracuda_runtime::{Device, DeviceBuffer, Library, Stream};

const VECTOR_ADD_PTX: &str = include_str!("kernels/vector_add.ptx");

#[test]
#[ignore = "requires an NVIDIA GPU; run with `cargo test -- --ignored`"]
fn device_enumeration() {
    let count = Device::count().expect("cudaGetDeviceCount");
    assert!(count >= 1, "no CUDA devices detected");
    let device = Device::from_ordinal(0);
    device.set_current().expect("cudaSetDevice");
    let (major, minor) = device.compute_capability().unwrap();
    eprintln!("device 0: cc {major}.{minor}");
    eprintln!(
        "multiprocessors: {}",
        device.multiprocessor_count().unwrap()
    );
    eprintln!("warp size: {}", device.warp_size().unwrap());
}

#[test]
#[ignore = "requires an NVIDIA GPU; run with `cargo test -- --ignored`"]
fn memcpy_roundtrip() {
    Device::from_ordinal(0)
        .set_current()
        .expect("cudaSetDevice");
    let host: Vec<f32> = (0..4096).map(|i| i as f32 * 0.25).collect();
    let device = DeviceBuffer::from_slice(&host).expect("cudaMalloc + cudaMemcpy");
    let mut back = vec![0.0f32; host.len()];
    device.copy_to_host(&mut back).expect("cudaMemcpy D2H");
    assert_eq!(host, back);
}

#[test]
#[ignore = "requires an NVIDIA GPU + CUDA 12.0+; run with `cargo test -- --ignored`"]
fn vector_add_via_library_api() {
    let device = Device::from_ordinal(0);
    device.set_current().unwrap();

    let n: u32 = 1 << 20;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

    let d_a = DeviceBuffer::from_slice(&a).unwrap();
    let d_b = DeviceBuffer::from_slice(&b).unwrap();
    let d_c: DeviceBuffer<f32> = DeviceBuffer::new(n as usize).unwrap();

    let lib = Library::load_ptx(VECTOR_ADD_PTX).expect("cudaLibraryLoadData");
    let kernel = lib.get_kernel("vector_add").expect("cudaLibraryGetKernel");

    let stream = Stream::new().unwrap();
    let block: u32 = 256;
    let grid: u32 = n.div_ceil(block);

    let a_ptr = d_a.as_device_ptr();
    let b_ptr = d_b.as_device_ptr();
    let c_ptr = d_c.as_device_ptr();

    // SAFETY: PTX signature is (const float*, const float*, float*, unsigned int).
    unsafe {
        kernel
            .launch()
            .grid((grid, 1, 1))
            .block((block, 1, 1))
            .stream(&stream)
            .arg(&a_ptr)
            .arg(&b_ptr)
            .arg(&c_ptr)
            .arg(&n)
            .launch()
            .expect("cudaLaunchKernel");
    }
    stream.synchronize().unwrap();

    let mut got = vec![0.0f32; n as usize];
    d_c.copy_to_host(&mut got).unwrap();
    assert_eq!(expected, got);
}

#[test]
#[ignore = "requires an NVIDIA GPU; run with `cargo test -- --ignored`"]
fn runtime_version_visible() {
    let v = baracuda_runtime::runtime_version().expect("cudaRuntimeGetVersion");
    let d = baracuda_runtime::driver_version().expect("cudaDriverGetVersion");
    eprintln!("runtime={v}, driver={d}");
    assert!(v.major() >= 11, "unexpectedly old runtime: {v}");
    assert!(d.major() >= 11, "unexpectedly old driver: {d}");
}
