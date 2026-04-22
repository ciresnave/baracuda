//! GPU-gated integration tests for `baracuda-driver`.
//!
//! These tests are marked `#[ignore]` so `cargo test` skips them by default.
//! Run them on a machine with an NVIDIA GPU via:
//!
//! ```text
//! cargo test -p baracuda-driver -- --ignored
//! ```
//!
//! They exercise the Day-11/12/13/14 happy path: device enumeration,
//! context creation, memory transfer round-trip, module load, and kernel
//! launch.

use baracuda_driver::{Context, Device, DeviceBuffer, Module, Stream};

const VECTOR_ADD_PTX: &str = include_str!("kernels/vector_add.ptx");

fn setup() -> baracuda_driver::Result<(Device, Context, Stream)> {
    baracuda_driver::init()?;
    let device = Device::get(0)?;
    let ctx = Context::new(&device)?;
    let stream = Stream::new(&ctx)?;
    Ok((device, ctx, stream))
}

#[test]
#[ignore = "requires an NVIDIA GPU; run with `cargo test -- --ignored`"]
fn device_enumeration() {
    baracuda_driver::init().expect("cuInit should succeed on a GPU host");
    let count = Device::count().expect("cuDeviceGetCount");
    assert!(count >= 1, "no CUDA devices detected");
    let device = Device::get(0).expect("cuDeviceGet(0)");
    let name = device.name().expect("cuDeviceGetName");
    assert!(!name.is_empty());
    eprintln!("device 0: {name}");
    let (major, minor) = device.compute_capability().unwrap();
    eprintln!("compute capability: {major}.{minor}");
    eprintln!(
        "multiprocessors: {}",
        device.multiprocessor_count().unwrap()
    );
    eprintln!("warp size: {}", device.warp_size().unwrap());
    eprintln!(
        "total memory: {:.2} GiB",
        device.total_memory().unwrap() as f64 / (1024.0 * 1024.0 * 1024.0)
    );
}

#[test]
#[ignore = "requires an NVIDIA GPU; run with `cargo test -- --ignored`"]
fn memcpy_roundtrip() {
    let (_dev, ctx, stream) = setup().expect("setup");
    let host: Vec<f32> = (0..4096).map(|i| i as f32 * 0.25).collect();
    let device = DeviceBuffer::from_slice(&ctx, &host).expect("alloc + H2D");
    let mut back = vec![0.0f32; host.len()];
    device.copy_to_host(&mut back).expect("D2H");
    stream.synchronize().expect("stream sync");
    assert_eq!(host, back);
}

#[test]
#[ignore = "requires an NVIDIA GPU; run with `cargo test -- --ignored`"]
fn vector_add_kernel_launch() {
    let (_dev, ctx, stream) = setup().expect("setup");

    let n: u32 = 1 << 20; // 1,048,576 elements
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

    let d_a = DeviceBuffer::from_slice(&ctx, &a).unwrap();
    let d_b = DeviceBuffer::from_slice(&ctx, &b).unwrap();
    let d_c: DeviceBuffer<f32> = DeviceBuffer::new(&ctx, n as usize).unwrap();

    let module = Module::load_ptx(&ctx, VECTOR_ADD_PTX).expect("cuModuleLoadData");
    let kernel = module
        .get_function("vector_add")
        .expect("cuModuleGetFunction");

    let block: u32 = 256;
    let grid: u32 = n.div_ceil(block);

    let a_ptr = d_a.as_raw();
    let b_ptr = d_b.as_raw();
    let c_ptr = d_c.as_raw();

    // SAFETY: PTX signature is (const float*, const float*, float*, unsigned int)
    // and our arg order matches.
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
            .expect("cuLaunchKernel");
    }

    stream.synchronize().expect("stream sync");

    let mut got = vec![0.0f32; n as usize];
    d_c.copy_to_host(&mut got).expect("D2H");
    assert_eq!(expected, got);
}
