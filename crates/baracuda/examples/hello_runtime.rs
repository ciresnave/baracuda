//! Minimal end-to-end baracuda **runtime** example: allocate buffers, load
//! PTX via the CUDA 12+ library API, launch `vector_add`, verify.
//!
//! Mirrors `hello_kernel.rs` but uses `baracuda::runtime` instead of
//! `baracuda::driver`. Requires an NVIDIA GPU and CUDA 12.0+ driver.
//!
//! ```text
//! cargo run --example hello_runtime --features "driver runtime"
//! ```
//!
//! (The `driver` feature is pulled in only so we share the vector_add PTX
//! fixture from `baracuda-driver`'s test directory; the runtime example
//! itself doesn't touch Driver APIs.)

use baracuda::runtime::{Device, DeviceBuffer, Library, Stream};

const PTX: &str = include_str!("../../baracuda-runtime/tests/kernels/vector_add.ptx");

fn main() -> baracuda::runtime::Result<()> {
    let runtime = baracuda::runtime::runtime_version()?;
    let driver = baracuda::runtime::driver_version()?;
    println!("CUDA runtime: {runtime} (driver: {driver})");

    let device = Device::from_ordinal(0);
    device.set_current()?;
    let (major, minor) = device.compute_capability()?;
    println!(
        "device 0: cc {major}.{minor}, {} SMs",
        device.multiprocessor_count()?,
    );

    let n: u32 = 1 << 16;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();

    let d_a = DeviceBuffer::from_slice(&a)?;
    let d_b = DeviceBuffer::from_slice(&b)?;
    let d_c: DeviceBuffer<f32> = DeviceBuffer::new(n as usize)?;

    let lib = Library::load_ptx(PTX)?;
    let kernel = lib.get_kernel("vector_add")?;

    let stream = Stream::new()?;
    let block: u32 = 256;
    let grid: u32 = n.div_ceil(block);

    let a_ptr = d_a.as_device_ptr();
    let b_ptr = d_b.as_device_ptr();
    let c_ptr = d_c.as_device_ptr();

    // SAFETY: arg types/order match the PTX signature.
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
            .launch()?;
    }
    stream.synchronize()?;

    let mut got = vec![0.0f32; n as usize];
    d_c.copy_to_host(&mut got)?;

    let mismatches: usize = a
        .iter()
        .zip(&b)
        .zip(&got)
        .filter(|((x, y), z)| (**x + **y - **z).abs() > 1e-3)
        .count();

    if mismatches == 0 {
        println!("runtime vector_add: {n} elements match");
    } else {
        eprintln!("runtime vector_add: {mismatches} mismatches out of {n}");
        std::process::exit(1);
    }

    Ok(())
}
