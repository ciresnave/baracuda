//! Minimal end-to-end baracuda driver example: allocate buffers, load PTX,
//! launch `vector_add`, verify.
//!
//! Requires an NVIDIA GPU. Run with:
//!
//! ```text
//! cargo run --example hello_kernel --features driver
//! ```

use baracuda::driver::{Context, Device, DeviceBuffer, Module, Stream};

const PTX: &str = include_str!("../../baracuda-driver/tests/kernels/vector_add.ptx");

fn main() -> baracuda::driver::Result<()> {
    baracuda::driver::init()?;
    let version = baracuda::driver::version()?;
    println!("CUDA driver: {version}");

    let device = Device::get(0)?;
    let (cc_major, cc_minor) = device.compute_capability()?;
    println!(
        "device 0: {} (cc {}.{}, {} SMs)",
        device.name()?,
        cc_major,
        cc_minor,
        device.multiprocessor_count()?,
    );

    let ctx = Context::new(&device)?;
    let stream = Stream::new(&ctx)?;

    let n: u32 = 1 << 16;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();

    let d_a = DeviceBuffer::from_slice(&ctx, &a)?;
    let d_b = DeviceBuffer::from_slice(&ctx, &b)?;
    let d_c: DeviceBuffer<f32> = DeviceBuffer::new(&ctx, n as usize)?;

    let module = Module::load_ptx(&ctx, PTX)?;
    let kernel = module.get_function("vector_add")?;

    let block: u32 = 256;
    let grid: u32 = n.div_ceil(block);

    let a_ptr = d_a.as_raw();
    let b_ptr = d_b.as_raw();
    let c_ptr = d_c.as_raw();

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
        println!("vector_add: {} elements match", n);
    } else {
        eprintln!("vector_add: {mismatches} mismatches out of {n}");
        std::process::exit(1);
    }

    Ok(())
}
