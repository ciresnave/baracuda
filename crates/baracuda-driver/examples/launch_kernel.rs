//! End-to-end Driver-API example: compile a tiny CUDA kernel at runtime
//! via NVRTC, load it through the Driver-API `Module`, launch it, copy
//! the result back, and verify.
//!
//! Demonstrates the full safe-wrapper lifecycle:
//! `init → Device → Context → Stream → NVRTC compile → Module::load_ptx
//! → Kernel::launch → DeviceBuffer::copy_to_host`.
//!
//! Run with:
//!
//! ```text
//! cargo run --example launch_kernel -p baracuda-driver
//! ```
//!
//! `baracuda-nvrtc` is a dev-dependency of `baracuda-driver`, so this
//! example is self-contained — no extra features needed.

use baracuda_driver::{Context, Device, DeviceBuffer, Module, Stream};
use baracuda_nvrtc::Program;

const SRC: &str = r#"
extern "C" __global__ void fill_squares(float* out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (float)i * (float)i;
    }
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    baracuda_driver::init()?;
    let device = Device::get(0)?;
    let (cc_major, cc_minor) = device.compute_capability()?;
    println!(
        "device 0: {} (cc {cc_major}.{cc_minor})",
        device.name()?
    );

    // Ask NVRTC to compile for this device's compute capability.
    let arch = format!("--gpu-architecture=compute_{cc_major}{cc_minor}");
    let ptx = Program::compile(SRC, "fill_squares.cu", &[&arch])?;
    println!("compiled PTX: {} bytes", ptx.len());

    let ctx = Context::new(&device)?;
    let stream = Stream::new(&ctx)?;
    let module = Module::load_ptx(&ctx, &ptx)?;
    let kernel = module.get_function("fill_squares")?;

    let n: u32 = 1024;
    let d_out: DeviceBuffer<f32> = DeviceBuffer::new(&ctx, n as usize)?;
    let out_ptr = d_out.as_raw();

    let block: u32 = 256;
    let grid: u32 = n.div_ceil(block);
    // SAFETY: argument types/order match the CUDA kernel signature
    // `(float*, unsigned int)`.
    unsafe {
        kernel
            .launch()
            .grid((grid, 1, 1))
            .block((block, 1, 1))
            .stream(&stream)
            .arg(&out_ptr)
            .arg(&n)
            .launch()?;
    }
    stream.synchronize()?;

    let mut host = vec![0.0f32; n as usize];
    d_out.copy_to_host(&mut host)?;
    for (i, v) in host.iter().enumerate() {
        let expected = (i as f32) * (i as f32);
        assert!(
            (v - expected).abs() < 1e-3,
            "fill_squares[{i}] = {v}, expected {expected}"
        );
    }
    println!("fill_squares verified: first 6 = {:?}", &host[..6]);
    println!("OK");
    Ok(())
}
