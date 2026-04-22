//! End-to-end example of the baracuda stack: compile CUDA C++ at runtime
//! via NVRTC, load the resulting PTX through the Driver API, launch the
//! kernel, verify output.
//!
//! ```text
//! cargo run --example hello_nvrtc --features "driver"
//! ```

use baracuda::driver::{Context, Device, DeviceBuffer, Module, Stream};
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
    baracuda::driver::init()?;
    let device = Device::get(0)?;
    let (cc_major, cc_minor) = device.compute_capability()?;
    println!("device: {} (cc {cc_major}.{cc_minor})", device.name()?);

    let (major, minor) = baracuda_nvrtc::version()?;
    println!("NVRTC: {major}.{minor}");

    let arch = format!("--gpu-architecture=compute_{cc_major}{cc_minor}");
    let ptx = Program::compile(SRC, "fill.cu", &[&arch])?;
    println!(
        "compiled PTX is {} bytes, {} lines",
        ptx.len(),
        ptx.lines().count()
    );

    let ctx = Context::new(&device)?;
    let stream = Stream::new(&ctx)?;
    let module = Module::load_ptx(&ctx, &ptx)?;
    let kernel = module.get_function("fill_squares")?;

    let n: u32 = 1024;
    let d_out: DeviceBuffer<f32> = DeviceBuffer::new(&ctx, n as usize)?;
    let out_ptr = d_out.as_raw();

    let block: u32 = 256;
    let grid: u32 = n.div_ceil(block);
    // SAFETY: args match the PTX signature.
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
            "mismatch at index {i}: {v} vs {expected}"
        );
    }
    println!("fill_squares verified: first few = {:?}", &host[..6]);
    Ok(())
}
