//! `forge_hello` — the canonical baracuda-forge end-to-end story.
//!
//! 1. `kernels/forge_hello.cu` defines `vector_add`.
//! 2. `examples/build.rs` invokes [`baracuda_forge::KernelBuilder`] to
//!    compile that `.cu` into PTX under `OUT_DIR` at build time.
//! 3. This binary loads the PTX through [`baracuda_driver::Module::load_ptx`]
//!    and launches the kernel against a device buffer pair.
//! 4. We copy the result back, compare against a CPU reference, and exit
//!    with a clear pass / fail signal.
//!
//! Run with:
//!
//! ```text
//! cargo run -p baracuda-examples --bin forge_hello --features forge-hello --release
//! ```
//!
//! Requires a working CUDA toolkit (for `nvcc` at build time) and an NVIDIA
//! GPU + driver (for kernel execution at runtime).

use baracuda_driver::{init, Context, Device, DeviceBuffer, Module};

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/forge_hello.ptx"));

const N: usize = 1 << 20; // 1Mi elements
const BLOCK: u32 = 256;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init()?;

    let device = Device::get(0)?;
    println!("forge_hello: device 0 = {}", device.name()?);

    let ctx = Context::new(&device)?;
    let module = Module::load_ptx(&ctx, PTX)?;
    let kernel = module.get_function("vector_add")?;

    let host_a: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let host_b: Vec<f32> = (0..N).map(|i| (2 * i) as f32).collect();

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a)?;
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b)?;
    let dev_out: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, N)?;

    let n_i32 = N as i32;
    let grid = ((N as u32).div_ceil(BLOCK), 1, 1);
    let block = (BLOCK, 1, 1);

    // SAFETY: vector_add takes (const float*, const float*, float*, int) —
    // we pass three live device buffers (lifetime tied to `&dev_*` borrows
    // that outlive launch+sync) plus an i32, in declaration order. The
    // grid covers all N elements with BLOCK threads each and the kernel
    // bounds-checks `i < n`.
    unsafe {
        kernel
            .launch()
            .grid(grid)
            .block(block)
            .arg(&dev_a)
            .arg(&dev_b)
            .arg(&dev_out)
            .arg(&n_i32)
            .launch()?;
    }

    let mut host_out = vec![0.0f32; N];
    dev_out.copy_to_host(&mut host_out)?;

    let mut max_err: f32 = 0.0;
    for i in 0..N {
        let expected = host_a[i] + host_b[i];
        let err = (host_out[i] - expected).abs();
        if err > max_err {
            max_err = err;
        }
    }

    if max_err == 0.0 {
        println!(
            "forge_hello: ✅ pass — {N} elements, exact match against CPU reference"
        );
        Ok(())
    } else {
        Err(format!(
            "forge_hello: ❌ fail — max abs error {max_err} across {N} elements"
        )
        .into())
    }
}
