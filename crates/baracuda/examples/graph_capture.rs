//! Capture a sequence of kernel launches into a CUDA Graph, then replay
//! it many times. Demonstrates the overhead reduction CUDA Graphs are
//! designed for — the per-launch CPU cost of the replay is much lower
//! than re-launching each kernel by hand.
//!
//! Requires an NVIDIA GPU. Run with:
//!
//! ```text
//! cargo run --example graph_capture --features driver
//! ```

use std::time::Instant;

use baracuda::driver::{CaptureMode, Context, Device, DeviceBuffer, Module, Stream};

const PTX: &str = include_str!("../../baracuda-driver/tests/kernels/vector_add.ptx");

fn main() -> baracuda::driver::Result<()> {
    baracuda::driver::init()?;
    let device = Device::get(0)?;
    println!("device: {}", device.name()?);

    let ctx = Context::new(&device)?;
    let stream = Stream::new(&ctx)?;

    let n: u32 = 1 << 18; // 262,144
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
    let d_a = DeviceBuffer::from_slice(&ctx, &a)?;
    let d_b = DeviceBuffer::from_slice(&ctx, &b)?;
    let d_c: DeviceBuffer<f32> = DeviceBuffer::new(&ctx, n as usize)?;

    let module = Module::load_ptx(&ctx, PTX)?;
    let kernel = module.get_function("vector_add")?;

    let block: u32 = 256;
    let grid: u32 = n.div_ceil(block);

    // Capture a chain of 8 kernel launches into a single graph.
    let chain_len = 8usize;
    let a_ptr = d_a.as_raw();
    let b_ptr = d_b.as_raw();
    let c_ptr = d_c.as_raw();

    let graph = stream.capture(CaptureMode::ThreadLocal, |s| {
        for _ in 0..chain_len {
            // SAFETY: args match the PTX signature.
            unsafe {
                kernel
                    .launch()
                    .grid((grid, 1, 1))
                    .block((block, 1, 1))
                    .stream(s)
                    .arg(&a_ptr)
                    .arg(&b_ptr)
                    .arg(&c_ptr)
                    .arg(&n)
                    .launch()?;
            }
        }
        Ok(())
    })?;
    println!("graph captured with {} nodes", graph.node_count()?);

    let exec = graph.instantiate()?;

    // Timed comparison: N individual launches vs N graph replays.
    let replays = 100;

    let t0 = Instant::now();
    for _ in 0..replays {
        for _ in 0..chain_len {
            // SAFETY: args match the PTX signature.
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
        }
    }
    stream.synchronize()?;
    let dt_manual = t0.elapsed();

    let t1 = Instant::now();
    for _ in 0..replays {
        exec.launch(&stream)?;
    }
    stream.synchronize()?;
    let dt_graph = t1.elapsed();

    println!(
        "manual launches:    {replays} × {chain_len} kernels = {} μs total ({:.2} μs / chain)",
        dt_manual.as_micros(),
        dt_manual.as_micros() as f64 / replays as f64
    );
    println!(
        "graph replays:      {replays} × 1 graph    = {} μs total ({:.2} μs / replay)",
        dt_graph.as_micros(),
        dt_graph.as_micros() as f64 / replays as f64
    );

    // Verify the final output is correct.
    let mut got = vec![0.0f32; n as usize];
    d_c.copy_to_host(&mut got)?;
    let mismatches = a
        .iter()
        .zip(&b)
        .zip(&got)
        .filter(|((x, y), z)| (**x + **y - **z).abs() > 1e-3)
        .count();
    if mismatches == 0 {
        println!("vector_add output verified ({n} elements)");
    } else {
        eprintln!("{mismatches} mismatches detected");
        std::process::exit(1);
    }

    Ok(())
}
