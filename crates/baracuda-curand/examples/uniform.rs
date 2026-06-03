//! Minimal cuRAND `Uniform(0, 1]` demo.
//!
//! Seeds a Philox 4x32-10 generator, fills a small device buffer with
//! uniform samples, copies them back, prints the first few values, and
//! sanity-checks the mean.
//!
//! Run with:
//!
//! ```text
//! cargo run --example uniform -p baracuda-curand
//! ```

use baracuda_curand::{Generator, RngKind};
use baracuda_driver::{Context, Device, DeviceBuffer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    baracuda_driver::init()?;
    let device = Device::get(0)?;
    println!("device: {}", device.name()?);
    let ctx = Context::new(&device)?;

    let n: usize = 4096;
    let mut buf: DeviceBuffer<f32> = DeviceBuffer::new(&ctx, n)?;

    // Philox is the recommended default for general-purpose RNG.
    let rng = Generator::new(RngKind::Philox4_32_10)?;
    rng.seed(0xDEAD_BEEF)?;
    rng.uniform(&mut buf)?;

    let mut host = vec![0.0f32; n];
    buf.copy_to_host(&mut host)?;

    println!("first 8 samples: {:?}", &host[..8]);

    // cuRAND uniform samples are in (0, 1].
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    for &v in &host {
        assert!(v > 0.0 && v <= 1.0, "sample {v} outside (0, 1]");
        min = min.min(v);
        max = max.max(v);
        sum += v as f64;
    }
    let mean = sum / n as f64;
    println!("min={min:.4}  max={max:.4}  mean={mean:.4} (expected ~0.5)");
    assert!((mean - 0.5).abs() < 0.05, "mean {mean} far from 0.5");
    println!("OK");
    Ok(())
}
