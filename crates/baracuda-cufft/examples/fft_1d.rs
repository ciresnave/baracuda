//! Minimal 1-D cuFFT R2C → C2R round-trip demo.
//!
//! Generates a small sine-wave-style real signal, runs a forward
//! real-to-complex FFT, prints the dominant bin, then inverse-FFTs back
//! to real and verifies the round-trip reproduces the input (modulo the
//! cuFFT C2R 1/N normalization the caller must apply).
//!
//! Run with:
//!
//! ```text
//! cargo run --example fft_1d -p baracuda-cufft
//! ```

use baracuda_cufft::{Plan1d, Transform};
use baracuda_driver::{Context, Device, DeviceBuffer};
use baracuda_types::Complex32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    baracuda_driver::init()?;
    let device = Device::get(0)?;
    println!("device: {}", device.name()?);
    let ctx = Context::new(&device)?;

    // Pure cosine at bin 4 over 64 samples (real-valued).
    let n: usize = 64;
    let host_in: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * 4.0 * 2.0 * std::f32::consts::PI / n as f32).cos())
        .collect();
    println!(
        "input (first 8): {:?}",
        &host_in.iter().take(8).copied().collect::<Vec<_>>()
    );

    let mut d_in = DeviceBuffer::from_slice(&ctx, &host_in)?;
    // R2C: output is the Hermitian half — `n/2 + 1` complex bins.
    let n_out_complex = n / 2 + 1;
    let mut d_freq: DeviceBuffer<Complex32> = DeviceBuffer::new(&ctx, n_out_complex)?;

    let fwd = Plan1d::new(n as i32, Transform::R2C, 1)?;
    fwd.exec_r2c(&mut d_in, &mut d_freq)?;

    // Pull the spectrum back and find the dominant bin.
    let mut freq_host = vec![Complex32 { re: 0.0, im: 0.0 }; n_out_complex];
    d_freq.copy_to_host(&mut freq_host)?;
    let mut best = (0usize, 0.0f32);
    for (i, c) in freq_host.iter().enumerate() {
        let mag = (c.re * c.re + c.im * c.im).sqrt();
        if mag > best.1 {
            best = (i, mag);
        }
    }
    println!("dominant bin: {} (magnitude {:.3})", best.0, best.1);
    assert_eq!(best.0, 4, "expected dominant frequency at bin 4");

    // Inverse FFT: C2R reconstructs the real signal scaled by N.
    let mut d_round: DeviceBuffer<f32> = DeviceBuffer::new(&ctx, n)?;
    let inv = Plan1d::new(n as i32, Transform::C2R, 1)?;
    inv.exec_c2r(&mut d_freq, &mut d_round)?;

    let mut round_host = vec![0.0f32; n];
    d_round.copy_to_host(&mut round_host)?;

    let mut max_err = 0.0f32;
    for (orig, got) in host_in.iter().zip(&round_host) {
        // Apply the missing 1/N normalization on the host side.
        let normalized = got / n as f32;
        max_err = max_err.max((orig - normalized).abs());
    }
    println!("round-trip max abs error vs original: {max_err}");
    assert!(max_err < 1e-4, "round-trip error too large: {max_err}");
    println!("OK");
    Ok(())
}
