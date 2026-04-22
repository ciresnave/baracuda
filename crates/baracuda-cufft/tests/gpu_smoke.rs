//! GPU-gated integration tests for cuFFT.

use baracuda_cufft::{Direction, Plan1d, Transform};
use baracuda_driver::{Context, Device, DeviceBuffer};
use baracuda_types::Complex32;

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn c2c_forward_then_inverse_roundtrip() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();

    let n: i32 = 1024;
    let input: Vec<Complex32> = (0..n as usize)
        .map(|i| Complex32::new((i as f32 * 0.1).sin(), (i as f32 * 0.1).cos()))
        .collect();

    let mut d_in = DeviceBuffer::from_slice(&ctx, &input).unwrap();
    let mut d_freq: DeviceBuffer<Complex32> = DeviceBuffer::new(&ctx, n as usize).unwrap();
    let mut d_out: DeviceBuffer<Complex32> = DeviceBuffer::new(&ctx, n as usize).unwrap();

    let plan = Plan1d::new(n, Transform::C2C, 1).unwrap();
    plan.exec_c2c(&mut d_in, &mut d_freq, Direction::Forward)
        .unwrap();
    plan.exec_c2c(&mut d_freq, &mut d_out, Direction::Inverse)
        .unwrap();

    let mut out_host = vec![Complex32::default(); n as usize];
    d_out.copy_to_host(&mut out_host).unwrap();

    // cuFFT does NOT normalize the inverse — result is N * original.
    let mut max_err = 0.0f32;
    for (orig, recov) in input.iter().zip(&out_host) {
        let recov_re = recov.re / n as f32;
        let recov_im = recov.im / n as f32;
        max_err = max_err.max((recov_re - orig.re).abs());
        max_err = max_err.max((recov_im - orig.im).abs());
    }
    assert!(max_err < 1e-3, "FFT roundtrip error too large: {max_err}");
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn version_is_queryable() {
    let _ = baracuda_cufft::version().expect("cufftGetVersion");
}
