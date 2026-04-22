//! GPU-gated integration tests for cuRAND.

use baracuda_curand::{Generator, RngKind};
use baracuda_driver::{Context, Device, DeviceBuffer};

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn uniform_deterministic_by_seed() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();

    let n = 4096;
    let mut d_buf: DeviceBuffer<f32> = DeviceBuffer::new(&ctx, n).unwrap();

    let gen1 = Generator::new(RngKind::Philox4_32_10).unwrap();
    gen1.seed(0xDEAD_BEEF).unwrap();
    gen1.uniform(&mut d_buf).unwrap();
    let mut buf1 = vec![0.0f32; n];
    d_buf.copy_to_host(&mut buf1).unwrap();

    let gen2 = Generator::new(RngKind::Philox4_32_10).unwrap();
    gen2.seed(0xDEAD_BEEF).unwrap();
    gen2.uniform(&mut d_buf).unwrap();
    let mut buf2 = vec![0.0f32; n];
    d_buf.copy_to_host(&mut buf2).unwrap();

    assert_eq!(buf1, buf2, "same seed should produce identical streams");

    // All samples should be in (0, 1].
    for &v in &buf1 {
        assert!(v > 0.0 && v <= 1.0, "out-of-range uniform sample: {v}");
    }
    // Mean should be roughly 0.5.
    let mean: f32 = buf1.iter().sum::<f32>() / buf1.len() as f32;
    assert!(
        (mean - 0.5).abs() < 0.05,
        "uniform mean looks wrong: {mean}"
    );
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn normal_has_expected_statistics() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();

    let n = 8192;
    let mut d_buf: DeviceBuffer<f32> = DeviceBuffer::new(&ctx, n).unwrap();

    let gen = Generator::new(RngKind::Philox4_32_10).unwrap();
    gen.seed(0xCAFE_BABE).unwrap();
    gen.normal(&mut d_buf, 0.0, 1.0).unwrap();
    let mut samples = vec![0.0f32; n];
    d_buf.copy_to_host(&mut samples).unwrap();

    let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
    let variance: f32 =
        samples.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / samples.len() as f32;
    assert!(mean.abs() < 0.1, "normal mean is off: {mean}");
    assert!(
        (variance - 1.0).abs() < 0.15,
        "normal variance is off: {variance}"
    );
}
