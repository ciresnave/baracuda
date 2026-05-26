//! Real-GPU smoke tests for the Phase 23 cuRAND FFI facade.
//!
//! Each cuRAND-backed plan family gets at least one FFI symbol smoke
//! test that verifies the symbol is callable, samples into a device
//! buffer, and (where applicable) sanity-checks distributional moments.
//!
//! Coverage matches Phase 23 deliverables:
//! - `curand_uniform_f32` — sample (low, high]; mean ≈ (low + high) / 2.
//! - `curand_uniform_f64` — sample (0, 1]; samples all in (0, 1].
//! - `curand_normal_f32` — sample N(mean, stddev); mean ≈ requested.
//! - `curand_normal_f64` — sample N(0, 1); std ≈ 1.
//!
//! Tolerances are wide (3σ × wide) — these are smoke checks, not
//! convergence tests; bias issues should surface as gross failures.
//!
//! `#[ignore]` by default — requires a real CUDA device.

#![allow(unused_mut)]

use core::ffi::c_void;

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels_sys::{
    baracuda_kernels_curand_normal_f32_run, baracuda_kernels_curand_normal_f32_workspace_size,
    baracuda_kernels_curand_normal_f64_run, baracuda_kernels_curand_normal_f64_workspace_size,
    baracuda_kernels_curand_uniform_f32_run, baracuda_kernels_curand_uniform_f32_workspace_size,
    baracuda_kernels_curand_uniform_f64_run, baracuda_kernels_curand_uniform_f64_workspace_size,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn curand_uniform_f32_ffi() {
    let (ctx, stream) = setup();
    let numel: i64 = 1 << 15; // 32768 cells — enough to bound moments tightly.
    let (low, high) = (-1.0f32, 2.0f32);

    let mut dev_y: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, numel as usize).expect("alloc y");

    let mut ws_bytes: usize = 0;
    let status = unsafe {
        baracuda_kernels_curand_uniform_f32_workspace_size(numel, &mut ws_bytes as *mut _)
    };
    assert_eq!(status, 0);
    assert_eq!(ws_bytes, 0);

    let status = unsafe {
        baracuda_kernels_curand_uniform_f32_run(
            numel,
            low,
            high,
            0xDEAD_BEEF_F00D_BAAD_u64,
            dev_y.as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "uniform f32 status");
    stream.synchronize().expect("sync");

    let mut y_host = vec![0f32; numel as usize];
    dev_y.copy_to_host(&mut y_host).expect("dl");

    let mut min_seen = f32::INFINITY;
    let mut max_seen = f32::NEG_INFINITY;
    let mut sum = 0.0;
    for v in &y_host {
        assert!(*v > low - 1e-3, "below low: {}", v);
        assert!(*v <= high + 1e-3, "above high: {}", v);
        min_seen = min_seen.min(*v);
        max_seen = max_seen.max(*v);
        sum += *v;
    }
    let mean = sum / (numel as f32);
    let expected = 0.5 * (low + high);
    // Width 3, n = 32768 → stddev of mean ≈ 3 / sqrt(12 * 32768) ≈ 0.0048.
    assert!(
        (mean - expected).abs() < 0.05,
        "mean {:.4} too far from {:.4}",
        mean,
        expected
    );
    // Sanity: range coverage is most of [low, high].
    assert!(min_seen < low + 0.1, "min {:.4} not near low", min_seen);
    assert!(max_seen > high - 0.1, "max {:.4} not near high", max_seen);
}

#[test]
#[ignore]
fn curand_uniform_f64_ffi() {
    let (ctx, stream) = setup();
    let numel: i64 = 4096;

    let mut dev_y: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, numel as usize).expect("alloc y");

    let mut ws_bytes: usize = 0;
    let status = unsafe {
        baracuda_kernels_curand_uniform_f64_workspace_size(numel, &mut ws_bytes as *mut _)
    };
    assert_eq!(status, 0);

    let status = unsafe {
        baracuda_kernels_curand_uniform_f64_run(
            numel,
            0.0,
            1.0,
            0xCAFE_F00D_F00D_BABA_u64,
            dev_y.as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "uniform f64 status");
    stream.synchronize().expect("sync");

    let mut y_host = vec![0f64; numel as usize];
    dev_y.copy_to_host(&mut y_host).expect("dl");

    for v in &y_host {
        assert!(*v > 0.0 && *v <= 1.0 + 1e-9, "out of (0, 1]: {}", v);
    }
}

#[test]
#[ignore]
fn curand_normal_f32_ffi() {
    let (ctx, stream) = setup();
    let numel: i64 = 1 << 16; // 65536 cells, even (Box-Muller).
    let (mean, stddev) = (1.5f32, 0.25f32);

    let mut dev_y: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, numel as usize).expect("alloc y");

    let mut ws_bytes: usize = 0;
    let status = unsafe {
        baracuda_kernels_curand_normal_f32_workspace_size(numel, &mut ws_bytes as *mut _)
    };
    assert_eq!(status, 0);

    let status = unsafe {
        baracuda_kernels_curand_normal_f32_run(
            numel,
            mean,
            stddev,
            0x12345678_9ABCDEF0_u64,
            dev_y.as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "normal f32 status");
    stream.synchronize().expect("sync");

    let mut y_host = vec![0f32; numel as usize];
    dev_y.copy_to_host(&mut y_host).expect("dl");

    let sum: f64 = y_host.iter().map(|&v| v as f64).sum();
    let sample_mean = sum / (numel as f64);
    // stderr of mean ≈ stddev / sqrt(n) ≈ 0.001; allow generous 0.02.
    assert!(
        (sample_mean - mean as f64).abs() < 0.02,
        "normal f32 mean {:.5} vs expected {:.5}",
        sample_mean,
        mean
    );
    let var: f64 = y_host
        .iter()
        .map(|&v| (v as f64 - sample_mean).powi(2))
        .sum::<f64>()
        / (numel as f64);
    let std = var.sqrt();
    // stddev should match within ~5%.
    assert!(
        (std - stddev as f64).abs() < 0.05,
        "normal f32 std {:.5} vs expected {:.5}",
        std,
        stddev
    );
}

#[test]
#[ignore]
fn curand_normal_f64_ffi() {
    let (ctx, stream) = setup();
    let numel: i64 = 1 << 15;

    let mut dev_y: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, numel as usize).expect("alloc y");

    let mut ws_bytes: usize = 0;
    let status = unsafe {
        baracuda_kernels_curand_normal_f64_workspace_size(numel, &mut ws_bytes as *mut _)
    };
    assert_eq!(status, 0);

    let status = unsafe {
        baracuda_kernels_curand_normal_f64_run(
            numel,
            0.0,
            1.0,
            0xFEED_FACE_DEAD_BEEF_u64,
            dev_y.as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "normal f64 status");
    stream.synchronize().expect("sync");

    let mut y_host = vec![0f64; numel as usize];
    dev_y.copy_to_host(&mut y_host).expect("dl");

    let sum: f64 = y_host.iter().sum();
    let sample_mean = sum / (numel as f64);
    assert!(
        sample_mean.abs() < 0.05,
        "normal f64 mean {:.5} far from 0",
        sample_mean
    );
    let var: f64 = y_host
        .iter()
        .map(|&v| (v - sample_mean).powi(2))
        .sum::<f64>()
        / (numel as f64);
    let std = var.sqrt();
    assert!(
        (std - 1.0).abs() < 0.05,
        "normal f64 std {:.5} far from 1",
        std
    );
}
