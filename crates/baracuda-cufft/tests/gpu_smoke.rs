//! GPU-gated integration tests for cuFFT.
#![cfg(feature = "device-tests")]

use baracuda_cufft::{Direction, Plan1d, PlanMany, Transform};
use baracuda_driver::{Context, Device, DeviceBuffer};
use baracuda_types::Complex32;

#[test]
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
fn version_is_queryable() {
    let _ = baracuda_cufft::version().expect("cufftGetVersion");
}

/// Advanced data layout: FFT down the columns of a row-major `rows x cols`
/// matrix (`istride = cols`, `idist = 1`) without packing, compared with
/// the same columns copied out and run through a contiguous `Plan1d`.
#[test]
fn strided_c2c_columns_match_contiguous() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();

    let (rows, cols) = (64usize, 5usize);
    let m: Vec<Complex32> = (0..rows * cols)
        .map(|i| Complex32::new((i as f32 * 0.37).sin(), (i as f32 * 0.11).cos()))
        .collect();

    // Strided: one plan, `cols` transforms of length `rows`, element stride `cols`.
    // cuFFT ignores stride/dist when `inembed`/`onembed` is NULL (basic layout),
    // so the embed arrays must be passed for a strided plan.
    let mut n = [rows as i32];
    let mut inembed = [rows as i32];
    let mut onembed = [rows as i32];
    let plan = PlanMany::new(
        1,
        &mut n,
        Some(&mut inembed),
        cols as i32,
        1,
        Some(&mut onembed),
        cols as i32,
        1,
        Transform::C2C,
        cols as i32,
    )
    .unwrap();
    let mut d_in = DeviceBuffer::from_slice(&ctx, &m).unwrap();
    let mut d_out: DeviceBuffer<Complex32> = DeviceBuffer::new(&ctx, rows * cols).unwrap();
    plan.exec_c2c(&mut d_in, &mut d_out, Direction::Forward)
        .unwrap();
    let mut got = vec![Complex32::default(); rows * cols];
    d_out.copy_to_host(&mut got).unwrap();

    // Reference: gather each column, contiguous transform.
    let ref_plan = Plan1d::new(rows as i32, Transform::C2C, 1).unwrap();
    let mut max_err = 0.0f32;
    for c in 0..cols {
        let col: Vec<Complex32> = (0..rows).map(|r| m[r * cols + c]).collect();
        let mut d_c = DeviceBuffer::from_slice(&ctx, &col).unwrap();
        let mut d_f: DeviceBuffer<Complex32> = DeviceBuffer::new(&ctx, rows).unwrap();
        ref_plan
            .exec_c2c(&mut d_c, &mut d_f, Direction::Forward)
            .unwrap();
        let mut want = vec![Complex32::default(); rows];
        d_f.copy_to_host(&mut want).unwrap();
        for r in 0..rows {
            let g = got[r * cols + c];
            max_err = max_err
                .max((g.re - want[r].re).abs())
                .max((g.im - want[r].im).abs());
        }
    }
    assert!(
        max_err < 1e-3,
        "strided vs contiguous FFT mismatch: {max_err}"
    );
}
