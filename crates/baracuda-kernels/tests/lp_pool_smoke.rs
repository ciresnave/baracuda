#![cfg(feature = "cudnn")]
//! Real-GPU smoke tests for `LpPool1dPlan` / `LpPool2dPlan` FW + BW
//! (Phase 16.2 — bespoke fused kernel).
//!
//! Covers the three canonical norm-p settings (p=1 sum-of-abs, p=2
//! L2-norm, p=3 general) plus a 1d sanity check and an f16 path. BW
//! is validated against a finite-difference reference for the most
//! common case (p=2, 2x2 window).
//!
//! All tests are `#[ignore]` by default (require real CUDA device).

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, LpPool1dBackwardPlan, LpPool1dBwArgs, LpPool1dDescriptor,
    LpPool1dFwArgs, LpPool1dPlan, LpPool2dBackwardPlan, LpPool2dBwArgs, LpPool2dDescriptor,
    LpPool2dFwArgs, LpPool2dPlan, PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::f16;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

// =============================================================================
// Host references
// =============================================================================

/// LpPool2d FW reference (no padding, ceil_mode = false). Layout NCHW.
fn host_lp_pool2d_fw(
    n: usize, c: usize, h_in: usize, w_in: usize, x: &[f32],
    kh: usize, kw: usize, sh: usize, sw: usize, p: f32,
) -> (Vec<f32>, usize, usize) {
    let h_out = (h_in - kh) / sh + 1;
    let w_out = (w_in - kw) / sw + 1;
    let mut y = vec![0f32; n * c * h_out * w_out];
    for ni in 0..n {
        for ci in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let h_start = oh * sh;
                    let w_start = ow * sw;
                    let mut acc = 0f64;
                    for kh_ in 0..kh {
                        for kw_ in 0..kw {
                            let ih = h_start + kh_;
                            let iw = w_start + kw_;
                            if ih < h_in && iw < w_in {
                                let v = x[((ni * c + ci) * h_in + ih) * w_in + iw];
                                acc += (v.abs() as f64).powf(p as f64);
                            }
                        }
                    }
                    let val = if acc == 0.0 { 0.0 } else { acc.powf(1.0 / p as f64) };
                    y[((ni * c + ci) * h_out + oh) * w_out + ow] = val as f32;
                }
            }
        }
    }
    (y, h_out, w_out)
}

/// LpPool1d FW reference (no padding).
fn host_lp_pool1d_fw(
    n: usize, c: usize, l_in: usize, x: &[f32],
    window: usize, stride: usize, p: f32,
) -> (Vec<f32>, usize) {
    let l_out = (l_in - window) / stride + 1;
    let mut y = vec![0f32; n * c * l_out];
    for ni in 0..n {
        for ci in 0..c {
            for ol in 0..l_out {
                let l_start = ol * stride;
                let mut acc = 0f64;
                for k in 0..window {
                    let il = l_start + k;
                    if il < l_in {
                        let v = x[(ni * c + ci) * l_in + il];
                        acc += (v.abs() as f64).powf(p as f64);
                    }
                }
                let val = if acc == 0.0 { 0.0 } else { acc.powf(1.0 / p as f64) };
                y[(ni * c + ci) * l_out + ol] = val as f32;
            }
        }
    }
    (y, l_out)
}

// =============================================================================
// Tests
// =============================================================================

#[test]
#[ignore]
fn lp_pool_p2_l2_2x2_kernel_f32() {
    let (ctx, stream) = setup();
    let (n, c, h_in, w_in) = (1i32, 2i32, 4i32, 4i32);
    let host_x: Vec<f32> = (1..=32).map(|k| (k as f32) * 0.5 - 4.0).collect();
    let (exp_y, h_out, w_out) =
        host_lp_pool2d_fw(n as usize, c as usize, h_in as usize, w_in as usize, &host_x, 2, 2, 2, 2, 2.0);
    assert_eq!(h_out, 2);
    assert_eq!(w_out, 2);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, exp_y.len()).expect("y");

    let desc =
        LpPool2dDescriptor::new(n, c, h_in, w_in, 2, 2, 2.0, ElementKind::F32);
    let plan =
        LpPool2dPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let x_shape = [n, c, h_in, w_in];
    let y_shape = [n, c, h_out as i32, w_out as i32];
    plan.run_fw(&stream, Workspace::None, LpPool2dFwArgs {
        x: TensorRef { data: dev_x.as_slice(), shape: x_shape, stride: contiguous_stride(x_shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: y_shape, stride: contiguous_stride(y_shape) },
    }).expect("fw");
    stream.synchronize().expect("sync fw");

    let mut got_y = vec![0f32; exp_y.len()];
    dev_y.copy_to_host(&mut got_y).expect("dl y");
    // 4 cells contribute (kh*kw = 4), each up to ~16 in magnitude;
    // f32 sqrt precision sufficient at ~64 * eps.
    let tol = 64.0 * f32::EPSILON;
    for i in 0..exp_y.len() {
        let t = (exp_y[i].abs() * tol).max(tol);
        assert!((got_y[i] - exp_y[i]).abs() <= t,
            "lp_pool2d p=2 f32 y @ {i}: got={} want={}", got_y[i], exp_y[i]);
    }
}

#[test]
#[ignore]
fn lp_pool_p1_l1_2x2_kernel_f32() {
    let (ctx, stream) = setup();
    let (n, c, h_in, w_in) = (1i32, 1i32, 4i32, 4i32);
    let host_x: Vec<f32> = (0..16).map(|k| (k as f32) - 7.5).collect();
    let (exp_y, h_out, w_out) =
        host_lp_pool2d_fw(n as usize, c as usize, h_in as usize, w_in as usize, &host_x, 2, 2, 2, 2, 1.0);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, exp_y.len()).expect("y");

    let desc =
        LpPool2dDescriptor::new(n, c, h_in, w_in, 2, 2, 1.0, ElementKind::F32);
    let plan =
        LpPool2dPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let x_shape = [n, c, h_in, w_in];
    let y_shape = [n, c, h_out as i32, w_out as i32];
    plan.run_fw(&stream, Workspace::None, LpPool2dFwArgs {
        x: TensorRef { data: dev_x.as_slice(), shape: x_shape, stride: contiguous_stride(x_shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: y_shape, stride: contiguous_stride(y_shape) },
    }).expect("fw");
    stream.synchronize().expect("sync fw");

    let mut got_y = vec![0f32; exp_y.len()];
    dev_y.copy_to_host(&mut got_y).expect("dl y");
    let tol = 32.0 * f32::EPSILON;
    for i in 0..exp_y.len() {
        let t = (exp_y[i].abs() * tol).max(tol);
        assert!((got_y[i] - exp_y[i]).abs() <= t,
            "lp_pool2d p=1 f32 y @ {i}: got={} want={}", got_y[i], exp_y[i]);
    }
}

#[test]
#[ignore]
fn lp_pool_p3_2x2_kernel_f32() {
    let (ctx, stream) = setup();
    let (n, c, h_in, w_in) = (1i32, 1i32, 4i32, 4i32);
    let host_x: Vec<f32> = (1..=16).map(|k| (k as f32) * 0.25).collect();
    let p_val = 3.0f32;
    let (exp_y, h_out, w_out) =
        host_lp_pool2d_fw(n as usize, c as usize, h_in as usize, w_in as usize, &host_x, 2, 2, 2, 2, p_val);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, exp_y.len()).expect("y");

    let desc =
        LpPool2dDescriptor::new(n, c, h_in, w_in, 2, 2, p_val, ElementKind::F32);
    let plan =
        LpPool2dPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let x_shape = [n, c, h_in, w_in];
    let y_shape = [n, c, h_out as i32, w_out as i32];
    plan.run_fw(&stream, Workspace::None, LpPool2dFwArgs {
        x: TensorRef { data: dev_x.as_slice(), shape: x_shape, stride: contiguous_stride(x_shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: y_shape, stride: contiguous_stride(y_shape) },
    }).expect("fw");
    stream.synchronize().expect("sync fw");

    let mut got_y = vec![0f32; exp_y.len()];
    dev_y.copy_to_host(&mut got_y).expect("dl y");
    // p=3 stresses the pow path on device; allow a looser tolerance.
    let tol = 1e-4f32;
    for i in 0..exp_y.len() {
        let t = (exp_y[i].abs() * tol).max(tol);
        assert!((got_y[i] - exp_y[i]).abs() <= t,
            "lp_pool2d p=3 f32 y @ {i}: got={} want={}", got_y[i], exp_y[i]);
    }
}

#[test]
#[ignore]
fn lp_pool_1d_p2_f32() {
    let (ctx, stream) = setup();
    let (n, c, l_in) = (1i32, 2i32, 8i32);
    let host_x: Vec<f32> = (0..16).map(|k| (k as f32) - 7.5).collect();
    let (exp_y, l_out) =
        host_lp_pool1d_fw(n as usize, c as usize, l_in as usize, &host_x, 2, 2, 2.0);
    assert_eq!(l_out, 4);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, exp_y.len()).expect("y");

    let desc = LpPool1dDescriptor::new(n, c, l_in, 2, 2.0, ElementKind::F32);
    let plan =
        LpPool1dPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let x_shape = [n, c, l_in];
    let y_shape = [n, c, l_out as i32];
    plan.run_fw(&stream, Workspace::None, LpPool1dFwArgs {
        x: TensorRef { data: dev_x.as_slice(), shape: x_shape, stride: contiguous_stride(x_shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: y_shape, stride: contiguous_stride(y_shape) },
    }).expect("fw");
    stream.synchronize().expect("sync fw");

    let mut got_y = vec![0f32; exp_y.len()];
    dev_y.copy_to_host(&mut got_y).expect("dl y");
    let tol = 64.0 * f32::EPSILON;
    for i in 0..exp_y.len() {
        let t = (exp_y[i].abs() * tol).max(tol);
        assert!((got_y[i] - exp_y[i]).abs() <= t,
            "lp_pool1d p=2 f32 y @ {i}: got={} want={}", got_y[i], exp_y[i]);
    }
}

#[test]
#[ignore]
fn lp_pool_bw_p2_2x2_f32() {
    let (ctx, stream) = setup();
    let (n, c, h_in, w_in) = (1i32, 1i32, 4i32, 4i32);
    // Use values away from zero to keep the FD reference well-conditioned.
    let host_x: Vec<f32> = (1..=16).map(|k| (k as f32) * 0.3 + 0.7).collect();
    let p_val = 2.0f32;
    let (host_y, h_out, w_out) =
        host_lp_pool2d_fw(n as usize, c as usize, h_in as usize, w_in as usize, &host_x, 2, 2, 2, 2, p_val);
    let numel_y = host_y.len();
    let numel_x = host_x.len();

    // Upstream gradient (deterministic, non-zero per cell).
    let host_dy: Vec<f32> =
        (0..numel_y).map(|i| 0.25 + 0.125 * (i as f32)).collect();

    // Finite-difference reference for dx.
    let eps = 1e-3f32;
    let mut exp_dx = vec![0f32; numel_x];
    let host_y_baseline = host_y.clone();
    for k in 0..numel_x {
        let mut xp = host_x.clone();
        let mut xn = host_x.clone();
        xp[k] += eps;
        xn[k] -= eps;
        let (yp, _, _) =
            host_lp_pool2d_fw(n as usize, c as usize, h_in as usize, w_in as usize, &xp, 2, 2, 2, 2, p_val);
        let (yn, _, _) =
            host_lp_pool2d_fw(n as usize, c as usize, h_in as usize, w_in as usize, &xn, 2, 2, 2, 2, p_val);
        let mut g = 0f32;
        for j in 0..numel_y {
            let dy_dx = (yp[j] - yn[j]) / (2.0 * eps);
            g += host_dy[j] * dy_dx;
        }
        exp_dx[k] = g;
    }
    // Sanity: host_y_baseline matches the original FW.
    for j in 0..numel_y {
        assert!((host_y[j] - host_y_baseline[j]).abs() < 1e-6);
    }

    // Drive the GPU FW + BW.
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_y_in = DeviceBuffer::from_slice(&ctx, &host_y).expect("up y");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel_x).expect("dx");

    let desc =
        LpPool2dDescriptor::new(n, c, h_in, w_in, 2, 2, p_val, ElementKind::F32);
    let bw =
        LpPool2dBackwardPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("bw sel");
    let x_shape = [n, c, h_in, w_in];
    let y_shape = [n, c, h_out as i32, w_out as i32];
    bw.run_bw(&stream, Workspace::None, LpPool2dBwArgs {
        x: TensorRef { data: dev_x.as_slice(), shape: x_shape, stride: contiguous_stride(x_shape) },
        y: TensorRef { data: dev_y_in.as_slice(), shape: y_shape, stride: contiguous_stride(y_shape) },
        dy: TensorRef { data: dev_dy.as_slice(), shape: y_shape, stride: contiguous_stride(y_shape) },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape: x_shape, stride: contiguous_stride(x_shape) },
    }).expect("bw");
    stream.synchronize().expect("sync bw");

    let mut got_dx = vec![0f32; numel_x];
    dev_dx.copy_to_host(&mut got_dx).expect("dl dx");

    // FD tolerance: O(eps^2) error from the central difference, plus
    // ~kh*kw ulps of summation noise in dx.
    let tol = 1e-2f32;
    for i in 0..numel_x {
        let t = (exp_dx[i].abs() * tol).max(tol);
        assert!((got_dx[i] - exp_dx[i]).abs() <= t,
            "lp_pool2d_bw p=2 f32 dx @ {i}: got={} want={}", got_dx[i], exp_dx[i]);
    }
}

#[test]
#[ignore]
fn lp_pool_f16_p2_2x2() {
    let (ctx, stream) = setup();
    let (n, c, h_in, w_in) = (1i32, 1i32, 4i32, 4i32);
    let host_x_f32: Vec<f32> = (1..=16).map(|k| (k as f32) * 0.5).collect();
    let host_x: Vec<f16> = host_x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let (exp_y_f32, h_out, w_out) =
        host_lp_pool2d_fw(n as usize, c as usize, h_in as usize, w_in as usize, &host_x_f32, 2, 2, 2, 2, 2.0);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, exp_y_f32.len()).expect("y");

    let desc =
        LpPool2dDescriptor::new(n, c, h_in, w_in, 2, 2, 2.0, ElementKind::F16);
    let plan =
        LpPool2dPlan::<f16>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let x_shape = [n, c, h_in, w_in];
    let y_shape = [n, c, h_out as i32, w_out as i32];
    plan.run_fw(&stream, Workspace::None, LpPool2dFwArgs {
        x: TensorRef { data: dev_x.as_slice(), shape: x_shape, stride: contiguous_stride(x_shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: y_shape, stride: contiguous_stride(y_shape) },
    }).expect("fw");
    stream.synchronize().expect("sync fw");

    let mut got_y_h = vec![f16::from_f32(0.0); exp_y_f32.len()];
    dev_y.copy_to_host(&mut got_y_h).expect("dl y");
    let tol = 4e-3f32;
    for i in 0..exp_y_f32.len() {
        let g = got_y_h[i].to_f32();
        let t = (exp_y_f32[i].abs() * tol).max(tol);
        assert!((g - exp_y_f32[i]).abs() <= t,
            "lp_pool2d p=2 f16 y @ {i}: got={} want={}", g, exp_y_f32[i]);
    }
}
