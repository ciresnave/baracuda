#![cfg(feature = "cudnn")]
//! Real-GPU smoke tests for `MaxPool2dPlan` / `AvgPool2dPlan` FW + BW.
//!
//! Covers all four FP dtypes (f32 / f64 / f16 / bf16). The base fixture
//! is the canonical "halve-resolution" pool: 1×2×4×4 input, 2×2 window,
//! stride 2, pad 0, producing a 1×2×2×2 output (each output cell is the
//! reduction over a disjoint 2×2 input tile).
//!
//! `#[ignore]` by default — requires a real CUDA device + cuDNN.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, AvgPool2dPlan, ElementKind, MaxPool2dPlan, PlanPreference, Pool2dBwArgs,
    Pool2dDescriptor, Pool2dFwArgs, PoolMode, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

const F16_TOL: f32 = 4e-3;
const BF16_TOL: f32 = 3e-2;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// CPU reference: max-pool FW over `[N, C, H, W]` NCHW row-major.
fn host_max_pool_fw_f32(
    n: usize, c: usize, h_in: usize, w_in: usize, x: &[f32],
    window_h: usize, window_w: usize,
    pad_h: usize, pad_w: usize,
    stride_h: usize, stride_w: usize,
) -> (Vec<f32>, usize, usize) {
    let h_out = (h_in + 2 * pad_h - window_h) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - window_w) / stride_w + 1;
    let mut y = vec![f32::NEG_INFINITY; n * c * h_out * w_out];
    for ni in 0..n {
        for ci in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut m = f32::NEG_INFINITY;
                    for kh in 0..window_h {
                        for kw in 0..window_w {
                            let ih = oh * stride_h + kh;
                            let iw = ow * stride_w + kw;
                            if ih < pad_h || iw < pad_w {
                                continue;
                            }
                            let ih = ih - pad_h;
                            let iw = iw - pad_w;
                            if ih >= h_in || iw >= w_in {
                                continue;
                            }
                            let v = x[((ni * c + ci) * h_in + ih) * w_in + iw];
                            if v > m {
                                m = v;
                            }
                        }
                    }
                    y[((ni * c + ci) * h_out + oh) * w_out + ow] = m;
                }
            }
        }
    }
    (y, h_out, w_out)
}

/// CPU reference: max-pool BW. Routes `dy` to whichever input cell was
/// the per-window argmax in the forward (ties: first occurrence wins —
/// scan order matches the FW reference above).
fn host_max_pool_bw_f32(
    n: usize, c: usize, h_in: usize, w_in: usize, x: &[f32], dy: &[f32],
    window_h: usize, window_w: usize,
    pad_h: usize, pad_w: usize,
    stride_h: usize, stride_w: usize,
) -> Vec<f32> {
    let h_out = (h_in + 2 * pad_h - window_h) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - window_w) / stride_w + 1;
    let mut dx = vec![0f32; n * c * h_in * w_in];
    for ni in 0..n {
        for ci in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut best_v = f32::NEG_INFINITY;
                    let mut best_idx: Option<usize> = None;
                    for kh in 0..window_h {
                        for kw in 0..window_w {
                            let ih = oh * stride_h + kh;
                            let iw = ow * stride_w + kw;
                            if ih < pad_h || iw < pad_w {
                                continue;
                            }
                            let ih = ih - pad_h;
                            let iw = iw - pad_w;
                            if ih >= h_in || iw >= w_in {
                                continue;
                            }
                            let idx = ((ni * c + ci) * h_in + ih) * w_in + iw;
                            let v = x[idx];
                            if v > best_v {
                                best_v = v;
                                best_idx = Some(idx);
                            }
                        }
                    }
                    if let Some(idx) = best_idx {
                        dx[idx] += dy[((ni * c + ci) * h_out + oh) * w_out + ow];
                    }
                }
            }
        }
    }
    dx
}

/// CPU reference: avg-pool FW (count-exclude-padding — PyTorch default).
fn host_avg_pool_fw_excl_f32(
    n: usize, c: usize, h_in: usize, w_in: usize, x: &[f32],
    window_h: usize, window_w: usize,
    pad_h: usize, pad_w: usize,
    stride_h: usize, stride_w: usize,
) -> (Vec<f32>, usize, usize) {
    let h_out = (h_in + 2 * pad_h - window_h) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - window_w) / stride_w + 1;
    let mut y = vec![0f32; n * c * h_out * w_out];
    for ni in 0..n {
        for ci in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut sum = 0f64;
                    let mut count = 0usize;
                    for kh in 0..window_h {
                        for kw in 0..window_w {
                            let ih = oh * stride_h + kh;
                            let iw = ow * stride_w + kw;
                            if ih < pad_h || iw < pad_w {
                                continue;
                            }
                            let ih = ih - pad_h;
                            let iw = iw - pad_w;
                            if ih >= h_in || iw >= w_in {
                                continue;
                            }
                            sum += x[((ni * c + ci) * h_in + ih) * w_in + iw] as f64;
                            count += 1;
                        }
                    }
                    y[((ni * c + ci) * h_out + oh) * w_out + ow] =
                        if count > 0 { (sum / count as f64) as f32 } else { 0f32 };
                }
            }
        }
    }
    (y, h_out, w_out)
}

/// CPU reference: avg-pool BW (count-exclude-padding).
fn host_avg_pool_bw_excl_f32(
    n: usize, c: usize, h_in: usize, w_in: usize, dy: &[f32],
    window_h: usize, window_w: usize,
    pad_h: usize, pad_w: usize,
    stride_h: usize, stride_w: usize,
) -> Vec<f32> {
    let h_out = (h_in + 2 * pad_h - window_h) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - window_w) / stride_w + 1;
    let mut dx = vec![0f32; n * c * h_in * w_in];
    for ni in 0..n {
        for ci in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    // First pass: compute the valid-count denominator
                    // for this window.
                    let mut count = 0usize;
                    for kh in 0..window_h {
                        for kw in 0..window_w {
                            let ih = oh * stride_h + kh;
                            let iw = ow * stride_w + kw;
                            if ih < pad_h || iw < pad_w {
                                continue;
                            }
                            let ih = ih - pad_h;
                            let iw = iw - pad_w;
                            if ih < h_in && iw < w_in {
                                count += 1;
                            }
                        }
                    }
                    if count == 0 {
                        continue;
                    }
                    let g = dy[((ni * c + ci) * h_out + oh) * w_out + ow] / count as f32;
                    // Second pass: distribute the per-cell gradient.
                    for kh in 0..window_h {
                        for kw in 0..window_w {
                            let ih = oh * stride_h + kh;
                            let iw = ow * stride_w + kw;
                            if ih < pad_h || iw < pad_w {
                                continue;
                            }
                            let ih = ih - pad_h;
                            let iw = iw - pad_w;
                            if ih < h_in && iw < w_in {
                                let idx = ((ni * c + ci) * h_in + ih) * w_in + iw;
                                dx[idx] += g;
                            }
                        }
                    }
                }
            }
        }
    }
    dx
}

/// Canonical "halve-resolution" fixture input: 1×2×4×4, distinct values
/// per channel chosen so the per-2×2-window argmax is unambiguous.
fn fixture_input_4x4_f32() -> Vec<f32> {
    // C=0 row-major H×W:
    //  1  2  3  4
    //  5  6  7  8
    //  9 10 11 12
    // 13 14 15 16
    // C=1 (flipped sign, distinct from C=0):
    // -16 -15 -14 -13
    // -12 -11 -10  -9
    //  -8  -7  -6  -5
    //  -4  -3  -2  -1
    let mut v = Vec::with_capacity(32);
    for k in 1..=16 {
        v.push(k as f32);
    }
    for k in (1..=16).rev() {
        v.push(-(k as f32));
    }
    v
}

#[test]
#[ignore]
fn max_pool2d_f32() {
    let (ctx, stream) = setup();
    let (n, c, h_in, w_in) = (1i32, 2i32, 4i32, 4i32);
    let host_x = fixture_input_4x4_f32();
    let numel_x = host_x.len();
    let (exp_y, h_out, w_out) = host_max_pool_fw_f32(
        n as usize, c as usize, h_in as usize, w_in as usize, &host_x,
        2, 2, 0, 0, 2, 2,
    );
    let numel_y = exp_y.len();
    assert_eq!(h_out, 2);
    assert_eq!(w_out, 2);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel_y).expect("y");

    let desc = Pool2dDescriptor {
        batch: n, channels: c, h_in, w_in,
        window_h: 2, window_w: 2,
        pad_h: 0, pad_w: 0,
        stride_h: 2, stride_w: 2,
        mode: PoolMode::Max,
        element: ElementKind::F32,
    };
    let plan = MaxPool2dPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let y_shape = [n, c, h_out as i32, w_out as i32];
    let x_shape = [n, c, h_in, w_in];
    plan.run_fw(&stream, Workspace::None, Pool2dFwArgs {
        x: TensorRef { data: dev_x.as_slice(), shape: x_shape, stride: contiguous_stride(x_shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: y_shape, stride: contiguous_stride(y_shape) },
    }).expect("fw");
    stream.synchronize().expect("sync fw");

    let mut got_y = vec![0f32; numel_y];
    dev_y.copy_to_host(&mut got_y).expect("dl y");
    let tol = 16.0 * f32::EPSILON;
    for i in 0..numel_y {
        let t = (exp_y[i].abs() * tol).max(tol);
        assert!((got_y[i] - exp_y[i]).abs() <= t,
            "max_pool f32 y @ {i}: got={} want={}", got_y[i], exp_y[i]);
    }

    // BW: random-ish dy, propagate to dx, compare with host argmax-based
    // reference.
    let host_dy: Vec<f32> = (0..numel_y).map(|i| 0.5 + (i as f32) * 0.25).collect();
    let exp_dx = host_max_pool_bw_f32(
        n as usize, c as usize, h_in as usize, w_in as usize, &host_x, &host_dy,
        2, 2, 0, 0, 2, 2,
    );
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel_x).expect("dx");
    plan.run_bw(&stream, Workspace::None, Pool2dBwArgs {
        y: TensorRef { data: dev_y.as_slice(), shape: y_shape, stride: contiguous_stride(y_shape) },
        dy: TensorRef { data: dev_dy.as_slice(), shape: y_shape, stride: contiguous_stride(y_shape) },
        x: TensorRef { data: dev_x.as_slice(), shape: x_shape, stride: contiguous_stride(x_shape) },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape: x_shape, stride: contiguous_stride(x_shape) },
    }).expect("bw");
    stream.synchronize().expect("sync bw");

    let mut got_dx = vec![0f32; numel_x];
    dev_dx.copy_to_host(&mut got_dx).expect("dl dx");
    for i in 0..numel_x {
        let t = (exp_dx[i].abs() * tol).max(tol);
        assert!((got_dx[i] - exp_dx[i]).abs() <= t,
            "max_pool f32 dx @ {i}: got={} want={}", got_dx[i], exp_dx[i]);
    }
}

#[test]
#[ignore]
fn max_pool2d_f64() {
    let (ctx, stream) = setup();
    let (n, c, h_in, w_in) = (1i32, 2i32, 4i32, 4i32);
    let host_x_f32 = fixture_input_4x4_f32();
    let host_x: Vec<f64> = host_x_f32.iter().map(|&v| v as f64).collect();
    let numel_x = host_x.len();
    let (exp_y_f32, h_out, w_out) = host_max_pool_fw_f32(
        n as usize, c as usize, h_in as usize, w_in as usize, &host_x_f32,
        2, 2, 0, 0, 2, 2,
    );
    let numel_y = exp_y_f32.len();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel_y).expect("y");
    let desc = Pool2dDescriptor {
        batch: n, channels: c, h_in, w_in,
        window_h: 2, window_w: 2,
        pad_h: 0, pad_w: 0,
        stride_h: 2, stride_w: 2,
        mode: PoolMode::Max,
        element: ElementKind::F64,
    };
    let plan = MaxPool2dPlan::<f64>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let y_shape = [n, c, h_out as i32, w_out as i32];
    let x_shape = [n, c, h_in, w_in];
    plan.run_fw(&stream, Workspace::None, Pool2dFwArgs {
        x: TensorRef { data: dev_x.as_slice(), shape: x_shape, stride: contiguous_stride(x_shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: y_shape, stride: contiguous_stride(y_shape) },
    }).expect("fw");
    stream.synchronize().expect("sync fw");

    let mut got_y = vec![0f64; numel_y];
    dev_y.copy_to_host(&mut got_y).expect("dl y");
    let tol = 32.0 * f64::EPSILON;
    for i in 0..numel_y {
        let want = exp_y_f32[i] as f64;
        let t = (want.abs() * tol).max(tol);
        assert!((got_y[i] - want).abs() <= t,
            "max_pool f64 y @ {i}: got={} want={}", got_y[i], want);
    }

    let host_dy_f32: Vec<f32> = (0..numel_y).map(|i| 0.5 + (i as f32) * 0.25).collect();
    let host_dy: Vec<f64> = host_dy_f32.iter().map(|&v| v as f64).collect();
    let exp_dx_f32 = host_max_pool_bw_f32(
        n as usize, c as usize, h_in as usize, w_in as usize, &host_x_f32, &host_dy_f32,
        2, 2, 0, 0, 2, 2,
    );
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_dx: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel_x).expect("dx");
    plan.run_bw(&stream, Workspace::None, Pool2dBwArgs {
        y: TensorRef { data: dev_y.as_slice(), shape: y_shape, stride: contiguous_stride(y_shape) },
        dy: TensorRef { data: dev_dy.as_slice(), shape: y_shape, stride: contiguous_stride(y_shape) },
        x: TensorRef { data: dev_x.as_slice(), shape: x_shape, stride: contiguous_stride(x_shape) },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape: x_shape, stride: contiguous_stride(x_shape) },
    }).expect("bw");
    stream.synchronize().expect("sync bw");

    let mut got_dx = vec![0f64; numel_x];
    dev_dx.copy_to_host(&mut got_dx).expect("dl dx");
    for i in 0..numel_x {
        let want = exp_dx_f32[i] as f64;
        let t = (want.abs() * tol).max(tol);
        assert!((got_dx[i] - want).abs() <= t,
            "max_pool f64 dx @ {i}: got={} want={}", got_dx[i], want);
    }
}

#[test]
#[ignore]
fn avg_pool2d_f32() {
    let (ctx, stream) = setup();
    let (n, c, h_in, w_in) = (1i32, 2i32, 4i32, 4i32);
    let host_x = fixture_input_4x4_f32();
    let numel_x = host_x.len();
    let (exp_y, h_out, w_out) = host_avg_pool_fw_excl_f32(
        n as usize, c as usize, h_in as usize, w_in as usize, &host_x,
        2, 2, 0, 0, 2, 2,
    );
    let numel_y = exp_y.len();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel_y).expect("y");
    let desc = Pool2dDescriptor {
        batch: n, channels: c, h_in, w_in,
        window_h: 2, window_w: 2,
        pad_h: 0, pad_w: 0,
        stride_h: 2, stride_w: 2,
        mode: PoolMode::AvgExcludePad,
        element: ElementKind::F32,
    };
    let plan = AvgPool2dPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let y_shape = [n, c, h_out as i32, w_out as i32];
    let x_shape = [n, c, h_in, w_in];
    plan.run_fw(&stream, Workspace::None, Pool2dFwArgs {
        x: TensorRef { data: dev_x.as_slice(), shape: x_shape, stride: contiguous_stride(x_shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: y_shape, stride: contiguous_stride(y_shape) },
    }).expect("fw");
    stream.synchronize().expect("sync fw");

    let mut got_y = vec![0f32; numel_y];
    dev_y.copy_to_host(&mut got_y).expect("dl y");
    let tol = 32.0 * f32::EPSILON;
    for i in 0..numel_y {
        let t = (exp_y[i].abs() * tol).max(tol);
        assert!((got_y[i] - exp_y[i]).abs() <= t,
            "avg_pool f32 y @ {i}: got={} want={}", got_y[i], exp_y[i]);
    }

    let host_dy: Vec<f32> = (0..numel_y).map(|i| 0.5 + (i as f32) * 0.25).collect();
    let exp_dx = host_avg_pool_bw_excl_f32(
        n as usize, c as usize, h_in as usize, w_in as usize, &host_dy,
        2, 2, 0, 0, 2, 2,
    );
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel_x).expect("dx");
    plan.run_bw(&stream, Workspace::None, Pool2dBwArgs {
        y: TensorRef { data: dev_y.as_slice(), shape: y_shape, stride: contiguous_stride(y_shape) },
        dy: TensorRef { data: dev_dy.as_slice(), shape: y_shape, stride: contiguous_stride(y_shape) },
        x: TensorRef { data: dev_x.as_slice(), shape: x_shape, stride: contiguous_stride(x_shape) },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape: x_shape, stride: contiguous_stride(x_shape) },
    }).expect("bw");
    stream.synchronize().expect("sync bw");

    let mut got_dx = vec![0f32; numel_x];
    dev_dx.copy_to_host(&mut got_dx).expect("dl dx");
    for i in 0..numel_x {
        let t = (exp_dx[i].abs() * tol).max(tol);
        assert!((got_dx[i] - exp_dx[i]).abs() <= t,
            "avg_pool f32 dx @ {i}: got={} want={}", got_dx[i], exp_dx[i]);
    }
}

#[test]
#[ignore]
fn avg_pool2d_f64() {
    let (ctx, stream) = setup();
    let (n, c, h_in, w_in) = (1i32, 2i32, 4i32, 4i32);
    let host_x_f32 = fixture_input_4x4_f32();
    let host_x: Vec<f64> = host_x_f32.iter().map(|&v| v as f64).collect();
    let numel_x = host_x.len();
    let (exp_y_f32, h_out, w_out) = host_avg_pool_fw_excl_f32(
        n as usize, c as usize, h_in as usize, w_in as usize, &host_x_f32,
        2, 2, 0, 0, 2, 2,
    );
    let numel_y = exp_y_f32.len();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel_y).expect("y");
    let desc = Pool2dDescriptor {
        batch: n, channels: c, h_in, w_in,
        window_h: 2, window_w: 2,
        pad_h: 0, pad_w: 0,
        stride_h: 2, stride_w: 2,
        mode: PoolMode::AvgExcludePad,
        element: ElementKind::F64,
    };
    let plan = AvgPool2dPlan::<f64>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let y_shape = [n, c, h_out as i32, w_out as i32];
    let x_shape = [n, c, h_in, w_in];
    plan.run_fw(&stream, Workspace::None, Pool2dFwArgs {
        x: TensorRef { data: dev_x.as_slice(), shape: x_shape, stride: contiguous_stride(x_shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: y_shape, stride: contiguous_stride(y_shape) },
    }).expect("fw");
    stream.synchronize().expect("sync fw");

    let mut got_y = vec![0f64; numel_y];
    dev_y.copy_to_host(&mut got_y).expect("dl y");
    let tol = 32.0 * f64::EPSILON;
    for i in 0..numel_y {
        let want = exp_y_f32[i] as f64;
        let t = (want.abs() * tol).max(tol);
        assert!((got_y[i] - want).abs() <= t,
            "avg_pool f64 y @ {i}: got={} want={}", got_y[i], want);
    }

    let host_dy_f32: Vec<f32> = (0..numel_y).map(|i| 0.5 + (i as f32) * 0.25).collect();
    let host_dy: Vec<f64> = host_dy_f32.iter().map(|&v| v as f64).collect();
    let exp_dx_f32 = host_avg_pool_bw_excl_f32(
        n as usize, c as usize, h_in as usize, w_in as usize, &host_dy_f32,
        2, 2, 0, 0, 2, 2,
    );
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_dx: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel_x).expect("dx");
    plan.run_bw(&stream, Workspace::None, Pool2dBwArgs {
        y: TensorRef { data: dev_y.as_slice(), shape: y_shape, stride: contiguous_stride(y_shape) },
        dy: TensorRef { data: dev_dy.as_slice(), shape: y_shape, stride: contiguous_stride(y_shape) },
        x: TensorRef { data: dev_x.as_slice(), shape: x_shape, stride: contiguous_stride(x_shape) },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape: x_shape, stride: contiguous_stride(x_shape) },
    }).expect("bw");
    stream.synchronize().expect("sync bw");

    let mut got_dx = vec![0f64; numel_x];
    dev_dx.copy_to_host(&mut got_dx).expect("dl dx");
    for i in 0..numel_x {
        let want = exp_dx_f32[i] as f64;
        let t = (want.abs() * tol).max(tol);
        assert!((got_dx[i] - want).abs() <= t,
            "avg_pool f64 dx @ {i}: got={} want={}", got_dx[i], want);
    }
}

#[test]
#[ignore]
fn pool2d_f16_smoke() {
    let (ctx, stream) = setup();
    let (n, c, h_in, w_in) = (1i32, 2i32, 4i32, 4i32);
    let host_x_f32 = fixture_input_4x4_f32();
    let host_x: Vec<f16> = host_x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let (exp_y_max_f32, h_out, w_out) = host_max_pool_fw_f32(
        n as usize, c as usize, h_in as usize, w_in as usize, &host_x_f32,
        2, 2, 0, 0, 2, 2,
    );
    let (exp_y_avg_f32, _, _) = host_avg_pool_fw_excl_f32(
        n as usize, c as usize, h_in as usize, w_in as usize, &host_x_f32,
        2, 2, 0, 0, 2, 2,
    );
    let numel_y = exp_y_max_f32.len();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y_max: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel_y).expect("y_max");
    let mut dev_y_avg: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel_y).expect("y_avg");
    let y_shape = [n, c, h_out as i32, w_out as i32];
    let x_shape = [n, c, h_in, w_in];

    // max-pool FW
    let max_desc = Pool2dDescriptor {
        batch: n, channels: c, h_in, w_in,
        window_h: 2, window_w: 2, pad_h: 0, pad_w: 0, stride_h: 2, stride_w: 2,
        mode: PoolMode::Max, element: ElementKind::F16,
    };
    let max_plan = MaxPool2dPlan::<f16>::select(&stream, &max_desc, PlanPreference::default())
        .expect("max sel");
    max_plan.run_fw(&stream, Workspace::None, Pool2dFwArgs {
        x: TensorRef { data: dev_x.as_slice(), shape: x_shape, stride: contiguous_stride(x_shape) },
        y: TensorMut { data: dev_y_max.as_slice_mut(), shape: y_shape, stride: contiguous_stride(y_shape) },
    }).expect("max fw");

    // avg-pool FW
    let avg_desc = Pool2dDescriptor {
        batch: n, channels: c, h_in, w_in,
        window_h: 2, window_w: 2, pad_h: 0, pad_w: 0, stride_h: 2, stride_w: 2,
        mode: PoolMode::AvgExcludePad, element: ElementKind::F16,
    };
    let avg_plan = AvgPool2dPlan::<f16>::select(&stream, &avg_desc, PlanPreference::default())
        .expect("avg sel");
    avg_plan.run_fw(&stream, Workspace::None, Pool2dFwArgs {
        x: TensorRef { data: dev_x.as_slice(), shape: x_shape, stride: contiguous_stride(x_shape) },
        y: TensorMut { data: dev_y_avg.as_slice_mut(), shape: y_shape, stride: contiguous_stride(y_shape) },
    }).expect("avg fw");

    stream.synchronize().expect("sync");

    let mut got_max = vec![f16::ZERO; numel_y];
    let mut got_avg = vec![f16::ZERO; numel_y];
    dev_y_max.copy_to_host(&mut got_max).expect("dl max");
    dev_y_avg.copy_to_host(&mut got_avg).expect("dl avg");
    for i in 0..numel_y {
        let got = got_max[i].to_f32();
        let want = exp_y_max_f32[i];
        let t = (want.abs() * F16_TOL).max(F16_TOL);
        assert!((got - want).abs() <= t,
            "max_pool f16 y @ {i}: got={got} want={want}");
    }
    for i in 0..numel_y {
        let got = got_avg[i].to_f32();
        let want = exp_y_avg_f32[i];
        let t = (want.abs() * F16_TOL).max(F16_TOL);
        assert!((got - want).abs() <= t,
            "avg_pool f16 y @ {i}: got={got} want={want}");
    }
}

#[test]
#[ignore]
fn pool2d_bf16_smoke() {
    let (ctx, stream) = setup();
    let (n, c, h_in, w_in) = (1i32, 2i32, 4i32, 4i32);
    let host_x_f32 = fixture_input_4x4_f32();
    let host_x: Vec<bf16> = host_x_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let (exp_y_max_f32, h_out, w_out) = host_max_pool_fw_f32(
        n as usize, c as usize, h_in as usize, w_in as usize, &host_x_f32,
        2, 2, 0, 0, 2, 2,
    );
    let (exp_y_avg_f32, _, _) = host_avg_pool_fw_excl_f32(
        n as usize, c as usize, h_in as usize, w_in as usize, &host_x_f32,
        2, 2, 0, 0, 2, 2,
    );
    let numel_y = exp_y_max_f32.len();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y_max: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel_y).expect("y_max");
    let mut dev_y_avg: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel_y).expect("y_avg");
    let y_shape = [n, c, h_out as i32, w_out as i32];
    let x_shape = [n, c, h_in, w_in];

    let max_desc = Pool2dDescriptor {
        batch: n, channels: c, h_in, w_in,
        window_h: 2, window_w: 2, pad_h: 0, pad_w: 0, stride_h: 2, stride_w: 2,
        mode: PoolMode::Max, element: ElementKind::Bf16,
    };
    let max_plan = MaxPool2dPlan::<bf16>::select(&stream, &max_desc, PlanPreference::default())
        .expect("max sel");
    max_plan.run_fw(&stream, Workspace::None, Pool2dFwArgs {
        x: TensorRef { data: dev_x.as_slice(), shape: x_shape, stride: contiguous_stride(x_shape) },
        y: TensorMut { data: dev_y_max.as_slice_mut(), shape: y_shape, stride: contiguous_stride(y_shape) },
    }).expect("max fw");

    let avg_desc = Pool2dDescriptor {
        batch: n, channels: c, h_in, w_in,
        window_h: 2, window_w: 2, pad_h: 0, pad_w: 0, stride_h: 2, stride_w: 2,
        mode: PoolMode::AvgExcludePad, element: ElementKind::Bf16,
    };
    let avg_plan = AvgPool2dPlan::<bf16>::select(&stream, &avg_desc, PlanPreference::default())
        .expect("avg sel");
    avg_plan.run_fw(&stream, Workspace::None, Pool2dFwArgs {
        x: TensorRef { data: dev_x.as_slice(), shape: x_shape, stride: contiguous_stride(x_shape) },
        y: TensorMut { data: dev_y_avg.as_slice_mut(), shape: y_shape, stride: contiguous_stride(y_shape) },
    }).expect("avg fw");

    stream.synchronize().expect("sync");

    let mut got_max = vec![bf16::ZERO; numel_y];
    let mut got_avg = vec![bf16::ZERO; numel_y];
    dev_y_max.copy_to_host(&mut got_max).expect("dl max");
    dev_y_avg.copy_to_host(&mut got_avg).expect("dl avg");
    for i in 0..numel_y {
        let got = got_max[i].to_f32();
        let want = exp_y_max_f32[i];
        let t = (want.abs() * BF16_TOL).max(BF16_TOL);
        assert!((got - want).abs() <= t,
            "max_pool bf16 y @ {i}: got={got} want={want}");
    }
    for i in 0..numel_y {
        let got = got_avg[i].to_f32();
        let want = exp_y_avg_f32[i];
        let t = (want.abs() * BF16_TOL).max(BF16_TOL);
        assert!((got - want).abs() <= t,
            "avg_pool bf16 y @ {i}: got={got} want={want}");
    }
}

#[test]
#[ignore]
fn max_pool2d_with_padding_f32() {
    // 3x3 input, 3x3 window, stride 1, pad 1 → 3x3 output. Each output
    // cell is the max over a 3x3 neighborhood (with zero padding on the
    // border).
    let (ctx, stream) = setup();
    let (n, c, h_in, w_in) = (1i32, 1i32, 3i32, 3i32);
    let host_x: Vec<f32> = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ];
    let numel_x = host_x.len();
    let (exp_y, h_out, w_out) = host_max_pool_fw_f32(
        n as usize, c as usize, h_in as usize, w_in as usize, &host_x,
        3, 3, 1, 1, 1, 1,
    );
    assert_eq!(h_out, 3);
    assert_eq!(w_out, 3);
    let numel_y = exp_y.len();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel_y).expect("y");
    let desc = Pool2dDescriptor {
        batch: n, channels: c, h_in, w_in,
        window_h: 3, window_w: 3, pad_h: 1, pad_w: 1, stride_h: 1, stride_w: 1,
        mode: PoolMode::Max, element: ElementKind::F32,
    };
    let plan = MaxPool2dPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let y_shape = [n, c, h_out as i32, w_out as i32];
    let x_shape = [n, c, h_in, w_in];
    plan.run_fw(&stream, Workspace::None, Pool2dFwArgs {
        x: TensorRef { data: dev_x.as_slice(), shape: x_shape, stride: contiguous_stride(x_shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: y_shape, stride: contiguous_stride(y_shape) },
    }).expect("fw");
    stream.synchronize().expect("sync fw");

    let mut got_y = vec![0f32; numel_y];
    dev_y.copy_to_host(&mut got_y).expect("dl y");
    let tol = 16.0 * f32::EPSILON;
    for i in 0..numel_y {
        let t = (exp_y[i].abs() * tol).max(tol);
        assert!((got_y[i] - exp_y[i]).abs() <= t,
            "max_pool_pad f32 y @ {i}: got={} want={}", got_y[i], exp_y[i]);
    }

    // BW round-trip
    let host_dy: Vec<f32> = (0..numel_y).map(|i| 1.0 + (i as f32) * 0.5).collect();
    let exp_dx = host_max_pool_bw_f32(
        n as usize, c as usize, h_in as usize, w_in as usize, &host_x, &host_dy,
        3, 3, 1, 1, 1, 1,
    );
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel_x).expect("dx");
    plan.run_bw(&stream, Workspace::None, Pool2dBwArgs {
        y: TensorRef { data: dev_y.as_slice(), shape: y_shape, stride: contiguous_stride(y_shape) },
        dy: TensorRef { data: dev_dy.as_slice(), shape: y_shape, stride: contiguous_stride(y_shape) },
        x: TensorRef { data: dev_x.as_slice(), shape: x_shape, stride: contiguous_stride(x_shape) },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape: x_shape, stride: contiguous_stride(x_shape) },
    }).expect("bw");
    stream.synchronize().expect("sync bw");

    let mut got_dx = vec![0f32; numel_x];
    dev_dx.copy_to_host(&mut got_dx).expect("dl dx");
    for i in 0..numel_x {
        let t = (exp_dx[i].abs() * tol).max(tol);
        assert!((got_dx[i] - exp_dx[i]).abs() <= t,
            "max_pool_pad f32 dx @ {i}: got={} want={}", got_dx[i], exp_dx[i]);
    }
}
