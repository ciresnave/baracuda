#![cfg(feature = "cudnn")]
//! Real-GPU smoke tests for `FractionalMaxPool2dPlan` /
//! `FractionalMaxPool3dPlan` (Phase 16.3 — bespoke kernel).
//!
//! Two regimes exercised:
//!   * Deterministic samples — caller-provided f32 α buffer, output is
//!     verified against a host reference computing the same per-cell
//!     window placement formula (evenly-spaced base + α perturbation).
//!   * cuRAND-driven — fills α via `RandomPlan + RandomKind::Uniform` and
//!     verifies output cells are valid input values + indices are in
//!     range (no exact equality — schedule is RNG-driven).
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features cudnn,sm89 \
//!   --test fractional_max_pool_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, FractionalMaxPool2dBwArgs, FractionalMaxPool2dDescriptor,
    FractionalMaxPool2dFwArgs, FractionalMaxPool2dPlan, FractionalMaxPool3dDescriptor,
    FractionalMaxPool3dFwArgs, FractionalMaxPool3dPlan, PlanPreference, RandomArgs,
    RandomDescriptor, RandomKind, RandomPlan, TensorMut, TensorRef, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Host port of the kernel's evenly-spaced-base + α perturbation
/// formula. Matches `compute_window_start` in
/// `baracuda_fractional_max_pool.cuh` bit-for-bit (single-precision
/// math throughout).
fn host_window_start(out_idx: i32, out_size: i32, in_size: i32, k: i32, alpha: f32) -> i32 {
    let max_start = in_size - k;
    if max_start <= 0 {
        return 0;
    }
    let start = if out_size <= 1 {
        (alpha * max_start as f32).floor() as i32
    } else {
        let base = out_idx as f32 * max_start as f32 / (out_size - 1) as f32;
        (base + alpha).floor() as i32
    };
    start.clamp(0, max_start)
}

/// CPU FW reference for the 2D fractional max-pool.
fn host_fmp2d_f32(
    n: usize, c: usize, h_in: usize, w_in: usize, x: &[f32],
    h_out: usize, w_out: usize,
    kh: usize, kw: usize,
    samples: &[f32],     // [N, C, 2]
) -> (Vec<f32>, Vec<i64>) {
    let mut y = vec![f32::NEG_INFINITY; n * c * h_out * w_out];
    let mut indices = vec![0i64; n * c * h_out * w_out];
    for ni in 0..n {
        for ci in 0..c {
            let alpha_h = samples[(ni * c + ci) * 2 + 0];
            let alpha_w = samples[(ni * c + ci) * 2 + 1];
            for oh in 0..h_out {
                let start_h = host_window_start(
                    oh as i32, h_out as i32, h_in as i32, kh as i32, alpha_h,
                ) as usize;
                for ow in 0..w_out {
                    let start_w = host_window_start(
                        ow as i32, w_out as i32, w_in as i32, kw as i32, alpha_w,
                    ) as usize;
                    let mut best_v = f32::NEG_INFINITY;
                    let mut best_idx = ((ni * c + ci) * h_in + start_h) * w_in + start_w;
                    for dh in 0..kh {
                        for dw in 0..kw {
                            let idx = ((ni * c + ci) * h_in + start_h + dh) * w_in
                                + start_w + dw;
                            let v = x[idx];
                            if v > best_v {
                                best_v = v;
                                best_idx = idx;
                            }
                        }
                    }
                    let out_off = ((ni * c + ci) * h_out + oh) * w_out + ow;
                    y[out_off] = best_v;
                    indices[out_off] = best_idx as i64;
                }
            }
        }
    }
    (y, indices)
}

/// CPU FW reference for the 3D fractional max-pool.
fn host_fmp3d_f32(
    n: usize, c: usize, d_in: usize, h_in: usize, w_in: usize, x: &[f32],
    d_out: usize, h_out: usize, w_out: usize,
    kd: usize, kh: usize, kw: usize,
    samples: &[f32],     // [N, C, 3]
) -> (Vec<f32>, Vec<i64>) {
    let in_plane = h_in * w_in;
    let in_volume = d_in * in_plane;
    let mut y = vec![f32::NEG_INFINITY; n * c * d_out * h_out * w_out];
    let mut indices = vec![0i64; n * c * d_out * h_out * w_out];
    for ni in 0..n {
        for ci in 0..c {
            let alpha_d = samples[(ni * c + ci) * 3 + 0];
            let alpha_h = samples[(ni * c + ci) * 3 + 1];
            let alpha_w = samples[(ni * c + ci) * 3 + 2];
            for od in 0..d_out {
                let start_d = host_window_start(
                    od as i32, d_out as i32, d_in as i32, kd as i32, alpha_d,
                ) as usize;
                for oh in 0..h_out {
                    let start_h = host_window_start(
                        oh as i32, h_out as i32, h_in as i32, kh as i32, alpha_h,
                    ) as usize;
                    for ow in 0..w_out {
                        let start_w = host_window_start(
                            ow as i32, w_out as i32, w_in as i32, kw as i32, alpha_w,
                        ) as usize;
                        let mut best_v = f32::NEG_INFINITY;
                        let mut best_idx = (ni * c + ci) * in_volume
                            + start_d * in_plane
                            + start_h * w_in
                            + start_w;
                        for dd in 0..kd {
                            for dh in 0..kh {
                                for dw in 0..kw {
                                    let idx = (ni * c + ci) * in_volume
                                        + (start_d + dd) * in_plane
                                        + (start_h + dh) * w_in
                                        + start_w + dw;
                                    let v = x[idx];
                                    if v > best_v {
                                        best_v = v;
                                        best_idx = idx;
                                    }
                                }
                            }
                        }
                        let out_off = (((ni * c + ci) * d_out + od) * h_out + oh) * w_out + ow;
                        y[out_off] = best_v;
                        indices[out_off] = best_idx as i64;
                    }
                }
            }
        }
    }
    (y, indices)
}

/// Fill `[N, C, num_axes]` random samples via cuRAND uniform.
fn curand_uniform(
    ctx: &Context, stream: &Stream, n: i32, c: i32, num_axes: i32, seed: u64,
) -> DeviceBuffer<f32> {
    let total = (n * c * num_axes) as usize;
    let mut dev = DeviceBuffer::<f32>::zeros(ctx, total).expect("alloc samples");
    let shape = [total as i32];
    let stride = contiguous_stride(shape);
    let desc = RandomDescriptor {
        kind: RandomKind::Uniform,
        shape,
        element: ElementKind::F32,
        param1: 0.0,
        param2: 1.0,
        seed,
    };
    let plan = RandomPlan::<f32, 1>::select(stream, &desc, PlanPreference::default())
        .expect("RandomPlan select");
    let args = RandomArgs::<f32, 1> {
        y: TensorMut { data: dev.as_slice_mut(), shape, stride },
    };
    plan.run(stream, Workspace::None, args).expect("uniform run");
    dev
}

#[test]
#[ignore]
fn fmp2d_f32_4x4_to_3x3() {
    // 1×1×4×4 → 1×1×3×3 with kh=kw=2. Deterministic samples [0.0, 0.0]
    // — driver picks start_h = floor(oh * 2 / 2 + 0) ∈ {0, 1, 2} for
    // oh ∈ {0, 1, 2}, similarly for w; each output cell sees a distinct
    // 2×2 window.
    let (ctx, stream) = setup();
    let (n, c, h_in, w_in) = (1i32, 1i32, 4i32, 4i32);
    let (h_out, w_out, kh, kw) = (3i32, 3i32, 2i32, 2i32);
    let host_x: Vec<f32> = (1..=(h_in * w_in)).map(|v| v as f32).collect();
    // Two distinct sample sets to exercise both edge-α and mid-α.
    let host_samples: Vec<f32> = vec![0.0, 0.0];  // [N=1, C=1, 2]
    let (exp_y, exp_idx) = host_fmp2d_f32(
        n as usize, c as usize, h_in as usize, w_in as usize, &host_x,
        h_out as usize, w_out as usize, kh as usize, kw as usize, &host_samples);

    let numel_x = (n * c * h_in * w_in) as usize;
    let numel_y = (n * c * h_out * w_out) as usize;
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel_y).expect("y");
    let mut dev_indices: DeviceBuffer<i64> = DeviceBuffer::zeros(&ctx, numel_y).expect("indices");
    let dev_samples = DeviceBuffer::from_slice(&ctx, &host_samples).expect("up samples");

    let desc = FractionalMaxPool2dDescriptor::new(
        n, c, h_in, w_in, kh, kw, h_out, w_out, ElementKind::F32,
    );
    let plan = FractionalMaxPool2dPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let x_shape = [n, c, h_in, w_in];
    let y_shape = [n, c, h_out, w_out];
    let s_shape = [n, c, 2i32];

    plan.run_fw(&stream, Workspace::None, FractionalMaxPool2dFwArgs {
        x: TensorRef { data: dev_x.as_slice(), shape: x_shape, stride: contiguous_stride(x_shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: y_shape, stride: contiguous_stride(y_shape) },
        indices: TensorMut { data: dev_indices.as_slice_mut(), shape: y_shape, stride: contiguous_stride(y_shape) },
        random_samples: TensorRef { data: dev_samples.as_slice(), shape: s_shape, stride: contiguous_stride(s_shape) },
    }).expect("fw");
    stream.synchronize().expect("sync fw");

    let mut got_y = vec![0f32; numel_y];
    let mut got_idx = vec![0i64; numel_y];
    dev_y.copy_to_host(&mut got_y).expect("dl y");
    dev_indices.copy_to_host(&mut got_idx).expect("dl idx");
    for i in 0..numel_y {
        let tol = (exp_y[i].abs() * 1e-5).max(1e-5);
        assert!((got_y[i] - exp_y[i]).abs() <= tol,
            "fmp2d f32 y @ {i}: got={} want={}", got_y[i], exp_y[i]);
        assert_eq!(got_idx[i], exp_idx[i],
            "fmp2d f32 idx @ {i}: got={} want={}", got_idx[i], exp_idx[i]);
    }
    // Sanity: each output cell is a valid input value.
    for i in 0..numel_y {
        let idx = got_idx[i] as usize;
        assert!(idx < numel_x);
        assert!((host_x[idx] - got_y[i]).abs() <= 1e-6);
    }
}

#[test]
#[ignore]
fn fmp2d_f32_8x8_to_5x5_random_samples() {
    // 1×2×8×8 → 1×2×5×5 with kh=kw=3. Samples generated by cuRAND.
    // We only verify (a) every output value is in the input range, and
    // (b) every index is in [0, numel_x). The schedule is RNG-driven so
    // no exact equality.
    let (ctx, stream) = setup();
    let (n, c, h_in, w_in) = (1i32, 2i32, 8i32, 8i32);
    let (h_out, w_out, kh, kw) = (5i32, 5i32, 3i32, 3i32);
    let host_x: Vec<f32> = (0..(n * c * h_in * w_in))
        .map(|v| (v as f32) * 0.5 - 10.0)
        .collect();
    let numel_x = host_x.len();
    let numel_y = (n * c * h_out * w_out) as usize;

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel_y).expect("y");
    let mut dev_indices: DeviceBuffer<i64> = DeviceBuffer::zeros(&ctx, numel_y).expect("indices");
    let dev_samples = curand_uniform(&ctx, &stream, n, c, 2, 0xA1B2_C3D4_E5F6_7788);

    let desc = FractionalMaxPool2dDescriptor::new(
        n, c, h_in, w_in, kh, kw, h_out, w_out, ElementKind::F32,
    );
    let plan = FractionalMaxPool2dPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let x_shape = [n, c, h_in, w_in];
    let y_shape = [n, c, h_out, w_out];
    let s_shape = [n, c, 2i32];

    plan.run_fw(&stream, Workspace::None, FractionalMaxPool2dFwArgs {
        x: TensorRef { data: dev_x.as_slice(), shape: x_shape, stride: contiguous_stride(x_shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: y_shape, stride: contiguous_stride(y_shape) },
        indices: TensorMut { data: dev_indices.as_slice_mut(), shape: y_shape, stride: contiguous_stride(y_shape) },
        random_samples: TensorRef { data: dev_samples.as_slice(), shape: s_shape, stride: contiguous_stride(s_shape) },
    }).expect("fw");
    stream.synchronize().expect("sync fw");

    let mut got_y = vec![0f32; numel_y];
    let mut got_idx = vec![0i64; numel_y];
    dev_y.copy_to_host(&mut got_y).expect("dl y");
    dev_indices.copy_to_host(&mut got_idx).expect("dl idx");

    let x_min = host_x.iter().cloned().fold(f32::INFINITY, f32::min);
    let x_max = host_x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    for i in 0..numel_y {
        let idx = got_idx[i];
        assert!(idx >= 0 && (idx as usize) < numel_x,
            "fmp2d rand idx @ {i}: {idx} out of range");
        assert!(got_y[i].is_finite() && got_y[i] >= x_min && got_y[i] <= x_max,
            "fmp2d rand y @ {i}: {} out of input range [{}, {}]", got_y[i], x_min, x_max);
        // Each output cell equals the input value at its argmax index.
        let expected = host_x[idx as usize];
        assert!((got_y[i] - expected).abs() <= 1e-6,
            "fmp2d rand y @ {i}: got={} expected x[{}]={}", got_y[i], idx, expected);
    }
}

#[test]
#[ignore]
fn fmp3d_f32_4x4x4_to_3x3x3() {
    // 1×1×4×4×4 → 1×1×3×3×3 with kd=kh=kw=2.
    let (ctx, stream) = setup();
    let (n, c, d_in, h_in, w_in) = (1i32, 1i32, 4i32, 4i32, 4i32);
    let (d_out, h_out, w_out, kd, kh, kw) = (3i32, 3i32, 3i32, 2i32, 2i32, 2i32);
    let numel_x = (n * c * d_in * h_in * w_in) as usize;
    let numel_y = (n * c * d_out * h_out * w_out) as usize;
    let host_x: Vec<f32> = (1..=numel_x as i32).map(|v| v as f32).collect();
    // α = [0.5, 0.25, 0.75] — mid / lower / upper.
    let host_samples: Vec<f32> = vec![0.5, 0.25, 0.75];
    let (exp_y, exp_idx) = host_fmp3d_f32(
        n as usize, c as usize, d_in as usize, h_in as usize, w_in as usize, &host_x,
        d_out as usize, h_out as usize, w_out as usize,
        kd as usize, kh as usize, kw as usize, &host_samples);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel_y).expect("y");
    let mut dev_indices: DeviceBuffer<i64> = DeviceBuffer::zeros(&ctx, numel_y).expect("indices");
    let dev_samples = DeviceBuffer::from_slice(&ctx, &host_samples).expect("up samples");

    let desc = FractionalMaxPool3dDescriptor::new(
        n, c, d_in, h_in, w_in, kd, kh, kw, d_out, h_out, w_out, ElementKind::F32,
    );
    let plan = FractionalMaxPool3dPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let x_shape = [n, c, d_in, h_in, w_in];
    let y_shape = [n, c, d_out, h_out, w_out];
    let s_shape = [n, c, 3i32];

    plan.run_fw(&stream, Workspace::None, FractionalMaxPool3dFwArgs {
        x: TensorRef { data: dev_x.as_slice(), shape: x_shape, stride: contiguous_stride(x_shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: y_shape, stride: contiguous_stride(y_shape) },
        indices: TensorMut { data: dev_indices.as_slice_mut(), shape: y_shape, stride: contiguous_stride(y_shape) },
        random_samples: TensorRef { data: dev_samples.as_slice(), shape: s_shape, stride: contiguous_stride(s_shape) },
    }).expect("fw");
    stream.synchronize().expect("sync fw");

    let mut got_y = vec![0f32; numel_y];
    let mut got_idx = vec![0i64; numel_y];
    dev_y.copy_to_host(&mut got_y).expect("dl y");
    dev_indices.copy_to_host(&mut got_idx).expect("dl idx");
    for i in 0..numel_y {
        let tol = (exp_y[i].abs() * 1e-5).max(1e-5);
        assert!((got_y[i] - exp_y[i]).abs() <= tol,
            "fmp3d f32 y @ {i}: got={} want={}", got_y[i], exp_y[i]);
        assert_eq!(got_idx[i], exp_idx[i],
            "fmp3d f32 idx @ {i}: got={} want={}", got_idx[i], exp_idx[i]);
    }
}

#[test]
#[ignore]
fn fmp2d_bw_f32_4x4_to_3x3() {
    // BW round-trip: run FW to populate y + indices, then BW with a
    // distinct dy, and verify dx matches a host scatter-via-indices
    // reference (which is straightforward — atomicAdd order doesn't
    // matter on f32 sums of small distinct contributions; we just
    // accumulate naively on the host).
    let (ctx, stream) = setup();
    let (n, c, h_in, w_in) = (1i32, 1i32, 4i32, 4i32);
    let (h_out, w_out, kh, kw) = (3i32, 3i32, 2i32, 2i32);
    let numel_x = (n * c * h_in * w_in) as usize;
    let numel_y = (n * c * h_out * w_out) as usize;
    let host_x: Vec<f32> = (1..=(h_in * w_in)).map(|v| v as f32).collect();
    let host_samples: Vec<f32> = vec![0.5, 0.5];
    let (_, exp_idx) = host_fmp2d_f32(
        n as usize, c as usize, h_in as usize, w_in as usize, &host_x,
        h_out as usize, w_out as usize, kh as usize, kw as usize, &host_samples);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel_y).expect("y");
    let mut dev_indices: DeviceBuffer<i64> = DeviceBuffer::zeros(&ctx, numel_y).expect("indices");
    let dev_samples = DeviceBuffer::from_slice(&ctx, &host_samples).expect("up samples");

    let desc = FractionalMaxPool2dDescriptor::new(
        n, c, h_in, w_in, kh, kw, h_out, w_out, ElementKind::F32,
    );
    let plan = FractionalMaxPool2dPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let x_shape = [n, c, h_in, w_in];
    let y_shape = [n, c, h_out, w_out];
    let s_shape = [n, c, 2i32];

    plan.run_fw(&stream, Workspace::None, FractionalMaxPool2dFwArgs {
        x: TensorRef { data: dev_x.as_slice(), shape: x_shape, stride: contiguous_stride(x_shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: y_shape, stride: contiguous_stride(y_shape) },
        indices: TensorMut { data: dev_indices.as_slice_mut(), shape: y_shape, stride: contiguous_stride(y_shape) },
        random_samples: TensorRef { data: dev_samples.as_slice(), shape: s_shape, stride: contiguous_stride(s_shape) },
    }).expect("fw");
    stream.synchronize().expect("sync fw");

    // BW round-trip
    let host_dy: Vec<f32> = (0..numel_y).map(|i| 0.25 + (i as f32) * 0.5).collect();
    let mut exp_dx = vec![0f32; numel_x];
    for i in 0..numel_y {
        let idx = exp_idx[i] as usize;
        exp_dx[idx] += host_dy[i];
    }
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel_x).expect("dx");

    plan.run_bw(&stream, Workspace::None, FractionalMaxPool2dBwArgs {
        dy: TensorRef { data: dev_dy.as_slice(), shape: y_shape, stride: contiguous_stride(y_shape) },
        indices: TensorRef { data: dev_indices.as_slice(), shape: y_shape, stride: contiguous_stride(y_shape) },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape: x_shape, stride: contiguous_stride(x_shape) },
    }).expect("bw");
    stream.synchronize().expect("sync bw");

    let mut got_dx = vec![0f32; numel_x];
    dev_dx.copy_to_host(&mut got_dx).expect("dl dx");
    for i in 0..numel_x {
        let tol = (exp_dx[i].abs() * 1e-5).max(1e-5);
        assert!((got_dx[i] - exp_dx[i]).abs() <= tol,
            "fmp2d bw f32 dx @ {i}: got={} want={}", got_dx[i], exp_dx[i]);
    }
}
