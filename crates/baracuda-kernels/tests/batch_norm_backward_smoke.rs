//! Real-GPU smoke test for `BatchNormBackwardPlan`.
//!
//! Covers all four FP dtypes (f32 / f16 / bf16 / f64).
//!
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, BatchNormBackwardArgs, BatchNormBackwardDescriptor,
    BatchNormBackwardPlan, ElementKind, PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

const F16_EPS: f32 = 9.77e-4;
const BF16_EPS: f32 = 7.81e-3;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// CPU reference for BN BW. Returns `(dx, dgamma, dbeta)`.
fn host_bn_bw_f32(
    n: usize, c: usize, s: usize, dy: &[f32], x: &[f32],
    gamma: Option<&[f32]>, saved_mean: &[f32], saved_rstd: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let m = (n * s) as f64;
    let mut sum_dxh = vec![0f64; c];
    let mut sum_dxhxh = vec![0f64; c];
    for ni in 0..n {
        for ci in 0..c {
            let mean = saved_mean[ci] as f64;
            let rstd = saved_rstd[ci] as f64;
            let gv = gamma.map(|g| g[ci] as f64).unwrap_or(1.0);
            for si in 0..s {
                let xv = x[ni * c * s + ci * s + si] as f64;
                let dy_v = dy[ni * c * s + ci * s + si] as f64;
                let dxh = dy_v * gv;
                let xh = (xv - mean) * rstd;
                sum_dxh[ci] += dxh;
                sum_dxhxh[ci] += dxh * xh;
            }
        }
    }
    let mut dx = vec![0f32; n * c * s];
    for ni in 0..n {
        for ci in 0..c {
            let mean = saved_mean[ci] as f64;
            let rstd = saved_rstd[ci] as f64;
            let gv = gamma.map(|g| g[ci] as f64).unwrap_or(1.0);
            for si in 0..s {
                let xv = x[ni * c * s + ci * s + si] as f64;
                let dy_v = dy[ni * c * s + ci * s + si] as f64;
                let dxh = dy_v * gv;
                let xh = (xv - mean) * rstd;
                let v = rstd * (dxh - sum_dxh[ci] / m - xh * sum_dxhxh[ci] / m);
                dx[ni * c * s + ci * s + si] = v as f32;
            }
        }
    }
    // dgamma[c] = sum dy * x_hat; dbeta[c] = sum dy
    let mut dgamma = vec![0f64; c];
    let mut dbeta = vec![0f64; c];
    for ni in 0..n {
        for ci in 0..c {
            let mean = saved_mean[ci] as f64;
            let rstd = saved_rstd[ci] as f64;
            for si in 0..s {
                let xv = x[ni * c * s + ci * s + si] as f64;
                let dy_v = dy[ni * c * s + ci * s + si] as f64;
                let xh = (xv - mean) * rstd;
                dgamma[ci] += dy_v * xh;
                dbeta[ci] += dy_v;
            }
        }
    }
    (
        dx,
        dgamma.into_iter().map(|v| v as f32).collect(),
        dbeta.into_iter().map(|v| v as f32).collect(),
    )
}

/// Compute saved_mean / saved_rstd from x at f32 precision (matches the
/// training-mode FW that would have produced these saves).
fn compute_saves_f32(
    n: usize, c: usize, s: usize, x: &[f32], eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let m = (n * s) as f64;
    let mut sums = vec![0f64; c];
    let mut sqs = vec![0f64; c];
    for ni in 0..n {
        for ci in 0..c {
            for si in 0..s {
                let v = x[ni * c * s + ci * s + si] as f64;
                sums[ci] += v;
                sqs[ci] += v * v;
            }
        }
    }
    let mut saved_mean = vec![0f32; c];
    let mut saved_rstd = vec![0f32; c];
    for ci in 0..c {
        let mean = sums[ci] / m;
        let var = (sqs[ci] / m - mean * mean).max(0.0);
        saved_mean[ci] = mean as f32;
        saved_rstd[ci] = (1.0 / (var + eps as f64).sqrt()) as f32;
    }
    (saved_mean, saved_rstd)
}

#[test]
#[ignore]
fn batch_norm_bw_f32_with_affine() {
    let (ctx, stream) = setup();
    let n = 3i32;
    let c = 4i32;
    let h = 3i32;
    let w = 3i32;
    let shape = [n, c, h, w];
    let s = (h * w) as usize;
    let numel = (n * c * h * w) as usize;
    let host_x: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.13 + 0.5).sin() * 0.9)
        .collect();
    let host_dy: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.07 - 1.1).cos() * 0.4)
        .collect();
    let host_gamma: Vec<f32> = (0..c).map(|i| 0.7 + 0.1 * i as f32).collect();
    let (saved_mean, saved_rstd) =
        compute_saves_f32(n as usize, c as usize, s, &host_x, 1e-5);

    let (exp_dx, exp_dgamma, exp_dbeta) = host_bn_bw_f32(
        n as usize, c as usize, s, &host_dy, &host_x,
        Some(&host_gamma), &saved_mean, &saved_rstd,
    );

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_g = DeviceBuffer::from_slice(&ctx, &host_gamma).expect("up gamma");
    let dev_sm = DeviceBuffer::from_slice(&ctx, &saved_mean).expect("up mean");
    let dev_sr = DeviceBuffer::from_slice(&ctx, &saved_rstd).expect("up rstd");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("dx");
    let mut dev_dg: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, c as usize).expect("dg");
    let mut dev_db: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, c as usize).expect("db");
    let ws_bytes = 2 * c as usize * core::mem::size_of::<f32>();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("ws");

    let desc = BatchNormBackwardDescriptor::<4> {
        input_shape: shape,
        channel_axis: 1,
        has_affine: true,
        element: ElementKind::F32,
    };
    let plan = BatchNormBackwardPlan::<f32, 4>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        BatchNormBackwardArgs {
            dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
            x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            gamma: Some(TensorRef { data: dev_g.as_slice(), shape: [c], stride: [1] }),
            saved_mean: TensorRef { data: dev_sm.as_slice(), shape: [c], stride: [1] },
            saved_rstd: TensorRef { data: dev_sr.as_slice(), shape: [c], stride: [1] },
            dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
            dgamma: Some(TensorMut { data: dev_dg.as_slice_mut(), shape: [c], stride: [1] }),
            dbeta: Some(TensorMut { data: dev_db.as_slice_mut(), shape: [c], stride: [1] }),
        },
    ).expect("run");
    stream.synchronize().expect("sync");

    let mut got_dx = vec![0f32; numel];
    let mut got_dg = vec![0f32; c as usize];
    let mut got_db = vec![0f32; c as usize];
    dev_dx.copy_to_host(&mut got_dx).expect("dl dx");
    dev_dg.copy_to_host(&mut got_dg).expect("dl dg");
    dev_db.copy_to_host(&mut got_db).expect("dl db");

    let eps_tol = 64.0 * f32::EPSILON;
    for i in 0..numel {
        let tol = (exp_dx[i].abs() * eps_tol).max(eps_tol);
        assert!((got_dx[i] - exp_dx[i]).abs() <= tol,
            "bn bw f32 dx @ {i}: got={} want={}", got_dx[i], exp_dx[i]);
    }
    for i in 0..c as usize {
        let tol = (exp_dgamma[i].abs() * eps_tol).max(eps_tol);
        assert!((got_dg[i] - exp_dgamma[i]).abs() <= tol,
            "bn bw f32 dgamma @ {i}: got={} want={}", got_dg[i], exp_dgamma[i]);
        let tol2 = (exp_dbeta[i].abs() * eps_tol).max(eps_tol);
        assert!((got_db[i] - exp_dbeta[i]).abs() <= tol2,
            "bn bw f32 dbeta @ {i}: got={} want={}", got_db[i], exp_dbeta[i]);
    }
}

#[test]
#[ignore]
fn batch_norm_bw_f64_with_affine() {
    let (ctx, stream) = setup();
    let n = 3i32;
    let c = 4i32;
    let h = 3i32;
    let w = 3i32;
    let shape = [n, c, h, w];
    let s = (h * w) as usize;
    let numel = (n * c * h * w) as usize;
    let host_x_f32: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.13 + 0.5).sin() * 0.9)
        .collect();
    let host_dy_f32: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.07 - 1.1).cos() * 0.4)
        .collect();
    let host_gamma_f32: Vec<f32> = (0..c).map(|i| 0.7 + 0.1 * i as f32).collect();
    let (saved_mean_f32, saved_rstd_f32) =
        compute_saves_f32(n as usize, c as usize, s, &host_x_f32, 1e-5);

    let (exp_dx_f32, exp_dgamma_f32, exp_dbeta_f32) = host_bn_bw_f32(
        n as usize, c as usize, s, &host_dy_f32, &host_x_f32,
        Some(&host_gamma_f32), &saved_mean_f32, &saved_rstd_f32,
    );

    let host_x: Vec<f64> = host_x_f32.iter().map(|&v| v as f64).collect();
    let host_dy: Vec<f64> = host_dy_f32.iter().map(|&v| v as f64).collect();
    let host_gamma: Vec<f64> = host_gamma_f32.iter().map(|&v| v as f64).collect();
    let saved_mean: Vec<f64> = saved_mean_f32.iter().map(|&v| v as f64).collect();
    let saved_rstd: Vec<f64> = saved_rstd_f32.iter().map(|&v| v as f64).collect();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_g = DeviceBuffer::from_slice(&ctx, &host_gamma).expect("up gamma");
    let dev_sm = DeviceBuffer::from_slice(&ctx, &saved_mean).expect("up mean");
    let dev_sr = DeviceBuffer::from_slice(&ctx, &saved_rstd).expect("up rstd");
    let mut dev_dx: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("dx");
    let mut dev_dg: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, c as usize).expect("dg");
    let mut dev_db: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, c as usize).expect("db");
    let ws_bytes = 2 * c as usize * core::mem::size_of::<f64>();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("ws");

    let desc = BatchNormBackwardDescriptor::<4> {
        input_shape: shape,
        channel_axis: 1,
        has_affine: true,
        element: ElementKind::F64,
    };
    let plan = BatchNormBackwardPlan::<f64, 4>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        BatchNormBackwardArgs {
            dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
            x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            gamma: Some(TensorRef { data: dev_g.as_slice(), shape: [c], stride: [1] }),
            saved_mean: TensorRef { data: dev_sm.as_slice(), shape: [c], stride: [1] },
            saved_rstd: TensorRef { data: dev_sr.as_slice(), shape: [c], stride: [1] },
            dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
            dgamma: Some(TensorMut { data: dev_dg.as_slice_mut(), shape: [c], stride: [1] }),
            dbeta: Some(TensorMut { data: dev_db.as_slice_mut(), shape: [c], stride: [1] }),
        },
    ).expect("run");
    stream.synchronize().expect("sync");

    let mut got_dx = vec![0f64; numel];
    let mut got_dg = vec![0f64; c as usize];
    let mut got_db = vec![0f64; c as usize];
    dev_dx.copy_to_host(&mut got_dx).expect("dl dx");
    dev_dg.copy_to_host(&mut got_dg).expect("dl dg");
    dev_db.copy_to_host(&mut got_db).expect("dl db");

    // f32 reference upcast to f64 — widen by f32 eps.
    let eps_tol = 64.0 * f32::EPSILON as f64;
    for i in 0..numel {
        let want = exp_dx_f32[i] as f64;
        let tol = (want.abs() * eps_tol).max(eps_tol);
        assert!((got_dx[i] - want).abs() <= tol,
            "bn bw f64 dx @ {i}: got={} want={}", got_dx[i], want);
    }
    for i in 0..c as usize {
        let want_g = exp_dgamma_f32[i] as f64;
        let tol = (want_g.abs() * eps_tol).max(eps_tol);
        assert!((got_dg[i] - want_g).abs() <= tol,
            "bn bw f64 dgamma @ {i}");
        let want_b = exp_dbeta_f32[i] as f64;
        let tol2 = (want_b.abs() * eps_tol).max(eps_tol);
        assert!((got_db[i] - want_b).abs() <= tol2,
            "bn bw f64 dbeta @ {i}");
    }
}

#[test]
#[ignore]
fn batch_norm_bw_f16_with_affine() {
    let (ctx, stream) = setup();
    let n = 3i32;
    let c = 4i32;
    let h = 3i32;
    let w = 3i32;
    let shape = [n, c, h, w];
    let s = (h * w) as usize;
    let numel = (n * c * h * w) as usize;
    let host_x_f32: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.13 + 0.5).sin() * 0.9)
        .collect();
    let host_dy_f32: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.07 - 1.1).cos() * 0.4)
        .collect();
    let host_gamma_f32: Vec<f32> = (0..c).map(|i| 0.7 + 0.1 * i as f32).collect();
    let (saved_mean_f32, saved_rstd_f32) =
        compute_saves_f32(n as usize, c as usize, s, &host_x_f32, 1e-5);

    let (exp_dx_f32, exp_dgamma_f32, exp_dbeta_f32) = host_bn_bw_f32(
        n as usize, c as usize, s, &host_dy_f32, &host_x_f32,
        Some(&host_gamma_f32), &saved_mean_f32, &saved_rstd_f32,
    );

    let host_x: Vec<f16> = host_x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_dy: Vec<f16> = host_dy_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_gamma: Vec<f16> = host_gamma_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let saved_mean: Vec<f16> = saved_mean_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let saved_rstd: Vec<f16> = saved_rstd_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_g = DeviceBuffer::from_slice(&ctx, &host_gamma).expect("up gamma");
    let dev_sm = DeviceBuffer::from_slice(&ctx, &saved_mean).expect("up mean");
    let dev_sr = DeviceBuffer::from_slice(&ctx, &saved_rstd).expect("up rstd");
    let mut dev_dx: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("dx");
    let mut dev_dg: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, c as usize).expect("dg");
    let mut dev_db: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, c as usize).expect("db");
    let ws_bytes = 2 * c as usize * core::mem::size_of::<f32>();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("ws");

    let desc = BatchNormBackwardDescriptor::<4> {
        input_shape: shape,
        channel_axis: 1,
        has_affine: true,
        element: ElementKind::F16,
    };
    let plan = BatchNormBackwardPlan::<f16, 4>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        BatchNormBackwardArgs {
            dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
            x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            gamma: Some(TensorRef { data: dev_g.as_slice(), shape: [c], stride: [1] }),
            saved_mean: TensorRef { data: dev_sm.as_slice(), shape: [c], stride: [1] },
            saved_rstd: TensorRef { data: dev_sr.as_slice(), shape: [c], stride: [1] },
            dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
            dgamma: Some(TensorMut { data: dev_dg.as_slice_mut(), shape: [c], stride: [1] }),
            dbeta: Some(TensorMut { data: dev_db.as_slice_mut(), shape: [c], stride: [1] }),
        },
    ).expect("run");
    stream.synchronize().expect("sync");

    let mut got_dx = vec![f16::ZERO; numel];
    let mut got_dg = vec![f16::ZERO; c as usize];
    let mut got_db = vec![f16::ZERO; c as usize];
    dev_dx.copy_to_host(&mut got_dx).expect("dl dx");
    dev_dg.copy_to_host(&mut got_dg).expect("dl dg");
    dev_db.copy_to_host(&mut got_db).expect("dl db");

    // ~16 ULP-equivalent for BW; aggregations + chain rule drift more.
    let eps_tol = 16.0 * F16_EPS;
    for i in 0..numel {
        let tol = (exp_dx_f32[i].abs() * eps_tol).max(eps_tol);
        let diff = (got_dx[i].to_f32() - exp_dx_f32[i]).abs();
        assert!(diff <= tol,
            "bn bw f16 dx @ {i}: diff={diff} got={} want={}",
            got_dx[i].to_f32(), exp_dx_f32[i]);
    }
    for i in 0..c as usize {
        let tol = (exp_dgamma_f32[i].abs() * eps_tol).max(eps_tol);
        let diff = (got_dg[i].to_f32() - exp_dgamma_f32[i]).abs();
        assert!(diff <= tol, "bn bw f16 dgamma @ {i}: diff={diff}");
        let tol2 = (exp_dbeta_f32[i].abs() * eps_tol).max(eps_tol);
        let diff2 = (got_db[i].to_f32() - exp_dbeta_f32[i]).abs();
        assert!(diff2 <= tol2, "bn bw f16 dbeta @ {i}: diff={diff2}");
    }
}

#[test]
#[ignore]
fn batch_norm_bw_bf16_with_affine() {
    let (ctx, stream) = setup();
    let n = 3i32;
    let c = 4i32;
    let h = 3i32;
    let w = 3i32;
    let shape = [n, c, h, w];
    let s = (h * w) as usize;
    let numel = (n * c * h * w) as usize;
    let host_x_f32: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.13 + 0.5).sin() * 0.9)
        .collect();
    let host_dy_f32: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.07 - 1.1).cos() * 0.4)
        .collect();
    let host_gamma_f32: Vec<f32> = (0..c).map(|i| 0.7 + 0.1 * i as f32).collect();
    let (saved_mean_f32, saved_rstd_f32) =
        compute_saves_f32(n as usize, c as usize, s, &host_x_f32, 1e-5);

    let (exp_dx_f32, exp_dgamma_f32, exp_dbeta_f32) = host_bn_bw_f32(
        n as usize, c as usize, s, &host_dy_f32, &host_x_f32,
        Some(&host_gamma_f32), &saved_mean_f32, &saved_rstd_f32,
    );

    let host_x: Vec<bf16> = host_x_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_dy: Vec<bf16> = host_dy_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_gamma: Vec<bf16> = host_gamma_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let saved_mean: Vec<bf16> = saved_mean_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let saved_rstd: Vec<bf16> = saved_rstd_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_g = DeviceBuffer::from_slice(&ctx, &host_gamma).expect("up gamma");
    let dev_sm = DeviceBuffer::from_slice(&ctx, &saved_mean).expect("up mean");
    let dev_sr = DeviceBuffer::from_slice(&ctx, &saved_rstd).expect("up rstd");
    let mut dev_dx: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("dx");
    let mut dev_dg: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, c as usize).expect("dg");
    let mut dev_db: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, c as usize).expect("db");
    let ws_bytes = 2 * c as usize * core::mem::size_of::<f32>();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("ws");

    let desc = BatchNormBackwardDescriptor::<4> {
        input_shape: shape,
        channel_axis: 1,
        has_affine: true,
        element: ElementKind::Bf16,
    };
    let plan = BatchNormBackwardPlan::<bf16, 4>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        BatchNormBackwardArgs {
            dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
            x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            gamma: Some(TensorRef { data: dev_g.as_slice(), shape: [c], stride: [1] }),
            saved_mean: TensorRef { data: dev_sm.as_slice(), shape: [c], stride: [1] },
            saved_rstd: TensorRef { data: dev_sr.as_slice(), shape: [c], stride: [1] },
            dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
            dgamma: Some(TensorMut { data: dev_dg.as_slice_mut(), shape: [c], stride: [1] }),
            dbeta: Some(TensorMut { data: dev_db.as_slice_mut(), shape: [c], stride: [1] }),
        },
    ).expect("run");
    stream.synchronize().expect("sync");

    let mut got_dx = vec![bf16::ZERO; numel];
    let mut got_dg = vec![bf16::ZERO; c as usize];
    let mut got_db = vec![bf16::ZERO; c as usize];
    dev_dx.copy_to_host(&mut got_dx).expect("dl dx");
    dev_dg.copy_to_host(&mut got_dg).expect("dl dg");
    dev_db.copy_to_host(&mut got_db).expect("dl db");

    // ~16 ULP-equivalent for BW; aggregations + chain rule drift more.
    let eps_tol = 16.0 * BF16_EPS;
    for i in 0..numel {
        let tol = (exp_dx_f32[i].abs() * eps_tol).max(eps_tol);
        let diff = (got_dx[i].to_f32() - exp_dx_f32[i]).abs();
        assert!(diff <= tol,
            "bn bw bf16 dx @ {i}: diff={diff} got={} want={}",
            got_dx[i].to_f32(), exp_dx_f32[i]);
    }
    for i in 0..c as usize {
        let tol = (exp_dgamma_f32[i].abs() * eps_tol).max(eps_tol);
        let diff = (got_dg[i].to_f32() - exp_dgamma_f32[i]).abs();
        assert!(diff <= tol, "bn bw bf16 dgamma @ {i}: diff={diff}");
        let tol2 = (exp_dbeta_f32[i].abs() * eps_tol).max(eps_tol);
        let diff2 = (got_db[i].to_f32() - exp_dbeta_f32[i]).abs();
        assert!(diff2 <= tol2, "bn bw bf16 dbeta @ {i}: diff={diff2}");
    }
}
