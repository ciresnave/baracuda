//! Real-GPU smoke test for `BatchNormPlan` FW (training mode).
//!
//! Covers all four FP dtypes (f32 / f16 / bf16 / f64).
//!
//! `#[ignore]` by default — requires a real CUDA device.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, BatchNormArgs, BatchNormDescriptor, BatchNormPlan, ElementKind,
    PlanPreference, TensorMut, TensorRef, Workspace,
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

/// CPU reference for BatchNorm training-mode FW on `[N, C, S]` row-major
/// memory. Returns `(y, mean_per_chan, rstd_per_chan)`.
fn host_batch_norm_f32(
    n: usize, c: usize, s: usize, x: &[f32],
    gamma: Option<&[f32]>, beta: Option<&[f32]>, eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut mean = vec![0f64; c];
    let mut sumsq = vec![0f64; c];
    let m = (n * s) as f64;
    for ni in 0..n {
        for ci in 0..c {
            for si in 0..s {
                let v = x[ni * c * s + ci * s + si] as f64;
                mean[ci] += v;
                sumsq[ci] += v * v;
            }
        }
    }
    for ci in 0..c {
        mean[ci] /= m;
    }
    let mut rstd = vec![0f64; c];
    for ci in 0..c {
        let var = sumsq[ci] / m - mean[ci] * mean[ci];
        let var = var.max(0.0);
        rstd[ci] = 1.0 / (var + eps as f64).sqrt();
    }
    let mut y = vec![0f32; n * c * s];
    for ni in 0..n {
        for ci in 0..c {
            for si in 0..s {
                let xv = x[ni * c * s + ci * s + si] as f64;
                let xh = (xv - mean[ci]) * rstd[ci];
                let gv = gamma.map(|g| g[ci] as f64).unwrap_or(1.0);
                let bv = beta.map(|b| b[ci] as f64).unwrap_or(0.0);
                y[ni * c * s + ci * s + si] = (xh * gv + bv) as f32;
            }
        }
    }
    (
        y,
        mean.into_iter().map(|v| v as f32).collect(),
        rstd.into_iter().map(|v| v as f32).collect(),
    )
}

#[test]
#[ignore]
fn batch_norm_f32_with_affine() {
    let (ctx, stream) = setup();
    let n = 4i32;
    let c = 6i32;
    let h = 3i32;
    let w = 3i32;
    let shape = [n, c, h, w];
    let s = (h * w) as usize;
    let numel = (n * c * h * w) as usize;
    let host_x: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.07 - 1.2).sin() * 1.3)
        .collect();
    let host_gamma: Vec<f32> = (0..c).map(|i| 0.8 + 0.07 * i as f32).collect();
    let host_beta: Vec<f32> = (0..c).map(|i| 0.1 * i as f32 - 0.2).collect();
    let eps = 1e-5f32;

    let (exp_y, exp_mean, exp_rstd) = host_batch_norm_f32(
        n as usize, c as usize, s, &host_x,
        Some(&host_gamma), Some(&host_beta), eps,
    );

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_g = DeviceBuffer::from_slice(&ctx, &host_gamma).expect("up gamma");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_beta).expect("up beta");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("y");
    let mut dev_mean: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, c as usize).expect("mean");
    let mut dev_rstd: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, c as usize).expect("rstd");

    let desc = BatchNormDescriptor::<4> {
        input_shape: shape,
        channel_axis: 1,
        eps,
        has_affine: true,
        element: ElementKind::F32,
    };
    let plan = BatchNormPlan::<f32, 4>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run(&stream, Workspace::None, BatchNormArgs {
        x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
        gamma: Some(TensorRef { data: dev_g.as_slice(), shape: [c], stride: [1] }),
        beta: Some(TensorRef { data: dev_b.as_slice(), shape: [c], stride: [1] }),
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        saved_mean: TensorMut { data: dev_mean.as_slice_mut(), shape: [c], stride: [1] },
        saved_rstd: TensorMut { data: dev_rstd.as_slice_mut(), shape: [c], stride: [1] },
    }).expect("run");
    stream.synchronize().expect("sync");

    let mut got_y = vec![0f32; numel];
    let mut got_mean = vec![0f32; c as usize];
    let mut got_rstd = vec![0f32; c as usize];
    dev_y.copy_to_host(&mut got_y).expect("dl y");
    dev_mean.copy_to_host(&mut got_mean).expect("dl mean");
    dev_rstd.copy_to_host(&mut got_rstd).expect("dl rstd");

    let eps_tol = 32.0 * f32::EPSILON;
    for i in 0..numel {
        let tol = (exp_y[i].abs() * eps_tol).max(eps_tol);
        assert!((got_y[i] - exp_y[i]).abs() <= tol,
            "bn f32 y @ {i}: got={} want={}", got_y[i], exp_y[i]);
    }
    for i in 0..c as usize {
        let tol = (exp_mean[i].abs() * eps_tol).max(eps_tol);
        assert!((got_mean[i] - exp_mean[i]).abs() <= tol, "mean @ {i}");
        let tol2 = (exp_rstd[i].abs() * eps_tol).max(eps_tol);
        assert!((got_rstd[i] - exp_rstd[i]).abs() <= tol2, "rstd @ {i}");
    }
}

#[test]
#[ignore]
fn batch_norm_f32_no_affine() {
    let (ctx, stream) = setup();
    let n = 3i32;
    let c = 4i32;
    let h = 2i32;
    let w = 2i32;
    let shape = [n, c, h, w];
    let s = (h * w) as usize;
    let numel = (n * c * h * w) as usize;
    let host_x: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.11 - 0.5).cos() * 0.7)
        .collect();
    let eps = 1e-5f32;

    let (exp_y, _, _) = host_batch_norm_f32(
        n as usize, c as usize, s, &host_x, None, None, eps,
    );

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("y");
    let mut dev_mean: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, c as usize).expect("mean");
    let mut dev_rstd: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, c as usize).expect("rstd");

    let desc = BatchNormDescriptor::<4> {
        input_shape: shape,
        channel_axis: 1,
        eps,
        has_affine: false,
        element: ElementKind::F32,
    };
    let plan = BatchNormPlan::<f32, 4>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run(&stream, Workspace::None, BatchNormArgs {
        x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
        gamma: None,
        beta: None,
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        saved_mean: TensorMut { data: dev_mean.as_slice_mut(), shape: [c], stride: [1] },
        saved_rstd: TensorMut { data: dev_rstd.as_slice_mut(), shape: [c], stride: [1] },
    }).expect("run");
    stream.synchronize().expect("sync");

    let mut got_y = vec![0f32; numel];
    dev_y.copy_to_host(&mut got_y).expect("dl");
    let eps_tol = 32.0 * f32::EPSILON;
    for i in 0..numel {
        let tol = (exp_y[i].abs() * eps_tol).max(eps_tol);
        assert!((got_y[i] - exp_y[i]).abs() <= tol,
            "bn f32 no-affine y @ {i}: got={} want={}", got_y[i], exp_y[i]);
    }
}

#[test]
#[ignore]
fn batch_norm_f64_with_affine() {
    let (ctx, stream) = setup();
    let n = 4i32;
    let c = 6i32;
    let h = 3i32;
    let w = 3i32;
    let shape = [n, c, h, w];
    let s = (h * w) as usize;
    let numel = (n * c * h * w) as usize;
    let host_x_f32: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.07 - 1.2).sin() * 1.3)
        .collect();
    let host_gamma_f32: Vec<f32> = (0..c).map(|i| 0.8 + 0.07 * i as f32).collect();
    let host_beta_f32: Vec<f32> = (0..c).map(|i| 0.1 * i as f32 - 0.2).collect();
    let eps = 1e-5f32;

    let (exp_y_f32, exp_mean_f32, exp_rstd_f32) = host_batch_norm_f32(
        n as usize, c as usize, s, &host_x_f32,
        Some(&host_gamma_f32), Some(&host_beta_f32), eps,
    );
    let host_x: Vec<f64> = host_x_f32.iter().map(|&v| v as f64).collect();
    let host_gamma: Vec<f64> = host_gamma_f32.iter().map(|&v| v as f64).collect();
    let host_beta: Vec<f64> = host_beta_f32.iter().map(|&v| v as f64).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_g = DeviceBuffer::from_slice(&ctx, &host_gamma).expect("up gamma");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_beta).expect("up beta");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("y");
    let mut dev_mean: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, c as usize).expect("mean");
    let mut dev_rstd: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, c as usize).expect("rstd");

    let desc = BatchNormDescriptor::<4> {
        input_shape: shape,
        channel_axis: 1,
        eps,
        has_affine: true,
        element: ElementKind::F64,
    };
    let plan = BatchNormPlan::<f64, 4>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run(&stream, Workspace::None, BatchNormArgs {
        x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
        gamma: Some(TensorRef { data: dev_g.as_slice(), shape: [c], stride: [1] }),
        beta: Some(TensorRef { data: dev_b.as_slice(), shape: [c], stride: [1] }),
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        saved_mean: TensorMut { data: dev_mean.as_slice_mut(), shape: [c], stride: [1] },
        saved_rstd: TensorMut { data: dev_rstd.as_slice_mut(), shape: [c], stride: [1] },
    }).expect("run");
    stream.synchronize().expect("sync");

    let mut got_y = vec![0f64; numel];
    let mut got_mean = vec![0f64; c as usize];
    let mut got_rstd = vec![0f64; c as usize];
    dev_y.copy_to_host(&mut got_y).expect("dl y");
    dev_mean.copy_to_host(&mut got_mean).expect("dl mean");
    dev_rstd.copy_to_host(&mut got_rstd).expect("dl rstd");

    // f32 ref is upcast to f64 expectations; widen by f32 eps since the
    // reference itself was computed at f32 precision.
    let eps_tol = 32.0 * f32::EPSILON as f64;
    for i in 0..numel {
        let want = exp_y_f32[i] as f64;
        let tol = (want.abs() * eps_tol).max(eps_tol);
        assert!((got_y[i] - want).abs() <= tol,
            "bn f64 y @ {i}: got={} want={}", got_y[i], want);
    }
    for i in 0..c as usize {
        let want_m = exp_mean_f32[i] as f64;
        let tol = (want_m.abs() * eps_tol).max(eps_tol);
        assert!((got_mean[i] - want_m).abs() <= tol, "bn f64 mean @ {i}");
        let want_r = exp_rstd_f32[i] as f64;
        let tol2 = (want_r.abs() * eps_tol).max(eps_tol);
        assert!((got_rstd[i] - want_r).abs() <= tol2, "bn f64 rstd @ {i}");
    }
}

#[test]
#[ignore]
fn batch_norm_f16_with_affine() {
    let (ctx, stream) = setup();
    let n = 4i32;
    let c = 6i32;
    let h = 3i32;
    let w = 3i32;
    let shape = [n, c, h, w];
    let s = (h * w) as usize;
    let numel = (n * c * h * w) as usize;
    let host_x_f32: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.07 - 1.2).sin() * 1.3)
        .collect();
    let host_gamma_f32: Vec<f32> = (0..c).map(|i| 0.8 + 0.07 * i as f32).collect();
    let host_beta_f32: Vec<f32> = (0..c).map(|i| 0.1 * i as f32 - 0.2).collect();
    let eps = 1e-5f32;

    let (exp_y_f32, _, _) = host_batch_norm_f32(
        n as usize, c as usize, s, &host_x_f32,
        Some(&host_gamma_f32), Some(&host_beta_f32), eps,
    );

    let host_x: Vec<f16> = host_x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_gamma: Vec<f16> = host_gamma_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_beta: Vec<f16> = host_beta_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_g = DeviceBuffer::from_slice(&ctx, &host_gamma).expect("up gamma");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_beta).expect("up beta");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("y");
    let mut dev_mean: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, c as usize).expect("mean");
    let mut dev_rstd: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, c as usize).expect("rstd");

    let desc = BatchNormDescriptor::<4> {
        input_shape: shape,
        channel_axis: 1,
        eps,
        has_affine: true,
        element: ElementKind::F16,
    };
    let plan = BatchNormPlan::<f16, 4>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run(&stream, Workspace::None, BatchNormArgs {
        x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
        gamma: Some(TensorRef { data: dev_g.as_slice(), shape: [c], stride: [1] }),
        beta: Some(TensorRef { data: dev_b.as_slice(), shape: [c], stride: [1] }),
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        saved_mean: TensorMut { data: dev_mean.as_slice_mut(), shape: [c], stride: [1] },
        saved_rstd: TensorMut { data: dev_rstd.as_slice_mut(), shape: [c], stride: [1] },
    }).expect("run");
    stream.synchronize().expect("sync");

    let mut got_y = vec![f16::ZERO; numel];
    dev_y.copy_to_host(&mut got_y).expect("dl");

    // ~12 ULP-equivalent: F16_EPS is ~1 ULP near 1.0.
    let eps_tol = 12.0 * F16_EPS;
    for i in 0..numel {
        let tol = (exp_y_f32[i].abs() * eps_tol).max(eps_tol);
        let diff = (got_y[i].to_f32() - exp_y_f32[i]).abs();
        assert!(diff <= tol,
            "bn f16 y @ {i}: diff={diff} got={} want={}",
            got_y[i].to_f32(), exp_y_f32[i]);
    }
}

#[test]
#[ignore]
fn batch_norm_bf16_with_affine() {
    let (ctx, stream) = setup();
    let n = 4i32;
    let c = 6i32;
    let h = 3i32;
    let w = 3i32;
    let shape = [n, c, h, w];
    let s = (h * w) as usize;
    let numel = (n * c * h * w) as usize;
    let host_x_f32: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.07 - 1.2).sin() * 1.3)
        .collect();
    let host_gamma_f32: Vec<f32> = (0..c).map(|i| 0.8 + 0.07 * i as f32).collect();
    let host_beta_f32: Vec<f32> = (0..c).map(|i| 0.1 * i as f32 - 0.2).collect();
    let eps = 1e-5f32;

    let (exp_y_f32, _, _) = host_batch_norm_f32(
        n as usize, c as usize, s, &host_x_f32,
        Some(&host_gamma_f32), Some(&host_beta_f32), eps,
    );

    let host_x: Vec<bf16> = host_x_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_gamma: Vec<bf16> = host_gamma_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_beta: Vec<bf16> = host_beta_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_g = DeviceBuffer::from_slice(&ctx, &host_gamma).expect("up gamma");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_beta).expect("up beta");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("y");
    let mut dev_mean: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, c as usize).expect("mean");
    let mut dev_rstd: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, c as usize).expect("rstd");

    let desc = BatchNormDescriptor::<4> {
        input_shape: shape,
        channel_axis: 1,
        eps,
        has_affine: true,
        element: ElementKind::Bf16,
    };
    let plan = BatchNormPlan::<bf16, 4>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run(&stream, Workspace::None, BatchNormArgs {
        x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
        gamma: Some(TensorRef { data: dev_g.as_slice(), shape: [c], stride: [1] }),
        beta: Some(TensorRef { data: dev_b.as_slice(), shape: [c], stride: [1] }),
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        saved_mean: TensorMut { data: dev_mean.as_slice_mut(), shape: [c], stride: [1] },
        saved_rstd: TensorMut { data: dev_rstd.as_slice_mut(), shape: [c], stride: [1] },
    }).expect("run");
    stream.synchronize().expect("sync");

    let mut got_y = vec![bf16::ZERO; numel];
    dev_y.copy_to_host(&mut got_y).expect("dl");

    // ~12 ULP-equivalent: BF16_EPS is ~1 ULP near 1.0.
    let eps_tol = 12.0 * BF16_EPS;
    for i in 0..numel {
        let tol = (exp_y_f32[i].abs() * eps_tol).max(eps_tol);
        let diff = (got_y[i].to_f32() - exp_y_f32[i]).abs();
        assert!(diff <= tol,
            "bn bf16 y @ {i}: diff={diff} got={} want={}",
            got_y[i].to_f32(), exp_y_f32[i]);
    }
}
