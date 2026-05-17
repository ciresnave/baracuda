//! Real-GPU smoke test for `InstanceNormPlan` FW.
//!
//! Verifies that InstanceNorm = GroupNorm with `num_groups == C` (wires
//! through the same kernel symbols via the GN dispatch).
//!
//! Covers all four FP dtypes (f32 / f16 / bf16 / f64).
//!
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, InstanceNormArgs, InstanceNormDescriptor, InstanceNormPlan,
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

/// CPU reference: per-`(n, c)` stats across spatial; per-channel affine.
fn host_instance_norm_f32(
    n: usize, c: usize, s: usize, x: &[f32],
    gamma: Option<&[f32]>, beta: Option<&[f32]>, eps: f32,
) -> Vec<f32> {
    let mut y = vec![0f32; n * c * s];
    for ni in 0..n {
        for ci in 0..c {
            let mut sum = 0f64;
            let mut sq = 0f64;
            for si in 0..s {
                let v = x[ni * c * s + ci * s + si] as f64;
                sum += v;
                sq += v * v;
            }
            let mean = sum / s as f64;
            let var = (sq / s as f64 - mean * mean).max(0.0);
            let rstd = 1.0 / (var + eps as f64).sqrt();
            let gv = gamma.map(|g| g[ci] as f64).unwrap_or(1.0);
            let bv = beta.map(|b| b[ci] as f64).unwrap_or(0.0);
            for si in 0..s {
                let xv = x[ni * c * s + ci * s + si] as f64;
                y[ni * c * s + ci * s + si] = ((xv - mean) * rstd * gv + bv) as f32;
            }
        }
    }
    y
}

#[test]
#[ignore]
fn instance_norm_f32_with_affine() {
    let (ctx, stream) = setup();
    let n = 2i32;
    let c = 4i32;
    let h = 4i32;
    let w = 4i32;
    let shape = [n, c, h, w];
    let s = (h * w) as usize;
    let numel = (n * c * h * w) as usize;
    let host_x: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.13 + 1.7).sin() * 1.4)
        .collect();
    let host_gamma: Vec<f32> = (0..c).map(|i| 0.9 + 0.1 * i as f32).collect();
    let host_beta: Vec<f32> = (0..c).map(|i| 0.2 - 0.05 * i as f32).collect();
    let eps = 1e-5f32;
    let exp_y = host_instance_norm_f32(
        n as usize, c as usize, s, &host_x,
        Some(&host_gamma), Some(&host_beta), eps,
    );

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_g = DeviceBuffer::from_slice(&ctx, &host_gamma).expect("up gamma");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_beta).expect("up beta");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("y");
    let g_count = (n as usize) * (c as usize);
    let mut dev_mean: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, g_count).expect("mean");
    let mut dev_rstd: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, g_count).expect("rstd");

    let desc = InstanceNormDescriptor::<4> {
        input_shape: shape,
        channel_axis: 1,
        eps,
        has_affine: true,
        element: ElementKind::F32,
    };
    let plan = InstanceNormPlan::<f32, 4>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run(&stream, Workspace::None, InstanceNormArgs {
        x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
        gamma: Some(TensorRef { data: dev_g.as_slice(), shape: [c], stride: [1] }),
        beta: Some(TensorRef { data: dev_b.as_slice(), shape: [c], stride: [1] }),
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        saved_mean: TensorMut {
            data: dev_mean.as_slice_mut(), shape: [g_count as i32], stride: [1]
        },
        saved_rstd: TensorMut {
            data: dev_rstd.as_slice_mut(), shape: [g_count as i32], stride: [1]
        },
    }).expect("run");
    stream.synchronize().expect("sync");

    let mut got_y = vec![0f32; numel];
    dev_y.copy_to_host(&mut got_y).expect("dl");
    let eps_tol = 32.0 * f32::EPSILON;
    for i in 0..numel {
        let tol = (exp_y[i].abs() * eps_tol).max(eps_tol);
        assert!((got_y[i] - exp_y[i]).abs() <= tol,
            "in f32 y @ {i}: got={} want={}", got_y[i], exp_y[i]);
    }
}

#[test]
#[ignore]
fn instance_norm_f64_with_affine() {
    let (ctx, stream) = setup();
    let n = 2i32;
    let c = 4i32;
    let h = 4i32;
    let w = 4i32;
    let shape = [n, c, h, w];
    let s = (h * w) as usize;
    let numel = (n * c * h * w) as usize;
    let host_x_f32: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.13 + 1.7).sin() * 1.4)
        .collect();
    let host_gamma_f32: Vec<f32> = (0..c).map(|i| 0.9 + 0.1 * i as f32).collect();
    let host_beta_f32: Vec<f32> = (0..c).map(|i| 0.2 - 0.05 * i as f32).collect();
    let eps = 1e-5f32;
    let exp_y_f32 = host_instance_norm_f32(
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
    let g_count = (n as usize) * (c as usize);
    let mut dev_mean: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, g_count).expect("mean");
    let mut dev_rstd: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, g_count).expect("rstd");

    let desc = InstanceNormDescriptor::<4> {
        input_shape: shape,
        channel_axis: 1,
        eps,
        has_affine: true,
        element: ElementKind::F64,
    };
    let plan = InstanceNormPlan::<f64, 4>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run(&stream, Workspace::None, InstanceNormArgs {
        x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
        gamma: Some(TensorRef { data: dev_g.as_slice(), shape: [c], stride: [1] }),
        beta: Some(TensorRef { data: dev_b.as_slice(), shape: [c], stride: [1] }),
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        saved_mean: TensorMut {
            data: dev_mean.as_slice_mut(), shape: [g_count as i32], stride: [1]
        },
        saved_rstd: TensorMut {
            data: dev_rstd.as_slice_mut(), shape: [g_count as i32], stride: [1]
        },
    }).expect("run");
    stream.synchronize().expect("sync");

    let mut got_y = vec![0f64; numel];
    dev_y.copy_to_host(&mut got_y).expect("dl");
    // f32 reference upcast — bound by f32 eps.
    let eps_tol = 32.0 * f32::EPSILON as f64;
    for i in 0..numel {
        let want = exp_y_f32[i] as f64;
        let tol = (want.abs() * eps_tol).max(eps_tol);
        assert!((got_y[i] - want).abs() <= tol,
            "in f64 y @ {i}: got={} want={}", got_y[i], want);
    }
}

#[test]
#[ignore]
fn instance_norm_f16_with_affine() {
    let (ctx, stream) = setup();
    let n = 2i32;
    let c = 4i32;
    let h = 4i32;
    let w = 4i32;
    let shape = [n, c, h, w];
    let s = (h * w) as usize;
    let numel = (n * c * h * w) as usize;
    let host_x_f32: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.13 + 1.7).sin() * 1.4)
        .collect();
    let host_gamma_f32: Vec<f32> = (0..c).map(|i| 0.9 + 0.1 * i as f32).collect();
    let host_beta_f32: Vec<f32> = (0..c).map(|i| 0.2 - 0.05 * i as f32).collect();
    let eps = 1e-5f32;
    let exp_y_f32 = host_instance_norm_f32(
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
    let g_count = (n as usize) * (c as usize);
    let mut dev_mean: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, g_count).expect("mean");
    let mut dev_rstd: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, g_count).expect("rstd");

    let desc = InstanceNormDescriptor::<4> {
        input_shape: shape,
        channel_axis: 1,
        eps,
        has_affine: true,
        element: ElementKind::F16,
    };
    let plan = InstanceNormPlan::<f16, 4>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run(&stream, Workspace::None, InstanceNormArgs {
        x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
        gamma: Some(TensorRef { data: dev_g.as_slice(), shape: [c], stride: [1] }),
        beta: Some(TensorRef { data: dev_b.as_slice(), shape: [c], stride: [1] }),
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        saved_mean: TensorMut {
            data: dev_mean.as_slice_mut(), shape: [g_count as i32], stride: [1]
        },
        saved_rstd: TensorMut {
            data: dev_rstd.as_slice_mut(), shape: [g_count as i32], stride: [1]
        },
    }).expect("run");
    stream.synchronize().expect("sync");

    let mut got_y = vec![f16::ZERO; numel];
    dev_y.copy_to_host(&mut got_y).expect("dl");
    let eps_tol = 12.0 * F16_EPS;
    for i in 0..numel {
        let tol = (exp_y_f32[i].abs() * eps_tol).max(eps_tol);
        let diff = (got_y[i].to_f32() - exp_y_f32[i]).abs();
        assert!(diff <= tol,
            "in f16 y @ {i}: diff={diff} got={} want={}",
            got_y[i].to_f32(), exp_y_f32[i]);
    }
}

#[test]
#[ignore]
fn instance_norm_bf16_with_affine() {
    let (ctx, stream) = setup();
    let n = 2i32;
    let c = 4i32;
    let h = 4i32;
    let w = 4i32;
    let shape = [n, c, h, w];
    let s = (h * w) as usize;
    let numel = (n * c * h * w) as usize;
    let host_x_f32: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.13 + 1.7).sin() * 1.4)
        .collect();
    let host_gamma_f32: Vec<f32> = (0..c).map(|i| 0.9 + 0.1 * i as f32).collect();
    let host_beta_f32: Vec<f32> = (0..c).map(|i| 0.2 - 0.05 * i as f32).collect();
    let eps = 1e-5f32;
    let exp_y_f32 = host_instance_norm_f32(
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
    let g_count = (n as usize) * (c as usize);
    let mut dev_mean: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, g_count).expect("mean");
    let mut dev_rstd: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, g_count).expect("rstd");

    let desc = InstanceNormDescriptor::<4> {
        input_shape: shape,
        channel_axis: 1,
        eps,
        has_affine: true,
        element: ElementKind::Bf16,
    };
    let plan = InstanceNormPlan::<bf16, 4>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run(&stream, Workspace::None, InstanceNormArgs {
        x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
        gamma: Some(TensorRef { data: dev_g.as_slice(), shape: [c], stride: [1] }),
        beta: Some(TensorRef { data: dev_b.as_slice(), shape: [c], stride: [1] }),
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        saved_mean: TensorMut {
            data: dev_mean.as_slice_mut(), shape: [g_count as i32], stride: [1]
        },
        saved_rstd: TensorMut {
            data: dev_rstd.as_slice_mut(), shape: [g_count as i32], stride: [1]
        },
    }).expect("run");
    stream.synchronize().expect("sync");

    let mut got_y = vec![bf16::ZERO; numel];
    dev_y.copy_to_host(&mut got_y).expect("dl");
    let eps_tol = 12.0 * BF16_EPS;
    for i in 0..numel {
        let tol = (exp_y_f32[i].abs() * eps_tol).max(eps_tol);
        let diff = (got_y[i].to_f32() - exp_y_f32[i]).abs();
        assert!(diff <= tol,
            "in bf16 y @ {i}: diff={diff} got={} want={}",
            got_y[i].to_f32(), exp_y_f32[i]);
    }
}
