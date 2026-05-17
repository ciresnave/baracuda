//! Real-GPU smoke test for `GroupNormPlan` FW.
//!
//! Covers all four FP dtypes (f32 / f16 / bf16 / f64).
//!
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, GroupNormArgs, GroupNormDescriptor, GroupNormPlan,
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

/// CPU reference: `y[n, c, s] = (x - mean[n, g]) * rstd[n, g] * gamma[c] + beta[c]`
/// where `g = c / (C / num_groups)`.
fn host_group_norm_f32(
    n: usize, c: usize, s: usize, num_groups: usize, x: &[f32],
    gamma: Option<&[f32]>, beta: Option<&[f32]>, eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let group_size = c / num_groups;
    let group_extent = group_size * s;
    let g_count = n * num_groups;
    let mut mean = vec![0f64; g_count];
    let mut sumsq = vec![0f64; g_count];
    for ni in 0..n {
        for gi in 0..num_groups {
            for cc in 0..group_size {
                let ci = gi * group_size + cc;
                for si in 0..s {
                    let v = x[ni * c * s + ci * s + si] as f64;
                    mean[ni * num_groups + gi] += v;
                    sumsq[ni * num_groups + gi] += v * v;
                }
            }
        }
    }
    let mut rstd = vec![0f64; g_count];
    let m = group_extent as f64;
    for k in 0..g_count {
        mean[k] /= m;
        let var = (sumsq[k] / m - mean[k] * mean[k]).max(0.0);
        rstd[k] = 1.0 / (var + eps as f64).sqrt();
    }
    let mut y = vec![0f32; n * c * s];
    for ni in 0..n {
        for ci in 0..c {
            let gi = ci / group_size;
            let k = ni * num_groups + gi;
            for si in 0..s {
                let xv = x[ni * c * s + ci * s + si] as f64;
                let xh = (xv - mean[k]) * rstd[k];
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
fn group_norm_f32_with_affine() {
    let (ctx, stream) = setup();
    let n = 2i32;
    let c = 8i32;
    let h = 3i32;
    let w = 3i32;
    let num_groups = 4u32;
    let shape = [n, c, h, w];
    let s = (h * w) as usize;
    let numel = (n * c * h * w) as usize;
    let host_x: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.09 + 0.3).sin() * 1.1)
        .collect();
    let host_gamma: Vec<f32> = (0..c).map(|i| 0.5 + 0.08 * i as f32).collect();
    let host_beta: Vec<f32> = (0..c).map(|i| -0.1 + 0.05 * i as f32).collect();
    let eps = 1e-5f32;
    let (exp_y, _, _) = host_group_norm_f32(
        n as usize, c as usize, s, num_groups as usize,
        &host_x, Some(&host_gamma), Some(&host_beta), eps,
    );

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_g = DeviceBuffer::from_slice(&ctx, &host_gamma).expect("up gamma");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_beta).expect("up beta");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("y");
    let g_count = (n as usize) * (num_groups as usize);
    let mut dev_mean: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, g_count).expect("mean");
    let mut dev_rstd: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, g_count).expect("rstd");

    let desc = GroupNormDescriptor::<4> {
        input_shape: shape,
        channel_axis: 1,
        num_groups,
        eps,
        has_affine: true,
        element: ElementKind::F32,
    };
    let plan = GroupNormPlan::<f32, 4>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run(&stream, Workspace::None, GroupNormArgs {
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
    // GN aggregates across (group_size * spatial) cells; f32 partial-sums
    // drift faster than f32::EPSILON suggests. Use a coarser bound that
    // still catches systematic errors.
    let eps_tol = 128.0 * f32::EPSILON;
    for i in 0..numel {
        let tol = (exp_y[i].abs() * eps_tol).max(eps_tol);
        assert!((got_y[i] - exp_y[i]).abs() <= tol,
            "gn f32 y @ {i}: got={} want={}", got_y[i], exp_y[i]);
    }
}

#[test]
#[ignore]
fn group_norm_f64_with_affine() {
    let (ctx, stream) = setup();
    let n = 2i32;
    let c = 8i32;
    let h = 3i32;
    let w = 3i32;
    let num_groups = 4u32;
    let shape = [n, c, h, w];
    let s = (h * w) as usize;
    let numel = (n * c * h * w) as usize;
    let host_x_f32: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.09 + 0.3).sin() * 1.1)
        .collect();
    let host_gamma_f32: Vec<f32> = (0..c).map(|i| 0.5 + 0.08 * i as f32).collect();
    let host_beta_f32: Vec<f32> = (0..c).map(|i| -0.1 + 0.05 * i as f32).collect();
    let eps = 1e-5f32;
    let (exp_y_f32, _, _) = host_group_norm_f32(
        n as usize, c as usize, s, num_groups as usize,
        &host_x_f32, Some(&host_gamma_f32), Some(&host_beta_f32), eps,
    );

    let host_x: Vec<f64> = host_x_f32.iter().map(|&v| v as f64).collect();
    let host_gamma: Vec<f64> = host_gamma_f32.iter().map(|&v| v as f64).collect();
    let host_beta: Vec<f64> = host_beta_f32.iter().map(|&v| v as f64).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_g = DeviceBuffer::from_slice(&ctx, &host_gamma).expect("up gamma");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_beta).expect("up beta");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("y");
    let g_count = (n as usize) * (num_groups as usize);
    let mut dev_mean: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, g_count).expect("mean");
    let mut dev_rstd: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, g_count).expect("rstd");

    let desc = GroupNormDescriptor::<4> {
        input_shape: shape,
        channel_axis: 1,
        num_groups,
        eps,
        has_affine: true,
        element: ElementKind::F64,
    };
    let plan = GroupNormPlan::<f64, 4>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run(&stream, Workspace::None, GroupNormArgs {
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
    // f32 reference upcast — widen by f32 eps + GN aggregation factor.
    let eps_tol = 128.0 * f32::EPSILON as f64;
    for i in 0..numel {
        let want = exp_y_f32[i] as f64;
        let tol = (want.abs() * eps_tol).max(eps_tol);
        assert!((got_y[i] - want).abs() <= tol,
            "gn f64 y @ {i}: got={} want={}", got_y[i], want);
    }
}

#[test]
#[ignore]
fn group_norm_f16_with_affine() {
    let (ctx, stream) = setup();
    let n = 2i32;
    let c = 8i32;
    let h = 3i32;
    let w = 3i32;
    let num_groups = 4u32;
    let shape = [n, c, h, w];
    let s = (h * w) as usize;
    let numel = (n * c * h * w) as usize;
    let host_x_f32: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.09 + 0.3).sin() * 1.1)
        .collect();
    let host_gamma_f32: Vec<f32> = (0..c).map(|i| 0.5 + 0.08 * i as f32).collect();
    let host_beta_f32: Vec<f32> = (0..c).map(|i| -0.1 + 0.05 * i as f32).collect();
    let eps = 1e-5f32;
    let (exp_y_f32, _, _) = host_group_norm_f32(
        n as usize, c as usize, s, num_groups as usize,
        &host_x_f32, Some(&host_gamma_f32), Some(&host_beta_f32), eps,
    );

    let host_x: Vec<f16> = host_x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_gamma: Vec<f16> = host_gamma_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_beta: Vec<f16> = host_beta_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_g = DeviceBuffer::from_slice(&ctx, &host_gamma).expect("up gamma");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_beta).expect("up beta");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("y");
    let g_count = (n as usize) * (num_groups as usize);
    let mut dev_mean: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, g_count).expect("mean");
    let mut dev_rstd: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, g_count).expect("rstd");

    let desc = GroupNormDescriptor::<4> {
        input_shape: shape,
        channel_axis: 1,
        num_groups,
        eps,
        has_affine: true,
        element: ElementKind::F16,
    };
    let plan = GroupNormPlan::<f16, 4>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run(&stream, Workspace::None, GroupNormArgs {
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
    // ~12 ULP-equivalent.
    let eps_tol = 12.0 * F16_EPS;
    for i in 0..numel {
        let tol = (exp_y_f32[i].abs() * eps_tol).max(eps_tol);
        let diff = (got_y[i].to_f32() - exp_y_f32[i]).abs();
        assert!(diff <= tol,
            "gn f16 y @ {i}: diff={diff} got={} want={}",
            got_y[i].to_f32(), exp_y_f32[i]);
    }
}

#[test]
#[ignore]
fn group_norm_bf16_with_affine() {
    let (ctx, stream) = setup();
    let n = 2i32;
    let c = 8i32;
    let h = 3i32;
    let w = 3i32;
    let num_groups = 4u32;
    let shape = [n, c, h, w];
    let s = (h * w) as usize;
    let numel = (n * c * h * w) as usize;
    let host_x_f32: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.09 + 0.3).sin() * 1.1)
        .collect();
    let host_gamma_f32: Vec<f32> = (0..c).map(|i| 0.5 + 0.08 * i as f32).collect();
    let host_beta_f32: Vec<f32> = (0..c).map(|i| -0.1 + 0.05 * i as f32).collect();
    let eps = 1e-5f32;
    let (exp_y_f32, _, _) = host_group_norm_f32(
        n as usize, c as usize, s, num_groups as usize,
        &host_x_f32, Some(&host_gamma_f32), Some(&host_beta_f32), eps,
    );

    let host_x: Vec<bf16> = host_x_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_gamma: Vec<bf16> = host_gamma_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_beta: Vec<bf16> = host_beta_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_g = DeviceBuffer::from_slice(&ctx, &host_gamma).expect("up gamma");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_beta).expect("up beta");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("y");
    let g_count = (n as usize) * (num_groups as usize);
    let mut dev_mean: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, g_count).expect("mean");
    let mut dev_rstd: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, g_count).expect("rstd");

    let desc = GroupNormDescriptor::<4> {
        input_shape: shape,
        channel_axis: 1,
        num_groups,
        eps,
        has_affine: true,
        element: ElementKind::Bf16,
    };
    let plan = GroupNormPlan::<bf16, 4>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run(&stream, Workspace::None, GroupNormArgs {
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
    // ~12 ULP-equivalent.
    let eps_tol = 12.0 * BF16_EPS;
    for i in 0..numel {
        let tol = (exp_y_f32[i].abs() * eps_tol).max(eps_tol);
        let diff = (got_y[i].to_f32() - exp_y_f32[i]).abs();
        assert!(diff <= tol,
            "gn bf16 y @ {i}: diff={diff} got={} want={}",
            got_y[i].to_f32(), exp_y_f32[i]);
    }
}
