//! Real-GPU smoke test for `RMSNormBackwardPlan`.
//!
//! Backward formulas:
//!   `dx[..., i] = (dy[..., i] · gamma[i]) / rms`
//!              `- x[..., i] · (Σ_j dy[..., j] · gamma[j] · x[..., j]) / (rms³ · N)`
//!   `dgamma[i] = Σ over non-norm-axis cells dy[..., i] · (x[..., i] / rms[..., 0, ...])`
//!
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, RMSNormBackwardArgs,
    RMSNormBackwardDescriptor, RMSNormBackwardPlan, TensorMut, TensorRef, Workspace,
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

/// CPU reference. Returns `(dx, dgamma_opt)`. `dgamma_opt` is `None` if
/// `gamma` is `None`.
fn host_rms_norm_bw_f32(
    shape: &[i32], norm_axis: usize,
    dy: &[f32], x: &[f32], gamma: Option<&[f32]>, rms_per_row: &[f32],
) -> (Vec<f32>, Option<Vec<f32>>) {
    let n = shape.len();
    let mut stride = vec![1i64; n];
    for d in (0..n - 1).rev() {
        stride[d] = stride[d + 1] * shape[d + 1] as i64;
    }
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let extent = shape[norm_axis] as usize;
    let mut outer_shape: Vec<i32> = shape.to_vec();
    outer_shape[norm_axis] = 1;
    let outer_numel: usize = outer_shape.iter().map(|&d| d as usize).product();
    let mut outer_stride = vec![1i64; n];
    for d in (0..n - 1).rev() {
        outer_stride[d] = outer_stride[d + 1] * outer_shape[d + 1] as i64;
    }
    let inv_n = 1.0 / extent as f64;
    let mut dx = vec![0f32; numel];
    let mut dgamma = vec![0f64; extent];

    for outer_lin in 0..outer_numel {
        let mut coord = vec![0i32; n];
        let mut rem = outer_lin;
        for d in (0..n).rev() {
            if d == norm_axis {
                coord[d] = 0;
            } else {
                coord[d] = (rem % outer_shape[d] as usize) as i32;
                rem /= outer_shape[d] as usize;
            }
        }
        let mut off_rms = 0i64;
        for d in 0..n { off_rms += coord[d] as i64 * outer_stride[d]; }
        let rms = rms_per_row[off_rms as usize] as f64;
        let inv_rms = 1.0 / rms;
        let inv_rms3 = inv_rms * inv_rms * inv_rms;
        // dot = Σ dy · gamma · x
        let mut dot = 0f64;
        for j in 0..extent {
            coord[norm_axis] = j as i32;
            let mut idx = 0i64;
            for d in 0..n { idx += coord[d] as i64 * stride[d]; }
            let gj = gamma.map(|g| g[j]).unwrap_or(1.0) as f64;
            dot += (dy[idx as usize] as f64) * gj * (x[idx as usize] as f64);
        }
        for j in 0..extent {
            coord[norm_axis] = j as i32;
            let mut idx = 0i64;
            for d in 0..n { idx += coord[d] as i64 * stride[d]; }
            let gj = gamma.map(|g| g[j]).unwrap_or(1.0) as f64;
            let dyj = dy[idx as usize] as f64;
            let xj  = x[idx as usize] as f64;
            let out = dyj * gj * inv_rms - xj * dot * inv_rms3 * inv_n;
            dx[idx as usize] = out as f32;
            if gamma.is_some() {
                dgamma[j] += dyj * (xj * inv_rms);
            }
        }
    }
    let dgamma_out = gamma.map(|_| dgamma.iter().map(|&v| v as f32).collect());
    (dx, dgamma_out)
}

fn host_rms_per_row_f32(shape: &[i32], norm_axis: usize, x: &[f32], eps: f32) -> Vec<f32> {
    let n = shape.len();
    let mut stride = vec![1i64; n];
    for d in (0..n - 1).rev() {
        stride[d] = stride[d + 1] * shape[d + 1] as i64;
    }
    let extent = shape[norm_axis] as usize;
    let mut outer_shape: Vec<i32> = shape.to_vec();
    outer_shape[norm_axis] = 1;
    let outer_numel: usize = outer_shape.iter().map(|&d| d as usize).product();
    let mut outer_stride = vec![1i64; n];
    for d in (0..n - 1).rev() {
        outer_stride[d] = outer_stride[d + 1] * outer_shape[d + 1] as i64;
    }
    let mut rms = vec![0f32; outer_numel];
    for outer_lin in 0..outer_numel {
        let mut coord = vec![0i32; n];
        let mut rem = outer_lin;
        for d in (0..n).rev() {
            if d == norm_axis { coord[d] = 0; }
            else { coord[d] = (rem % outer_shape[d] as usize) as i32; rem /= outer_shape[d] as usize; }
        }
        let mut off_rms = 0i64;
        for d in 0..n { off_rms += coord[d] as i64 * outer_stride[d]; }
        let mut sum_sq = 0f64;
        for j in 0..extent {
            coord[norm_axis] = j as i32;
            let mut idx = 0i64;
            for d in 0..n { idx += coord[d] as i64 * stride[d]; }
            let v = x[idx as usize] as f64;
            sum_sq += v * v;
        }
        let mean_sq = sum_sq / extent as f64;
        rms[off_rms as usize] = ((mean_sq + eps as f64).sqrt()) as f32;
    }
    rms
}

#[test]
#[ignore]
fn rms_norm_bw_f32_with_gamma() {
    let (ctx, stream) = setup();
    let shape = [3i32, 5, 8];
    let numel = 120usize;
    let extent = 8usize;
    let host_x: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.11 - 1.0).sin() * 1.3).collect();
    let host_dy: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.07 + 0.3).cos()).collect();
    let host_gamma: Vec<f32> = (0..extent).map(|i| 0.4 + 0.1 * i as f32).collect();
    let eps = 1e-5f32;
    let host_rms = host_rms_per_row_f32(&shape, 2, &host_x, eps);
    let (expected_dx, expected_dgamma) =
        host_rms_norm_bw_f32(&shape, 2, &host_dy, &host_x, Some(&host_gamma), &host_rms);
    let expected_dgamma = expected_dgamma.expect("gamma supplied");

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_gamma = DeviceBuffer::from_slice(&ctx, &host_gamma).expect("up gamma");
    let dev_rms = DeviceBuffer::from_slice(&ctx, &host_rms).expect("up rms");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");
    let mut dev_dgamma: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, extent).expect("alloc dgamma");

    let desc = RMSNormBackwardDescriptor {
        input_shape: shape,
        norm_axes_mask: 4,
        has_gamma: true,
        element: ElementKind::F32,
    };
    let plan = RMSNormBackwardPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let rms_shape = desc.rms_shape();
    plan.run(&stream, Workspace::None, RMSNormBackwardArgs {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
        x:  TensorRef { data: dev_x.as_slice(),  shape, stride: contiguous_stride(shape) },
        gamma: Some(TensorRef { data: dev_gamma.as_slice(), shape: [extent as i32], stride: [1] }),
        rms: TensorRef {
            data: dev_rms.as_slice(), shape: rms_shape, stride: contiguous_stride(rms_shape),
        },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        dgamma: Some(TensorMut { data: dev_dgamma.as_slice_mut(), shape: [extent as i32], stride: [1] }),
    }).expect("run");
    stream.synchronize().expect("sync");

    let mut got_dx = vec![0f32; numel];
    let mut got_dgamma = vec![0f32; extent];
    dev_dx.copy_to_host(&mut got_dx).expect("dl dx");
    dev_dgamma.copy_to_host(&mut got_dgamma).expect("dl dgamma");

    let eps_tol = 32.0 * f32::EPSILON;
    for i in 0..numel {
        let tol = (expected_dx[i].abs() * eps_tol).max(eps_tol);
        assert!((got_dx[i] - expected_dx[i]).abs() <= tol,
            "f32 rms_norm_bw dx @ {i}: got={} want={}", got_dx[i], expected_dx[i]);
    }
    // dgamma accumulates over (3*5)=15 cells per feature — looser tol.
    let dg_tol = 64.0 * f32::EPSILON;
    for i in 0..extent {
        let tol = (expected_dgamma[i].abs() * dg_tol).max(dg_tol);
        assert!((got_dgamma[i] - expected_dgamma[i]).abs() <= tol,
            "f32 rms_norm_bw dgamma @ {i}: got={} want={}", got_dgamma[i], expected_dgamma[i]);
    }
}

#[test]
#[ignore]
fn rms_norm_bw_f64_no_gamma() {
    let (ctx, stream) = setup();
    let shape = [2i32, 6];
    let numel = 12usize;
    let extent = 6usize;
    let host_x: Vec<f64> = (0..numel).map(|i| ((i as f64) * 0.21 - 1.1).cos() * 0.8).collect();
    let host_dy: Vec<f64> = (0..numel).map(|i| ((i as f64) * 0.13 + 0.5).sin() * 0.6).collect();
    let eps = 1e-6f32;
    // CPU f64 ref
    let mut rms = vec![0f64; 2];
    for i in 0..2 {
        let mut sum_sq = 0f64;
        for j in 0..extent { let v = host_x[i * 6 + j]; sum_sq += v * v; }
        rms[i] = (sum_sq / extent as f64 + eps as f64).sqrt();
    }
    let inv_n = 1.0 / extent as f64;
    let mut expected_dx = vec![0f64; numel];
    for i in 0..2 {
        let inv_rms = 1.0 / rms[i];
        let inv_rms3 = inv_rms * inv_rms * inv_rms;
        let mut dot = 0f64;
        for j in 0..extent { dot += host_dy[i * 6 + j] * host_x[i * 6 + j]; }
        for j in 0..extent {
            expected_dx[i * 6 + j] = host_dy[i * 6 + j] * inv_rms
                - host_x[i * 6 + j] * dot * inv_rms3 * inv_n;
        }
    }
    let host_rms: Vec<f64> = rms.clone();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_rms = DeviceBuffer::from_slice(&ctx, &host_rms).expect("up rms");
    let mut dev_dx: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");

    let desc = RMSNormBackwardDescriptor {
        input_shape: shape,
        norm_axes_mask: 2,
        has_gamma: false,
        element: ElementKind::F64,
    };
    let plan = RMSNormBackwardPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let rms_shape = desc.rms_shape();
    plan.run(&stream, Workspace::None, RMSNormBackwardArgs {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
        x:  TensorRef { data: dev_x.as_slice(),  shape, stride: contiguous_stride(shape) },
        gamma: None,
        rms: TensorRef {
            data: dev_rms.as_slice(), shape: rms_shape, stride: contiguous_stride(rms_shape),
        },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        dgamma: None,
    }).expect("run");
    stream.synchronize().expect("sync");

    let mut got_dx = vec![0f64; numel];
    dev_dx.copy_to_host(&mut got_dx).expect("dl");
    let eps_tol = 32.0 * f64::EPSILON;
    for i in 0..numel {
        let tol = (expected_dx[i].abs() * eps_tol).max(eps_tol);
        assert!((got_dx[i] - expected_dx[i]).abs() <= tol,
            "f64 rms_norm_bw dx @ {i}: got={} want={}", got_dx[i], expected_dx[i]);
    }
}

#[test]
#[ignore]
fn rms_norm_bw_f16_with_gamma() {
    let (ctx, stream) = setup();
    let shape = [3i32, 8];
    let numel = 24usize;
    let extent = 8usize;
    let host_x_f32: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.18 - 1.4).sin()).collect();
    let host_dy_f32: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.09 + 0.2).cos()).collect();
    let host_gamma_f32: Vec<f32> = (0..extent).map(|i| 1.0 + 0.05 * i as f32).collect();
    let eps = 1e-5f32;
    let host_rms_f32 = host_rms_per_row_f32(&shape, 1, &host_x_f32, eps);
    let (expected_dx_f32, expected_dgamma_f32) =
        host_rms_norm_bw_f32(&shape, 1, &host_dy_f32, &host_x_f32,
                              Some(&host_gamma_f32), &host_rms_f32);
    let expected_dgamma_f32 = expected_dgamma_f32.unwrap();

    let host_x: Vec<f16> = host_x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_dy: Vec<f16> = host_dy_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_gamma: Vec<f16> = host_gamma_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_rms: Vec<f16> = host_rms_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up");
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let dev_gamma = DeviceBuffer::from_slice(&ctx, &host_gamma).expect("up");
    let dev_rms = DeviceBuffer::from_slice(&ctx, &host_rms).expect("up");
    let mut dev_dx: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let mut dev_dgamma: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, extent).expect("alloc");

    let desc = RMSNormBackwardDescriptor {
        input_shape: shape,
        norm_axes_mask: 2,
        has_gamma: true,
        element: ElementKind::F16,
    };
    let plan = RMSNormBackwardPlan::<f16, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let rms_shape = desc.rms_shape();
    plan.run(&stream, Workspace::None, RMSNormBackwardArgs {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
        x:  TensorRef { data: dev_x.as_slice(),  shape, stride: contiguous_stride(shape) },
        gamma: Some(TensorRef { data: dev_gamma.as_slice(), shape: [extent as i32], stride: [1] }),
        rms: TensorRef {
            data: dev_rms.as_slice(), shape: rms_shape, stride: contiguous_stride(rms_shape),
        },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        dgamma: Some(TensorMut { data: dev_dgamma.as_slice_mut(), shape: [extent as i32], stride: [1] }),
    }).expect("run");
    stream.synchronize().expect("sync");

    let mut got_dx = vec![f16::ZERO; numel];
    let mut got_dgamma = vec![f16::ZERO; extent];
    dev_dx.copy_to_host(&mut got_dx).expect("dl dx");
    dev_dgamma.copy_to_host(&mut got_dgamma).expect("dl dgamma");

    let eps_tol = 16.0 * F16_EPS;
    for i in 0..numel {
        let tol = (expected_dx_f32[i].abs() * eps_tol).max(eps_tol);
        let diff = (got_dx[i].to_f32() - expected_dx_f32[i]).abs();
        assert!(diff <= tol, "f16 rms_norm_bw dx @ {i}: diff={diff}");
    }
    let dg_tol = 32.0 * F16_EPS;
    for i in 0..extent {
        let tol = (expected_dgamma_f32[i].abs() * dg_tol).max(dg_tol);
        let diff = (got_dgamma[i].to_f32() - expected_dgamma_f32[i]).abs();
        assert!(diff <= tol, "f16 rms_norm_bw dgamma @ {i}: diff={diff}");
    }
}

#[test]
#[ignore]
fn rms_norm_bw_bf16_no_gamma() {
    let (ctx, stream) = setup();
    let shape = [2i32, 8];
    let numel = 16usize;
    let host_x_f32: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.27 - 0.4).cos()).collect();
    let host_dy_f32: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.15 + 0.6).sin()).collect();
    let eps = 1e-5f32;
    let host_rms_f32 = host_rms_per_row_f32(&shape, 1, &host_x_f32, eps);
    let (expected_dx_f32, _) =
        host_rms_norm_bw_f32(&shape, 1, &host_dy_f32, &host_x_f32, None, &host_rms_f32);

    let host_x: Vec<bf16> = host_x_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_dy: Vec<bf16> = host_dy_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_rms: Vec<bf16> = host_rms_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up");
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let dev_rms = DeviceBuffer::from_slice(&ctx, &host_rms).expect("up");
    let mut dev_dx: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");

    let desc = RMSNormBackwardDescriptor {
        input_shape: shape,
        norm_axes_mask: 2,
        has_gamma: false,
        element: ElementKind::Bf16,
    };
    let plan = RMSNormBackwardPlan::<bf16, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let rms_shape = desc.rms_shape();
    plan.run(&stream, Workspace::None, RMSNormBackwardArgs {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
        x:  TensorRef { data: dev_x.as_slice(),  shape, stride: contiguous_stride(shape) },
        gamma: None,
        rms: TensorRef {
            data: dev_rms.as_slice(), shape: rms_shape, stride: contiguous_stride(rms_shape),
        },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        dgamma: None,
    }).expect("run");
    stream.synchronize().expect("sync");

    let mut got_dx = vec![bf16::ZERO; numel];
    dev_dx.copy_to_host(&mut got_dx).expect("dl");

    let eps_tol = 16.0 * BF16_EPS;
    for i in 0..numel {
        let tol = (expected_dx_f32[i].abs() * eps_tol).max(eps_tol);
        let diff = (got_dx[i].to_f32() - expected_dx_f32[i]).abs();
        assert!(diff <= tol, "bf16 rms_norm_bw dx @ {i}: diff={diff}");
    }
}
