//! Real-GPU smoke test for `LayerNormBackwardPlan`.
//!
//! BW formulas (biased variance):
//!   `x_hat[i] = (x[i] - mean) * inv_std`
//!   `dx_hat[i] = dy[i] · gamma[i]`
//!   `dx[i] = inv_std · (dx_hat[i] - Σ_j dx_hat[j] / N - x_hat[i] · Σ_j dx_hat[j] · x_hat[j] / N)`
//!   `dgamma[i] = Σ over non-norm-axis cells dy[..., i] · x_hat[..., i]`
//!   `dbeta[i]  = Σ over non-norm-axis cells dy[..., i]`
//!
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, LayerNormBackwardArgs, LayerNormBackwardDescriptor,
    LayerNormBackwardPlan, PlanPreference, TensorMut, TensorRef, Workspace,
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

/// CPU reference. Returns `(dx, dgamma_opt, dbeta_opt)`. Saves are
/// computed inline (FW work duplicated to keep the test self-contained).
fn host_layer_norm_bw_f32(
    shape: &[i32], norm_axis: usize,
    dy: &[f32], x: &[f32],
    gamma: Option<&[f32]>, has_beta: bool, eps: f32,
) -> (Vec<f32>, Option<Vec<f32>>, Option<Vec<f32>>) {
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
    let inv_n = 1.0 / extent as f64;
    let mut dx = vec![0f32; numel];
    let mut dgamma = vec![0f64; extent];
    let mut dbeta  = vec![0f64; extent];

    for outer_lin in 0..outer_numel {
        let mut coord = vec![0i32; n];
        let mut rem = outer_lin;
        for d in (0..n).rev() {
            if d == norm_axis { coord[d] = 0; }
            else { coord[d] = (rem % outer_shape[d] as usize) as i32;
                   rem /= outer_shape[d] as usize; }
        }
        // mean, inv_std
        let mut sum = 0f64;
        for j in 0..extent {
            coord[norm_axis] = j as i32;
            let mut idx = 0i64;
            for d in 0..n { idx += coord[d] as i64 * stride[d]; }
            sum += x[idx as usize] as f64;
        }
        let mean = sum * inv_n;
        let mut sumsq = 0f64;
        for j in 0..extent {
            coord[norm_axis] = j as i32;
            let mut idx = 0i64;
            for d in 0..n { idx += coord[d] as i64 * stride[d]; }
            let v = x[idx as usize] as f64 - mean;
            sumsq += v * v;
        }
        let var = sumsq * inv_n;
        let inv_std = 1.0 / (var + eps as f64).sqrt();
        // sums
        let mut sum_dxh = 0f64;
        let mut sum_dxhxh = 0f64;
        for j in 0..extent {
            coord[norm_axis] = j as i32;
            let mut idx = 0i64;
            for d in 0..n { idx += coord[d] as i64 * stride[d]; }
            let dyj = dy[idx as usize] as f64;
            let xj  = x[idx as usize] as f64;
            let gj  = gamma.map(|g| g[j]).unwrap_or(1.0) as f64;
            let dxh = dyj * gj;
            let xh  = (xj - mean) * inv_std;
            sum_dxh   += dxh;
            sum_dxhxh += dxh * xh;
        }
        for j in 0..extent {
            coord[norm_axis] = j as i32;
            let mut idx = 0i64;
            for d in 0..n { idx += coord[d] as i64 * stride[d]; }
            let dyj = dy[idx as usize] as f64;
            let xj  = x[idx as usize] as f64;
            let gj  = gamma.map(|g| g[j]).unwrap_or(1.0) as f64;
            let dxh = dyj * gj;
            let xh  = (xj - mean) * inv_std;
            let out = inv_std * (dxh - sum_dxh * inv_n - xh * sum_dxhxh * inv_n);
            dx[idx as usize] = out as f32;
            if gamma.is_some() { dgamma[j] += dyj * xh; }
            if has_beta { dbeta[j] += dyj; }
        }
    }
    let dg_out = gamma.map(|_| dgamma.iter().map(|&v| v as f32).collect());
    let db_out = if has_beta { Some(dbeta.iter().map(|&v| v as f32).collect()) } else { None };
    (dx, dg_out, db_out)
}

fn host_mean_invstd_f32(shape: &[i32], norm_axis: usize, x: &[f32], eps: f32)
    -> (Vec<f32>, Vec<f32>)
{
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
    let inv_n = 1.0 / extent as f64;
    let mut mean = vec![0f32; outer_numel];
    let mut inv_std = vec![0f32; outer_numel];
    for outer_lin in 0..outer_numel {
        let mut coord = vec![0i32; n];
        let mut rem = outer_lin;
        for d in (0..n).rev() {
            if d == norm_axis { coord[d] = 0; }
            else { coord[d] = (rem % outer_shape[d] as usize) as i32; rem /= outer_shape[d] as usize; }
        }
        let mut off_save = 0i64;
        for d in 0..n { off_save += coord[d] as i64 * outer_stride[d]; }
        let mut sum = 0f64;
        for j in 0..extent {
            coord[norm_axis] = j as i32;
            let mut idx = 0i64;
            for d in 0..n { idx += coord[d] as i64 * stride[d]; }
            sum += x[idx as usize] as f64;
        }
        let m = sum * inv_n;
        let mut sumsq = 0f64;
        for j in 0..extent {
            coord[norm_axis] = j as i32;
            let mut idx = 0i64;
            for d in 0..n { idx += coord[d] as i64 * stride[d]; }
            let v = x[idx as usize] as f64 - m;
            sumsq += v * v;
        }
        let var = sumsq * inv_n;
        mean[off_save as usize] = m as f32;
        inv_std[off_save as usize] = (1.0 / (var + eps as f64).sqrt()) as f32;
    }
    (mean, inv_std)
}

#[test]
#[ignore]
fn layer_norm_bw_f32_full_affine() {
    let (ctx, stream) = setup();
    let shape = [2i32, 4, 8];
    let numel = 64usize;
    let extent = 8usize;
    let host_x: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.11 - 1.0).sin() * 1.2).collect();
    let host_dy: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.07 + 0.2).cos()).collect();
    let host_gamma: Vec<f32> = (0..extent).map(|i| 0.6 + 0.05 * i as f32).collect();
    let host_beta:  Vec<f32> = (0..extent).map(|i| -0.1 + 0.02 * i as f32).collect();
    let eps = 1e-5f32;
    let (host_mean, host_std) = host_mean_invstd_f32(&shape, 2, &host_x, eps);
    let _ = host_beta; // beta participates only in FW; BW uses dy/x/gamma + mean/inv_std.
    let (expected_dx, expected_dgamma, expected_dbeta) =
        host_layer_norm_bw_f32(&shape, 2, &host_dy, &host_x, Some(&host_gamma), true, eps);
    let expected_dgamma = expected_dgamma.unwrap();
    let expected_dbeta = expected_dbeta.unwrap();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_gamma = DeviceBuffer::from_slice(&ctx, &host_gamma).expect("up gamma");
    let dev_mean = DeviceBuffer::from_slice(&ctx, &host_mean).expect("up mean");
    let dev_std = DeviceBuffer::from_slice(&ctx, &host_std).expect("up std");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");
    let mut dev_dgamma: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, extent).expect("alloc dgamma");
    let mut dev_dbeta: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, extent).expect("alloc dbeta");

    let desc = LayerNormBackwardDescriptor {
        input_shape: shape,
        norm_axes_mask: 4,
        has_gamma: true,
        has_beta: true,
        element: ElementKind::F32,
    };
    let plan = LayerNormBackwardPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let save_shape = desc.save_shape();
    plan.run(&stream, Workspace::None, LayerNormBackwardArgs {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
        x:  TensorRef { data: dev_x.as_slice(),  shape, stride: contiguous_stride(shape) },
        gamma: Some(TensorRef { data: dev_gamma.as_slice(), shape: [extent as i32], stride: [1] }),
        mean: TensorRef {
            data: dev_mean.as_slice(), shape: save_shape, stride: contiguous_stride(save_shape),
        },
        inv_std: TensorRef {
            data: dev_std.as_slice(), shape: save_shape, stride: contiguous_stride(save_shape),
        },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        dgamma: Some(TensorMut { data: dev_dgamma.as_slice_mut(), shape: [extent as i32], stride: [1] }),
        dbeta:  Some(TensorMut { data: dev_dbeta.as_slice_mut(),  shape: [extent as i32], stride: [1] }),
    }).expect("run");
    stream.synchronize().expect("sync");

    let mut got_dx = vec![0f32; numel];
    let mut got_dgamma = vec![0f32; extent];
    let mut got_dbeta = vec![0f32; extent];
    dev_dx.copy_to_host(&mut got_dx).expect("dl dx");
    dev_dgamma.copy_to_host(&mut got_dgamma).expect("dl dgamma");
    dev_dbeta.copy_to_host(&mut got_dbeta).expect("dl dbeta");

    let eps_tol = 32.0 * f32::EPSILON;
    for i in 0..numel {
        let tol = (expected_dx[i].abs() * eps_tol).max(eps_tol);
        assert!((got_dx[i] - expected_dx[i]).abs() <= tol,
            "f32 layer_norm_bw dx @ {i}: got={} want={}", got_dx[i], expected_dx[i]);
    }
    let aff_tol = 64.0 * f32::EPSILON;
    for i in 0..extent {
        let tol = (expected_dgamma[i].abs() * aff_tol).max(aff_tol);
        assert!((got_dgamma[i] - expected_dgamma[i]).abs() <= tol,
            "f32 layer_norm_bw dgamma @ {i}");
        let tol2 = (expected_dbeta[i].abs() * aff_tol).max(aff_tol);
        assert!((got_dbeta[i] - expected_dbeta[i]).abs() <= tol2,
            "f32 layer_norm_bw dbeta @ {i}");
    }
}

#[test]
#[ignore]
fn layer_norm_bw_f64_no_affine() {
    let (ctx, stream) = setup();
    let shape = [3i32, 6];
    let numel = 18usize;
    let extent = 6usize;
    let host_x: Vec<f64> = (0..numel).map(|i| ((i as f64) * 0.2 - 1.0).cos() * 0.9).collect();
    let host_dy: Vec<f64> = (0..numel).map(|i| ((i as f64) * 0.13 + 0.5).sin() * 0.6).collect();
    let eps = 1e-6f32;
    // CPU f64 ref
    let mut host_mean = vec![0f64; 3];
    let mut host_std = vec![0f64; 3];
    let inv_n = 1.0 / extent as f64;
    for i in 0..3 {
        let mut sum = 0f64;
        for j in 0..extent { sum += host_x[i * 6 + j]; }
        let mean = sum * inv_n;
        let mut sumsq = 0f64;
        for j in 0..extent { let v = host_x[i * 6 + j] - mean; sumsq += v * v; }
        host_mean[i] = mean;
        host_std[i] = 1.0 / (sumsq * inv_n + eps as f64).sqrt();
    }
    let mut expected_dx = vec![0f64; numel];
    for i in 0..3 {
        let mean = host_mean[i];
        let inv_std = host_std[i];
        let mut sum_dxh = 0f64;
        let mut sum_dxhxh = 0f64;
        for j in 0..extent {
            let dxh = host_dy[i * 6 + j];
            let xh = (host_x[i * 6 + j] - mean) * inv_std;
            sum_dxh += dxh;
            sum_dxhxh += dxh * xh;
        }
        for j in 0..extent {
            let dxh = host_dy[i * 6 + j];
            let xh  = (host_x[i * 6 + j] - mean) * inv_std;
            expected_dx[i * 6 + j] = inv_std * (dxh - sum_dxh * inv_n - xh * sum_dxhxh * inv_n);
        }
    }

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_mean = DeviceBuffer::from_slice(&ctx, &host_mean).expect("up mean");
    let dev_std = DeviceBuffer::from_slice(&ctx, &host_std).expect("up std");
    let mut dev_dx: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");

    let desc = LayerNormBackwardDescriptor {
        input_shape: shape, norm_axes_mask: 2,
        has_gamma: false, has_beta: false,
        element: ElementKind::F64,
    };
    let plan = LayerNormBackwardPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let save_shape = desc.save_shape();
    plan.run(&stream, Workspace::None, LayerNormBackwardArgs {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
        x:  TensorRef { data: dev_x.as_slice(),  shape, stride: contiguous_stride(shape) },
        gamma: None,
        mean: TensorRef {
            data: dev_mean.as_slice(), shape: save_shape, stride: contiguous_stride(save_shape),
        },
        inv_std: TensorRef {
            data: dev_std.as_slice(), shape: save_shape, stride: contiguous_stride(save_shape),
        },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        dgamma: None,
        dbeta:  None,
    }).expect("run");
    stream.synchronize().expect("sync");

    let mut got_dx = vec![0f64; numel];
    dev_dx.copy_to_host(&mut got_dx).expect("dl");

    let eps_tol = 32.0 * f64::EPSILON;
    for i in 0..numel {
        let tol = (expected_dx[i].abs() * eps_tol).max(eps_tol);
        assert!((got_dx[i] - expected_dx[i]).abs() <= tol,
            "f64 layer_norm_bw dx @ {i}: got={} want={}", got_dx[i], expected_dx[i]);
    }
}

#[test]
#[ignore]
fn layer_norm_bw_f16_full_affine() {
    let (ctx, stream) = setup();
    let shape = [3i32, 8];
    let numel = 24usize;
    let extent = 8usize;
    let host_x_f32: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.18 - 1.4).sin()).collect();
    let host_dy_f32: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.09 + 0.2).cos()).collect();
    let host_gamma_f32: Vec<f32> = (0..extent).map(|i| 0.8 + 0.04 * i as f32).collect();
    let eps = 1e-5f32;
    let (host_mean_f32, host_std_f32) = host_mean_invstd_f32(&shape, 1, &host_x_f32, eps);
    let (expected_dx_f32, expected_dgamma_f32, expected_dbeta_f32) =
        host_layer_norm_bw_f32(&shape, 1, &host_dy_f32, &host_x_f32,
                                Some(&host_gamma_f32), true, eps);
    let expected_dgamma_f32 = expected_dgamma_f32.unwrap();
    let expected_dbeta_f32 = expected_dbeta_f32.unwrap();

    let host_x: Vec<f16> = host_x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_dy: Vec<f16> = host_dy_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_gamma: Vec<f16> = host_gamma_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_mean: Vec<f16> = host_mean_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_std:  Vec<f16> = host_std_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up");
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let dev_gamma = DeviceBuffer::from_slice(&ctx, &host_gamma).expect("up");
    let dev_mean = DeviceBuffer::from_slice(&ctx, &host_mean).expect("up");
    let dev_std = DeviceBuffer::from_slice(&ctx, &host_std).expect("up");
    let mut dev_dx: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let mut dev_dgamma: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, extent).expect("alloc");
    let mut dev_dbeta: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, extent).expect("alloc");

    let desc = LayerNormBackwardDescriptor {
        input_shape: shape, norm_axes_mask: 2,
        has_gamma: true, has_beta: true,
        element: ElementKind::F16,
    };
    let plan = LayerNormBackwardPlan::<f16, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let save_shape = desc.save_shape();
    plan.run(&stream, Workspace::None, LayerNormBackwardArgs {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
        x:  TensorRef { data: dev_x.as_slice(),  shape, stride: contiguous_stride(shape) },
        gamma: Some(TensorRef { data: dev_gamma.as_slice(), shape: [extent as i32], stride: [1] }),
        mean: TensorRef {
            data: dev_mean.as_slice(), shape: save_shape, stride: contiguous_stride(save_shape),
        },
        inv_std: TensorRef {
            data: dev_std.as_slice(), shape: save_shape, stride: contiguous_stride(save_shape),
        },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        dgamma: Some(TensorMut { data: dev_dgamma.as_slice_mut(), shape: [extent as i32], stride: [1] }),
        dbeta:  Some(TensorMut { data: dev_dbeta.as_slice_mut(),  shape: [extent as i32], stride: [1] }),
    }).expect("run");
    stream.synchronize().expect("sync");

    let mut got_dx = vec![f16::ZERO; numel];
    let mut got_dgamma = vec![f16::ZERO; extent];
    let mut got_dbeta = vec![f16::ZERO; extent];
    dev_dx.copy_to_host(&mut got_dx).expect("dl");
    dev_dgamma.copy_to_host(&mut got_dgamma).expect("dl");
    dev_dbeta.copy_to_host(&mut got_dbeta).expect("dl");

    let eps_tol = 16.0 * F16_EPS;
    for i in 0..numel {
        let tol = (expected_dx_f32[i].abs() * eps_tol).max(eps_tol);
        let diff = (got_dx[i].to_f32() - expected_dx_f32[i]).abs();
        assert!(diff <= tol, "f16 layer_norm_bw dx @ {i}: diff={diff}");
    }
    let aff_tol = 32.0 * F16_EPS;
    for i in 0..extent {
        let tol = (expected_dgamma_f32[i].abs() * aff_tol).max(aff_tol);
        let diff = (got_dgamma[i].to_f32() - expected_dgamma_f32[i]).abs();
        assert!(diff <= tol, "f16 layer_norm_bw dgamma @ {i}: diff={diff}");
        let tol2 = (expected_dbeta_f32[i].abs() * aff_tol).max(aff_tol);
        let diff2 = (got_dbeta[i].to_f32() - expected_dbeta_f32[i]).abs();
        assert!(diff2 <= tol2, "f16 layer_norm_bw dbeta @ {i}: diff={diff2}");
    }
}

#[test]
#[ignore]
fn layer_norm_bw_bf16_no_affine() {
    let (ctx, stream) = setup();
    let shape = [2i32, 8];
    let numel = 16usize;
    let host_x_f32: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.22 - 1.0).cos()).collect();
    let host_dy_f32: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.15 + 0.6).sin()).collect();
    let eps = 1e-5f32;
    let (host_mean_f32, host_std_f32) = host_mean_invstd_f32(&shape, 1, &host_x_f32, eps);
    let (expected_dx_f32, _, _) =
        host_layer_norm_bw_f32(&shape, 1, &host_dy_f32, &host_x_f32, None, false, eps);
    let host_x: Vec<bf16> = host_x_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_dy: Vec<bf16> = host_dy_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_mean: Vec<bf16> = host_mean_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_std:  Vec<bf16> = host_std_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up");
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let dev_mean = DeviceBuffer::from_slice(&ctx, &host_mean).expect("up");
    let dev_std = DeviceBuffer::from_slice(&ctx, &host_std).expect("up");
    let mut dev_dx: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");

    let desc = LayerNormBackwardDescriptor {
        input_shape: shape, norm_axes_mask: 2,
        has_gamma: false, has_beta: false,
        element: ElementKind::Bf16,
    };
    let plan = LayerNormBackwardPlan::<bf16, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let save_shape = desc.save_shape();
    plan.run(&stream, Workspace::None, LayerNormBackwardArgs {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
        x:  TensorRef { data: dev_x.as_slice(),  shape, stride: contiguous_stride(shape) },
        gamma: None,
        mean: TensorRef {
            data: dev_mean.as_slice(), shape: save_shape, stride: contiguous_stride(save_shape),
        },
        inv_std: TensorRef {
            data: dev_std.as_slice(), shape: save_shape, stride: contiguous_stride(save_shape),
        },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        dgamma: None,
        dbeta:  None,
    }).expect("run");
    stream.synchronize().expect("sync");

    let mut got_dx = vec![bf16::ZERO; numel];
    dev_dx.copy_to_host(&mut got_dx).expect("dl");

    let eps_tol = 16.0 * BF16_EPS;
    for i in 0..numel {
        let tol = (expected_dx_f32[i].abs() * eps_tol).max(eps_tol);
        let diff = (got_dx[i].to_f32() - expected_dx_f32[i]).abs();
        assert!(diff <= tol, "bf16 layer_norm_bw dx @ {i}: diff={diff}");
    }
}
