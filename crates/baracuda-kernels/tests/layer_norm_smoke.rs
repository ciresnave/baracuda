//! Real-GPU smoke test for `LayerNormPlan` FW.
//!
//! Forward: `y = (x - mean) / sqrt(var + eps) * gamma + beta` with
//! biased (population) variance, matching PyTorch's `nn.LayerNorm`.
//! Tests cover affine on/off across all four dtypes.
//!
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, LayerNormArgs, LayerNormDescriptor, LayerNormPlan,
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

/// CPU reference. Returns `(y, mean_per_row, inv_std_per_row)`.
fn host_layer_norm_f32(
    shape: &[i32], norm_axis: usize, x: &[f32],
    gamma: Option<&[f32]>, beta: Option<&[f32]>, eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
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

    let mut y = vec![0f32; numel];
    let mut mean_out = vec![0f32; outer_numel];
    let mut inv_std_out = vec![0f32; outer_numel];

    for outer_lin in 0..outer_numel {
        let mut coord = vec![0i32; n];
        let mut rem = outer_lin;
        for d in (0..n).rev() {
            if d == norm_axis { coord[d] = 0; }
            else { coord[d] = (rem % outer_shape[d] as usize) as i32;
                   rem /= outer_shape[d] as usize; }
        }
        let mut off_save = 0i64;
        for d in 0..n { off_save += coord[d] as i64 * outer_stride[d]; }
        // mean
        let mut sum = 0f64;
        for j in 0..extent {
            coord[norm_axis] = j as i32;
            let mut idx = 0i64;
            for d in 0..n { idx += coord[d] as i64 * stride[d]; }
            sum += x[idx as usize] as f64;
        }
        let mean = sum / extent as f64;
        // biased variance
        let mut sumsq = 0f64;
        for j in 0..extent {
            coord[norm_axis] = j as i32;
            let mut idx = 0i64;
            for d in 0..n { idx += coord[d] as i64 * stride[d]; }
            let v = x[idx as usize] as f64 - mean;
            sumsq += v * v;
        }
        let var = sumsq / extent as f64;
        let inv_std = 1.0 / (var + eps as f64).sqrt();
        mean_out[off_save as usize] = mean as f32;
        inv_std_out[off_save as usize] = inv_std as f32;
        // y
        for j in 0..extent {
            coord[norm_axis] = j as i32;
            let mut idx = 0i64;
            for d in 0..n { idx += coord[d] as i64 * stride[d]; }
            let xh = (x[idx as usize] as f64 - mean) * inv_std;
            let g = gamma.map(|g| g[j]).unwrap_or(1.0) as f64;
            let b = beta.map(|b| b[j]).unwrap_or(0.0) as f64;
            y[idx as usize] = (xh * g + b) as f32;
        }
    }
    (y, mean_out, inv_std_out)
}

#[test]
#[ignore]
fn layer_norm_f32_full_affine() {
    let (ctx, stream) = setup();
    let shape = [2i32, 3, 8];
    let numel = 48usize;
    let extent = 8usize;
    let host_x: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.12 - 1.3).sin() * 1.4).collect();
    let host_gamma: Vec<f32> = (0..extent).map(|i| 0.7 + 0.05 * i as f32).collect();
    let host_beta:  Vec<f32> = (0..extent).map(|i| -0.2 + 0.03 * i as f32).collect();
    let eps = 1e-5f32;
    let (expected_y, expected_mean, expected_std) =
        host_layer_norm_f32(&shape, 2, &host_x, Some(&host_gamma), Some(&host_beta), eps);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_gamma = DeviceBuffer::from_slice(&ctx, &host_gamma).expect("up gamma");
    let dev_beta = DeviceBuffer::from_slice(&ctx, &host_beta).expect("up beta");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let mut dev_mean: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, expected_mean.len()).expect("alloc mean");
    let mut dev_std: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, expected_std.len()).expect("alloc std");

    let desc = LayerNormDescriptor {
        input_shape: shape,
        norm_axes_mask: 4,
        eps,
        has_gamma: true,
        has_beta: true,
        element: ElementKind::F32,
    };
    let plan = LayerNormPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let save_shape = desc.save_shape();
    plan.run(&stream, Workspace::None, LayerNormArgs {
        x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
        gamma: Some(TensorRef { data: dev_gamma.as_slice(), shape: [extent as i32], stride: [1] }),
        beta:  Some(TensorRef { data: dev_beta.as_slice(),  shape: [extent as i32], stride: [1] }),
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        mean: TensorMut {
            data: dev_mean.as_slice_mut(), shape: save_shape, stride: contiguous_stride(save_shape),
        },
        inv_std: TensorMut {
            data: dev_std.as_slice_mut(), shape: save_shape, stride: contiguous_stride(save_shape),
        },
    }).expect("run");
    stream.synchronize().expect("sync");

    let mut got_y = vec![0f32; numel];
    let mut got_mean = vec![0f32; expected_mean.len()];
    let mut got_std = vec![0f32; expected_std.len()];
    dev_y.copy_to_host(&mut got_y).expect("dl y");
    dev_mean.copy_to_host(&mut got_mean).expect("dl mean");
    dev_std.copy_to_host(&mut got_std).expect("dl std");

    let eps_tol = 16.0 * f32::EPSILON;
    for i in 0..numel {
        let tol = (expected_y[i].abs() * eps_tol).max(eps_tol);
        assert!((got_y[i] - expected_y[i]).abs() <= tol,
            "f32 layer_norm y @ {i}: got={} want={}", got_y[i], expected_y[i]);
    }
    for i in 0..expected_mean.len() {
        let tol = (expected_mean[i].abs() * eps_tol).max(eps_tol);
        assert!((got_mean[i] - expected_mean[i]).abs() <= tol,
            "f32 layer_norm mean @ {i}");
        let tol2 = (expected_std[i].abs() * eps_tol).max(eps_tol);
        assert!((got_std[i] - expected_std[i]).abs() <= tol2,
            "f32 layer_norm inv_std @ {i}");
    }
}

#[test]
#[ignore]
fn layer_norm_f64_no_affine() {
    let (ctx, stream) = setup();
    let shape = [3i32, 6];
    let numel = 18usize;
    let extent = 6usize;
    let host_x: Vec<f64> = (0..numel).map(|i| ((i as f64) * 0.2 - 1.0).cos() * 0.9).collect();
    let eps = 1e-6f32;
    // CPU f64 ref
    let mut expected_y = vec![0f64; numel];
    let mut expected_mean = vec![0f64; 3];
    let mut expected_std = vec![0f64; 3];
    for i in 0..3 {
        let mut sum = 0f64;
        for j in 0..extent { sum += host_x[i * 6 + j]; }
        let mean = sum / extent as f64;
        let mut sumsq = 0f64;
        for j in 0..extent { let v = host_x[i * 6 + j] - mean; sumsq += v * v; }
        let var = sumsq / extent as f64;
        let inv_std = 1.0 / (var + eps as f64).sqrt();
        expected_mean[i] = mean;
        expected_std[i] = inv_std;
        for j in 0..extent {
            expected_y[i * 6 + j] = (host_x[i * 6 + j] - mean) * inv_std;
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let mut dev_mean: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, 3).expect("alloc mean");
    let mut dev_std: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, 3).expect("alloc std");

    let desc = LayerNormDescriptor {
        input_shape: shape, norm_axes_mask: 2, eps,
        has_gamma: false, has_beta: false,
        element: ElementKind::F64,
    };
    let plan = LayerNormPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let save_shape = desc.save_shape();
    plan.run(&stream, Workspace::None, LayerNormArgs {
        x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
        gamma: None, beta: None,
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        mean: TensorMut {
            data: dev_mean.as_slice_mut(), shape: save_shape, stride: contiguous_stride(save_shape),
        },
        inv_std: TensorMut {
            data: dev_std.as_slice_mut(), shape: save_shape, stride: contiguous_stride(save_shape),
        },
    }).expect("run");
    stream.synchronize().expect("sync");

    let mut got_y = vec![0f64; numel];
    let mut got_mean = vec![0f64; 3];
    let mut got_std = vec![0f64; 3];
    dev_y.copy_to_host(&mut got_y).expect("dl y");
    dev_mean.copy_to_host(&mut got_mean).expect("dl mean");
    dev_std.copy_to_host(&mut got_std).expect("dl std");

    let eps_tol = 16.0 * f64::EPSILON;
    for i in 0..numel {
        let tol = (expected_y[i].abs() * eps_tol).max(eps_tol);
        assert!((got_y[i] - expected_y[i]).abs() <= tol,
            "f64 layer_norm y @ {i}");
    }
    for i in 0..3 {
        let tol = (expected_mean[i].abs() * eps_tol).max(eps_tol);
        assert!((got_mean[i] - expected_mean[i]).abs() <= tol, "f64 layer_norm mean @ {i}");
        let tol2 = (expected_std[i].abs() * eps_tol).max(eps_tol);
        assert!((got_std[i] - expected_std[i]).abs() <= tol2, "f64 layer_norm inv_std @ {i}");
    }
}

#[test]
#[ignore]
fn layer_norm_f16_full_affine() {
    let (ctx, stream) = setup();
    let shape = [3i32, 8];
    let numel = 24usize;
    let extent = 8usize;
    let host_x_f32: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.18 - 1.4).sin()).collect();
    let host_gamma_f32: Vec<f32> = (0..extent).map(|i| 0.8 + 0.04 * i as f32).collect();
    let host_beta_f32:  Vec<f32> = (0..extent).map(|i| -0.1 + 0.02 * i as f32).collect();
    let eps = 1e-5f32;
    let (expected_y_f32, _, _) =
        host_layer_norm_f32(&shape, 1, &host_x_f32, Some(&host_gamma_f32),
                              Some(&host_beta_f32), eps);

    let host_x: Vec<f16> = host_x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_gamma: Vec<f16> = host_gamma_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_beta:  Vec<f16> = host_beta_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let dev_gamma = DeviceBuffer::from_slice(&ctx, &host_gamma).expect("up");
    let dev_beta = DeviceBuffer::from_slice(&ctx, &host_beta).expect("up");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let mut dev_mean: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, 3).expect("alloc");
    let mut dev_std: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, 3).expect("alloc");

    let desc = LayerNormDescriptor {
        input_shape: shape, norm_axes_mask: 2, eps,
        has_gamma: true, has_beta: true,
        element: ElementKind::F16,
    };
    let plan = LayerNormPlan::<f16, 2>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let save_shape = desc.save_shape();
    plan.run(&stream, Workspace::None, LayerNormArgs {
        x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
        gamma: Some(TensorRef { data: dev_gamma.as_slice(), shape: [extent as i32], stride: [1] }),
        beta:  Some(TensorRef { data: dev_beta.as_slice(),  shape: [extent as i32], stride: [1] }),
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        mean: TensorMut {
            data: dev_mean.as_slice_mut(), shape: save_shape, stride: contiguous_stride(save_shape),
        },
        inv_std: TensorMut {
            data: dev_std.as_slice_mut(), shape: save_shape, stride: contiguous_stride(save_shape),
        },
    }).expect("run");
    stream.synchronize().expect("sync");

    let mut got_y = vec![f16::ZERO; numel];
    dev_y.copy_to_host(&mut got_y).expect("dl");

    let eps_tol = 8.0 * F16_EPS;
    for i in 0..numel {
        let tol = (expected_y_f32[i].abs() * eps_tol).max(eps_tol);
        let diff = (got_y[i].to_f32() - expected_y_f32[i]).abs();
        assert!(diff <= tol, "f16 layer_norm y @ {i}: diff={diff}");
    }
}

#[test]
#[ignore]
fn layer_norm_bf16_no_affine() {
    let (ctx, stream) = setup();
    let shape = [2i32, 8];
    let numel = 16usize;
    let host_x_f32: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.22 - 1.0).cos()).collect();
    let eps = 1e-5f32;
    let (expected_y_f32, _, _) = host_layer_norm_f32(&shape, 1, &host_x_f32, None, None, eps);
    let host_x: Vec<bf16> = host_x_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let mut dev_mean: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, 2).expect("alloc");
    let mut dev_std: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, 2).expect("alloc");

    let desc = LayerNormDescriptor {
        input_shape: shape, norm_axes_mask: 2, eps,
        has_gamma: false, has_beta: false,
        element: ElementKind::Bf16,
    };
    let plan = LayerNormPlan::<bf16, 2>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let save_shape = desc.save_shape();
    plan.run(&stream, Workspace::None, LayerNormArgs {
        x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
        gamma: None, beta: None,
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        mean: TensorMut {
            data: dev_mean.as_slice_mut(), shape: save_shape, stride: contiguous_stride(save_shape),
        },
        inv_std: TensorMut {
            data: dev_std.as_slice_mut(), shape: save_shape, stride: contiguous_stride(save_shape),
        },
    }).expect("run");
    stream.synchronize().expect("sync");

    let mut got_y = vec![bf16::ZERO; numel];
    dev_y.copy_to_host(&mut got_y).expect("dl");

    let eps_tol = 8.0 * BF16_EPS;
    for i in 0..numel {
        let tol = (expected_y_f32[i].abs() * eps_tol).max(eps_tol);
        let diff = (got_y[i].to_f32() - expected_y_f32[i]).abs();
        assert!(diff <= tol, "bf16 layer_norm y @ {i}: diff={diff}");
    }
}
