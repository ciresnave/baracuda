//! Real-GPU smoke test for `RMSNormPlan + NormalizationKind::RMSNorm` FW.
//!
//! Forward: `y = x / sqrt(mean(x², dim=norm_axis) + eps) * gamma`. We
//! verify the per-cell formula against a host f32 reference, and also
//! that the `rms` save buffer carries the expected per-row RMS for
//! reuse by BW. Each test covers both `has_gamma=true` and a no-gamma
//! variant.
//!
//! `#[ignore]` by default — requires a real CUDA device.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, RMSNormArgs, RMSNormDescriptor, RMSNormPlan,
    TensorMut, TensorRef, Workspace,
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

/// CPU reference for RMSNorm on rank-N tensors (`N <= 4`).
///
/// Returns `(y, rms_per_row)`. `rms_per_row` shape == input with
/// `norm_axis` collapsed to 1.
fn host_rms_norm_f32(
    shape: &[i32], norm_axis: usize, x: &[f32], gamma: Option<&[f32]>, eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = shape.len();
    // row-major strides
    let mut stride = vec![1i64; n];
    for d in (0..n - 1).rev() {
        stride[d] = stride[d + 1] * shape[d + 1] as i64;
    }
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let extent = shape[norm_axis] as usize;
    // outer shape — collapse norm_axis to 1
    let mut outer_shape: Vec<i32> = shape.to_vec();
    outer_shape[norm_axis] = 1;
    let outer_numel: usize = outer_shape.iter().map(|&d| d as usize).product();
    let mut outer_stride = vec![1i64; n];
    for d in (0..n - 1).rev() {
        outer_stride[d] = outer_stride[d + 1] * outer_shape[d + 1] as i64;
    }

    let mut y = vec![0f32; numel];
    let mut rms_out = vec![0f32; outer_numel];

    for outer_lin in 0..outer_numel {
        // unravel outer_lin over outer_shape
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
        for d in 0..n {
            off_rms += coord[d] as i64 * outer_stride[d];
        }
        // sum(x²)
        let mut sum_sq = 0f64;
        for j in 0..extent {
            coord[norm_axis] = j as i32;
            let mut idx = 0i64;
            for d in 0..n { idx += coord[d] as i64 * stride[d]; }
            let v = x[idx as usize] as f64;
            sum_sq += v * v;
        }
        let mean_sq = sum_sq / extent as f64;
        let rms = (mean_sq + eps as f64).sqrt() as f32;
        rms_out[off_rms as usize] = rms;
        // write y
        for j in 0..extent {
            coord[norm_axis] = j as i32;
            let mut idx = 0i64;
            for d in 0..n { idx += coord[d] as i64 * stride[d]; }
            let g = gamma.map(|g| g[j]).unwrap_or(1.0);
            y[idx as usize] = (x[idx as usize] / rms) * g;
        }
    }
    (y, rms_out)
}

#[test]
#[ignore]
fn rms_norm_f32_with_gamma() {
    let (ctx, stream) = setup();
    let shape = [3i32, 4, 8];
    let numel = 96usize;
    let extent = 8usize;
    let host_x: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.13 - 1.7).sin() * 1.5).collect();
    let host_gamma: Vec<f32> = (0..extent).map(|i| 0.5 + 0.1 * i as f32).collect();
    let eps = 1e-5f32;
    let (expected_y, expected_rms) =
        host_rms_norm_f32(&shape, 2, &host_x, Some(&host_gamma), eps);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_gamma = DeviceBuffer::from_slice(&ctx, &host_gamma).expect("up gamma");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let mut dev_rms: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, expected_rms.len())
        .expect("alloc rms");

    let desc = RMSNormDescriptor {
        input_shape: shape,
        norm_axes_mask: 4,
        eps,
        has_gamma: true,
        element: ElementKind::F32,
    };
    let plan = RMSNormPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let rms_shape = desc.rms_shape();
    plan.run(&stream, Workspace::None, RMSNormArgs {
        x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
        gamma: Some(TensorRef {
            data: dev_gamma.as_slice(), shape: [extent as i32], stride: [1],
        }),
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        rms: TensorMut {
            data: dev_rms.as_slice_mut(), shape: rms_shape, stride: contiguous_stride(rms_shape),
        },
    }).expect("run");
    stream.synchronize().expect("sync");

    let mut got_y = vec![0f32; numel];
    let mut got_rms = vec![0f32; expected_rms.len()];
    dev_y.copy_to_host(&mut got_y).expect("dl y");
    dev_rms.copy_to_host(&mut got_rms).expect("dl rms");

    // 16·eps relative tolerance for FW.
    let eps_tol = 16.0 * f32::EPSILON;
    for i in 0..numel {
        let tol = (expected_y[i].abs() * eps_tol).max(eps_tol);
        assert!((got_y[i] - expected_y[i]).abs() <= tol,
            "f32 rms_norm y @ {i}: got={} want={}", got_y[i], expected_y[i]);
    }
    for i in 0..expected_rms.len() {
        let tol = (expected_rms[i].abs() * eps_tol).max(eps_tol);
        assert!((got_rms[i] - expected_rms[i]).abs() <= tol,
            "f32 rms_norm rms @ {i}: got={} want={}", got_rms[i], expected_rms[i]);
    }
}

#[test]
#[ignore]
fn rms_norm_f64_no_gamma() {
    let (ctx, stream) = setup();
    let shape = [2i32, 6];
    let numel = 12usize;
    let extent = 6usize;
    let host_x: Vec<f64> = (0..numel).map(|i| ((i as f64) * 0.21 - 1.1).cos() * 0.8).collect();
    let eps = 1e-6f32;
    // CPU ref in f64
    let mut expected_y = vec![0f64; numel];
    let mut expected_rms = vec![0f64; 2];
    for i in 0..2 {
        let mut sum_sq = 0f64;
        for j in 0..extent {
            let v = host_x[i * 6 + j];
            sum_sq += v * v;
        }
        let mean_sq = sum_sq / extent as f64;
        let rms = (mean_sq + eps as f64).sqrt();
        expected_rms[i] = rms;
        for j in 0..extent {
            expected_y[i * 6 + j] = host_x[i * 6 + j] / rms;
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let mut dev_rms: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, 2).expect("alloc rms");

    let desc = RMSNormDescriptor {
        input_shape: shape,
        norm_axes_mask: 2,
        eps,
        has_gamma: false,
        element: ElementKind::F64,
    };
    let plan = RMSNormPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let rms_shape = desc.rms_shape();
    plan.run(&stream, Workspace::None, RMSNormArgs {
        x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
        gamma: None,
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        rms: TensorMut {
            data: dev_rms.as_slice_mut(), shape: rms_shape, stride: contiguous_stride(rms_shape),
        },
    }).expect("run");
    stream.synchronize().expect("sync");

    let mut got_y = vec![0f64; numel];
    let mut got_rms = vec![0f64; 2];
    dev_y.copy_to_host(&mut got_y).expect("dl y");
    dev_rms.copy_to_host(&mut got_rms).expect("dl rms");

    let eps_tol = 16.0 * f64::EPSILON;
    for i in 0..numel {
        let tol = (expected_y[i].abs() * eps_tol).max(eps_tol);
        assert!((got_y[i] - expected_y[i]).abs() <= tol,
            "f64 rms_norm y @ {i}: got={} want={}", got_y[i], expected_y[i]);
    }
    for i in 0..2 {
        let tol = (expected_rms[i].abs() * eps_tol).max(eps_tol);
        assert!((got_rms[i] - expected_rms[i]).abs() <= tol, "f64 rms_norm rms @ {i}");
    }
}

#[test]
#[ignore]
fn rms_norm_f16_with_gamma() {
    let (ctx, stream) = setup();
    let shape = [4i32, 8];
    let numel = 32usize;
    let extent = 8usize;
    let host_x_f32: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.18 - 1.4).sin()).collect();
    let host_gamma_f32: Vec<f32> = (0..extent).map(|i| 1.0 + 0.05 * i as f32).collect();
    let eps = 1e-5f32;
    let (expected_y_f32, _) =
        host_rms_norm_f32(&shape, 1, &host_x_f32, Some(&host_gamma_f32), eps);

    let host_x: Vec<f16> = host_x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_gamma: Vec<f16> = host_gamma_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_gamma = DeviceBuffer::from_slice(&ctx, &host_gamma).expect("up gamma");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let mut dev_rms: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, 4).expect("alloc rms");

    let desc = RMSNormDescriptor {
        input_shape: shape,
        norm_axes_mask: 2,
        eps,
        has_gamma: true,
        element: ElementKind::F16,
    };
    let plan = RMSNormPlan::<f16, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let rms_shape = desc.rms_shape();
    plan.run(&stream, Workspace::None, RMSNormArgs {
        x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
        gamma: Some(TensorRef {
            data: dev_gamma.as_slice(), shape: [extent as i32], stride: [1],
        }),
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        rms: TensorMut {
            data: dev_rms.as_slice_mut(), shape: rms_shape, stride: contiguous_stride(rms_shape),
        },
    }).expect("run");
    stream.synchronize().expect("sync");

    let mut got_y = vec![f16::ZERO; numel];
    dev_y.copy_to_host(&mut got_y).expect("dl");

    let eps_tol = 8.0 * F16_EPS;
    for i in 0..numel {
        let tol = (expected_y_f32[i].abs() * eps_tol).max(eps_tol);
        let diff = (got_y[i].to_f32() - expected_y_f32[i]).abs();
        assert!(diff <= tol, "f16 rms_norm y @ {i}: diff={diff}");
    }
}

#[test]
#[ignore]
fn rms_norm_bf16_no_gamma() {
    let (ctx, stream) = setup();
    let shape = [3i32, 8];
    let numel = 24usize;
    let host_x_f32: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.22 - 1.0).cos()).collect();
    let eps = 1e-5f32;
    let (expected_y_f32, _) = host_rms_norm_f32(&shape, 1, &host_x_f32, None, eps);
    let host_x: Vec<bf16> = host_x_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let mut dev_rms: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, 3).expect("alloc rms");

    let desc = RMSNormDescriptor {
        input_shape: shape,
        norm_axes_mask: 2,
        eps,
        has_gamma: false,
        element: ElementKind::Bf16,
    };
    let plan = RMSNormPlan::<bf16, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let rms_shape = desc.rms_shape();
    plan.run(&stream, Workspace::None, RMSNormArgs {
        x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
        gamma: None,
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        rms: TensorMut {
            data: dev_rms.as_slice_mut(), shape: rms_shape, stride: contiguous_stride(rms_shape),
        },
    }).expect("run");
    stream.synchronize().expect("sync");

    let mut got_y = vec![bf16::ZERO; numel];
    dev_y.copy_to_host(&mut got_y).expect("dl");

    let eps_tol = 8.0 * BF16_EPS;
    for i in 0..numel {
        let tol = (expected_y_f32[i].abs() * eps_tol).max(eps_tol);
        let diff = (got_y[i].to_f32() - expected_y_f32[i]).abs();
        assert!(diff <= tol, "bf16 rms_norm y @ {i}: diff={diff}");
    }
}
