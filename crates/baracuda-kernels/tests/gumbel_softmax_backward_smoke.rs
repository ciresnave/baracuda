//! Real-GPU smoke test for `GumbelSoftmaxBackwardPlan` (Milestone 5.4).
//!
//! BW formula: `dx = y_soft · (dy - Σ y_soft · dy)` (modulo the
//! per-temperature `1/τ` scaling — callers bake it into their loss
//! chain, see the BW plan docs). The kernel reuses the existing
//! softmax_backward kernel, so we verify against the analytic softmax
//! BW formula directly.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, GumbelSoftmaxBackwardArgs,
    GumbelSoftmaxBackwardDescriptor, GumbelSoftmaxBackwardPlan, PlanPreference,
    TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn host_softmax_bw_f32(shape: [i32; 2], axis: usize, y: &[f32], dy: &[f32]) -> Vec<f32> {
    let extent = shape[axis] as usize;
    let other = shape[1 - axis] as usize;
    let numel = (shape[0] * shape[1]) as usize;
    let mut dx = vec![0f32; numel];
    for o in 0..other {
        let idx = |j: usize| -> usize {
            if axis == 1 { o * shape[1] as usize + j } else { j * shape[1] as usize + o }
        };
        let mut dot = 0f32;
        for j in 0..extent {
            dot += y[idx(j)] * dy[idx(j)];
        }
        for j in 0..extent {
            dx[idx(j)] = y[idx(j)] * (dy[idx(j)] - dot);
        }
    }
    dx
}

#[test]
#[ignore]
fn gumbel_softmax_bw_f32_2d() {
    let (ctx, stream) = setup();
    let shape = [3i32, 6];
    let numel = 18usize;
    // y is a softmax output (rows sum to 1).
    let host_y: Vec<f32> = {
        let mut v = vec![0f32; numel];
        for row in 0..3 {
            let mut row_sum = 0f32;
            for j in 0..6 {
                let raw = ((row * 6 + j) as f32 * 0.3 - 1.0).sin().exp();
                v[row * 6 + j] = raw;
                row_sum += raw;
            }
            for j in 0..6 {
                v[row * 6 + j] /= row_sum;
            }
        }
        v
    };
    let host_dy: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.1 - 0.5).cos()).collect();
    let expected = host_softmax_bw_f32(shape, 1, &host_y, &host_dy);

    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up y");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");

    let desc = GumbelSoftmaxBackwardDescriptor {
        input_shape: shape,
        softmax_axis: 1,
        temperature: 1.0,
        element: ElementKind::F32,
    };
    let plan = GumbelSoftmaxBackwardPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run(
        &stream,
        Workspace::None,
        GumbelSoftmaxBackwardArgs {
            dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
            y: TensorRef { data: dev_y.as_slice(), shape, stride: contiguous_stride(shape) },
            dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; numel];
    dev_dx.copy_to_host(&mut got).expect("dl");
    let eps = 8.0 * f32::EPSILON;
    for i in 0..numel {
        let tol = (expected[i].abs() * eps).max(eps);
        assert!((got[i] - expected[i]).abs() <= tol,
            "f32 gumbel BW @ {i}: got={} want={}", got[i], expected[i]);
    }
}

#[test]
#[ignore]
fn gumbel_softmax_bw_f64_2d() {
    let (ctx, stream) = setup();
    let shape = [3i32, 4];
    let numel = 12usize;
    let host_y: Vec<f64> = {
        let mut v = vec![0f64; numel];
        for row in 0..3 {
            let mut row_sum = 0f64;
            for j in 0..4 {
                let raw = ((row * 4 + j) as f64 * 0.3 - 1.0).sin().exp();
                v[row * 4 + j] = raw;
                row_sum += raw;
            }
            for j in 0..4 {
                v[row * 4 + j] /= row_sum;
            }
        }
        v
    };
    let host_dy: Vec<f64> = (0..numel).map(|i| ((i as f64) * 0.1 - 0.5).cos()).collect();
    let mut expected = vec![0f64; numel];
    for row in 0..3 {
        let mut dot = 0f64;
        for j in 0..4 { dot += host_y[row * 4 + j] * host_dy[row * 4 + j]; }
        for j in 0..4 {
            expected[row * 4 + j] = host_y[row * 4 + j] * (host_dy[row * 4 + j] - dot);
        }
    }

    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up y");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_dx: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");

    let desc = GumbelSoftmaxBackwardDescriptor {
        input_shape: shape,
        softmax_axis: 1,
        temperature: 1.0,
        element: ElementKind::F64,
    };
    let plan = GumbelSoftmaxBackwardPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run(
        &stream,
        Workspace::None,
        GumbelSoftmaxBackwardArgs {
            dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
            y: TensorRef { data: dev_y.as_slice(), shape, stride: contiguous_stride(shape) },
            dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f64; numel];
    dev_dx.copy_to_host(&mut got).expect("dl");
    let eps = 8.0 * f64::EPSILON;
    for i in 0..numel {
        let tol = (expected[i].abs() * eps).max(eps);
        assert!((got[i] - expected[i]).abs() <= tol, "f64 gumbel BW @ {i}");
    }
}

#[test]
#[ignore]
fn gumbel_softmax_bw_f16_2d() {
    let (ctx, stream) = setup();
    let shape = [3i32, 6];
    let numel = 18usize;
    let host_y_f32: Vec<f32> = {
        let mut v = vec![0f32; numel];
        for row in 0..3 {
            let mut s = 0f32;
            for j in 0..6 {
                let raw = ((row * 6 + j) as f32 * 0.3 - 1.0).sin().exp();
                v[row * 6 + j] = raw;
                s += raw;
            }
            for j in 0..6 { v[row * 6 + j] /= s; }
        }
        v
    };
    let host_dy_f32: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.1).cos()).collect();
    let expected_f32 = host_softmax_bw_f32(shape, 1, &host_y_f32, &host_dy_f32);

    let host_y: Vec<f16> = host_y_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_dy: Vec<f16> = host_dy_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up y");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_dx: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");

    let desc = GumbelSoftmaxBackwardDescriptor {
        input_shape: shape,
        softmax_axis: 1,
        temperature: 1.0,
        element: ElementKind::F16,
    };
    let plan = GumbelSoftmaxBackwardPlan::<f16, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run(
        &stream,
        Workspace::None,
        GumbelSoftmaxBackwardArgs {
            dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
            y: TensorRef { data: dev_y.as_slice(), shape, stride: contiguous_stride(shape) },
            dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![f16::ZERO; numel];
    dev_dx.copy_to_host(&mut got).expect("dl");
    let eps = 8.0 * 9.77e-4_f32;
    for i in 0..numel {
        let tol = (expected_f32[i].abs() * eps).max(eps);
        let diff = (got[i].to_f32() - expected_f32[i]).abs();
        assert!(diff <= tol, "f16 gumbel BW @ {i}: diff={diff}");
    }
}

#[test]
#[ignore]
fn gumbel_softmax_bw_bf16_2d() {
    let (ctx, stream) = setup();
    let shape = [3i32, 6];
    let numel = 18usize;
    let host_y_f32: Vec<f32> = {
        let mut v = vec![0f32; numel];
        for row in 0..3 {
            let mut s = 0f32;
            for j in 0..6 {
                let raw = ((row * 6 + j) as f32 * 0.25 - 0.5).cos().exp();
                v[row * 6 + j] = raw;
                s += raw;
            }
            for j in 0..6 { v[row * 6 + j] /= s; }
        }
        v
    };
    let host_dy_f32: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.1).sin()).collect();
    let expected_f32 = host_softmax_bw_f32(shape, 1, &host_y_f32, &host_dy_f32);

    let host_y: Vec<bf16> = host_y_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_dy: Vec<bf16> = host_dy_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up y");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_dx: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");

    let desc = GumbelSoftmaxBackwardDescriptor {
        input_shape: shape,
        softmax_axis: 1,
        temperature: 1.0,
        element: ElementKind::Bf16,
    };
    let plan = GumbelSoftmaxBackwardPlan::<bf16, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run(
        &stream,
        Workspace::None,
        GumbelSoftmaxBackwardArgs {
            dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
            y: TensorRef { data: dev_y.as_slice(), shape, stride: contiguous_stride(shape) },
            dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![bf16::ZERO; numel];
    dev_dx.copy_to_host(&mut got).expect("dl");
    let eps = 8.0 * 7.81e-3_f32;
    for i in 0..numel {
        let tol = (expected_f32[i].abs() * eps).max(eps);
        let diff = (got[i].to_f32() - expected_f32[i]).abs();
        assert!(diff <= tol, "bf16 gumbel BW @ {i}: diff={diff}");
    }
}
