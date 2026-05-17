//! Real-GPU smoke test for `SoftmaxBackwardPlan + SoftmaxKind::LogSoftmax`.
//!
//! Forward: `y = log_softmax(x, axis)`. Backward:
//! `dx[k] = dy[k] - exp(y[k]) · Σ_j dy[j]`. Needs saved log-softmax
//! output `y` (so `exp(y) ∈ [0, 1]` recovers the softmax probabilities).

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, SoftmaxBackwardArgs,
    SoftmaxBackwardDescriptor, SoftmaxBackwardPlan, SoftmaxKind, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn host_log_softmax_f32(shape: [i32; 2], axis: usize, x: &[f32]) -> Vec<f32> {
    let extent = shape[axis] as usize;
    let other = shape[1 - axis] as usize;
    let numel = (shape[0] * shape[1]) as usize;
    let mut y = vec![0f32; numel];
    let (row_stride, col_stride) = (shape[1] as usize, 1usize);
    for o in 0..other {
        let idx = |j: usize| -> usize {
            if axis == 1 { o * row_stride + j * col_stride }
            else { j * row_stride + o * col_stride }
        };
        let mut max = f32::NEG_INFINITY;
        for j in 0..extent {
            let v = x[idx(j)];
            if v > max { max = v; }
        }
        let mut sum = 0f32;
        for j in 0..extent {
            sum += (x[idx(j)] - max).exp();
        }
        let log_sum = sum.ln();
        for j in 0..extent {
            y[idx(j)] = (x[idx(j)] - max) - log_sum;
        }
    }
    y
}

fn host_log_softmax_bw_f32(
    shape: [i32; 2],
    axis: usize,
    y: &[f32],
    dy: &[f32],
) -> Vec<f32> {
    let extent = shape[axis] as usize;
    let other = shape[1 - axis] as usize;
    let numel = (shape[0] * shape[1]) as usize;
    let mut dx = vec![0f32; numel];
    for o in 0..other {
        let idx = |j: usize| -> usize {
            if axis == 1 { o * shape[1] as usize + j }
            else { j * shape[1] as usize + o }
        };
        let mut dy_sum = 0f32;
        for j in 0..extent {
            dy_sum += dy[idx(j)];
        }
        for j in 0..extent {
            dx[idx(j)] = dy[idx(j)] - y[idx(j)].exp() * dy_sum;
        }
    }
    dx
}

#[test]
#[ignore]
fn log_softmax_bw_f32_2d_axis_1() {
    let (ctx, stream) = setup();
    let shape = [4i32, 8];
    let numel = 32;
    let host_x: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.3 - 4.0).sin()).collect();
    let host_y = host_log_softmax_f32(shape, 1, &host_x);
    let host_dy: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.1 - 0.5).cos()).collect();
    let expected = host_log_softmax_bw_f32(shape, 1, &host_y, &host_dy);

    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up y");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let desc = SoftmaxBackwardDescriptor {
        kind: SoftmaxKind::LogSoftmax,
        input_shape: shape,
        softmax_axis: 1,
        element: ElementKind::F32,
    };
    let plan = SoftmaxBackwardPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run(&stream, Workspace::None, SoftmaxBackwardArgs {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
        y: TensorRef { data: dev_y.as_slice(), shape, stride: contiguous_stride(shape) },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
    }).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; numel];
    dev_dx.copy_to_host(&mut got).expect("dl");
    let eps = 8.0 * f32::EPSILON;
    for i in 0..numel {
        let tol = (expected[i].abs() * eps).max(eps);
        assert!((got[i] - expected[i]).abs() <= tol,
            "f32 log_softmax BW @ {i}: got={} want={}", got[i], expected[i]);
    }
}

#[test]
#[ignore]
fn log_softmax_bw_f64_2d_axis_0() {
    let (ctx, stream) = setup();
    let shape = [6i32, 5];
    let numel = 30;
    let host_x: Vec<f64> = (0..numel).map(|i| ((i as f64) * 0.2 - 3.0).sin()).collect();
    let host_dy: Vec<f64> = (0..numel).map(|i| ((i as f64) * 0.1 - 0.5).cos()).collect();
    // f64 reference log-softmax (axis=0).
    let mut host_y = vec![0f64; numel];
    for o in 0..5 {
        let mut max = f64::NEG_INFINITY;
        for j in 0..6 {
            let v = host_x[j * 5 + o];
            if v > max { max = v; }
        }
        let mut sum = 0f64;
        for j in 0..6 {
            sum += (host_x[j * 5 + o] - max).exp();
        }
        let log_sum = sum.ln();
        for j in 0..6 {
            host_y[j * 5 + o] = (host_x[j * 5 + o] - max) - log_sum;
        }
    }
    let mut expected = vec![0f64; numel];
    for o in 0..5 {
        let mut dy_sum = 0f64;
        for j in 0..6 {
            dy_sum += host_dy[j * 5 + o];
        }
        for j in 0..6 {
            expected[j * 5 + o] = host_dy[j * 5 + o] - host_y[j * 5 + o].exp() * dy_sum;
        }
    }

    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up y");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_dx: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let desc = SoftmaxBackwardDescriptor {
        kind: SoftmaxKind::LogSoftmax,
        input_shape: shape,
        softmax_axis: 0,
        element: ElementKind::F64,
    };
    let plan = SoftmaxBackwardPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run(&stream, Workspace::None, SoftmaxBackwardArgs {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
        y: TensorRef { data: dev_y.as_slice(), shape, stride: contiguous_stride(shape) },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
    }).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f64; numel];
    dev_dx.copy_to_host(&mut got).expect("dl");
    let eps = 8.0 * f64::EPSILON;
    for i in 0..numel {
        let tol = (expected[i].abs() * eps).max(eps);
        assert!((got[i] - expected[i]).abs() <= tol, "f64 log_softmax BW @ {i}");
    }
}

#[test]
#[ignore]
fn log_softmax_bw_f16_2d() {
    let (ctx, stream) = setup();
    let shape = [3i32, 6];
    let numel = 18;
    let host_x_f32: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.2 - 2.0).sin()).collect();
    let host_y_f32 = host_log_softmax_f32(shape, 1, &host_x_f32);
    let host_dy_f32: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.15 - 0.5).cos()).collect();
    let expected_f32 = host_log_softmax_bw_f32(shape, 1, &host_y_f32, &host_dy_f32);

    let host_y: Vec<f16> = host_y_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_dy: Vec<f16> = host_dy_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up y");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_dx: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let desc = SoftmaxBackwardDescriptor {
        kind: SoftmaxKind::LogSoftmax,
        input_shape: shape,
        softmax_axis: 1,
        element: ElementKind::F16,
    };
    let plan = SoftmaxBackwardPlan::<f16, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run(&stream, Workspace::None, SoftmaxBackwardArgs {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
        y: TensorRef { data: dev_y.as_slice(), shape, stride: contiguous_stride(shape) },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
    }).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![f16::ZERO; numel];
    dev_dx.copy_to_host(&mut got).expect("dl");
    let eps = 8.0 * 9.77e-4_f32;
    for i in 0..numel {
        let tol = (expected_f32[i].abs() * eps).max(eps);
        let diff = (got[i].to_f32() - expected_f32[i]).abs();
        assert!(diff <= tol, "f16 log_softmax BW @ {i}: diff={diff}");
    }
}

#[test]
#[ignore]
fn log_softmax_bw_bf16_2d() {
    let (ctx, stream) = setup();
    let shape = [4i32, 8];
    let numel = 32;
    let host_x_f32: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.25 - 3.0).cos()).collect();
    let host_y_f32 = host_log_softmax_f32(shape, 1, &host_x_f32);
    let host_dy_f32: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.1 - 0.5).sin()).collect();
    let expected_f32 = host_log_softmax_bw_f32(shape, 1, &host_y_f32, &host_dy_f32);

    let host_y: Vec<bf16> = host_y_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_dy: Vec<bf16> = host_dy_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up y");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_dx: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let desc = SoftmaxBackwardDescriptor {
        kind: SoftmaxKind::LogSoftmax,
        input_shape: shape,
        softmax_axis: 1,
        element: ElementKind::Bf16,
    };
    let plan = SoftmaxBackwardPlan::<bf16, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run(&stream, Workspace::None, SoftmaxBackwardArgs {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
        y: TensorRef { data: dev_y.as_slice(), shape, stride: contiguous_stride(shape) },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
    }).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![bf16::ZERO; numel];
    dev_dx.copy_to_host(&mut got).expect("dl");
    let eps = 8.0 * 7.81e-3_f32;
    for i in 0..numel {
        let tol = (expected_f32[i].abs() * eps).max(eps);
        let diff = (got[i].to_f32() - expected_f32[i]).abs();
        assert!(diff <= tol, "bf16 log_softmax BW @ {i}: diff={diff}");
    }
}
