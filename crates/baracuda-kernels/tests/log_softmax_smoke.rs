//! Real-GPU smoke test for `SoftmaxPlan + SoftmaxKind::LogSoftmax`.
//!
//! Forward: `y[k] = (x[k] - max(x)) - log(Σ_j exp(x[j] - max(x)))`,
//! numerically stable via max subtraction. Each y is a log-probability
//! and lives in `[-INF, 0]`; we verify the per-cell formula and that
//! `Σ_j exp(y[j]) = 1` (invariant).

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, SoftmaxArgs, SoftmaxDescriptor,
    SoftmaxKind, SoftmaxPlan, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn host_log_softmax_f32(input_shape: [i32; 3], axis: usize, x: &[f32]) -> Vec<f32> {
    let mut stride = [1usize; 3];
    for d in (0..3).rev().skip(1) {
        stride[d] = stride[d + 1] * input_shape[d + 1] as usize;
    }
    let numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let mut y = vec![0f32; numel];
    let extent = input_shape[axis] as usize;
    let mut outer_shape = input_shape;
    outer_shape[axis] = 1;
    let outer_numel: usize = outer_shape.iter().map(|&d| d as usize).product();
    for outer_lin in 0..outer_numel {
        let mut coord = [0i32; 3];
        let mut rem = outer_lin;
        for d in (0..3).rev() {
            if d == axis {
                coord[d] = 0;
            } else {
                coord[d] = (rem % outer_shape[d] as usize) as i32;
                rem /= outer_shape[d] as usize;
            }
        }
        let mut max = f32::NEG_INFINITY;
        for j in 0..extent {
            coord[axis] = j as i32;
            let mut idx = 0usize;
            for d in 0..3 {
                idx += coord[d] as usize * stride[d];
            }
            if x[idx] > max {
                max = x[idx];
            }
        }
        let mut sum = 0f32;
        for j in 0..extent {
            coord[axis] = j as i32;
            let mut idx = 0usize;
            for d in 0..3 {
                idx += coord[d] as usize * stride[d];
            }
            sum += (x[idx] - max).exp();
        }
        let log_sum = sum.ln();
        for j in 0..extent {
            coord[axis] = j as i32;
            let mut idx = 0usize;
            for d in 0..3 {
                idx += coord[d] as usize * stride[d];
            }
            y[idx] = (x[idx] - max) - log_sum;
        }
    }
    y
}

#[test]
#[ignore]
fn log_softmax_f32_3d_axis_1() {
    let (ctx, stream) = setup();
    let shape = [2i32, 5, 4];
    let numel = 40;
    let host_x: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.3 - 4.0).sin()).collect();
    let expected = host_log_softmax_f32(shape, 1, &host_x);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let desc = SoftmaxDescriptor {
        kind: SoftmaxKind::LogSoftmax,
        input_shape: shape,
        softmax_axis: 1,
        element: ElementKind::F32,
    };
    let plan =
        SoftmaxPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    plan.run(&stream, Workspace::None, SoftmaxArgs {
        x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
    }).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("dl");
    let eps = 8.0 * f32::EPSILON;
    for i in 0..numel {
        // log-softmax outputs live near 0..-15 so absolute eps is fine.
        let tol = (expected[i].abs() * eps).max(eps);
        assert!((got[i] - expected[i]).abs() <= tol,
            "f32 log_softmax @ {i}: got={} want={}", got[i], expected[i]);
    }
    // Σ_j exp(y[j]) = 1 invariant.
    for outer_i in 0..(2 * 4) {
        let i = outer_i / 4;
        let k = outer_i % 4;
        let mut sum = 0f32;
        for j in 0..5 {
            sum += got[(i * 5 + j) * 4 + k].exp();
        }
        assert!((sum - 1.0).abs() <= 1e-5, "f32 exp-row-sum @ ({i},{k}) = {sum}");
    }
}

#[test]
#[ignore]
fn log_softmax_f64_2d_axis_1() {
    let (ctx, stream) = setup();
    let shape = [4i32, 8];
    let numel = 32;
    let host_x: Vec<f64> = (0..numel).map(|i| ((i as f64) * 0.2 - 3.0).sin()).collect();
    let extent = 8;
    let mut expected = vec![0f64; numel];
    for i in 0..4 {
        let mut max = f64::NEG_INFINITY;
        for j in 0..extent {
            let v = host_x[i * 8 + j];
            if v > max { max = v; }
        }
        let mut sum = 0f64;
        for j in 0..extent {
            sum += (host_x[i * 8 + j] - max).exp();
        }
        let log_sum = sum.ln();
        for j in 0..extent {
            expected[i * 8 + j] = (host_x[i * 8 + j] - max) - log_sum;
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let desc = SoftmaxDescriptor {
        kind: SoftmaxKind::LogSoftmax,
        input_shape: shape,
        softmax_axis: 1,
        element: ElementKind::F64,
    };
    let plan =
        SoftmaxPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    plan.run(&stream, Workspace::None, SoftmaxArgs {
        x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
    }).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f64; numel];
    dev_y.copy_to_host(&mut got).expect("dl");
    let eps = 8.0 * f64::EPSILON;
    for i in 0..numel {
        let tol = (expected[i].abs() * eps).max(eps);
        assert!((got[i] - expected[i]).abs() <= tol, "f64 log_softmax @ {i}");
    }
}

#[test]
#[ignore]
fn log_softmax_f16_2d_axis_1() {
    let (ctx, stream) = setup();
    let shape = [3i32, 8];
    let numel = 24;
    // Small input range keeps log-softmax output within f16's well-behaved zone.
    let host_x_f32: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.2 - 2.0).sin()).collect();
    let expected_f32 = host_log_softmax_f32([1i32, shape[0], shape[1]], 2, &host_x_f32);
    let host_x: Vec<f16> = host_x_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let desc = SoftmaxDescriptor {
        kind: SoftmaxKind::LogSoftmax,
        input_shape: shape,
        softmax_axis: 1,
        element: ElementKind::F16,
    };
    let plan =
        SoftmaxPlan::<f16, 2>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    plan.run(&stream, Workspace::None, SoftmaxArgs {
        x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
    }).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![f16::ZERO; numel];
    dev_y.copy_to_host(&mut got).expect("dl");
    let eps = 4.0 * 9.77e-4_f32;
    for i in 0..numel {
        let tol = (expected_f32[i].abs() * eps).max(eps);
        let diff = (got[i].to_f32() - expected_f32[i]).abs();
        assert!(diff <= tol, "f16 log_softmax @ {i}: diff={diff}");
    }
}

#[test]
#[ignore]
fn log_softmax_bf16_2d_axis_0() {
    let (ctx, stream) = setup();
    let shape = [6i32, 4];
    let numel = 24;
    let host_x_f32: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.25 - 3.0).cos()).collect();
    let expected_f32 = host_log_softmax_f32([1i32, shape[0], shape[1]], 1, &host_x_f32);
    let host_x: Vec<bf16> = host_x_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let desc = SoftmaxDescriptor {
        kind: SoftmaxKind::LogSoftmax,
        input_shape: shape,
        softmax_axis: 0,
        element: ElementKind::Bf16,
    };
    let plan =
        SoftmaxPlan::<bf16, 2>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    plan.run(&stream, Workspace::None, SoftmaxArgs {
        x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
    }).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![bf16::ZERO; numel];
    dev_y.copy_to_host(&mut got).expect("dl");
    let eps = 4.0 * 7.81e-3_f32;
    for i in 0..numel {
        let tol = (expected_f32[i].abs() * eps).max(eps);
        let diff = (got[i].to_f32() - expected_f32[i]).abs();
        assert!(diff <= tol, "bf16 log_softmax @ {i}: diff={diff}");
    }
}
