//! Real-GPU smoke test for `SparsemaxBackwardPlan` (Milestone 5.4).
//!
//! BW formula: for active positions (`y > 0`),
//!   `dx[i] = dy[i] - mean(dy[active])`;
//! else 0.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, SparsemaxBackwardArgs,
    SparsemaxBackwardDescriptor, SparsemaxBackwardPlan, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn host_sparsemax_bw_f32(
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
            if axis == 1 { o * shape[1] as usize + j } else { j * shape[1] as usize + o }
        };
        let mut sum_active = 0f32;
        let mut n_active = 0usize;
        for j in 0..extent {
            if y[idx(j)] > 0.0 {
                sum_active += dy[idx(j)];
                n_active += 1;
            }
        }
        let avg = if n_active > 0 { sum_active / n_active as f32 } else { 0.0 };
        for j in 0..extent {
            dx[idx(j)] = if y[idx(j)] > 0.0 { dy[idx(j)] - avg } else { 0.0 };
        }
    }
    dx
}

#[test]
#[ignore]
fn sparsemax_bw_f32_known() {
    let (ctx, stream) = setup();
    let shape = [1i32, 3];
    // Sparsemax of [2,1,0] = [0.5, 0.5, 0]. Active set: {0, 1}.
    // dy = [a, b, c]. Active mean = (a + b)/2. dx = [a - (a+b)/2, b - (a+b)/2, 0]
    //                                            = [(a-b)/2, (b-a)/2, 0].
    let host_y = vec![0.5_f32, 0.5, 0.0];
    let host_dy = vec![3.0_f32, 1.0, 2.0];
    let expected = vec![1.0_f32, -1.0, 0.0];

    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up y");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 3).expect("alloc");
    let desc = SparsemaxBackwardDescriptor {
        input_shape: shape,
        softmax_axis: 1,
        element: ElementKind::F32,
    };
    let plan = SparsemaxBackwardPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run(&stream, Workspace::None, SparsemaxBackwardArgs {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
        y: TensorRef { data: dev_y.as_slice(), shape, stride: contiguous_stride(shape) },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
    })
    .expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; 3];
    dev_dx.copy_to_host(&mut got).expect("dl");
    let eps = 8.0 * f32::EPSILON;
    for i in 0..3 {
        let tol = (expected[i].abs() * eps).max(eps);
        assert!((got[i] - expected[i]).abs() <= tol,
            "sparsemax BW @ {i}: got={} want={}", got[i], expected[i]);
    }
}

#[test]
#[ignore]
fn sparsemax_bw_f32_2d() {
    let (ctx, stream) = setup();
    let shape = [4i32, 6];
    let numel = 24usize;
    // Build a plausible `y` (some zeros, some nonzero, rows sum to 1).
    let mut host_y = vec![0f32; numel];
    for row in 0..4 {
        // Active for first 4 elements, others zero.
        for j in 0..4 {
            host_y[row * 6 + j] = 0.25;
        }
    }
    let host_dy: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.1 - 0.5).cos()).collect();
    let expected = host_sparsemax_bw_f32(shape, 1, &host_y, &host_dy);

    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up y");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let desc = SparsemaxBackwardDescriptor {
        input_shape: shape,
        softmax_axis: 1,
        element: ElementKind::F32,
    };
    let plan = SparsemaxBackwardPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run(&stream, Workspace::None, SparsemaxBackwardArgs {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
        y: TensorRef { data: dev_y.as_slice(), shape, stride: contiguous_stride(shape) },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
    })
    .expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; numel];
    dev_dx.copy_to_host(&mut got).expect("dl");
    let eps = 8.0 * f32::EPSILON;
    for i in 0..numel {
        let tol = (expected[i].abs() * eps).max(eps);
        assert!((got[i] - expected[i]).abs() <= tol,
            "f32 sparsemax BW @ {i}: got={} want={}", got[i], expected[i]);
    }
}

#[test]
#[ignore]
fn sparsemax_bw_f64_2d() {
    let (ctx, stream) = setup();
    let shape = [3i32, 5];
    let numel = 15usize;
    let mut host_y = vec![0f64; numel];
    for row in 0..3 {
        // First 3 cells active, last 2 zero.
        for j in 0..3 { host_y[row * 5 + j] = 1.0 / 3.0; }
    }
    let host_dy: Vec<f64> = (0..numel).map(|i| ((i as f64) * 0.1 - 0.5).cos()).collect();

    let mut expected = vec![0f64; numel];
    for row in 0..3 {
        let mut sum_active = 0f64;
        let mut n_active = 0usize;
        for j in 0..5 {
            if host_y[row * 5 + j] > 0.0 {
                sum_active += host_dy[row * 5 + j];
                n_active += 1;
            }
        }
        let avg = if n_active > 0 { sum_active / n_active as f64 } else { 0.0 };
        for j in 0..5 {
            expected[row * 5 + j] = if host_y[row * 5 + j] > 0.0 { host_dy[row * 5 + j] - avg } else { 0.0 };
        }
    }

    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up y");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_dx: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let desc = SparsemaxBackwardDescriptor {
        input_shape: shape,
        softmax_axis: 1,
        element: ElementKind::F64,
    };
    let plan = SparsemaxBackwardPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run(&stream, Workspace::None, SparsemaxBackwardArgs {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
        y: TensorRef { data: dev_y.as_slice(), shape, stride: contiguous_stride(shape) },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
    })
    .expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f64; numel];
    dev_dx.copy_to_host(&mut got).expect("dl");
    let eps = 8.0 * f64::EPSILON;
    for i in 0..numel {
        let tol = (expected[i].abs() * eps).max(eps);
        assert!((got[i] - expected[i]).abs() <= tol, "f64 sparsemax BW @ {i}");
    }
}

#[test]
#[ignore]
fn sparsemax_bw_f16_2d() {
    let (ctx, stream) = setup();
    let shape = [3i32, 5];
    let numel = 15usize;
    let mut host_y_f32 = vec![0f32; numel];
    for row in 0..3 {
        for j in 0..3 { host_y_f32[row * 5 + j] = 1.0 / 3.0; }
    }
    let host_dy_f32: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.1).cos()).collect();
    let expected_f32 = host_sparsemax_bw_f32(shape, 1, &host_y_f32, &host_dy_f32);

    let host_y: Vec<f16> = host_y_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_dy: Vec<f16> = host_dy_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up y");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_dx: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let desc = SparsemaxBackwardDescriptor {
        input_shape: shape,
        softmax_axis: 1,
        element: ElementKind::F16,
    };
    let plan = SparsemaxBackwardPlan::<f16, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run(&stream, Workspace::None, SparsemaxBackwardArgs {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
        y: TensorRef { data: dev_y.as_slice(), shape, stride: contiguous_stride(shape) },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
    })
    .expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![f16::ZERO; numel];
    dev_dx.copy_to_host(&mut got).expect("dl");
    let eps = 8.0 * 9.77e-4_f32;
    for i in 0..numel {
        let tol = (expected_f32[i].abs() * eps).max(eps);
        let diff = (got[i].to_f32() - expected_f32[i]).abs();
        assert!(diff <= tol, "f16 sparsemax BW @ {i}: diff={diff}");
    }
}

#[test]
#[ignore]
fn sparsemax_bw_bf16_2d() {
    let (ctx, stream) = setup();
    let shape = [3i32, 5];
    let numel = 15usize;
    let mut host_y_f32 = vec![0f32; numel];
    for row in 0..3 {
        for j in 0..3 { host_y_f32[row * 5 + j] = 1.0 / 3.0; }
    }
    let host_dy_f32: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.1).sin()).collect();
    let expected_f32 = host_sparsemax_bw_f32(shape, 1, &host_y_f32, &host_dy_f32);

    let host_y: Vec<bf16> = host_y_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_dy: Vec<bf16> = host_dy_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up y");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_dx: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let desc = SparsemaxBackwardDescriptor {
        input_shape: shape,
        softmax_axis: 1,
        element: ElementKind::Bf16,
    };
    let plan = SparsemaxBackwardPlan::<bf16, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    plan.run(&stream, Workspace::None, SparsemaxBackwardArgs {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
        y: TensorRef { data: dev_y.as_slice(), shape, stride: contiguous_stride(shape) },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
    })
    .expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![bf16::ZERO; numel];
    dev_dx.copy_to_host(&mut got).expect("dl");
    let eps = 8.0 * 7.81e-3_f32;
    for i in 0..numel {
        let tol = (expected_f32[i].abs() * eps).max(eps);
        let diff = (got[i].to_f32() - expected_f32[i]).abs();
        assert!(diff <= tol, "bf16 sparsemax BW @ {i}: diff={diff}");
    }
}
