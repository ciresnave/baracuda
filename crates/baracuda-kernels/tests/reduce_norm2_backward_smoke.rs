//! Real-GPU smoke test for reduce-norm2 backward
//! (`ReduceBackwardPlan<T, N> + ReduceKind::Norm2`).
//!
//! Forward: `y = sqrt(sum(x², dim=k))` (keepdim). Backward:
//! `dx[c] = dy[c_reduced] * x[c] / y[c_reduced]`. Needs BOTH saved
//! `x` (full shape) and saved `y` (keepdim shape). Caller must ensure
//! `y != 0` (which means at least one `x` in each reduced group is
//! non-zero).

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, ReduceBackwardArgs,
    ReduceBackwardDescriptor, ReduceBackwardPlan, ReduceKind, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn for_each_coord<const N: usize, F: FnMut([i32; N], i64)>(shape: [i32; N], mut f: F) {
    let numel: i64 = shape.iter().map(|&d| d as i64).product();
    for linear in 0..numel {
        let mut coord = [0i32; N];
        let mut rem = linear;
        for d in (0..N).rev() {
            coord[d] = (rem % shape[d] as i64) as i32;
            rem /= shape[d] as i64;
        }
        f(coord, linear);
    }
}

fn dy_index<const N: usize>(dx_coord: [i32; N], axis: usize, dy_shape: [i32; N]) -> i64 {
    let mut idx = 0i64;
    let mut stride = 1i64;
    for d in (0..N).rev() {
        let c = if d == axis { 0 } else { dx_coord[d] };
        idx += (c as i64) * stride;
        stride *= dy_shape[d] as i64;
    }
    idx
}

// CPU reference for Norm2 BW: y = sqrt(sum(x²)); dx = dy * x / y.
fn host_norm2_bw_f32(
    input_shape: [i32; 3],
    axis: usize,
    x: &[f32],
    dy: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let dy_shape = {
        let mut s = input_shape;
        s[axis] = 1;
        s
    };
    let dy_numel: usize = dy_shape.iter().map(|&d| d as usize).product();
    let dx_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let mut y_sq = vec![0f32; dy_numel];
    for_each_coord::<3, _>(input_shape, |coord, x_linear| {
        let dy_lin = dy_index(coord, axis, dy_shape) as usize;
        y_sq[dy_lin] += x[x_linear as usize] * x[x_linear as usize];
    });
    let y: Vec<f32> = y_sq.iter().map(|v| v.sqrt()).collect();
    let mut dx = vec![0f32; dx_numel];
    for_each_coord::<3, _>(input_shape, |coord, x_linear| {
        let dy_lin = dy_index(coord, axis, dy_shape) as usize;
        dx[x_linear as usize] = dy[dy_lin] * x[x_linear as usize] / y[dy_lin];
    });
    (y, dx)
}

#[test]
#[ignore]
fn norm2_backward_f32_3d_axis1() {
    let (ctx, stream) = setup();
    let input_shape = [3i32, 4, 5];
    let axis: usize = 1;
    let dx_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let mut dy_shape = input_shape;
    dy_shape[axis] = 1;
    let dy_numel: usize = dy_shape.iter().map(|&d| d as usize).product();

    let host_x: Vec<f32> = (0..dx_numel).map(|i| 0.5 + 0.1 * (i as f32)).collect();
    let host_dy: Vec<f32> = (0..dy_numel).map(|i| 0.3 + 0.15 * (i as f32)).collect();
    let (host_y, expected_dx) = host_norm2_bw_f32(input_shape, axis, &host_x, &host_dy);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, dx_numel).expect("alloc");
    let desc = ReduceBackwardDescriptor {
        kind: ReduceKind::Norm2,
        input_shape,
        reduce_axis: axis as u8,
        element: ElementKind::F32,
        correction: 1,
    };
    let plan = ReduceBackwardPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = ReduceBackwardArgs::<f32, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape: dy_shape, stride: contiguous_stride(dy_shape) },
        x: Some(TensorRef { data: dev_x.as_slice(), shape: input_shape, stride: contiguous_stride(input_shape) }),
        y: Some(TensorRef { data: dev_y.as_slice(), shape: dy_shape, stride: contiguous_stride(dy_shape) }),
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape: input_shape, stride: contiguous_stride(input_shape) },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; dx_numel];
    dev_dx.copy_to_host(&mut got).expect("dl");
    let eps = 4.0 * f32::EPSILON;
    for i in 0..dx_numel {
        let tol = (expected_dx[i].abs() * eps).max(eps);
        assert!((got[i] - expected_dx[i]).abs() <= tol,
            "f32 norm2 BW @ {i}: got={} want={}", got[i], expected_dx[i]);
    }
}

#[test]
#[ignore]
fn norm2_backward_f64_3d_axis2() {
    let (ctx, stream) = setup();
    let input_shape = [3i32, 4, 5];
    let axis: usize = 2;
    let dx_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let mut dy_shape = input_shape;
    dy_shape[axis] = 1;
    let dy_numel: usize = dy_shape.iter().map(|&d| d as usize).product();

    let host_x: Vec<f64> = (0..dx_numel).map(|i| 0.5 + 0.1 * (i as f64)).collect();
    let host_dy: Vec<f64> = (0..dy_numel).map(|i| 0.3 + 0.15 * (i as f64)).collect();

    let mut host_y_sq = vec![0f64; dy_numel];
    for_each_coord::<3, _>(input_shape, |coord, x_lin| {
        let dy_lin = dy_index(coord, axis, dy_shape) as usize;
        host_y_sq[dy_lin] += host_x[x_lin as usize] * host_x[x_lin as usize];
    });
    let host_y: Vec<f64> = host_y_sq.iter().map(|v| v.sqrt()).collect();
    let mut expected_dx = vec![0f64; dx_numel];
    for_each_coord::<3, _>(input_shape, |coord, x_lin| {
        let dy_lin = dy_index(coord, axis, dy_shape) as usize;
        expected_dx[x_lin as usize] = host_dy[dy_lin] * host_x[x_lin as usize] / host_y[dy_lin];
    });

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up");
    let mut dev_dx: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, dx_numel).expect("alloc");
    let desc = ReduceBackwardDescriptor {
        kind: ReduceKind::Norm2,
        input_shape,
        reduce_axis: axis as u8,
        element: ElementKind::F64,
        correction: 1,
    };
    let plan = ReduceBackwardPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = ReduceBackwardArgs::<f64, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape: dy_shape, stride: contiguous_stride(dy_shape) },
        x: Some(TensorRef { data: dev_x.as_slice(), shape: input_shape, stride: contiguous_stride(input_shape) }),
        y: Some(TensorRef { data: dev_y.as_slice(), shape: dy_shape, stride: contiguous_stride(dy_shape) }),
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape: input_shape, stride: contiguous_stride(input_shape) },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f64; dx_numel];
    dev_dx.copy_to_host(&mut got).expect("dl");
    let eps = 4.0 * f64::EPSILON;
    for i in 0..dx_numel {
        let tol = (expected_dx[i].abs() * eps).max(eps);
        assert!((got[i] - expected_dx[i]).abs() <= tol, "f64 norm2 BW @ {i}");
    }
}

#[test]
#[ignore]
fn norm2_backward_f16_3d_axis0() {
    let (ctx, stream) = setup();
    let input_shape = [3i32, 4, 5];
    let axis: usize = 0;
    let dx_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let mut dy_shape = input_shape;
    dy_shape[axis] = 1;
    let dy_numel: usize = dy_shape.iter().map(|&d| d as usize).product();

    let host_x_f32: Vec<f32> = (0..dx_numel).map(|i| 0.5 + 0.1 * (i as f32)).collect();
    let host_dy_f32: Vec<f32> = (0..dy_numel).map(|i| 0.3 + 0.15 * (i as f32)).collect();
    let (host_y_f32, expected_dx_f32) = host_norm2_bw_f32(input_shape, axis, &host_x_f32, &host_dy_f32);

    let host_x: Vec<f16> = host_x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_y: Vec<f16> = host_y_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_dy: Vec<f16> = host_dy_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up");
    let mut dev_dx: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, dx_numel).expect("alloc");
    let desc = ReduceBackwardDescriptor {
        kind: ReduceKind::Norm2,
        input_shape,
        reduce_axis: axis as u8,
        element: ElementKind::F16,
        correction: 1,
    };
    let plan = ReduceBackwardPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = ReduceBackwardArgs::<f16, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape: dy_shape, stride: contiguous_stride(dy_shape) },
        x: Some(TensorRef { data: dev_x.as_slice(), shape: input_shape, stride: contiguous_stride(input_shape) }),
        y: Some(TensorRef { data: dev_y.as_slice(), shape: dy_shape, stride: contiguous_stride(dy_shape) }),
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape: input_shape, stride: contiguous_stride(input_shape) },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![f16::ZERO; dx_numel];
    dev_dx.copy_to_host(&mut got).expect("dl");
    let eps = 4.0 * 9.77e-4_f32;
    for i in 0..dx_numel {
        let tol = (expected_dx_f32[i].abs() * eps).max(eps);
        let diff = (got[i].to_f32() - expected_dx_f32[i]).abs();
        assert!(diff <= tol, "f16 norm2 BW @ {i}: diff={diff}");
    }
}

#[test]
#[ignore]
fn norm2_backward_bf16_3d_axis1() {
    let (ctx, stream) = setup();
    let input_shape = [3i32, 4, 5];
    let axis: usize = 1;
    let dx_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let mut dy_shape = input_shape;
    dy_shape[axis] = 1;
    let dy_numel: usize = dy_shape.iter().map(|&d| d as usize).product();

    let host_x_f32: Vec<f32> = (0..dx_numel).map(|i| 0.5 + 0.1 * (i as f32)).collect();
    let host_dy_f32: Vec<f32> = (0..dy_numel).map(|i| 0.3 + 0.15 * (i as f32)).collect();
    let (host_y_f32, expected_dx_f32) = host_norm2_bw_f32(input_shape, axis, &host_x_f32, &host_dy_f32);

    let host_x: Vec<bf16> = host_x_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_y: Vec<bf16> = host_y_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_dy: Vec<bf16> = host_dy_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up");
    let mut dev_dx: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, dx_numel).expect("alloc");
    let desc = ReduceBackwardDescriptor {
        kind: ReduceKind::Norm2,
        input_shape,
        reduce_axis: axis as u8,
        element: ElementKind::Bf16,
        correction: 1,
    };
    let plan = ReduceBackwardPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = ReduceBackwardArgs::<bf16, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape: dy_shape, stride: contiguous_stride(dy_shape) },
        x: Some(TensorRef { data: dev_x.as_slice(), shape: input_shape, stride: contiguous_stride(input_shape) }),
        y: Some(TensorRef { data: dev_y.as_slice(), shape: dy_shape, stride: contiguous_stride(dy_shape) }),
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape: input_shape, stride: contiguous_stride(input_shape) },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![bf16::ZERO; dx_numel];
    dev_dx.copy_to_host(&mut got).expect("dl");
    let eps = 4.0 * 7.81e-3_f32;
    for i in 0..dx_numel {
        let tol = (expected_dx_f32[i].abs() * eps).max(eps);
        let diff = (got[i].to_f32() - expected_dx_f32[i]).abs();
        assert!(diff <= tol, "bf16 norm2 BW @ {i}: diff={diff}");
    }
}
