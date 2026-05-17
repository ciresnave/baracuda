//! Real-GPU smoke test for reduce-LogSumExp backward
//! (`ReduceBackwardPlan<T, N> + ReduceKind::LogSumExp`).
//!
//! Forward: `y = log(sum(exp(x - max), dim=k)) + max` (keepdim).
//! Backward: `dx[c] = dy[c_reduced] * exp(x[c] - y[c_reduced])`. Needs
//! BOTH saved `x` (full shape) and saved `y` (keepdim shape).
//!
//! Numerically safe at every dtype: `y = lse(x) ≥ max(x) ≥ x[c]`, so
//! `x - y ∈ (-∞, 0]` and `exp(x - y) ∈ (0, 1]` — no overflow possible.
//! We feed inputs with `|x| ≤ ~5` so `exp(x - y)` stays comfortably
//! above the f16 subnormal floor even for off-max coords.
//!
//! Tolerance: `8 * eps` relative for f32 / f64 (one exp + one mul on
//! top of the saved-value loads). f16 / bf16 absorb additional
//! rounding through the f32-detour and the final cast back — bump to
//! `4 * dtype_eps` (matches the existing FW LSE smoke convention).

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

// CPU reference for LSE BW. Computes y (the numerically-stable lse)
// from x, then dx = dy * exp(x - y).
fn host_lse_bw_f32(
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

    // Pass 1: max per output cell.
    let mut m = vec![f32::NEG_INFINITY; dy_numel];
    for_each_coord::<3, _>(input_shape, |coord, x_linear| {
        let dy_lin = dy_index(coord, axis, dy_shape) as usize;
        let v = x[x_linear as usize];
        if v > m[dy_lin] {
            m[dy_lin] = v;
        }
    });
    // Pass 2: sum(exp(x - m)).
    let mut s = vec![0f32; dy_numel];
    for_each_coord::<3, _>(input_shape, |coord, x_linear| {
        let dy_lin = dy_index(coord, axis, dy_shape) as usize;
        s[dy_lin] += (x[x_linear as usize] - m[dy_lin]).exp();
    });
    let y: Vec<f32> = s.iter().zip(m.iter()).map(|(&si, &mi)| si.ln() + mi).collect();

    // Backward: dx = dy * exp(x - y).
    let mut dx = vec![0f32; dx_numel];
    for_each_coord::<3, _>(input_shape, |coord, x_linear| {
        let dy_lin = dy_index(coord, axis, dy_shape) as usize;
        dx[x_linear as usize] = dy[dy_lin] * (x[x_linear as usize] - y[dy_lin]).exp();
    });
    (y, dx)
}

#[test]
#[ignore]
fn logsumexp_backward_f32_3d_axis1() {
    let (ctx, stream) = setup();
    let input_shape = [3i32, 4, 5];
    let axis: usize = 1;
    let dx_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let mut dy_shape = input_shape;
    dy_shape[axis] = 1;
    let dy_numel: usize = dy_shape.iter().map(|&d| d as usize).product();

    // Keep |x| <= ~4 so exp(x - y) stays in a comfortable range for
    // f16/bf16 paths (this f32 test reuses the same generator).
    let host_x: Vec<f32> = (0..dx_numel)
        .map(|i| -2.0 + 0.1 * ((i % 41) as f32))
        .collect();
    let host_dy: Vec<f32> = (0..dy_numel).map(|i| 0.3 + 0.15 * (i as f32)).collect();
    let (host_y, expected_dx) = host_lse_bw_f32(input_shape, axis, &host_x, &host_dy);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, dx_numel).expect("alloc");
    let desc = ReduceBackwardDescriptor {
        kind: ReduceKind::LogSumExp,
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
    let eps = 8.0 * f32::EPSILON;
    for i in 0..dx_numel {
        let tol = (expected_dx[i].abs() * eps).max(eps);
        assert!((got[i] - expected_dx[i]).abs() <= tol,
            "f32 lse BW @ {i}: got={} want={}", got[i], expected_dx[i]);
    }
}

#[test]
#[ignore]
fn logsumexp_backward_f64_3d_axis2() {
    let (ctx, stream) = setup();
    let input_shape = [3i32, 4, 5];
    let axis: usize = 2;
    let dx_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let mut dy_shape = input_shape;
    dy_shape[axis] = 1;
    let dy_numel: usize = dy_shape.iter().map(|&d| d as usize).product();

    let host_x: Vec<f64> = (0..dx_numel)
        .map(|i| -2.0 + 0.1 * ((i % 41) as f64))
        .collect();
    let host_dy: Vec<f64> = (0..dy_numel).map(|i| 0.3 + 0.15 * (i as f64)).collect();

    // Reference in f64.
    let dy_shape_usize = dy_shape;
    let mut m = vec![f64::NEG_INFINITY; dy_numel];
    for_each_coord::<3, _>(input_shape, |coord, x_lin| {
        let dy_lin = dy_index(coord, axis, dy_shape_usize) as usize;
        let v = host_x[x_lin as usize];
        if v > m[dy_lin] {
            m[dy_lin] = v;
        }
    });
    let mut s = vec![0f64; dy_numel];
    for_each_coord::<3, _>(input_shape, |coord, x_lin| {
        let dy_lin = dy_index(coord, axis, dy_shape_usize) as usize;
        s[dy_lin] += (host_x[x_lin as usize] - m[dy_lin]).exp();
    });
    let host_y: Vec<f64> = s.iter().zip(m.iter()).map(|(&si, &mi)| si.ln() + mi).collect();
    let mut expected_dx = vec![0f64; dx_numel];
    for_each_coord::<3, _>(input_shape, |coord, x_lin| {
        let dy_lin = dy_index(coord, axis, dy_shape_usize) as usize;
        expected_dx[x_lin as usize] =
            host_dy[dy_lin] * (host_x[x_lin as usize] - host_y[dy_lin]).exp();
    });

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up");
    let mut dev_dx: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, dx_numel).expect("alloc");
    let desc = ReduceBackwardDescriptor {
        kind: ReduceKind::LogSumExp,
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
    let eps = 8.0 * f64::EPSILON;
    for i in 0..dx_numel {
        let tol = (expected_dx[i].abs() * eps).max(eps);
        assert!((got[i] - expected_dx[i]).abs() <= tol, "f64 lse BW @ {i}");
    }
}

#[test]
#[ignore]
fn logsumexp_backward_f16_3d_axis0() {
    let (ctx, stream) = setup();
    let input_shape = [3i32, 4, 5];
    let axis: usize = 0;
    let dx_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let mut dy_shape = input_shape;
    dy_shape[axis] = 1;
    let dy_numel: usize = dy_shape.iter().map(|&d| d as usize).product();

    let host_x_f32: Vec<f32> = (0..dx_numel)
        .map(|i| -2.0 + 0.1 * ((i % 41) as f32))
        .collect();
    let host_dy_f32: Vec<f32> = (0..dy_numel).map(|i| 0.3 + 0.15 * (i as f32)).collect();
    let (host_y_f32, expected_dx_f32) =
        host_lse_bw_f32(input_shape, axis, &host_x_f32, &host_dy_f32);

    let host_x: Vec<f16> = host_x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_y: Vec<f16> = host_y_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_dy: Vec<f16> = host_dy_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up");
    let mut dev_dx: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, dx_numel).expect("alloc");
    let desc = ReduceBackwardDescriptor {
        kind: ReduceKind::LogSumExp,
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
        assert!(diff <= tol, "f16 lse BW @ {i}: diff={diff}");
    }
}

#[test]
#[ignore]
fn logsumexp_backward_bf16_3d_axis1() {
    let (ctx, stream) = setup();
    let input_shape = [3i32, 4, 5];
    let axis: usize = 1;
    let dx_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let mut dy_shape = input_shape;
    dy_shape[axis] = 1;
    let dy_numel: usize = dy_shape.iter().map(|&d| d as usize).product();

    let host_x_f32: Vec<f32> = (0..dx_numel)
        .map(|i| -2.0 + 0.1 * ((i % 41) as f32))
        .collect();
    let host_dy_f32: Vec<f32> = (0..dy_numel).map(|i| 0.3 + 0.15 * (i as f32)).collect();
    let (host_y_f32, expected_dx_f32) =
        host_lse_bw_f32(input_shape, axis, &host_x_f32, &host_dy_f32);

    let host_x: Vec<bf16> = host_x_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_y: Vec<bf16> = host_y_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_dy: Vec<bf16> = host_dy_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up");
    let mut dev_dx: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, dx_numel).expect("alloc");
    let desc = ReduceBackwardDescriptor {
        kind: ReduceKind::LogSumExp,
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
        assert!(diff <= tol, "bf16 lse BW @ {i}: diff={diff}");
    }
}
