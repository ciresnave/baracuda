//! Real-GPU smoke test for reduce-mean backward
//! (`ReduceBackwardPlan<T, N> + ReduceKind::Mean`).
//!
//! Forward: `y = mean(x, dim=k)` (keepdim). Backward:
//! `dx[c] = dy[c with c[k] = 0] / k_extent`. The Rust dispatcher passes
//! `inv_extent = 1.0_f64 / k_extent` to the kernel; the kernel casts to
//! T at use. We tolerance-compare (the inv_extent rounding + the
//! multiplication round may diverge by 1 ULP from the host's `dy / k`
//! single-rounded division).

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, ReduceBackwardArgs,
    ReduceBackwardDescriptor, ReduceBackwardPlan, ReduceKind, TensorMut, TensorRef, Workspace,
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

#[test]
#[ignore]
fn mean_backward_f32_3d_axis1() {
    let (ctx, stream) = setup();
    let input_shape = [4i32, 8, 6];
    let axis: usize = 1;
    let dx_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let mut dy_shape = input_shape;
    dy_shape[axis] = 1;
    let dy_numel: usize = dy_shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<f32> = (0..dy_numel).map(|i| (i as f32) * 0.5 - 5.0).collect();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, dx_numel).expect("alloc dx");
    let desc = ReduceBackwardDescriptor {
        kind: ReduceKind::Mean,
        input_shape,
        reduce_axis: axis as u8,
        element: ElementKind::F32,
        correction: 1,
    };
    let plan = ReduceBackwardPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = ReduceBackwardArgs::<f32, 3> {
        dy: TensorRef {
            data: dev_dy.as_slice(),
            shape: dy_shape,
            stride: contiguous_stride(dy_shape),
        },
        x: None,
        y: None,
        dx: TensorMut {
            data: dev_dx.as_slice_mut(),
            shape: input_shape,
            stride: contiguous_stride(input_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; dx_numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    let extent = input_shape[axis] as f32;
    let inv = 1.0_f32 / extent;
    for_each_coord::<3, _>(input_shape, |coord, dx_linear| {
        let dy_linear = dy_index(coord, axis, dy_shape);
        let exp = host_dy[dy_linear as usize] * inv;
        let g = got[dx_linear as usize];
        let tol = exp.abs().max(1.0) * 4.0 * f32::EPSILON;
        assert!((g - exp).abs() <= tol, "mean bw f32 @ {dx_linear}: got {g} exp {exp}");
    });
}

#[test]
#[ignore]
fn mean_backward_f64_3d_axis2() {
    let (ctx, stream) = setup();
    let input_shape = [4i32, 8, 6];
    let axis: usize = 2;
    let dx_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let mut dy_shape = input_shape;
    dy_shape[axis] = 1;
    let dy_numel: usize = dy_shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<f64> = (0..dy_numel).map(|i| (i as f64) * 0.5 - 5.0).collect();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, dx_numel).expect("alloc dx");
    let desc = ReduceBackwardDescriptor {
        kind: ReduceKind::Mean,
        input_shape,
        reduce_axis: axis as u8,
        element: ElementKind::F64,
        correction: 1,
    };
    let plan = ReduceBackwardPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = ReduceBackwardArgs::<f64, 3> {
        dy: TensorRef {
            data: dev_dy.as_slice(),
            shape: dy_shape,
            stride: contiguous_stride(dy_shape),
        },
        x: None,
        y: None,
        dx: TensorMut {
            data: dev_dx.as_slice_mut(),
            shape: input_shape,
            stride: contiguous_stride(input_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f64; dx_numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    let extent = input_shape[axis] as f64;
    let inv = 1.0_f64 / extent;
    for_each_coord::<3, _>(input_shape, |coord, dx_linear| {
        let dy_linear = dy_index(coord, axis, dy_shape);
        let exp = host_dy[dy_linear as usize] * inv;
        let g = got[dx_linear as usize];
        let tol = exp.abs().max(1.0) * 4.0 * f64::EPSILON;
        assert!((g - exp).abs() <= tol, "mean bw f64 @ {dx_linear}");
    });
}

#[test]
#[ignore]
fn mean_backward_f16_3d_axis0() {
    let (ctx, stream) = setup();
    let input_shape = [4i32, 8, 6];
    let axis: usize = 0;
    let dx_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let mut dy_shape = input_shape;
    dy_shape[axis] = 1;
    let dy_numel: usize = dy_shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<f16> = (0..dy_numel)
        .map(|i| f16::from_f32((i as f32) * 0.125 - 1.0))
        .collect();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, dx_numel).expect("alloc dx");
    let desc = ReduceBackwardDescriptor {
        kind: ReduceKind::Mean,
        input_shape,
        reduce_axis: axis as u8,
        element: ElementKind::F16,
        correction: 1,
    };
    let plan = ReduceBackwardPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = ReduceBackwardArgs::<f16, 3> {
        dy: TensorRef {
            data: dev_dy.as_slice(),
            shape: dy_shape,
            stride: contiguous_stride(dy_shape),
        },
        x: None,
        y: None,
        dx: TensorMut {
            data: dev_dx.as_slice_mut(),
            shape: input_shape,
            stride: contiguous_stride(input_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![f16::from_f32(0.0); dx_numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    let extent = input_shape[axis] as f32;
    let inv = 1.0_f32 / extent;
    for_each_coord::<3, _>(input_shape, |coord, dx_linear| {
        let dy_linear = dy_index(coord, axis, dy_shape);
        let dy_v = host_dy[dy_linear as usize].to_f32();
        let exp = dy_v * inv;
        let g = got[dx_linear as usize].to_f32();
        let tol = exp.abs().max(1.0) * 4.0 * F16_EPS;
        assert!((g - exp).abs() <= tol, "mean bw f16 @ {dx_linear}: got {g} exp {exp}");
    });
}

#[test]
#[ignore]
fn mean_backward_bf16_3d_axis1() {
    let (ctx, stream) = setup();
    let input_shape = [4i32, 8, 6];
    let axis: usize = 1;
    let dx_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let mut dy_shape = input_shape;
    dy_shape[axis] = 1;
    let dy_numel: usize = dy_shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<bf16> = (0..dy_numel)
        .map(|i| bf16::from_f32((i as f32) * 0.125 - 1.0))
        .collect();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, dx_numel).expect("alloc dx");
    let desc = ReduceBackwardDescriptor {
        kind: ReduceKind::Mean,
        input_shape,
        reduce_axis: axis as u8,
        element: ElementKind::Bf16,
        correction: 1,
    };
    let plan = ReduceBackwardPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = ReduceBackwardArgs::<bf16, 3> {
        dy: TensorRef {
            data: dev_dy.as_slice(),
            shape: dy_shape,
            stride: contiguous_stride(dy_shape),
        },
        x: None,
        y: None,
        dx: TensorMut {
            data: dev_dx.as_slice_mut(),
            shape: input_shape,
            stride: contiguous_stride(input_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![bf16::from_f32(0.0); dx_numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    let extent = input_shape[axis] as f32;
    let inv = 1.0_f32 / extent;
    for_each_coord::<3, _>(input_shape, |coord, dx_linear| {
        let dy_linear = dy_index(coord, axis, dy_shape);
        let dy_v = host_dy[dy_linear as usize].to_f32();
        let exp = dy_v * inv;
        let g = got[dx_linear as usize].to_f32();
        let tol = exp.abs().max(1.0) * 4.0 * BF16_EPS;
        assert!((g - exp).abs() <= tol, "mean bw bf16 @ {dx_linear}: got {g} exp {exp}");
    });
}
