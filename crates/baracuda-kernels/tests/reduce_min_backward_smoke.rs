//! Real-GPU smoke test for reduce-min backward
//! (`ReduceBackwardPlan<T, N> + ReduceKind::Min`).
//!
//! Forward: `y = min(x, dim=k)` (keepdim). Backward:
//! `dx[c] = dy[c_reduced] if x[c] == y[c_reduced] else 0`. Shares the
//! same kernel as Max BW — the routing logic is identical, only y
//! differs (min vs max). Tests fill x with strictly-increasing values
//! so the min is always at the first position along the reduced axis.

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

#[test]
#[ignore]
fn min_backward_f32_3d_axis1() {
    let (ctx, stream) = setup();
    let input_shape = [4i32, 8, 6];
    let axis: usize = 1;
    let dx_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let mut dy_shape = input_shape;
    dy_shape[axis] = 1;
    let dy_numel: usize = dy_shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f32> = (0..dx_numel).map(|i| i as f32 * 0.5).collect();
    let mut host_y = vec![f32::INFINITY; dy_numel];
    for_each_coord::<3, _>(input_shape, |coord, x_linear| {
        let dy_lin = dy_index(coord, axis, dy_shape) as usize;
        if host_x[x_linear as usize] < host_y[dy_lin] {
            host_y[dy_lin] = host_x[x_linear as usize];
        }
    });
    let host_dy: Vec<f32> = (0..dy_numel).map(|i| (i as f32) * 0.25 + 1.0).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("upload y");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, dx_numel).expect("alloc dx");
    let desc = ReduceBackwardDescriptor {
        kind: ReduceKind::Min,
        input_shape,
        reduce_axis: axis as u8,
        element: ElementKind::F32,
        correction: 1,
    };
    let plan = ReduceBackwardPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = ReduceBackwardArgs::<f32, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape: dy_shape, stride: contiguous_stride(dy_shape) },
        x: Some(TensorRef { data: dev_x.as_slice(), shape: input_shape, stride: contiguous_stride(input_shape) }),
        y: Some(TensorRef { data: dev_y.as_slice(), shape: dy_shape, stride: contiguous_stride(dy_shape) }),
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape: input_shape, stride: contiguous_stride(input_shape) },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; dx_numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for_each_coord::<3, _>(input_shape, |coord, dx_linear| {
        let dy_lin = dy_index(coord, axis, dy_shape) as usize;
        let exp = if host_x[dx_linear as usize] == host_y[dy_lin] { host_dy[dy_lin] } else { 0.0 };
        assert_eq!(got[dx_linear as usize].to_bits(), exp.to_bits(),
                   "min bw f32 @ {dx_linear}");
    });
}

#[test]
#[ignore]
fn min_backward_f64_3d_axis0() {
    let (ctx, stream) = setup();
    let input_shape = [4i32, 8, 6];
    let axis: usize = 0;
    let dx_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let mut dy_shape = input_shape;
    dy_shape[axis] = 1;
    let dy_numel: usize = dy_shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f64> = (0..dx_numel).map(|i| i as f64 * 0.5).collect();
    let mut host_y = vec![f64::INFINITY; dy_numel];
    for_each_coord::<3, _>(input_shape, |coord, x_linear| {
        let dy_lin = dy_index(coord, axis, dy_shape) as usize;
        if host_x[x_linear as usize] < host_y[dy_lin] {
            host_y[dy_lin] = host_x[x_linear as usize];
        }
    });
    let host_dy: Vec<f64> = (0..dy_numel).map(|i| (i as f64) * 0.25 + 1.0).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("upload y");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, dx_numel).expect("alloc dx");
    let desc = ReduceBackwardDescriptor {
        kind: ReduceKind::Min,
        input_shape,
        reduce_axis: axis as u8,
        element: ElementKind::F64,
        correction: 1,
    };
    let plan = ReduceBackwardPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = ReduceBackwardArgs::<f64, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape: dy_shape, stride: contiguous_stride(dy_shape) },
        x: Some(TensorRef { data: dev_x.as_slice(), shape: input_shape, stride: contiguous_stride(input_shape) }),
        y: Some(TensorRef { data: dev_y.as_slice(), shape: dy_shape, stride: contiguous_stride(dy_shape) }),
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape: input_shape, stride: contiguous_stride(input_shape) },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f64; dx_numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for_each_coord::<3, _>(input_shape, |coord, dx_linear| {
        let dy_lin = dy_index(coord, axis, dy_shape) as usize;
        let exp = if host_x[dx_linear as usize] == host_y[dy_lin] { host_dy[dy_lin] } else { 0.0 };
        assert_eq!(got[dx_linear as usize].to_bits(), exp.to_bits(),
                   "min bw f64 @ {dx_linear}");
    });
}

#[test]
#[ignore]
fn min_backward_f16_3d_axis2() {
    let (ctx, stream) = setup();
    let input_shape = [4i32, 8, 6];
    let axis: usize = 2;
    let dx_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let mut dy_shape = input_shape;
    dy_shape[axis] = 1;
    let dy_numel: usize = dy_shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f16> = (0..dx_numel).map(|i| f16::from_f32(i as f32 * 0.0625)).collect();
    let mut host_y = vec![f16::from_f32(f32::INFINITY); dy_numel];
    for_each_coord::<3, _>(input_shape, |coord, x_linear| {
        let dy_lin = dy_index(coord, axis, dy_shape) as usize;
        if host_x[x_linear as usize].to_f32() < host_y[dy_lin].to_f32() {
            host_y[dy_lin] = host_x[x_linear as usize];
        }
    });
    let host_dy: Vec<f16> = (0..dy_numel).map(|i| f16::from_f32((i as f32) * 0.125 + 0.5)).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("upload y");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, dx_numel).expect("alloc dx");
    let desc = ReduceBackwardDescriptor {
        kind: ReduceKind::Min,
        input_shape,
        reduce_axis: axis as u8,
        element: ElementKind::F16,
        correction: 1,
    };
    let plan = ReduceBackwardPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = ReduceBackwardArgs::<f16, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape: dy_shape, stride: contiguous_stride(dy_shape) },
        x: Some(TensorRef { data: dev_x.as_slice(), shape: input_shape, stride: contiguous_stride(input_shape) }),
        y: Some(TensorRef { data: dev_y.as_slice(), shape: dy_shape, stride: contiguous_stride(dy_shape) }),
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape: input_shape, stride: contiguous_stride(input_shape) },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![f16::from_f32(0.0); dx_numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for_each_coord::<3, _>(input_shape, |coord, dx_linear| {
        let dy_lin = dy_index(coord, axis, dy_shape) as usize;
        let exp = if host_x[dx_linear as usize] == host_y[dy_lin] {
            host_dy[dy_lin]
        } else {
            f16::from_f32(0.0)
        };
        assert_eq!(got[dx_linear as usize].to_bits(), exp.to_bits(),
                   "min bw f16 @ {dx_linear}");
    });
}

#[test]
#[ignore]
fn min_backward_bf16_3d_axis1() {
    let (ctx, stream) = setup();
    let input_shape = [4i32, 8, 6];
    let axis: usize = 1;
    let dx_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let mut dy_shape = input_shape;
    dy_shape[axis] = 1;
    let dy_numel: usize = dy_shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<bf16> = (0..dx_numel).map(|i| bf16::from_f32(i as f32 * 0.0625)).collect();
    let mut host_y = vec![bf16::from_f32(f32::INFINITY); dy_numel];
    for_each_coord::<3, _>(input_shape, |coord, x_linear| {
        let dy_lin = dy_index(coord, axis, dy_shape) as usize;
        if host_x[x_linear as usize].to_f32() < host_y[dy_lin].to_f32() {
            host_y[dy_lin] = host_x[x_linear as usize];
        }
    });
    let host_dy: Vec<bf16> = (0..dy_numel).map(|i| bf16::from_f32((i as f32) * 0.125 + 0.5)).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("upload y");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, dx_numel).expect("alloc dx");
    let desc = ReduceBackwardDescriptor {
        kind: ReduceKind::Min,
        input_shape,
        reduce_axis: axis as u8,
        element: ElementKind::Bf16,
        correction: 1,
    };
    let plan = ReduceBackwardPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = ReduceBackwardArgs::<bf16, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape: dy_shape, stride: contiguous_stride(dy_shape) },
        x: Some(TensorRef { data: dev_x.as_slice(), shape: input_shape, stride: contiguous_stride(input_shape) }),
        y: Some(TensorRef { data: dev_y.as_slice(), shape: dy_shape, stride: contiguous_stride(dy_shape) }),
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape: input_shape, stride: contiguous_stride(input_shape) },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![bf16::from_f32(0.0); dx_numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for_each_coord::<3, _>(input_shape, |coord, dx_linear| {
        let dy_lin = dy_index(coord, axis, dy_shape) as usize;
        let exp = if host_x[dx_linear as usize] == host_y[dy_lin] {
            host_dy[dy_lin]
        } else {
            bf16::from_f32(0.0)
        };
        assert_eq!(got[dx_linear as usize].to_bits(), exp.to_bits(),
                   "min bw bf16 @ {dx_linear}");
    });
}
