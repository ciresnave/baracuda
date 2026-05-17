//! Real-GPU smoke test for the Phase 4 reduce-sum backward trailblazer
//! (`ReduceBackwardPlan<T, N> + ReduceKind::Sum`).
//!
//! Forward: `y = sum(x, dim=k)` (keepdim). Backward: `dx[c] = dy[c with
//! c[k] = 0]` — broadcast dy across the reduced axis. Pure copy, no
//! math — bit-exact compare for every dtype.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test reduce_sum_backward_smoke -- --ignored`.

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

/// Compute the dy index for a given dx coord. The reduced axis maps to
/// dy index 0; all other axes pass through.
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

/// Iterate every coord in `shape` in row-major (rightmost-fastest) order.
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

// -------- f32 --------

#[test]
#[ignore]
fn sum_backward_f32_3d_axis1() {
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
        kind: ReduceKind::Sum,
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
    for_each_coord::<3, _>(input_shape, |coord, dx_linear| {
        let dy_linear = dy_index(coord, axis, dy_shape);
        let exp = host_dy[dy_linear as usize];
        assert_eq!(
            got[dx_linear as usize].to_bits(),
            exp.to_bits(),
            "sum bw f32 mismatch @ dx_linear={dx_linear} coord={coord:?}"
        );
    });
}

#[test]
#[ignore]
fn sum_backward_f32_2d_axis0() {
    let (ctx, stream) = setup();
    let input_shape = [16i32, 32];
    let axis: usize = 0;
    let dx_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let mut dy_shape = input_shape;
    dy_shape[axis] = 1;
    let dy_numel: usize = dy_shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<f32> = (0..dy_numel).map(|i| (i as f32) * 0.25 - 2.0).collect();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, dx_numel).expect("alloc dx");
    let desc = ReduceBackwardDescriptor {
        kind: ReduceKind::Sum,
        input_shape,
        reduce_axis: axis as u8,
        element: ElementKind::F32,
        correction: 1,
    };
    let plan = ReduceBackwardPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = ReduceBackwardArgs::<f32, 2> {
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
    for_each_coord::<2, _>(input_shape, |coord, dx_linear| {
        let dy_linear = dy_index(coord, axis, dy_shape);
        let exp = host_dy[dy_linear as usize];
        assert_eq!(
            got[dx_linear as usize].to_bits(),
            exp.to_bits(),
            "sum bw f32 axis-0 mismatch @ {dx_linear}"
        );
    });
}

// -------- f16 / bf16 / f64 --------

#[test]
#[ignore]
fn sum_backward_f16_3d_axis2() {
    let (ctx, stream) = setup();
    let input_shape = [4i32, 8, 6];
    let axis: usize = 2;
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
        kind: ReduceKind::Sum,
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
    for_each_coord::<3, _>(input_shape, |coord, dx_linear| {
        let dy_linear = dy_index(coord, axis, dy_shape);
        let exp = host_dy[dy_linear as usize];
        assert_eq!(
            got[dx_linear as usize].to_bits(),
            exp.to_bits(),
            "sum bw f16 mismatch @ {dx_linear}"
        );
    });
}

#[test]
#[ignore]
fn sum_backward_bf16_3d_axis0() {
    let (ctx, stream) = setup();
    let input_shape = [4i32, 8, 6];
    let axis: usize = 0;
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
        kind: ReduceKind::Sum,
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
    for_each_coord::<3, _>(input_shape, |coord, dx_linear| {
        let dy_linear = dy_index(coord, axis, dy_shape);
        let exp = host_dy[dy_linear as usize];
        assert_eq!(
            got[dx_linear as usize].to_bits(),
            exp.to_bits(),
            "sum bw bf16 mismatch @ {dx_linear}"
        );
    });
}

#[test]
#[ignore]
fn sum_backward_f64_3d_axis1() {
    let (ctx, stream) = setup();
    let input_shape = [4i32, 8, 6];
    let axis: usize = 1;
    let dx_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let mut dy_shape = input_shape;
    dy_shape[axis] = 1;
    let dy_numel: usize = dy_shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<f64> = (0..dy_numel).map(|i| (i as f64) * 0.5 - 5.0).collect();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, dx_numel).expect("alloc dx");
    let desc = ReduceBackwardDescriptor {
        kind: ReduceKind::Sum,
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
    for_each_coord::<3, _>(input_shape, |coord, dx_linear| {
        let dy_linear = dy_index(coord, axis, dy_shape);
        let exp = host_dy[dy_linear as usize];
        assert_eq!(
            got[dx_linear as usize].to_bits(),
            exp.to_bits(),
            "sum bw f64 mismatch @ {dx_linear}"
        );
    });
}

/// `select` rejects reductions not yet wired today. `Any` / `All`
/// (boolean reductions) and `Argmax` / `Argmin` (index-output
/// reductions) need distinct plan shapes — their BW is either zero
/// (Any/All — boolean output, no gradient flows) or has no defined
/// gradient (Argmax/Argmin — discrete output) and they land in a
/// future fanout if needed at all.
#[test]
fn select_rejects_unwired_reduce_today() {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    let desc = ReduceBackwardDescriptor {
        kind: ReduceKind::Any,
        input_shape: [4, 4],
        reduce_axis: 0,
        element: ElementKind::F32,
        correction: 1,
    };
    let err = ReduceBackwardPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default());
    assert!(err.is_err(), "Any BW must be unwired in this wave");
}
