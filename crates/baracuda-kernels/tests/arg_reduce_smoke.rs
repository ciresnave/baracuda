//! Real-GPU smoke test for `ArgReducePlan<f32, N>` — argmax / argmin.
//! Output dtype is i64 (PyTorch convention). Ties broken by first
//! occurrence.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ArgReduceArgs, ArgReduceDescriptor, ArgReduceKind, ArgReducePlan,
    ElementKind, PlanPreference, TensorMut, TensorRef, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn argmax_1d() {
    let (ctx, stream) = setup();
    let input_shape = [128i32];
    // Place max at index 42.
    let mut host_x = vec![0f32; 128];
    for i in 0..128 {
        host_x[i] = -((i as f32 - 42.0).abs());
    }
    // host_x[42] = 0.0 (the max), all others negative.

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<i64> = DeviceBuffer::zeros(&ctx, 1).expect("alloc y");

    let desc = ArgReduceDescriptor {
        kind: ArgReduceKind::Argmax,
        input_shape,
        reduce_axis: 0,
        element: ElementKind::F32,
    };
    let plan = ArgReducePlan::<f32, 1>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let output_shape = [1i32];
    let args = ArgReduceArgs::<f32, 1> {
        x: TensorRef {
            data: dev_x.as_slice(),
            shape: input_shape,
            stride: contiguous_stride(input_shape),
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: output_shape,
            stride: contiguous_stride(output_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0i64; 1];
    dev_y.copy_to_host(&mut got).expect("download");
    assert_eq!(got[0], 42, "argmax 1d: expected index 42, got {}", got[0]);
}

#[test]
#[ignore]
fn argmin_1d() {
    let (ctx, stream) = setup();
    let input_shape = [128i32];
    let mut host_x = vec![0f32; 128];
    for i in 0..128 {
        host_x[i] = (i as f32 - 99.0).abs();
    }
    // host_x[99] = 0.0 (the min), all others positive.

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<i64> = DeviceBuffer::zeros(&ctx, 1).expect("alloc y");

    let desc = ArgReduceDescriptor {
        kind: ArgReduceKind::Argmin,
        input_shape,
        reduce_axis: 0,
        element: ElementKind::F32,
    };
    let plan = ArgReducePlan::<f32, 1>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let output_shape = [1i32];
    let args = ArgReduceArgs::<f32, 1> {
        x: TensorRef {
            data: dev_x.as_slice(),
            shape: input_shape,
            stride: contiguous_stride(input_shape),
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: output_shape,
            stride: contiguous_stride(output_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0i64; 1];
    dev_y.copy_to_host(&mut got).expect("download");
    assert_eq!(got[0], 99, "argmin 1d: expected index 99, got {}", got[0]);
}

#[test]
#[ignore]
fn argmax_3d_axis_1() {
    let (ctx, stream) = setup();
    let input_shape = [3i32, 16, 5];
    // For each (i, k), max along axis 1 lives at a known index.
    let in_numel = (3 * 16 * 5) as usize;
    let mut host_x = vec![0f32; in_numel];
    let in_strides = contiguous_stride(input_shape);
    for i in 0..3 {
        for j in 0..16 {
            for k in 0..5 {
                let off =
                    (i * in_strides[0] + j * in_strides[1] + k * in_strides[2]) as usize;
                // Max at j = i * 4 + k mod 16 (deterministic per (i, k)).
                let max_j = (i * 4 + k) % 16;
                host_x[off] = if j == max_j { 100.0 } else { (j as f32) * 0.1 };
            }
        }
    }

    let output_shape = [3i32, 1, 5];
    let out_numel = (3 * 1 * 5) as usize;
    let out_strides = contiguous_stride(output_shape);
    let mut expected = vec![0i64; out_numel];
    for i in 0..3 {
        for k in 0..5 {
            let max_j = (i * 4 + k) % 16;
            let out_off =
                (i * out_strides[0] + 0 * out_strides[1] + k * out_strides[2]) as usize;
            expected[out_off] = max_j as i64;
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<i64> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc y");

    let desc = ArgReduceDescriptor {
        kind: ArgReduceKind::Argmax,
        input_shape,
        reduce_axis: 1,
        element: ElementKind::F32,
    };
    let plan = ArgReducePlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = ArgReduceArgs::<f32, 3> {
        x: TensorRef {
            data: dev_x.as_slice(),
            shape: input_shape,
            stride: in_strides,
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: output_shape,
            stride: out_strides,
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0i64; out_numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(*g, *e, "argmax 3d axis 1 mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn argmax_ties_pick_first() {
    let (ctx, stream) = setup();
    let input_shape = [16i32];
    // All zeros — every index is tied. Expected: 0 (first index wins).
    let host_x = vec![0f32; 16];

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<i64> = DeviceBuffer::zeros(&ctx, 1).expect("alloc y");

    let desc = ArgReduceDescriptor {
        kind: ArgReduceKind::Argmax,
        input_shape,
        reduce_axis: 0,
        element: ElementKind::F32,
    };
    let plan = ArgReducePlan::<f32, 1>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let output_shape = [1i32];
    let args = ArgReduceArgs::<f32, 1> {
        x: TensorRef {
            data: dev_x.as_slice(),
            shape: input_shape,
            stride: contiguous_stride(input_shape),
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: output_shape,
            stride: contiguous_stride(output_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0i64; 1];
    dev_y.copy_to_host(&mut got).expect("download");
    assert_eq!(got[0], 0, "argmax tie: expected first index (0), got {}", got[0]);
}
