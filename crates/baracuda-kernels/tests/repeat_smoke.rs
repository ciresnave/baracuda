//! Real-GPU smoke test for `RepeatPlan<f32, N>` — per-axis tile.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, RepeatArgs, RepeatDescriptor, RepeatPlan,
    TensorMut, TensorRef, Workspace,
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
fn repeat_1d() {
    let (ctx, stream) = setup();
    let input_shape = [4i32];
    let repeats = [3i32];
    let output_shape = [12i32];
    let host_x: Vec<f32> = (0..4).map(|i| i as f32).collect();
    // Expected: [0,1,2,3, 0,1,2,3, 0,1,2,3]
    let mut expected = vec![0f32; 12];
    for i in 0..12 {
        expected[i] = host_x[i % 4];
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 12).expect("alloc y");

    let desc = RepeatDescriptor {
        input_shape,
        repeats,
        element: ElementKind::F32,
    };
    let plan = RepeatPlan::<f32, 1>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = RepeatArgs::<f32, 1> {
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

    let mut got = vec![0f32; 12];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "repeat 1d mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn repeat_2d() {
    let (ctx, stream) = setup();
    let input_shape = [2i32, 3];
    let repeats = [3i32, 2];
    let output_shape = [6i32, 6];
    let in_numel = 6;
    let host_x: Vec<f32> = (0..in_numel).map(|i| (i as f32) * 0.5).collect();
    // For each output coord (i, j): input coord = (i % 2, j % 3).
    let mut expected = vec![0f32; 36];
    for i in 0..6 {
        for j in 0..6 {
            let in_i = i % 2;
            let in_j = j % 3;
            expected[(i * 6 + j) as usize] = host_x[(in_i * 3 + in_j) as usize];
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 36).expect("alloc y");

    let desc = RepeatDescriptor {
        input_shape,
        repeats,
        element: ElementKind::F32,
    };
    let plan = RepeatPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = RepeatArgs::<f32, 2> {
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

    let mut got = vec![0f32; 36];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "repeat 2d mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn repeat_one_is_identity() {
    // repeats = [1, 1, ...] means output == input.
    let (ctx, stream) = setup();
    let shape = [4i32, 8];
    let repeats = [1i32, 1];
    let in_numel = (4 * 8) as usize;
    let host_x: Vec<f32> = (0..in_numel).map(|i| (i as f32) * 0.25 - 1.0).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, in_numel).expect("alloc y");

    let desc = RepeatDescriptor {
        input_shape: shape,
        repeats,
        element: ElementKind::F32,
    };
    let plan = RepeatPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = RepeatArgs::<f32, 2> {
        x: TensorRef {
            data: dev_x.as_slice(),
            shape,
            stride: contiguous_stride(shape),
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape,
            stride: contiguous_stride(shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; in_numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(host_x.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "repeat identity mismatch @ {i}");
    }
}
