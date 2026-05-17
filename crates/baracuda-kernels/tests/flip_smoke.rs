//! Real-GPU smoke test for `FlipPlan<f32, N>` — reverse along selected
//! axes. Bit-exact (no math, pure element copy).

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, FlipArgs, FlipDescriptor, FlipPlan, PlanPreference, TensorMut,
    TensorRef, Workspace,
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
fn flip_1d() {
    let (ctx, stream) = setup();
    let shape = [16i32];
    let host_x: Vec<f32> = (0..16).map(|i| (i as f32) - 7.0).collect();
    let mut expected = host_x.clone();
    expected.reverse();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 16).expect("alloc y");

    let desc = FlipDescriptor {
        shape,
        flip_axes: [true],
        element: ElementKind::F32,
    };
    let plan = FlipPlan::<f32, 1>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = FlipArgs::<f32, 1> {
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

    let mut got = vec![0f32; 16];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "flip 1d mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn flip_2d_axis_1() {
    let (ctx, stream) = setup();
    let shape = [4i32, 8];
    let host_x: Vec<f32> = (0..32).map(|i| (i as f32) * 0.5).collect();
    // Flip along axis 1 — reverse each row.
    let mut expected = vec![0f32; 32];
    for i in 0..4 {
        for j in 0..8 {
            expected[(i * 8 + j) as usize] = host_x[(i * 8 + (7 - j)) as usize];
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 32).expect("alloc y");

    let desc = FlipDescriptor {
        shape,
        flip_axes: [false, true],
        element: ElementKind::F32,
    };
    let plan = FlipPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = FlipArgs::<f32, 2> {
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

    let mut got = vec![0f32; 32];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "flip 2d axis 1 mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn flip_3d_both_axes() {
    let (ctx, stream) = setup();
    let shape = [2i32, 4, 8];
    let host_x: Vec<f32> = (0..64).map(|i| (i as f32) * 0.25).collect();
    // Flip axes 0 and 2 (not 1).
    let mut expected = vec![0f32; 64];
    for a in 0..2 {
        for b in 0..4 {
            for c in 0..8 {
                let src_a = 1 - a;
                let src_c = 7 - c;
                expected[(a * 32 + b * 8 + c) as usize] =
                    host_x[(src_a * 32 + b * 8 + src_c) as usize];
            }
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 64).expect("alloc y");

    let desc = FlipDescriptor {
        shape,
        flip_axes: [true, false, true],
        element: ElementKind::F32,
    };
    let plan = FlipPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = FlipArgs::<f32, 3> {
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

    let mut got = vec![0f32; 64];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "flip 3d both mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn flip_no_axes_is_copy() {
    let (ctx, stream) = setup();
    let shape = [4i32, 8];
    let host_x: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 32).expect("alloc y");

    let desc = FlipDescriptor {
        shape,
        flip_axes: [false, false],
        element: ElementKind::F32,
    };
    let plan = FlipPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = FlipArgs::<f32, 2> {
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

    let mut got = vec![0f32; 32];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(host_x.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "flip no-axes copy mismatch @ {i}");
    }
}
