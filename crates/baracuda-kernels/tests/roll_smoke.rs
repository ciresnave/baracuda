//! Real-GPU smoke test for `RollPlan<f32, N>` — cyclic shift along axes.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, RollArgs, RollDescriptor, RollPlan, TensorMut,
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
fn roll_1d_positive() {
    let (ctx, stream) = setup();
    let shape = [8i32];
    let shifts = [3i32];
    let host_x: Vec<f32> = (0..8).map(|i| i as f32).collect();
    // y[c] = x[(c - 3) mod 8]
    let mut expected = vec![0f32; 8];
    for c in 0i32..8 {
        let src = (c - 3).rem_euclid(8) as usize;
        expected[c as usize] = host_x[src];
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 8).expect("alloc y");

    let desc = RollDescriptor {
        shape,
        shifts,
        element: ElementKind::F32,
    };
    let plan = RollPlan::<f32, 1>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = RollArgs::<f32, 1> {
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

    let mut got = vec![0f32; 8];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "roll 1d+ mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn roll_1d_negative() {
    let (ctx, stream) = setup();
    let shape = [8i32];
    let shifts = [-2i32];
    let host_x: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let mut expected = vec![0f32; 8];
    for c in 0i32..8 {
        let src = (c - (-2)).rem_euclid(8) as usize;
        expected[c as usize] = host_x[src];
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 8).expect("alloc y");

    let desc = RollDescriptor {
        shape,
        shifts,
        element: ElementKind::F32,
    };
    let plan = RollPlan::<f32, 1>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = RollArgs::<f32, 1> {
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

    let mut got = vec![0f32; 8];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "roll 1d- mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn roll_2d_mixed() {
    let (ctx, stream) = setup();
    let shape = [4i32, 8];
    let shifts = [1i32, -3];
    let host_x: Vec<f32> = (0..32).map(|i| (i as f32) * 0.25).collect();
    let mut expected = vec![0f32; 32];
    for i in 0i32..4 {
        for j in 0i32..8 {
            let src_i = (i - 1).rem_euclid(4);
            let src_j = (j - (-3)).rem_euclid(8);
            expected[(i * 8 + j) as usize] = host_x[(src_i * 8 + src_j) as usize];
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 32).expect("alloc y");

    let desc = RollDescriptor {
        shape,
        shifts,
        element: ElementKind::F32,
    };
    let plan = RollPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = RollArgs::<f32, 2> {
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
        assert_eq!(g.to_bits(), e.to_bits(), "roll 2d mixed mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn roll_zero_shifts_is_copy() {
    let (ctx, stream) = setup();
    let shape = [4i32, 8];
    let shifts = [0i32, 0];
    let host_x: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 32).expect("alloc y");

    let desc = RollDescriptor {
        shape,
        shifts,
        element: ElementKind::F32,
    };
    let plan = RollPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = RollArgs::<f32, 2> {
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
        assert_eq!(g.to_bits(), e.to_bits(), "roll zero copy mismatch @ {i}");
    }
}
