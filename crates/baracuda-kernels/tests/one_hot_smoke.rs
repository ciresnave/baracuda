//! Real-GPU smoke test for `OneHotPlan<T, N>` (Phase 7 7.3).
//!
//! Output rank = input_rank + 1; last output axis = `num_classes`.
//! Output dtypes covered: f32, f64, i32, bool.
//!
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, Bool, ElementKind, OneHotArgs, OneHotDescriptor, OneHotPlan, PlanPreference,
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
fn one_hot_f32_2d() {
    let (ctx, stream) = setup();
    let num_classes = 5i32;
    // 4 batch items; output shape is [4, 5].
    let host_src: Vec<i32> = vec![0, 4, 2, 3];
    let out_shape = [4i32, num_classes];
    let out_numel: usize = 4 * num_classes as usize;
    let mut expected = vec![0f32; out_numel];
    for (i, &c) in host_src.iter().enumerate() {
        if c >= 0 && c < num_classes {
            expected[i * num_classes as usize + c as usize] = 1.0;
        }
    }
    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("up src");
    let mut dev_out: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc out");

    let desc = OneHotDescriptor {
        out_shape,
        num_classes,
        element: ElementKind::F32,
    };
    let plan = OneHotPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = OneHotArgs::<f32, 2> {
        src: TensorRef {
            data: dev_src.as_slice(),
            shape: [4i32],
            stride: contiguous_stride([4i32]),
        },
        out: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: out_shape,
            stride: contiguous_stride(out_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; out_numel];
    dev_out.copy_to_host(&mut got).expect("dl");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "one_hot f32 mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn one_hot_f64_2d_with_oob() {
    let (ctx, stream) = setup();
    let num_classes = 4i32;
    // include an out-of-range src value to verify it yields an all-zero row.
    let host_src: Vec<i32> = vec![2, -1, 0, 3, 7];
    let out_shape = [5i32, num_classes];
    let out_numel: usize = 5 * num_classes as usize;
    let mut expected = vec![0f64; out_numel];
    for (i, &c) in host_src.iter().enumerate() {
        if c >= 0 && c < num_classes {
            expected[i * num_classes as usize + c as usize] = 1.0;
        }
    }
    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("up src");
    let mut dev_out: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc out");

    let desc = OneHotDescriptor {
        out_shape,
        num_classes,
        element: ElementKind::F64,
    };
    let plan = OneHotPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = OneHotArgs::<f64, 2> {
        src: TensorRef {
            data: dev_src.as_slice(),
            shape: [5i32],
            stride: contiguous_stride([5i32]),
        },
        out: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: out_shape,
            stride: contiguous_stride(out_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; out_numel];
    dev_out.copy_to_host(&mut got).expect("dl");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "one_hot f64 mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn one_hot_i32_2d() {
    let (ctx, stream) = setup();
    let num_classes = 6i32;
    let host_src: Vec<i32> = vec![5, 1, 0, 3, 2, 4];
    let out_shape = [6i32, num_classes];
    let out_numel: usize = 6 * num_classes as usize;
    let mut expected = vec![0i32; out_numel];
    for (i, &c) in host_src.iter().enumerate() {
        if c >= 0 && c < num_classes {
            expected[i * num_classes as usize + c as usize] = 1;
        }
    }
    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("up src");
    let mut dev_out: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc out");

    let desc = OneHotDescriptor {
        out_shape,
        num_classes,
        element: ElementKind::I32,
    };
    let plan = OneHotPlan::<i32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = OneHotArgs::<i32, 2> {
        src: TensorRef {
            data: dev_src.as_slice(),
            shape: [6i32],
            stride: contiguous_stride([6i32]),
        },
        out: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: out_shape,
            stride: contiguous_stride(out_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0i32; out_numel];
    dev_out.copy_to_host(&mut got).expect("dl");
    assert_eq!(got, expected, "one_hot i32 mismatch");
}

#[test]
#[ignore]
fn one_hot_bool_2d() {
    let (ctx, stream) = setup();
    let num_classes = 3i32;
    let host_src: Vec<i32> = vec![0, 2, 1, 2];
    let out_shape = [4i32, num_classes];
    let out_numel: usize = 4 * num_classes as usize;
    let mut expected = vec![Bool(0); out_numel];
    for (i, &c) in host_src.iter().enumerate() {
        if c >= 0 && c < num_classes {
            expected[i * num_classes as usize + c as usize] = Bool(1);
        }
    }
    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("up src");
    let mut dev_out: DeviceBuffer<Bool> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc out");

    let desc = OneHotDescriptor {
        out_shape,
        num_classes,
        element: ElementKind::Bool,
    };
    let plan = OneHotPlan::<Bool, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = OneHotArgs::<Bool, 2> {
        src: TensorRef {
            data: dev_src.as_slice(),
            shape: [4i32],
            stride: contiguous_stride([4i32]),
        },
        out: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: out_shape,
            stride: contiguous_stride(out_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![Bool(0); out_numel];
    dev_out.copy_to_host(&mut got).expect("dl");
    assert_eq!(got, expected, "one_hot bool mismatch");
}
