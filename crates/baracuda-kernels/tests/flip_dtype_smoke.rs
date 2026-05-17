//! Real-GPU smoke tests for `FlipPlan<{f16,bf16,f64}, N>` — dtype
//! fanout of the f32 trailblazer in `flip_smoke.rs`.
//!
//! Flip does no math — bit-exact compare via `to_bits()`. Each test
//! mirrors `flip_2d_axis_1` from the f32 file (reverse each row of a
//! 4x8 matrix).
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test flip_dtype_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, FlipArgs, FlipDescriptor, FlipPlan, PlanPreference, TensorMut,
    TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn flip_2d_axis_1_f16() {
    let (ctx, stream) = setup();
    let shape = [4i32, 8];
    // Stay in [-10, 10] so values are exactly representable in f16.
    let host_x: Vec<f16> = (0..32)
        .map(|i| f16::from_f32(((i % 21) as f32) - 10.0))
        .collect();
    let mut expected = vec![f16::from_f32(0.0); 32];
    for i in 0..4 {
        for j in 0..8 {
            expected[(i * 8 + j) as usize] = host_x[(i * 8 + (7 - j)) as usize];
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, 32).expect("alloc y");

    let desc = FlipDescriptor {
        shape,
        flip_axes: [false, true],
        element: ElementKind::F16,
    };
    let plan = FlipPlan::<f16, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select f16");
    let args = FlipArgs::<f16, 2> {
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
    plan.run(&stream, Workspace::None, args).expect("run f16");
    stream.synchronize().expect("sync");

    let mut got = vec![f16::from_f32(0.0); 32];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "flip 2d axis 1 f16 mismatch @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn flip_2d_axis_1_bf16() {
    let (ctx, stream) = setup();
    let shape = [4i32, 8];
    let host_x: Vec<bf16> = (0..32)
        .map(|i| bf16::from_f32(((i % 21) as f32) - 10.0))
        .collect();
    let mut expected = vec![bf16::from_f32(0.0); 32];
    for i in 0..4 {
        for j in 0..8 {
            expected[(i * 8 + j) as usize] = host_x[(i * 8 + (7 - j)) as usize];
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, 32).expect("alloc y");

    let desc = FlipDescriptor {
        shape,
        flip_axes: [false, true],
        element: ElementKind::Bf16,
    };
    let plan = FlipPlan::<bf16, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select bf16");
    let args = FlipArgs::<bf16, 2> {
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
    plan.run(&stream, Workspace::None, args).expect("run bf16");
    stream.synchronize().expect("sync");

    let mut got = vec![bf16::from_f32(0.0); 32];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "flip 2d axis 1 bf16 mismatch @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn flip_2d_axis_1_f64() {
    let (ctx, stream) = setup();
    let shape = [4i32, 8];
    let host_x: Vec<f64> = (0..32).map(|i| (i as f64) * 0.5 - 8.0).collect();
    let mut expected = vec![0f64; 32];
    for i in 0..4 {
        for j in 0..8 {
            expected[(i * 8 + j) as usize] = host_x[(i * 8 + (7 - j)) as usize];
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, 32).expect("alloc y");

    let desc = FlipDescriptor {
        shape,
        flip_axes: [false, true],
        element: ElementKind::F64,
    };
    let plan = FlipPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select f64");
    let args = FlipArgs::<f64, 2> {
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
    plan.run(&stream, Workspace::None, args).expect("run f64");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; 32];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "flip 2d axis 1 f64 mismatch @ {i}: got {g} expected {e}"
        );
    }
}
