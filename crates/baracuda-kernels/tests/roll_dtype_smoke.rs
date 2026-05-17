//! Real-GPU smoke tests for `RollPlan<{f16,bf16,f64}, N>` — dtype
//! fanout of the f32 trailblazer in `roll_smoke.rs`.
//!
//! Roll does no math — bit-exact compare via `to_bits()`. Each test
//! mirrors `roll_2d_mixed` from the f32 file (4x8, shifts=[1, -3]).
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test roll_dtype_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, RollArgs, RollDescriptor, RollPlan, TensorMut,
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
fn roll_2d_mixed_f16() {
    let (ctx, stream) = setup();
    let shape = [4i32, 8];
    let shifts = [1i32, -3];
    // Stay in [-10, 10] for exact f16 representability.
    let host_x: Vec<f16> = (0..32)
        .map(|i| f16::from_f32(((i % 21) as f32) - 10.0))
        .collect();
    let mut expected = vec![f16::from_f32(0.0); 32];
    for i in 0i32..4 {
        for j in 0i32..8 {
            let src_i = (i - 1).rem_euclid(4);
            let src_j = (j - (-3)).rem_euclid(8);
            expected[(i * 8 + j) as usize] = host_x[(src_i * 8 + src_j) as usize];
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, 32).expect("alloc y");

    let desc = RollDescriptor {
        shape,
        shifts,
        element: ElementKind::F16,
    };
    let plan = RollPlan::<f16, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select f16");
    let args = RollArgs::<f16, 2> {
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
            "roll 2d mixed f16 mismatch @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn roll_2d_mixed_bf16() {
    let (ctx, stream) = setup();
    let shape = [4i32, 8];
    let shifts = [1i32, -3];
    let host_x: Vec<bf16> = (0..32)
        .map(|i| bf16::from_f32(((i % 21) as f32) - 10.0))
        .collect();
    let mut expected = vec![bf16::from_f32(0.0); 32];
    for i in 0i32..4 {
        for j in 0i32..8 {
            let src_i = (i - 1).rem_euclid(4);
            let src_j = (j - (-3)).rem_euclid(8);
            expected[(i * 8 + j) as usize] = host_x[(src_i * 8 + src_j) as usize];
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, 32).expect("alloc y");

    let desc = RollDescriptor {
        shape,
        shifts,
        element: ElementKind::Bf16,
    };
    let plan = RollPlan::<bf16, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select bf16");
    let args = RollArgs::<bf16, 2> {
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
            "roll 2d mixed bf16 mismatch @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn roll_2d_mixed_f64() {
    let (ctx, stream) = setup();
    let shape = [4i32, 8];
    let shifts = [1i32, -3];
    let host_x: Vec<f64> = (0..32).map(|i| (i as f64) * 0.25).collect();
    let mut expected = vec![0f64; 32];
    for i in 0i32..4 {
        for j in 0i32..8 {
            let src_i = (i - 1).rem_euclid(4);
            let src_j = (j - (-3)).rem_euclid(8);
            expected[(i * 8 + j) as usize] = host_x[(src_i * 8 + src_j) as usize];
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, 32).expect("alloc y");

    let desc = RollDescriptor {
        shape,
        shifts,
        element: ElementKind::F64,
    };
    let plan = RollPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select f64");
    let args = RollArgs::<f64, 2> {
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
            "roll 2d mixed f64 mismatch @ {i}: got {g} expected {e}"
        );
    }
}
