//! Real-GPU smoke test for the Phase 3 fill kernel family
//! (`FillPlan<T>`).
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test fill_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, FillArgs, FillDescriptor, FillPlan, PlanPreference,
    TensorMut, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn run_case<T>(value: T, kind: ElementKind, init_value: T)
where
    T: baracuda_kernels::Element + PartialEq + core::fmt::Debug,
{
    let (ctx, stream) = setup();
    let numel = 1024usize;
    let mut dev_y: DeviceBuffer<T> =
        DeviceBuffer::from_slice(&ctx, &vec![init_value; numel]).expect("alloc");

    let desc = FillDescriptor {
        numel: numel as i32,
        value,
        element: kind,
    };
    let plan = FillPlan::<T>::select(&stream, &desc, PlanPreference::default())
        .expect("select FillPlan");
    let args = FillArgs::<T> {
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: [numel as i32],
            stride: contiguous_stride([numel as i32]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("fill run");
    stream.synchronize().expect("sync");

    let mut got = vec![init_value; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, g) in got.iter().enumerate() {
        assert_eq!(*g, value, "fill {kind:?} mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn fill_f32() {
    run_case::<f32>(3.14159f32, ElementKind::F32, 0.0);
}

#[test]
#[ignore]
fn fill_f32_huge() {
    let (ctx, stream) = setup();
    let numel = 1 << 20;
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");

    let desc = FillDescriptor {
        numel: numel as i32,
        value: -2.5f32,
        element: ElementKind::F32,
    };
    let plan = FillPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = FillArgs::<f32> {
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: [numel as i32],
            stride: contiguous_stride([numel as i32]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, g) in got.iter().enumerate() {
        assert_eq!(*g, -2.5f32, "fill_f32_huge @ {i}");
    }
}

#[test]
#[ignore]
fn fill_f64() {
    run_case::<f64>(-1.7e-9f64, ElementKind::F64, 0.0);
}

#[test]
#[ignore]
fn fill_i32() {
    run_case::<i32>(-42i32, ElementKind::I32, 0);
}

#[test]
#[ignore]
fn fill_i64() {
    run_case::<i64>(1_234_567_890_123i64, ElementKind::I64, 0);
}

#[test]
#[ignore]
fn fill_f16() {
    run_case::<f16>(f16::from_f32(0.5f32), ElementKind::F16, f16::from_f32(0.0));
}

#[test]
#[ignore]
fn fill_bf16() {
    run_case::<bf16>(bf16::from_f32(-7.0f32), ElementKind::Bf16, bf16::from_f32(0.0));
}

#[test]
fn select_rejects_negative_numel() {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    let desc = FillDescriptor {
        numel: -1,
        value: 0.0f32,
        element: ElementKind::F32,
    };
    let err = FillPlan::<f32>::select(&stream, &desc, PlanPreference::default());
    assert!(err.is_err(), "negative numel must be rejected");
    let _ = ctx;
}
