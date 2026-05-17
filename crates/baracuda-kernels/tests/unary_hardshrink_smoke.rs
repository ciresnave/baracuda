//! Real-GPU smoke test for `UnaryPlan + UnaryKind::Hardshrink` across
//! f32 / f16 / bf16 / f64. λ hardcoded to 0.5.
//!
//! Bit-exact: kernel just compares and selects `x` or `0` — no arithmetic.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef, UnaryArgs,
    UnaryDescriptor, UnaryKind, UnaryPlan, Workspace,
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
fn hardshrink_f32_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    // Range covers both sides of ±0.5.
    let host_x: Vec<f32> = (0..numel).map(|i| ((i % 401) as f32) * 0.01 - 2.0).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let desc = UnaryDescriptor { kind: UnaryKind::Hardshrink, shape, element: ElementKind::F32 };
    let plan = UnaryPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = UnaryArgs::<f32, 3> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for i in 0..numel {
        let exp = if host_x[i].abs() > 0.5 { host_x[i] } else { 0.0 };
        assert_eq!(got[i].to_bits(), exp.to_bits(), "hardshrink f32 @ {i}: x={}, got {}, exp {}", host_x[i], got[i], exp);
    }
}

#[test]
#[ignore]
fn hardshrink_f64_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f64> = (0..numel).map(|i| ((i % 401) as f64) * 0.01 - 2.0).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let desc = UnaryDescriptor { kind: UnaryKind::Hardshrink, shape, element: ElementKind::F64 };
    let plan = UnaryPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = UnaryArgs::<f64, 3> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f64; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for i in 0..numel {
        let exp = if host_x[i].abs() > 0.5 { host_x[i] } else { 0.0 };
        assert_eq!(got[i].to_bits(), exp.to_bits(), "hardshrink f64 @ {i}");
    }
}

#[test]
#[ignore]
fn hardshrink_f16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f16> = (0..numel).map(|i| f16::from_f32(((i % 401) as f32) * 0.01 - 2.0)).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let desc = UnaryDescriptor { kind: UnaryKind::Hardshrink, shape, element: ElementKind::F16 };
    let plan = UnaryPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = UnaryArgs::<f16, 3> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![f16::from_f32(0.0); numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for i in 0..numel {
        let exp = if host_x[i].to_f32().abs() > 0.5 { host_x[i] } else { f16::from_f32(0.0) };
        assert_eq!(got[i].to_bits(), exp.to_bits(), "hardshrink f16 @ {i}");
    }
}

#[test]
#[ignore]
fn hardshrink_bf16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<bf16> = (0..numel).map(|i| bf16::from_f32(((i % 401) as f32) * 0.01 - 2.0)).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let desc = UnaryDescriptor { kind: UnaryKind::Hardshrink, shape, element: ElementKind::Bf16 };
    let plan = UnaryPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = UnaryArgs::<bf16, 3> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![bf16::from_f32(0.0); numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for i in 0..numel {
        let exp = if host_x[i].to_f32().abs() > 0.5 { host_x[i] } else { bf16::from_f32(0.0) };
        assert_eq!(got[i].to_bits(), exp.to_bits(), "hardshrink bf16 @ {i}");
    }
}
