//! Real-GPU smoke test for `UnaryPlan + UnaryKind::LeakyRelu` across
//! f32 / f16 / bf16 / f64. α is hardcoded to 0.01 in the kernel (PyTorch
//! default).
//!
//! Inputs straddle zero to exercise both branches. Tolerance: `4 * eps`
//! relative to expected magnitude — the negative branch involves a
//! 0.01 multiplication that introduces ~1 ULP of rounding error on
//! f32 / f64.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef, UnaryArgs,
    UnaryDescriptor, UnaryKind, UnaryPlan, Workspace,
};
use half::{bf16, f16};

const F16_EPS: f32 = 9.77e-4;
const BF16_EPS: f32 = 7.81e-3;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn cpu_ref_f32(x: f32) -> f32 { if x > 0.0 { x } else { 0.01 * x } }
fn cpu_ref_f64(x: f64) -> f64 { if x > 0.0 { x } else { 0.01 * x } }

fn assert_close_f32(got: f32, expected: f32, idx: usize) {
    let diff = (got - expected).abs();
    let allow = expected.abs().max(1.0) * 4.0 * f32::EPSILON;
    assert!(
        diff <= allow,
        "leaky_relu f32 mismatch @ {idx}: got {got} expected {expected} (diff {diff} > allow {allow})"
    );
}

fn assert_close_f64(got: f64, expected: f64, idx: usize) {
    let diff = (got - expected).abs();
    let allow = expected.abs().max(1.0) * 4.0 * f64::EPSILON;
    assert!(
        diff <= allow,
        "leaky_relu f64 mismatch @ {idx}: got {got} expected {expected} (diff {diff} > allow {allow})"
    );
}

fn assert_close_f16(got: f32, expected: f32, idx: usize) {
    let diff = (got - expected).abs();
    let allow = expected.abs().max(1.0) * 4.0 * F16_EPS;
    assert!(
        diff <= allow,
        "leaky_relu f16 mismatch @ {idx}: got {got} expected {expected} (diff {diff} > allow {allow})"
    );
}

fn assert_close_bf16(got: f32, expected: f32, idx: usize) {
    let diff = (got - expected).abs();
    let allow = expected.abs().max(1.0) * 4.0 * BF16_EPS;
    assert!(
        diff <= allow,
        "leaky_relu bf16 mismatch @ {idx}: got {got} expected {expected} (diff {diff} > allow {allow})"
    );
}

#[test]
#[ignore]
fn leaky_relu_f32_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.5 - 17.25).collect();
    let expected: Vec<f32> = host_x.iter().map(|&x| cpu_ref_f32(x)).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let desc = UnaryDescriptor { kind: UnaryKind::LeakyRelu, shape, element: ElementKind::F32 };
    let plan = UnaryPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = UnaryArgs::<f32, 3> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_close_f32(*g, *e, i);
    }
}

#[test]
#[ignore]
fn leaky_relu_f64_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.5 - 17.25).collect();
    let expected: Vec<f64> = host_x.iter().map(|&x| cpu_ref_f64(x)).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let desc = UnaryDescriptor { kind: UnaryKind::LeakyRelu, shape, element: ElementKind::F64 };
    let plan = UnaryPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = UnaryArgs::<f64, 3> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f64; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_close_f64(*g, *e, i);
    }
}

#[test]
#[ignore]
fn leaky_relu_f16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f16> = (0..numel).map(|i| f16::from_f32((i as f32) * 0.5 - 17.25)).collect();
    let host_expected: Vec<f16> = host_x.iter()
        .map(|x| f16::from_f32(cpu_ref_f32(x.to_f32())))
        .collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let desc = UnaryDescriptor { kind: UnaryKind::LeakyRelu, shape, element: ElementKind::F16 };
    let plan = UnaryPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = UnaryArgs::<f16, 3> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![f16::from_f32(0.0); numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(host_expected.iter()).enumerate() {
        assert_close_f16(g.to_f32(), e.to_f32(), i);
    }
}

#[test]
#[ignore]
fn leaky_relu_bf16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<bf16> = (0..numel).map(|i| bf16::from_f32((i as f32) * 0.5 - 17.25)).collect();
    let host_expected: Vec<bf16> = host_x.iter()
        .map(|x| bf16::from_f32(cpu_ref_f32(x.to_f32())))
        .collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let desc = UnaryDescriptor { kind: UnaryKind::LeakyRelu, shape, element: ElementKind::Bf16 };
    let plan = UnaryPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = UnaryArgs::<bf16, 3> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![bf16::from_f32(0.0); numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(host_expected.iter()).enumerate() {
        assert_close_bf16(g.to_f32(), e.to_f32(), i);
    }
}
