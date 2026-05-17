//! Real-GPU smoke test for `UnaryPlan + UnaryKind::Softshrink` across
//! f32 / f16 / bf16 / f64. λ hardcoded to 0.5.
//!
//! Forward: `y = x - λ if x > λ; x + λ if x < -λ; else 0`. f32 / f64 are
//! bit-exact (single subtract / add by exactly 0.5). f16 / bf16 use a
//! tight relative tolerance because the kernel routes through f32 then
//! rounds once on store — at boundary points the same exact value should
//! round identically, but we allow 4·ULP for safety.

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

fn cpu_ref_f32(x: f32) -> f32 {
    if x > 0.5 { x - 0.5 } else if x < -0.5 { x + 0.5 } else { 0.0 }
}
fn cpu_ref_f64(x: f64) -> f64 {
    if x > 0.5 { x - 0.5 } else if x < -0.5 { x + 0.5 } else { 0.0 }
}

#[test]
#[ignore]
fn softshrink_f32_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f32> = (0..numel).map(|i| ((i % 401) as f32) * 0.01 - 2.0).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let desc = UnaryDescriptor { kind: UnaryKind::Softshrink, shape, element: ElementKind::F32 };
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
        let exp = cpu_ref_f32(host_x[i]);
        assert_eq!(got[i].to_bits(), exp.to_bits(),
            "softshrink f32 @ {i}: x={} got {} exp {}", host_x[i], got[i], exp);
    }
}

#[test]
#[ignore]
fn softshrink_f64_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f64> = (0..numel).map(|i| ((i % 401) as f64) * 0.01 - 2.0).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let desc = UnaryDescriptor { kind: UnaryKind::Softshrink, shape, element: ElementKind::F64 };
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
        let exp = cpu_ref_f64(host_x[i]);
        assert_eq!(got[i].to_bits(), exp.to_bits(), "softshrink f64 @ {i}");
    }
}

#[test]
#[ignore]
fn softshrink_f16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f16> = (0..numel).map(|i| f16::from_f32(((i % 401) as f32) * 0.01 - 2.0)).collect();
    let host_expected: Vec<f16> = host_x.iter()
        .map(|x| f16::from_f32(cpu_ref_f32(x.to_f32())))
        .collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let desc = UnaryDescriptor { kind: UnaryKind::Softshrink, shape, element: ElementKind::F16 };
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
        let got_f = got[i].to_f32();
        let exp_f = host_expected[i].to_f32();
        let diff = (got_f - exp_f).abs();
        let allow = exp_f.abs().max(1.0) * 4.0 * F16_EPS;
        assert!(diff <= allow, "softshrink f16 @ {i}: got {} exp {} (diff {} > allow {})", got_f, exp_f, diff, allow);
    }
}

#[test]
#[ignore]
fn softshrink_bf16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<bf16> = (0..numel).map(|i| bf16::from_f32(((i % 401) as f32) * 0.01 - 2.0)).collect();
    let host_expected: Vec<bf16> = host_x.iter()
        .map(|x| bf16::from_f32(cpu_ref_f32(x.to_f32())))
        .collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let desc = UnaryDescriptor { kind: UnaryKind::Softshrink, shape, element: ElementKind::Bf16 };
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
        let got_f = got[i].to_f32();
        let exp_f = host_expected[i].to_f32();
        let diff = (got_f - exp_f).abs();
        let allow = exp_f.abs().max(1.0) * 4.0 * BF16_EPS;
        assert!(diff <= allow, "softshrink bf16 @ {i}: got {} exp {} (diff {} > allow {})", got_f, exp_f, diff, allow);
    }
}
