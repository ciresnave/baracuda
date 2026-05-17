//! Real-GPU smoke test for `UnaryBackwardPlan + UnaryKind::Elu`.
//!
//! Forward: `y = x if x > 0 else α·(exp(x) - 1)`. Backward:
//! `dx = dy if x > 0 else dy·α·exp(x)`. α hardcoded to 1.0.
//!
//! Tolerance: `4 * eps` (8 * eps on the negative branch, since exp is
//! involved); we use a uniform `8 * eps` for f32/f64 and the equivalent
//! ULP-eps for f16/bf16.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef, UnaryBackwardArgs,
    UnaryBackwardDescriptor, UnaryBackwardPlan, UnaryKind, Workspace,
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

#[test]
#[ignore]
fn elu_backward_f32_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f32> = (0..numel).map(|i| ((i % 401) as f32) * 0.02 - 4.0).collect();
    let host_dy: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.5 - 17.25).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");
    let stride = contiguous_stride(shape);
    let desc = UnaryBackwardDescriptor { kind: UnaryKind::Elu, shape, element: ElementKind::F32 };
    let plan = UnaryBackwardPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = UnaryBackwardArgs::<f32, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        x: Some(TensorRef { data: dev_x.as_slice(), shape, stride }),
        y: None,
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for i in 0..numel {
        let exp = if host_x[i] > 0.0 { host_dy[i] } else { host_dy[i] * host_x[i].exp() };
        let diff = (got[i] - exp).abs();
        let allow = exp.abs().max(1.0) * 8.0 * f32::EPSILON;
        assert!(diff <= allow, "elu bw f32 @ {i}: x={} dy={} got {} exp {} (diff {} > allow {})",
            host_x[i], host_dy[i], got[i], exp, diff, allow);
    }
}

#[test]
#[ignore]
fn elu_backward_f64_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f64> = (0..numel).map(|i| ((i % 401) as f64) * 0.02 - 4.0).collect();
    let host_dy: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.5 - 17.25).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");
    let stride = contiguous_stride(shape);
    let desc = UnaryBackwardDescriptor { kind: UnaryKind::Elu, shape, element: ElementKind::F64 };
    let plan = UnaryBackwardPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = UnaryBackwardArgs::<f64, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        x: Some(TensorRef { data: dev_x.as_slice(), shape, stride }),
        y: None,
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f64; numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for i in 0..numel {
        // GPU uses expf (f32) on the negative branch — match that path for f64 ref
        // is overkill; we just bound generously with 1e-5 absolute on the negative
        // branch (the ratio is at most 1e-7 relative due to expf precision).
        let exp_val = if host_x[i] > 0.0 { host_dy[i] } else { host_dy[i] * host_x[i].exp() };
        let diff = (got[i] - exp_val).abs();
        let allow = exp_val.abs().max(1.0) * 8.0 * f64::EPSILON;
        assert!(diff <= allow, "elu bw f64 @ {i}: diff {} > allow {}", diff, allow);
    }
}

#[test]
#[ignore]
fn elu_backward_f16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f16> = (0..numel).map(|i| f16::from_f32(((i % 401) as f32) * 0.02 - 4.0)).collect();
    let host_dy: Vec<f16> = (0..numel).map(|i| f16::from_f32(((i % 81) as f32) * 0.25 - 10.0)).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");
    let stride = contiguous_stride(shape);
    let desc = UnaryBackwardDescriptor { kind: UnaryKind::Elu, shape, element: ElementKind::F16 };
    let plan = UnaryBackwardPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = UnaryBackwardArgs::<f16, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        x: Some(TensorRef { data: dev_x.as_slice(), shape, stride }),
        y: None,
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![f16::from_f32(0.0); numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for i in 0..numel {
        let xf = host_x[i].to_f32();
        let dyf = host_dy[i].to_f32();
        let exp = if xf > 0.0 { dyf } else { dyf * xf.exp() };
        let got_f = got[i].to_f32();
        let diff = (got_f - exp).abs();
        let allow = exp.abs().max(1.0) * 8.0 * F16_EPS;
        assert!(diff <= allow, "elu bw f16 @ {i}: got {} exp {} (diff {} > allow {})", got_f, exp, diff, allow);
    }
}

#[test]
#[ignore]
fn elu_backward_bf16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<bf16> = (0..numel).map(|i| bf16::from_f32(((i % 401) as f32) * 0.02 - 4.0)).collect();
    let host_dy: Vec<bf16> = (0..numel).map(|i| bf16::from_f32(((i % 81) as f32) * 0.25 - 10.0)).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");
    let stride = contiguous_stride(shape);
    let desc = UnaryBackwardDescriptor { kind: UnaryKind::Elu, shape, element: ElementKind::Bf16 };
    let plan = UnaryBackwardPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = UnaryBackwardArgs::<bf16, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        x: Some(TensorRef { data: dev_x.as_slice(), shape, stride }),
        y: None,
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![bf16::from_f32(0.0); numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for i in 0..numel {
        let xf = host_x[i].to_f32();
        let dyf = host_dy[i].to_f32();
        let exp = if xf > 0.0 { dyf } else { dyf * xf.exp() };
        let got_f = got[i].to_f32();
        let diff = (got_f - exp).abs();
        let allow = exp.abs().max(1.0) * 8.0 * BF16_EPS;
        assert!(diff <= allow, "elu bw bf16 @ {i}: got {} exp {} (diff {} > allow {})", got_f, exp, diff, allow);
    }
}
