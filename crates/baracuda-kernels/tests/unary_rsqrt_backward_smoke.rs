//! Real-GPU smoke test for unary rsqrt backward
//! (`UnaryBackwardPlan<T, 3> + UnaryKind::Rsqrt`). Backward:
//! `dx = -0.5 * dy * y³`. Saved-y.

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
fn rsqrt_backward_f32_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    // y = 1/sqrt(x); bounded in (0, ~2] for x >= ~0.25. Use a small
    // positive range so y³ stays well within f16 / bf16 reach.
    let host_y: Vec<f32> = (0..numel).map(|i| ((i % 50) as f32) * 0.02 + 0.5).collect();
    let host_dy: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.5 - 17.25).collect();
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("upload y");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");
    let stride = contiguous_stride(shape);
    let desc = UnaryBackwardDescriptor { kind: UnaryKind::Rsqrt, shape, element: ElementKind::F32 };
    let plan = UnaryBackwardPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = UnaryBackwardArgs::<f32, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        x: None,
        y: Some(TensorRef { data: dev_y.as_slice(), shape, stride }),
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for i in 0..numel {
        let y = host_y[i];
        let exp = -0.5 * host_dy[i] * y * y * y;
        let tol = exp.abs().max(1.0) * 8.0 * f32::EPSILON;
        assert!((got[i] - exp).abs() <= tol, "rsqrt bw f32 @ {i}: got {} exp {}", got[i], exp);
    }
}

#[test]
#[ignore]
fn rsqrt_backward_f64_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_y: Vec<f64> = (0..numel).map(|i| ((i % 50) as f64) * 0.02 + 0.5).collect();
    let host_dy: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.5 - 17.25).collect();
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("upload y");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");
    let stride = contiguous_stride(shape);
    let desc = UnaryBackwardDescriptor { kind: UnaryKind::Rsqrt, shape, element: ElementKind::F64 };
    let plan = UnaryBackwardPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = UnaryBackwardArgs::<f64, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        x: None,
        y: Some(TensorRef { data: dev_y.as_slice(), shape, stride }),
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f64; numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for i in 0..numel {
        let y = host_y[i];
        let exp = -0.5 * host_dy[i] * y * y * y;
        let tol = exp.abs().max(1.0) * 8.0 * f64::EPSILON;
        assert!((got[i] - exp).abs() <= tol, "rsqrt bw f64 @ {i}");
    }
}

#[test]
#[ignore]
fn rsqrt_backward_f16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_y: Vec<f16> = (0..numel).map(|i| f16::from_f32(((i % 50) as f32) * 0.02 + 0.5)).collect();
    let host_dy: Vec<f16> = (0..numel).map(|i| f16::from_f32((i % 41) as f32 * 0.25 - 5.0)).collect();
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("upload y");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");
    let stride = contiguous_stride(shape);
    let desc = UnaryBackwardDescriptor { kind: UnaryKind::Rsqrt, shape, element: ElementKind::F16 };
    let plan = UnaryBackwardPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = UnaryBackwardArgs::<f16, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        x: None,
        y: Some(TensorRef { data: dev_y.as_slice(), shape, stride }),
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![f16::from_f32(0.0); numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for i in 0..numel {
        let y = host_y[i].to_f32();
        let dy = host_dy[i].to_f32();
        let exp = -0.5 * dy * y * y * y;
        let g = got[i].to_f32();
        let tol = exp.abs().max(1.0) * 8.0 * F16_EPS;
        assert!((g - exp).abs() <= tol, "rsqrt bw f16 @ {i}: got {g} exp {exp}");
    }
}

#[test]
#[ignore]
fn rsqrt_backward_bf16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_y: Vec<bf16> = (0..numel).map(|i| bf16::from_f32(((i % 50) as f32) * 0.02 + 0.5)).collect();
    let host_dy: Vec<bf16> = (0..numel).map(|i| bf16::from_f32((i % 41) as f32 * 0.25 - 5.0)).collect();
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("upload y");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");
    let stride = contiguous_stride(shape);
    let desc = UnaryBackwardDescriptor { kind: UnaryKind::Rsqrt, shape, element: ElementKind::Bf16 };
    let plan = UnaryBackwardPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = UnaryBackwardArgs::<bf16, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        x: None,
        y: Some(TensorRef { data: dev_y.as_slice(), shape, stride }),
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![bf16::from_f32(0.0); numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for i in 0..numel {
        let y = host_y[i].to_f32();
        let dy = host_dy[i].to_f32();
        let exp = -0.5 * dy * y * y * y;
        let g = got[i].to_f32();
        let tol = exp.abs().max(1.0) * 8.0 * BF16_EPS;
        assert!((g - exp).abs() <= tol, "rsqrt bw bf16 @ {i}: got {g} exp {exp}");
    }
}
