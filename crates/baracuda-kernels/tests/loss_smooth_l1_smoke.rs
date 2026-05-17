//! Real-GPU smoke test for `SmoothL1LossPlan`. FW × 4 dtypes × Mean.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, LossReduction, PlanPreference, SmoothL1LossArgs,
    SmoothL1LossDescriptor, SmoothL1LossPlan, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn host_smooth_l1_mean_f64(pred: &[f64], target: &[f64], beta: f64) -> f64 {
    let mut s = 0.0;
    for i in 0..pred.len() {
        let x = pred[i] - target[i];
        let ax = x.abs();
        s += if ax < beta { 0.5 * x * x / beta } else { ax - 0.5 * beta };
    }
    s / (pred.len() as f64)
}

#[test]
#[ignore]
fn loss_smooth_l1_f32_mean() {
    let (ctx, stream) = setup();
    let shape = [3i32, 5];
    let numel = 15usize;
    let beta = 1.0f32;
    let host_p: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.1 - 0.7).collect();
    let host_t: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.08).collect();
    let expected = host_smooth_l1_mean_f64(
        &host_p.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &host_t.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        beta as f64,
    ) as f32;

    let dev_p = DeviceBuffer::from_slice(&ctx, &host_p).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel * 4).unwrap();

    let desc = SmoothL1LossDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        beta,
        element: ElementKind::F32,
    };
    let plan =
        SmoothL1LossPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        SmoothL1LossArgs {
            pred: TensorRef { data: dev_p.as_slice(), shape, stride: contiguous_stride(shape) },
            target: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1, 1], stride: [1, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = [0f32; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let tol = expected.abs() * 8.0 * f32::EPSILON + 1e-6;
    assert!((got[0] - expected).abs() <= tol, "f32 SmoothL1: got={} want={}", got[0], expected);
}

#[test]
#[ignore]
fn loss_smooth_l1_f64_mean() {
    let (ctx, stream) = setup();
    let shape = [3i32, 4];
    let numel = 12usize;
    let beta = 0.5f32;
    let host_p: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.08 - 0.4).collect();
    let host_t: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.06).collect();
    let expected = host_smooth_l1_mean_f64(&host_p, &host_t, beta as f64);

    let dev_p = DeviceBuffer::from_slice(&ctx, &host_p).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel * 8).unwrap();

    let desc = SmoothL1LossDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        beta,
        element: ElementKind::F64,
    };
    let plan =
        SmoothL1LossPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        SmoothL1LossArgs {
            pred: TensorRef { data: dev_p.as_slice(), shape, stride: contiguous_stride(shape) },
            target: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1, 1], stride: [1, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = [0f64; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let tol = expected.abs() * 8.0 * f64::EPSILON + 1e-12;
    assert!((got[0] - expected).abs() <= tol);
}

#[test]
#[ignore]
fn loss_smooth_l1_f16_mean() {
    let (ctx, stream) = setup();
    let shape = [4i32, 6];
    let numel = 24usize;
    let beta = 1.0f32;
    let host_p_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05 - 0.5).collect();
    let host_t_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.04).collect();
    let host_p: Vec<f16> = host_p_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_t: Vec<f16> = host_t_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let p64: Vec<f64> = host_p.iter().map(|&v| v.to_f32() as f64).collect();
    let t64: Vec<f64> = host_t.iter().map(|&v| v.to_f32() as f64).collect();
    let expected = host_smooth_l1_mean_f64(&p64, &t64, beta as f64) as f32;

    let dev_p = DeviceBuffer::from_slice(&ctx, &host_p).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel * 2).unwrap();

    let desc = SmoothL1LossDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        beta,
        element: ElementKind::F16,
    };
    let plan =
        SmoothL1LossPlan::<f16, 2>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        SmoothL1LossArgs {
            pred: TensorRef { data: dev_p.as_slice(), shape, stride: contiguous_stride(shape) },
            target: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1, 1], stride: [1, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = [f16::ZERO; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let got_f32 = got[0].to_f32();
    let tol = expected.abs() * 16.0 * 9.77e-4_f32 + 5e-3;
    assert!((got_f32 - expected).abs() <= tol, "f16 SmoothL1: got={} want={}", got_f32, expected);
}

#[test]
#[ignore]
fn loss_smooth_l1_bf16_mean() {
    let (ctx, stream) = setup();
    let shape = [3i32, 7];
    let numel = 21usize;
    let beta = 1.0f32;
    let host_p_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.06 - 0.6).collect();
    let host_t_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05).collect();
    let host_p: Vec<bf16> = host_p_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_t: Vec<bf16> = host_t_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let p64: Vec<f64> = host_p.iter().map(|&v| v.to_f32() as f64).collect();
    let t64: Vec<f64> = host_t.iter().map(|&v| v.to_f32() as f64).collect();
    let expected = host_smooth_l1_mean_f64(&p64, &t64, beta as f64) as f32;

    let dev_p = DeviceBuffer::from_slice(&ctx, &host_p).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel * 2).unwrap();

    let desc = SmoothL1LossDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        beta,
        element: ElementKind::Bf16,
    };
    let plan =
        SmoothL1LossPlan::<bf16, 2>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        SmoothL1LossArgs {
            pred: TensorRef { data: dev_p.as_slice(), shape, stride: contiguous_stride(shape) },
            target: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1, 1], stride: [1, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = [bf16::ZERO; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let got_f32 = got[0].to_f32();
    let tol = expected.abs() * 16.0 * 7.81e-3_f32 + 2e-2;
    assert!((got_f32 - expected).abs() <= tol, "bf16 SmoothL1: got={} want={}", got_f32, expected);
}
