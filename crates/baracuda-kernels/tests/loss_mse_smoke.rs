//! Real-GPU smoke test for `MseLossPlan`. FW × 4 dtypes × Mean reduction.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, LossReduction, MseLossArgs, MseLossDescriptor, MseLossPlan,
    PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn host_mse_mean_f64(pred: &[f64], target: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..pred.len() {
        let d = pred[i] - target[i];
        s += d * d;
    }
    s / (pred.len() as f64)
}

#[test]
#[ignore]
fn loss_mse_f32_mean() {
    let (ctx, stream) = setup();
    let shape = [2i32, 8];
    let numel = 16usize;
    let host_p: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.1).collect();
    let host_t: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.1 + 0.05).collect();
    let expected = host_mse_mean_f64(
        &host_p.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &host_t.iter().map(|&v| v as f64).collect::<Vec<_>>(),
    ) as f32;

    let dev_p = DeviceBuffer::from_slice(&ctx, &host_p).expect("up p");
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).expect("up t");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc y");
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, numel * 4).expect("alloc ws");

    let desc = MseLossDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        element: ElementKind::F32,
    };
    let plan = MseLossPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        MseLossArgs {
            pred: TensorRef { data: dev_p.as_slice(), shape, stride: contiguous_stride(shape) },
            target: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1, 1], stride: [1, 1] },
        },
    )
    .expect("run");
    stream.synchronize().unwrap();
    let mut got = [0f32; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let tol = expected.abs() * 8.0 * f32::EPSILON + 1e-6;
    assert!((got[0] - expected).abs() <= tol, "f32 MSE: got={} want={}", got[0], expected);
}

#[test]
#[ignore]
fn loss_mse_f64_mean() {
    let (ctx, stream) = setup();
    let shape = [3i32, 5];
    let numel = 15usize;
    let host_p: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.07).collect();
    let host_t: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.07 + 0.03).collect();
    let expected = host_mse_mean_f64(&host_p, &host_t);

    let dev_p = DeviceBuffer::from_slice(&ctx, &host_p).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel * 8).unwrap();

    let desc = MseLossDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        element: ElementKind::F64,
    };
    let plan = MseLossPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        MseLossArgs {
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
fn loss_mse_f16_mean() {
    let (ctx, stream) = setup();
    let shape = [4i32, 6];
    let numel = 24usize;
    let host_p_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05).collect();
    let host_t_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05 + 0.02).collect();
    let expected = host_mse_mean_f64(
        &host_p_f32.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &host_t_f32.iter().map(|&v| v as f64).collect::<Vec<_>>(),
    ) as f32;
    let host_p: Vec<f16> = host_p_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_t: Vec<f16> = host_t_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let dev_p = DeviceBuffer::from_slice(&ctx, &host_p).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel * 2).unwrap();

    let desc = MseLossDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        element: ElementKind::F16,
    };
    let plan = MseLossPlan::<f16, 2>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        MseLossArgs {
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
    // f16 has ~3-4 decimal digits of precision; MSE is small, use absolute tol.
    let tol = expected.abs() * 16.0 * 9.77e-4_f32 + 1e-3;
    assert!((got_f32 - expected).abs() <= tol, "f16 MSE: got={} want={}", got_f32, expected);
}

#[test]
#[ignore]
fn loss_mse_bf16_mean() {
    let (ctx, stream) = setup();
    let shape = [3i32, 7];
    let numel = 21usize;
    let host_p_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.08).collect();
    let host_t_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.08 + 0.04).collect();
    let expected = host_mse_mean_f64(
        &host_p_f32.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &host_t_f32.iter().map(|&v| v as f64).collect::<Vec<_>>(),
    ) as f32;
    let host_p: Vec<bf16> = host_p_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_t: Vec<bf16> = host_t_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let dev_p = DeviceBuffer::from_slice(&ctx, &host_p).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel * 2).unwrap();

    let desc = MseLossDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        element: ElementKind::Bf16,
    };
    let plan = MseLossPlan::<bf16, 2>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        MseLossArgs {
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
    let tol = expected.abs() * 16.0 * 7.81e-3_f32 + 1e-2;
    assert!((got_f32 - expected).abs() <= tol, "bf16 MSE: got={} want={}", got_f32, expected);
}
