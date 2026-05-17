//! Real-GPU smoke test for `PoissonNllLossPlan`. FW × 4 dtypes × Mean
//! (log_input=true).

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, LossReduction, PlanPreference, PoissonNllLossArgs,
    PoissonNllLossDescriptor, PoissonNllLossPlan, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn host_poisson_nll_mean_log_f64(input: &[f64], target: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..input.len() {
        s += input[i].exp() - target[i] * input[i];
    }
    s / (input.len() as f64)
}

#[test]
#[ignore]
fn loss_poisson_nll_f32_mean() {
    let (ctx, stream) = setup();
    let shape = [3i32, 5];
    let numel = 15usize;
    let host_x: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.1 - 0.5).collect();
    let host_t: Vec<f32> = (0..numel).map(|i| ((i % 4) as f32) + 0.2).collect();
    let expected = host_poisson_nll_mean_log_f64(
        &host_x.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &host_t.iter().map(|&v| v as f64).collect::<Vec<_>>(),
    ) as f32;

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel * 4).unwrap();

    let desc = PoissonNllLossDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        log_input: true,
        element: ElementKind::F32,
    };
    let plan = PoissonNllLossPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        PoissonNllLossArgs {
            input: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            target: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1, 1], stride: [1, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = [0f32; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let tol = expected.abs() * 8.0 * f32::EPSILON + 1e-5;
    assert!((got[0] - expected).abs() <= tol, "f32 Poisson: got={} want={}", got[0], expected);
}

#[test]
#[ignore]
fn loss_poisson_nll_f64_mean() {
    let (ctx, stream) = setup();
    let shape = [3i32, 4];
    let numel = 12usize;
    let host_x: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.08 - 0.3).collect();
    let host_t: Vec<f64> = (0..numel).map(|i| ((i % 5) as f64) + 0.1).collect();
    let expected = host_poisson_nll_mean_log_f64(&host_x, &host_t);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel * 8).unwrap();

    let desc = PoissonNllLossDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        log_input: true,
        element: ElementKind::F64,
    };
    let plan = PoissonNllLossPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default())
        .unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        PoissonNllLossArgs {
            input: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            target: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1, 1], stride: [1, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = [0f64; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let tol = expected.abs() * 8.0 * f64::EPSILON + 1e-11;
    assert!((got[0] - expected).abs() <= tol);
}

#[test]
#[ignore]
fn loss_poisson_nll_f16_mean() {
    let (ctx, stream) = setup();
    let shape = [4i32, 5];
    let numel = 20usize;
    let host_x_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05 - 0.3).collect();
    let host_t_f32: Vec<f32> = (0..numel).map(|i| ((i % 4) as f32) + 0.2).collect();
    let host_x: Vec<f16> = host_x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_t: Vec<f16> = host_t_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let x64: Vec<f64> = host_x.iter().map(|&v| v.to_f32() as f64).collect();
    let t64: Vec<f64> = host_t.iter().map(|&v| v.to_f32() as f64).collect();
    let expected = host_poisson_nll_mean_log_f64(&x64, &t64) as f32;

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel * 2).unwrap();

    let desc = PoissonNllLossDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        log_input: true,
        element: ElementKind::F16,
    };
    let plan = PoissonNllLossPlan::<f16, 2>::select(&stream, &desc, PlanPreference::default())
        .unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        PoissonNllLossArgs {
            input: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
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
    assert!((got_f32 - expected).abs() <= tol, "f16 Poisson: got={} want={}", got_f32, expected);
}

#[test]
#[ignore]
fn loss_poisson_nll_bf16_mean() {
    let (ctx, stream) = setup();
    let shape = [3i32, 6];
    let numel = 18usize;
    let host_x_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.06 - 0.4).collect();
    let host_t_f32: Vec<f32> = (0..numel).map(|i| ((i % 4) as f32) + 0.2).collect();
    let host_x: Vec<bf16> = host_x_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_t: Vec<bf16> = host_t_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let x64: Vec<f64> = host_x.iter().map(|&v| v.to_f32() as f64).collect();
    let t64: Vec<f64> = host_t.iter().map(|&v| v.to_f32() as f64).collect();
    let expected = host_poisson_nll_mean_log_f64(&x64, &t64) as f32;

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel * 2).unwrap();

    let desc = PoissonNllLossDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        log_input: true,
        element: ElementKind::Bf16,
    };
    let plan = PoissonNllLossPlan::<bf16, 2>::select(&stream, &desc, PlanPreference::default())
        .unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        PoissonNllLossArgs {
            input: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
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
    assert!((got_f32 - expected).abs() <= tol, "bf16 Poisson: got={} want={}", got_f32, expected);
}
