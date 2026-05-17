//! Real-GPU smoke test for `GaussianNllLossPlan`. FW × 4 dtypes × Mean.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, GaussianNllLossArgs, GaussianNllLossDescriptor,
    GaussianNllLossPlan, LossReduction, PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn host_gauss_mean_f64(input: &[f64], target: &[f64], var: &[f64], eps: f64) -> f64 {
    let mut s = 0.0;
    for i in 0..input.len() {
        let ve = if var[i] > eps { var[i] } else { eps };
        let d = input[i] - target[i];
        s += 0.5 * (ve.ln() + d * d / ve);
    }
    s / (input.len() as f64)
}

#[test]
#[ignore]
fn loss_gaussian_nll_f32_mean() {
    let (ctx, stream) = setup();
    let shape = [3i32, 5];
    let numel = 15usize;
    let eps = 1e-6f32;
    let host_x: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.1 - 0.5).collect();
    let host_t: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.08).collect();
    let host_v: Vec<f32> = (0..numel).map(|i| 0.5 + (i as f32) * 0.05).collect();
    let expected = host_gauss_mean_f64(
        &host_x.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &host_t.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &host_v.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        eps as f64,
    ) as f32;

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_v = DeviceBuffer::from_slice(&ctx, &host_v).unwrap();
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel * 4).unwrap();

    let desc = GaussianNllLossDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        eps,
        element: ElementKind::F32,
    };
    let plan = GaussianNllLossPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        GaussianNllLossArgs {
            input: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            target: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            var: TensorRef { data: dev_v.as_slice(), shape, stride: contiguous_stride(shape) },
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1, 1], stride: [1, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = [0f32; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let tol = expected.abs() * 8.0 * f32::EPSILON + 1e-5;
    assert!((got[0] - expected).abs() <= tol, "f32 Gauss: got={} want={}", got[0], expected);
}

#[test]
#[ignore]
fn loss_gaussian_nll_f64_mean() {
    let (ctx, stream) = setup();
    let shape = [3i32, 4];
    let numel = 12usize;
    let eps = 1e-6f32;
    let host_x: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.08 - 0.3).collect();
    let host_t: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.07).collect();
    let host_v: Vec<f64> = (0..numel).map(|i| 0.5 + (i as f64) * 0.04).collect();
    let expected = host_gauss_mean_f64(&host_x, &host_t, &host_v, eps as f64);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_v = DeviceBuffer::from_slice(&ctx, &host_v).unwrap();
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel * 8).unwrap();

    let desc = GaussianNllLossDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        eps,
        element: ElementKind::F64,
    };
    let plan = GaussianNllLossPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default())
        .unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        GaussianNllLossArgs {
            input: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            target: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            var: TensorRef { data: dev_v.as_slice(), shape, stride: contiguous_stride(shape) },
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
fn loss_gaussian_nll_f16_mean() {
    let (ctx, stream) = setup();
    let shape = [4i32, 5];
    let numel = 20usize;
    let eps = 1e-3f32;
    let host_x_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05 - 0.3).collect();
    let host_t_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.04).collect();
    let host_v_f32: Vec<f32> = (0..numel).map(|i| 0.5 + (i as f32) * 0.03).collect();
    let host_x: Vec<f16> = host_x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_t: Vec<f16> = host_t_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_v: Vec<f16> = host_v_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let x64: Vec<f64> = host_x.iter().map(|&v| v.to_f32() as f64).collect();
    let t64: Vec<f64> = host_t.iter().map(|&v| v.to_f32() as f64).collect();
    let v64: Vec<f64> = host_v.iter().map(|&v| v.to_f32() as f64).collect();
    let expected = host_gauss_mean_f64(&x64, &t64, &v64, eps as f64) as f32;

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_v = DeviceBuffer::from_slice(&ctx, &host_v).unwrap();
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel * 2).unwrap();

    let desc = GaussianNllLossDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        eps,
        element: ElementKind::F16,
    };
    let plan = GaussianNllLossPlan::<f16, 2>::select(&stream, &desc, PlanPreference::default())
        .unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        GaussianNllLossArgs {
            input: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            target: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            var: TensorRef { data: dev_v.as_slice(), shape, stride: contiguous_stride(shape) },
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1, 1], stride: [1, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = [f16::ZERO; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let got_f32 = got[0].to_f32();
    let tol = expected.abs() * 16.0 * 9.77e-4_f32 + 1e-2;
    assert!((got_f32 - expected).abs() <= tol, "f16 Gauss: got={} want={}", got_f32, expected);
}

#[test]
#[ignore]
fn loss_gaussian_nll_bf16_mean() {
    let (ctx, stream) = setup();
    let shape = [3i32, 6];
    let numel = 18usize;
    let eps = 1e-3f32;
    let host_x_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.06 - 0.4).collect();
    let host_t_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05).collect();
    let host_v_f32: Vec<f32> = (0..numel).map(|i| 0.5 + (i as f32) * 0.04).collect();
    let host_x: Vec<bf16> = host_x_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_t: Vec<bf16> = host_t_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_v: Vec<bf16> = host_v_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let x64: Vec<f64> = host_x.iter().map(|&v| v.to_f32() as f64).collect();
    let t64: Vec<f64> = host_t.iter().map(|&v| v.to_f32() as f64).collect();
    let v64: Vec<f64> = host_v.iter().map(|&v| v.to_f32() as f64).collect();
    let expected = host_gauss_mean_f64(&x64, &t64, &v64, eps as f64) as f32;

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_v = DeviceBuffer::from_slice(&ctx, &host_v).unwrap();
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel * 2).unwrap();

    let desc = GaussianNllLossDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        eps,
        element: ElementKind::Bf16,
    };
    let plan = GaussianNllLossPlan::<bf16, 2>::select(&stream, &desc, PlanPreference::default())
        .unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        GaussianNllLossArgs {
            input: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            target: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            var: TensorRef { data: dev_v.as_slice(), shape, stride: contiguous_stride(shape) },
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1, 1], stride: [1, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = [bf16::ZERO; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let got_f32 = got[0].to_f32();
    let tol = expected.abs() * 16.0 * 7.81e-3_f32 + 3e-2;
    assert!((got_f32 - expected).abs() <= tol, "bf16 Gauss: got={} want={}", got_f32, expected);
}
