//! Real-GPU smoke test for `BceLossBackwardPlan`. BW × 4 dtypes × Mean.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, BceLossBackwardArgs, BceLossBackwardDescriptor, BceLossBackwardPlan,
    ElementKind, LossReduction, PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn host_bce_bw_f64(pred: &[f64], target: &[f64], dy: f64, n: usize) -> Vec<f64> {
    let scale = dy / (n as f64);
    pred.iter()
        .zip(target.iter())
        .map(|(&p, &t)| (p - t) / (p * (1.0 - p)) * scale)
        .collect()
}

#[test]
#[ignore]
fn loss_bce_backward_f32_mean() {
    let (ctx, stream) = setup();
    let shape = [3i32, 4];
    let numel = 12usize;
    let host_p: Vec<f32> = (0..numel).map(|i| 0.3 + (i as f32) * 0.04).collect();
    let host_t: Vec<f32> = (0..numel).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
    let dy_host = [1.0f32];

    let expected: Vec<f32> = host_bce_bw_f64(
        &host_p.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &host_t.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        1.0,
        numel,
    )
    .into_iter()
    .map(|v| v as f32)
    .collect();

    let dev_p = DeviceBuffer::from_slice(&ctx, &host_p).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_dp: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).unwrap();

    let desc = BceLossBackwardDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        element: ElementKind::F32,
    };
    let plan =
        BceLossBackwardPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        BceLossBackwardArgs {
            pred: TensorRef { data: dev_p.as_slice(), shape, stride: contiguous_stride(shape) },
            target: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            dy: TensorRef { data: dev_dy.as_slice(), shape: [1, 1], stride: [1, 1] },
            dpred: TensorMut {
                data: dev_dp.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = vec![0f32; numel];
    dev_dp.copy_to_host(&mut got).unwrap();
    for i in 0..numel {
        let tol = expected[i].abs() * 32.0 * f32::EPSILON + 1e-5;
        assert!((got[i] - expected[i]).abs() <= tol, "f32 BCE BW @{i}: got={} want={}",
            got[i], expected[i]);
    }
}

#[test]
#[ignore]
fn loss_bce_backward_f64_mean() {
    let (ctx, stream) = setup();
    let shape = [2i32, 5];
    let numel = 10usize;
    let host_p: Vec<f64> = (0..numel).map(|i| 0.3 + (i as f64) * 0.05).collect();
    let host_t: Vec<f64> = (0..numel).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
    let dy_host = [1.0f64];

    let expected = host_bce_bw_f64(&host_p, &host_t, 1.0, numel);

    let dev_p = DeviceBuffer::from_slice(&ctx, &host_p).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_dp: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).unwrap();

    let desc = BceLossBackwardDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        element: ElementKind::F64,
    };
    let plan =
        BceLossBackwardPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        BceLossBackwardArgs {
            pred: TensorRef { data: dev_p.as_slice(), shape, stride: contiguous_stride(shape) },
            target: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            dy: TensorRef { data: dev_dy.as_slice(), shape: [1, 1], stride: [1, 1] },
            dpred: TensorMut {
                data: dev_dp.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = vec![0f64; numel];
    dev_dp.copy_to_host(&mut got).unwrap();
    for i in 0..numel {
        let tol = expected[i].abs() * 32.0 * f64::EPSILON + 1e-12;
        assert!((got[i] - expected[i]).abs() <= tol);
    }
}

#[test]
#[ignore]
fn loss_bce_backward_f16_mean() {
    let (ctx, stream) = setup();
    let shape = [3i32, 4];
    let numel = 12usize;
    let host_p_f32: Vec<f32> = (0..numel).map(|i| 0.35 + (i as f32) * 0.03).collect();
    let host_t_f32: Vec<f32> = (0..numel).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
    let host_p: Vec<f16> = host_p_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_t: Vec<f16> = host_t_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let dy_host = [f16::from_f32(1.0)];

    let expected: Vec<f32> = host_bce_bw_f64(
        &host_p_f32.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &host_t_f32.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        1.0,
        numel,
    )
    .into_iter()
    .map(|v| v as f32)
    .collect();

    let dev_p = DeviceBuffer::from_slice(&ctx, &host_p).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_dp: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).unwrap();

    let desc = BceLossBackwardDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        element: ElementKind::F16,
    };
    let plan =
        BceLossBackwardPlan::<f16, 2>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        BceLossBackwardArgs {
            pred: TensorRef { data: dev_p.as_slice(), shape, stride: contiguous_stride(shape) },
            target: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            dy: TensorRef { data: dev_dy.as_slice(), shape: [1, 1], stride: [1, 1] },
            dpred: TensorMut {
                data: dev_dp.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = vec![f16::ZERO; numel];
    dev_dp.copy_to_host(&mut got).unwrap();
    for i in 0..numel {
        let tol = expected[i].abs() * 32.0 * 9.77e-4_f32 + 5e-3;
        let g = got[i].to_f32();
        assert!((g - expected[i]).abs() <= tol, "f16 BCE BW @{i}: got={} want={}",
            g, expected[i]);
    }
}

#[test]
#[ignore]
fn loss_bce_backward_bf16_mean() {
    let (ctx, stream) = setup();
    let shape = [3i32, 4];
    let numel = 12usize;
    let host_p_f32: Vec<f32> = (0..numel).map(|i| 0.3 + (i as f32) * 0.04).collect();
    let host_t_f32: Vec<f32> = (0..numel).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
    let host_p: Vec<bf16> = host_p_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_t: Vec<bf16> = host_t_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let dy_host = [bf16::from_f32(1.0)];

    let expected: Vec<f32> = host_bce_bw_f64(
        &host_p_f32.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &host_t_f32.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        1.0,
        numel,
    )
    .into_iter()
    .map(|v| v as f32)
    .collect();

    let dev_p = DeviceBuffer::from_slice(&ctx, &host_p).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_dp: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).unwrap();

    let desc = BceLossBackwardDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        element: ElementKind::Bf16,
    };
    let plan =
        BceLossBackwardPlan::<bf16, 2>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        BceLossBackwardArgs {
            pred: TensorRef { data: dev_p.as_slice(), shape, stride: contiguous_stride(shape) },
            target: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            dy: TensorRef { data: dev_dy.as_slice(), shape: [1, 1], stride: [1, 1] },
            dpred: TensorMut {
                data: dev_dp.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = vec![bf16::ZERO; numel];
    dev_dp.copy_to_host(&mut got).unwrap();
    for i in 0..numel {
        let tol = expected[i].abs() * 32.0 * 7.81e-3_f32 + 5e-2;
        let g = got[i].to_f32();
        assert!((g - expected[i]).abs() <= tol);
    }
}
