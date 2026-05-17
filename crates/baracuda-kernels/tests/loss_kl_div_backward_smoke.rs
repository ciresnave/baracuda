//! Real-GPU smoke test for `KlDivLossBackwardPlan`. BW × 4 dtypes × Mean.
//! `dinput[i] = -target[i] · dy / N`.

use baracuda_driver::{init, DeviceBuffer};
use baracuda_kernels::{
    contiguous_stride, ElementKind, KlDivLossBackwardArgs, KlDivLossBackwardDescriptor,
    KlDivLossBackwardPlan, LossReduction, PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (baracuda_driver::Context, baracuda_driver::Stream) {
    init().expect("driver init");
    let device = baracuda_driver::Device::get(0).expect("device 0");
    let ctx = baracuda_driver::Context::new(&device).expect("context");
    let stream = baracuda_driver::Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn loss_kl_div_backward_f32_mean() {
    let (ctx, stream) = setup();
    let shape = [3i32, 4];
    let numel = 12usize;
    let host_t: Vec<f32> = (0..numel).map(|i| 0.05 + (i as f32) * 0.02).collect();
    let dy_host = [1.0f32];
    let scale = 1.0f32 / (numel as f32);
    let expected: Vec<f32> = host_t.iter().map(|&t| -t * scale).collect();

    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_di: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).unwrap();

    let desc = KlDivLossBackwardDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        element: ElementKind::F32,
    };
    let plan =
        KlDivLossBackwardPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
            .unwrap();
    plan.run(
        &stream,
        Workspace::None,
        KlDivLossBackwardArgs {
            target: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            dy: TensorRef { data: dev_dy.as_slice(), shape: [1, 1], stride: [1, 1] },
            dinput: TensorMut {
                data: dev_di.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = vec![0f32; numel];
    dev_di.copy_to_host(&mut got).unwrap();
    for i in 0..numel {
        let tol = expected[i].abs() * 16.0 * f32::EPSILON + 1e-6;
        assert!((got[i] - expected[i]).abs() <= tol);
    }
}

#[test]
#[ignore]
fn loss_kl_div_backward_f64_mean() {
    let (ctx, stream) = setup();
    let shape = [4i32, 3];
    let numel = 12usize;
    let host_t: Vec<f64> = (0..numel).map(|i| 0.04 + (i as f64) * 0.02).collect();
    let dy_host = [1.0f64];
    let scale = 1.0f64 / (numel as f64);
    let expected: Vec<f64> = host_t.iter().map(|&t| -t * scale).collect();

    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_di: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).unwrap();

    let desc = KlDivLossBackwardDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        element: ElementKind::F64,
    };
    let plan =
        KlDivLossBackwardPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default())
            .unwrap();
    plan.run(
        &stream,
        Workspace::None,
        KlDivLossBackwardArgs {
            target: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            dy: TensorRef { data: dev_dy.as_slice(), shape: [1, 1], stride: [1, 1] },
            dinput: TensorMut {
                data: dev_di.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = vec![0f64; numel];
    dev_di.copy_to_host(&mut got).unwrap();
    for i in 0..numel {
        let tol = expected[i].abs() * 16.0 * f64::EPSILON + 1e-13;
        assert!((got[i] - expected[i]).abs() <= tol);
    }
}

#[test]
#[ignore]
fn loss_kl_div_backward_f16_mean() {
    let (ctx, stream) = setup();
    let shape = [3i32, 4];
    let numel = 12usize;
    let host_t_f32: Vec<f32> = (0..numel).map(|i| 0.06 + (i as f32) * 0.02).collect();
    let host_t: Vec<f16> = host_t_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let dy_host = [f16::from_f32(1.0)];
    let scale = 1.0f32 / (numel as f32);
    let expected: Vec<f32> = host_t_f32.iter().map(|&t| -t * scale).collect();

    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_di: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).unwrap();

    let desc = KlDivLossBackwardDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        element: ElementKind::F16,
    };
    let plan =
        KlDivLossBackwardPlan::<f16, 2>::select(&stream, &desc, PlanPreference::default())
            .unwrap();
    plan.run(
        &stream,
        Workspace::None,
        KlDivLossBackwardArgs {
            target: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            dy: TensorRef { data: dev_dy.as_slice(), shape: [1, 1], stride: [1, 1] },
            dinput: TensorMut {
                data: dev_di.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = vec![f16::ZERO; numel];
    dev_di.copy_to_host(&mut got).unwrap();
    for i in 0..numel {
        let tol = expected[i].abs() * 16.0 * 9.77e-4_f32 + 5e-4;
        let g = got[i].to_f32();
        assert!((g - expected[i]).abs() <= tol);
    }
}

#[test]
#[ignore]
fn loss_kl_div_backward_bf16_mean() {
    let (ctx, stream) = setup();
    let shape = [3i32, 4];
    let numel = 12usize;
    let host_t_f32: Vec<f32> = (0..numel).map(|i| 0.05 + (i as f32) * 0.02).collect();
    let host_t: Vec<bf16> = host_t_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let dy_host = [bf16::from_f32(1.0)];
    let scale = 1.0f32 / (numel as f32);
    let expected: Vec<f32> = host_t_f32.iter().map(|&t| -t * scale).collect();

    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_di: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).unwrap();

    let desc = KlDivLossBackwardDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        element: ElementKind::Bf16,
    };
    let plan =
        KlDivLossBackwardPlan::<bf16, 2>::select(&stream, &desc, PlanPreference::default())
            .unwrap();
    plan.run(
        &stream,
        Workspace::None,
        KlDivLossBackwardArgs {
            target: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            dy: TensorRef { data: dev_dy.as_slice(), shape: [1, 1], stride: [1, 1] },
            dinput: TensorMut {
                data: dev_di.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = vec![bf16::ZERO; numel];
    dev_di.copy_to_host(&mut got).unwrap();
    for i in 0..numel {
        let tol = expected[i].abs() * 16.0 * 7.81e-3_f32 + 3e-3;
        let g = got[i].to_f32();
        assert!((g - expected[i]).abs() <= tol);
    }
}
