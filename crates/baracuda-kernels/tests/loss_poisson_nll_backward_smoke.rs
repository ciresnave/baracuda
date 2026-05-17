//! Real-GPU smoke test for `PoissonNllLossBackwardPlan`. BW × 4 dtypes × Mean.
//!
//! `dinput = (exp(input) - target) · dy / N` (log_input=true).

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, LossReduction, PlanPreference, PoissonNllLossBackwardArgs,
    PoissonNllLossBackwardDescriptor, PoissonNllLossBackwardPlan, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn host_poisson_bw_f64(input: &[f64], target: &[f64], dy: f64, n: usize) -> Vec<f64> {
    let scale = dy / (n as f64);
    input
        .iter()
        .zip(target.iter())
        .map(|(&x, &t)| (x.exp() - t) * scale)
        .collect()
}

#[test]
#[ignore]
fn loss_poisson_nll_backward_f32_mean() {
    let (ctx, stream) = setup();
    let shape = [3i32, 5];
    let numel = 15usize;
    let host_x: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.1 - 0.5).collect();
    let host_t: Vec<f32> = (0..numel).map(|i| ((i % 4) as f32) + 0.2).collect();
    let dy_host = [1.0f32];
    let expected: Vec<f32> = host_poisson_bw_f64(
        &host_x.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &host_t.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        1.0, numel,
    )
    .into_iter()
    .map(|v| v as f32)
    .collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_di: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).unwrap();

    let desc = PoissonNllLossBackwardDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        log_input: true,
        element: ElementKind::F32,
    };
    let plan = PoissonNllLossBackwardPlan::<f32, 2>::select(
        &stream, &desc, PlanPreference::default()
    ).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        PoissonNllLossBackwardArgs {
            input: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
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
        let tol = expected[i].abs() * 8.0 * f32::EPSILON + 1e-6;
        assert!((got[i] - expected[i]).abs() <= tol, "f32 Poisson BW @{i}: got={} want={}",
            got[i], expected[i]);
    }
}

#[test]
#[ignore]
fn loss_poisson_nll_backward_f64_mean() {
    let (ctx, stream) = setup();
    let shape = [3i32, 4];
    let numel = 12usize;
    let host_x: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.08 - 0.3).collect();
    let host_t: Vec<f64> = (0..numel).map(|i| ((i % 5) as f64) + 0.1).collect();
    let dy_host = [2.0f64];
    let expected = host_poisson_bw_f64(&host_x, &host_t, 2.0, numel);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_di: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).unwrap();

    let desc = PoissonNllLossBackwardDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        log_input: true,
        element: ElementKind::F64,
    };
    let plan = PoissonNllLossBackwardPlan::<f64, 2>::select(
        &stream, &desc, PlanPreference::default()
    ).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        PoissonNllLossBackwardArgs {
            input: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
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
        let tol = expected[i].abs() * 8.0 * f64::EPSILON + 1e-11;
        assert!((got[i] - expected[i]).abs() <= tol);
    }
}

#[test]
#[ignore]
fn loss_poisson_nll_backward_f16_mean() {
    let (ctx, stream) = setup();
    let shape = [4i32, 5];
    let numel = 20usize;
    let host_x_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05 - 0.3).collect();
    let host_t_f32: Vec<f32> = (0..numel).map(|i| ((i % 4) as f32) + 0.2).collect();
    let host_x: Vec<f16> = host_x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_t: Vec<f16> = host_t_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let dy_host: Vec<f16> = [1.0f32].iter().map(|&v| f16::from_f32(v)).collect();
    let x64: Vec<f64> = host_x.iter().map(|&v| v.to_f32() as f64).collect();
    let t64: Vec<f64> = host_t.iter().map(|&v| v.to_f32() as f64).collect();
    let expected_f64 = host_poisson_bw_f64(&x64, &t64, 1.0, numel);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_di: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).unwrap();

    let desc = PoissonNllLossBackwardDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        log_input: true,
        element: ElementKind::F16,
    };
    let plan = PoissonNllLossBackwardPlan::<f16, 2>::select(
        &stream, &desc, PlanPreference::default()
    ).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        PoissonNllLossBackwardArgs {
            input: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
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
        let got_f32 = got[i].to_f32();
        let want = expected_f64[i] as f32;
        let tol = want.abs() * 8.0 * 9.77e-4_f32 + 5e-3;
        assert!((got_f32 - want).abs() <= tol, "f16 Poisson BW @{i}: got={} want={}", got_f32, want);
    }
}

#[test]
#[ignore]
fn loss_poisson_nll_backward_bf16_mean() {
    let (ctx, stream) = setup();
    let shape = [3i32, 6];
    let numel = 18usize;
    let host_x_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.06 - 0.4).collect();
    let host_t_f32: Vec<f32> = (0..numel).map(|i| ((i % 4) as f32) + 0.2).collect();
    let host_x: Vec<bf16> = host_x_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_t: Vec<bf16> = host_t_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let dy_host: Vec<bf16> = [1.0f32].iter().map(|&v| bf16::from_f32(v)).collect();
    let x64: Vec<f64> = host_x.iter().map(|&v| v.to_f32() as f64).collect();
    let t64: Vec<f64> = host_t.iter().map(|&v| v.to_f32() as f64).collect();
    let expected_f64 = host_poisson_bw_f64(&x64, &t64, 1.0, numel);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_di: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).unwrap();

    let desc = PoissonNllLossBackwardDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        log_input: true,
        element: ElementKind::Bf16,
    };
    let plan = PoissonNllLossBackwardPlan::<bf16, 2>::select(
        &stream, &desc, PlanPreference::default()
    ).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        PoissonNllLossBackwardArgs {
            input: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
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
        let got_f32 = got[i].to_f32();
        let want = expected_f64[i] as f32;
        let tol = want.abs() * 8.0 * 7.81e-3_f32 + 2e-2;
        assert!((got_f32 - want).abs() <= tol, "bf16 Poisson BW @{i}: got={} want={}", got_f32, want);
    }
}
