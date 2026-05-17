//! Real-GPU smoke test for `GaussianNllLossBackwardPlan`. BW × 4 dtypes × Mean.
//!
//! `dinput = (input - target) / max(var, eps) · dy / N`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, GaussianNllLossBackwardArgs,
    GaussianNllLossBackwardDescriptor, GaussianNllLossBackwardPlan, LossReduction,
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

fn host_gauss_bw_f64(
    input: &[f64],
    target: &[f64],
    var: &[f64],
    dy: f64,
    n: usize,
    eps: f64,
) -> Vec<f64> {
    let scale = dy / (n as f64);
    (0..input.len())
        .map(|i| {
            let ve = if var[i] > eps { var[i] } else { eps };
            (input[i] - target[i]) / ve * scale
        })
        .collect()
}

#[test]
#[ignore]
fn loss_gaussian_nll_backward_f32_mean() {
    let (ctx, stream) = setup();
    let shape = [3i32, 5];
    let numel = 15usize;
    let eps = 1e-6f32;
    let host_x: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.1 - 0.5).collect();
    let host_t: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.08).collect();
    let host_v: Vec<f32> = (0..numel).map(|i| 0.5 + (i as f32) * 0.05).collect();
    let dy_host = [1.0f32];
    let expected: Vec<f32> = host_gauss_bw_f64(
        &host_x.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &host_t.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &host_v.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        1.0, numel, eps as f64,
    )
    .into_iter()
    .map(|v| v as f32)
    .collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_v = DeviceBuffer::from_slice(&ctx, &host_v).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_di: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).unwrap();

    let desc = GaussianNllLossBackwardDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        eps,
        element: ElementKind::F32,
    };
    let plan = GaussianNllLossBackwardPlan::<f32, 2>::select(
        &stream, &desc, PlanPreference::default()
    ).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        GaussianNllLossBackwardArgs {
            input: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            target: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            var: TensorRef { data: dev_v.as_slice(), shape, stride: contiguous_stride(shape) },
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
        assert!((got[i] - expected[i]).abs() <= tol, "f32 Gauss BW @{i}: got={} want={}",
            got[i], expected[i]);
    }
}

#[test]
#[ignore]
fn loss_gaussian_nll_backward_f64_mean() {
    let (ctx, stream) = setup();
    let shape = [3i32, 4];
    let numel = 12usize;
    let eps = 1e-6f32;
    let host_x: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.08 - 0.3).collect();
    let host_t: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.07).collect();
    let host_v: Vec<f64> = (0..numel).map(|i| 0.5 + (i as f64) * 0.04).collect();
    let dy_host = [2.0f64];
    let expected = host_gauss_bw_f64(&host_x, &host_t, &host_v, 2.0, numel, eps as f64);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_v = DeviceBuffer::from_slice(&ctx, &host_v).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_di: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).unwrap();

    let desc = GaussianNllLossBackwardDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        eps,
        element: ElementKind::F64,
    };
    let plan = GaussianNllLossBackwardPlan::<f64, 2>::select(
        &stream, &desc, PlanPreference::default()
    ).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        GaussianNllLossBackwardArgs {
            input: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            target: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            var: TensorRef { data: dev_v.as_slice(), shape, stride: contiguous_stride(shape) },
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
fn loss_gaussian_nll_backward_f16_mean() {
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
    let dy_host: Vec<f16> = [1.0f32].iter().map(|&v| f16::from_f32(v)).collect();
    let x64: Vec<f64> = host_x.iter().map(|&v| v.to_f32() as f64).collect();
    let t64: Vec<f64> = host_t.iter().map(|&v| v.to_f32() as f64).collect();
    let v64: Vec<f64> = host_v.iter().map(|&v| v.to_f32() as f64).collect();
    let expected_f64 = host_gauss_bw_f64(&x64, &t64, &v64, 1.0, numel, eps as f64);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_v = DeviceBuffer::from_slice(&ctx, &host_v).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_di: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).unwrap();

    let desc = GaussianNllLossBackwardDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        eps,
        element: ElementKind::F16,
    };
    let plan = GaussianNllLossBackwardPlan::<f16, 2>::select(
        &stream, &desc, PlanPreference::default()
    ).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        GaussianNllLossBackwardArgs {
            input: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            target: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            var: TensorRef { data: dev_v.as_slice(), shape, stride: contiguous_stride(shape) },
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
        assert!((got_f32 - want).abs() <= tol, "f16 Gauss BW @{i}: got={} want={}", got_f32, want);
    }
}

#[test]
#[ignore]
fn loss_gaussian_nll_backward_bf16_mean() {
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
    let dy_host: Vec<bf16> = [1.0f32].iter().map(|&v| bf16::from_f32(v)).collect();
    let x64: Vec<f64> = host_x.iter().map(|&v| v.to_f32() as f64).collect();
    let t64: Vec<f64> = host_t.iter().map(|&v| v.to_f32() as f64).collect();
    let v64: Vec<f64> = host_v.iter().map(|&v| v.to_f32() as f64).collect();
    let expected_f64 = host_gauss_bw_f64(&x64, &t64, &v64, 1.0, numel, eps as f64);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_v = DeviceBuffer::from_slice(&ctx, &host_v).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_di: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).unwrap();

    let desc = GaussianNllLossBackwardDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        eps,
        element: ElementKind::Bf16,
    };
    let plan = GaussianNllLossBackwardPlan::<bf16, 2>::select(
        &stream, &desc, PlanPreference::default()
    ).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        GaussianNllLossBackwardArgs {
            input: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            target: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            var: TensorRef { data: dev_v.as_slice(), shape, stride: contiguous_stride(shape) },
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
        assert!((got_f32 - want).abs() <= tol, "bf16 Gauss BW @{i}: got={} want={}", got_f32, want);
    }
}
