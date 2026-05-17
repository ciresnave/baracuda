//! Real-GPU smoke test for `HingeEmbeddingLossBackwardPlan`. BW × 4 dtypes.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, HingeEmbeddingLossBackwardArgs,
    HingeEmbeddingLossBackwardDescriptor, HingeEmbeddingLossBackwardPlan, LossReduction,
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

fn host_he_bw_f64(input: &[f64], t: &[i64], margin: f64, dy: f64) -> Vec<f64> {
    let n = input.len();
    let sc = dy / (n as f64);
    let mut g = vec![0.0; n];
    for i in 0..n {
        g[i] = if t[i] == 1 {
            sc
        } else if margin > input[i] {
            -sc
        } else {
            0.0
        };
    }
    g
}

#[test]
#[ignore]
fn loss_hinge_embedding_backward_f32_mean() {
    let (ctx, stream) = setup();
    let shape = [3i32, 5];
    let numel = 15usize;
    let margin = 1.0f32;
    let host_in: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.1 - 0.7).collect();
    let host_t: Vec<i64> = (0..numel)
        .map(|i| if i % 2 == 0 { 1 } else { -1 })
        .collect();
    let dy_host = [1.0f32];
    let expected: Vec<f32> = host_he_bw_f64(
        &host_in.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &host_t,
        margin as f64,
        1.0,
    )
    .into_iter()
    .map(|v| v as f32)
    .collect();
    let dev_in = DeviceBuffer::from_slice(&ctx, &host_in).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let desc = HingeEmbeddingLossBackwardDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        margin,
        element: ElementKind::F32,
    };
    let plan = HingeEmbeddingLossBackwardPlan::<f32, 2>::select(
        &stream,
        &desc,
        PlanPreference::default(),
    )
    .unwrap();
    plan.run(
        &stream,
        Workspace::None,
        HingeEmbeddingLossBackwardArgs {
            input: TensorRef { data: dev_in.as_slice(), shape, stride: contiguous_stride(shape) },
            target: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            dy: TensorRef { data: dev_dy.as_slice(), shape: [1, 1], stride: [1, 1] },
            dinput: TensorMut {
                data: dev_dx.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = vec![0f32; numel];
    dev_dx.copy_to_host(&mut got).unwrap();
    for i in 0..numel {
        let tol = expected[i].abs() * 8.0 * f32::EPSILON + 1e-6;
        assert!((got[i] - expected[i]).abs() <= tol);
    }
}

#[test]
#[ignore]
fn loss_hinge_embedding_backward_f64_mean() {
    let (ctx, stream) = setup();
    let shape = [2i32, 5];
    let numel = 10usize;
    let margin = 0.5f32;
    let host_in: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.08 - 0.3).collect();
    let host_t: Vec<i64> = (0..numel)
        .map(|i| if i % 3 == 0 { -1 } else { 1 })
        .collect();
    let dy_host = [2.0f64];
    let expected = host_he_bw_f64(&host_in, &host_t, margin as f64, 2.0);
    let dev_in = DeviceBuffer::from_slice(&ctx, &host_in).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_dx: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let desc = HingeEmbeddingLossBackwardDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        margin,
        element: ElementKind::F64,
    };
    let plan = HingeEmbeddingLossBackwardPlan::<f64, 2>::select(
        &stream,
        &desc,
        PlanPreference::default(),
    )
    .unwrap();
    plan.run(
        &stream,
        Workspace::None,
        HingeEmbeddingLossBackwardArgs {
            input: TensorRef { data: dev_in.as_slice(), shape, stride: contiguous_stride(shape) },
            target: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            dy: TensorRef { data: dev_dy.as_slice(), shape: [1, 1], stride: [1, 1] },
            dinput: TensorMut {
                data: dev_dx.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = vec![0f64; numel];
    dev_dx.copy_to_host(&mut got).unwrap();
    for i in 0..numel {
        let tol = expected[i].abs() * 8.0 * f64::EPSILON + 1e-12;
        assert!((got[i] - expected[i]).abs() <= tol);
    }
}

#[test]
#[ignore]
fn loss_hinge_embedding_backward_f16_mean() {
    let (ctx, stream) = setup();
    let shape = [3i32, 4];
    let numel = 12usize;
    let margin = 1.0f32;
    let h_in_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05 - 0.3).collect();
    let host_in: Vec<f16> = h_in_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_t: Vec<i64> = (0..numel)
        .map(|i| if i % 2 == 0 { 1 } else { -1 })
        .collect();
    let dy_host = [f16::from_f32(1.0)];
    let in64: Vec<f64> = host_in.iter().map(|&v| v.to_f32() as f64).collect();
    let expected_f64 = host_he_bw_f64(&in64, &host_t, margin as f64, 1.0);
    let dev_in = DeviceBuffer::from_slice(&ctx, &host_in).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_dx: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let desc = HingeEmbeddingLossBackwardDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        margin,
        element: ElementKind::F16,
    };
    let plan = HingeEmbeddingLossBackwardPlan::<f16, 2>::select(
        &stream,
        &desc,
        PlanPreference::default(),
    )
    .unwrap();
    plan.run(
        &stream,
        Workspace::None,
        HingeEmbeddingLossBackwardArgs {
            input: TensorRef { data: dev_in.as_slice(), shape, stride: contiguous_stride(shape) },
            target: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            dy: TensorRef { data: dev_dy.as_slice(), shape: [1, 1], stride: [1, 1] },
            dinput: TensorMut {
                data: dev_dx.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = vec![f16::ZERO; numel];
    dev_dx.copy_to_host(&mut got).unwrap();
    for i in 0..numel {
        let want = expected_f64[i] as f32;
        let g = got[i].to_f32();
        let tol = want.abs().max(1.0) * 16.0 * 9.77e-4_f32 + 5e-3;
        assert!((g - want).abs() <= tol, "f16 HE BW @{i}: got={} want={}", g, want);
    }
}

#[test]
#[ignore]
fn loss_hinge_embedding_backward_bf16_mean() {
    let (ctx, stream) = setup();
    let shape = [3i32, 4];
    let numel = 12usize;
    let margin = 1.0f32;
    let h_in_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05 - 0.3).collect();
    let host_in: Vec<bf16> = h_in_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_t: Vec<i64> = (0..numel)
        .map(|i| if i % 2 == 0 { 1 } else { -1 })
        .collect();
    let dy_host = [bf16::from_f32(1.0)];
    let in64: Vec<f64> = host_in.iter().map(|&v| v.to_f32() as f64).collect();
    let expected_f64 = host_he_bw_f64(&in64, &host_t, margin as f64, 1.0);
    let dev_in = DeviceBuffer::from_slice(&ctx, &host_in).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_dx: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let desc = HingeEmbeddingLossBackwardDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        margin,
        element: ElementKind::Bf16,
    };
    let plan = HingeEmbeddingLossBackwardPlan::<bf16, 2>::select(
        &stream,
        &desc,
        PlanPreference::default(),
    )
    .unwrap();
    plan.run(
        &stream,
        Workspace::None,
        HingeEmbeddingLossBackwardArgs {
            input: TensorRef { data: dev_in.as_slice(), shape, stride: contiguous_stride(shape) },
            target: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            dy: TensorRef { data: dev_dy.as_slice(), shape: [1, 1], stride: [1, 1] },
            dinput: TensorMut {
                data: dev_dx.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = vec![bf16::ZERO; numel];
    dev_dx.copy_to_host(&mut got).unwrap();
    for i in 0..numel {
        let want = expected_f64[i] as f32;
        let g = got[i].to_f32();
        let tol = want.abs().max(1.0) * 16.0 * 7.81e-3_f32 + 2e-2;
        assert!((g - want).abs() <= tol, "bf16 HE BW @{i}: got={} want={}", g, want);
    }
}
