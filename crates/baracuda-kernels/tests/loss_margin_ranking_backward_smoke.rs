//! Real-GPU smoke test for `MarginRankingLossBackwardPlan`. BW × 4 dtypes × Mean.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, LossReduction, MarginRankingLossBackwardArgs,
    MarginRankingLossBackwardDescriptor, MarginRankingLossBackwardPlan, PlanPreference, TensorMut,
    TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn host_mr_bw_f64(x1: &[f64], x2: &[f64], t: &[f64], margin: f64, dy: f64) -> (Vec<f64>, Vec<f64>) {
    let n = x1.len();
    let mut d1 = vec![0.0; n];
    let mut d2 = vec![0.0; n];
    let scale = dy / (n as f64);
    for i in 0..n {
        let loss = -t[i] * (x1[i] - x2[i]) + margin;
        if loss > 0.0 {
            d1[i] = -t[i] * scale;
            d2[i] = -d1[i];
        }
    }
    (d1, d2)
}

#[test]
#[ignore]
fn loss_margin_ranking_backward_f32_mean() {
    let (ctx, stream) = setup();
    let shape = [3i32, 5];
    let numel = 15usize;
    let margin = 0.5f32;
    let host_x1: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.1 - 0.7).collect();
    let host_x2: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.08).collect();
    let host_t: Vec<f32> = (0..numel)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let dy_host = [1.0f32];
    let (e1, e2) = host_mr_bw_f64(
        &host_x1.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &host_x2.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &host_t.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        margin as f64,
        1.0,
    );
    let dev_x1 = DeviceBuffer::from_slice(&ctx, &host_x1).unwrap();
    let dev_x2 = DeviceBuffer::from_slice(&ctx, &host_x2).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_dx1: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let mut dev_dx2: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let desc = MarginRankingLossBackwardDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        margin,
        element: ElementKind::F32,
    };
    let plan = MarginRankingLossBackwardPlan::<f32, 2>::select(
        &stream,
        &desc,
        PlanPreference::default(),
    )
    .unwrap();
    plan.run(
        &stream,
        Workspace::None,
        MarginRankingLossBackwardArgs {
            x1: TensorRef { data: dev_x1.as_slice(), shape, stride: contiguous_stride(shape) },
            x2: TensorRef { data: dev_x2.as_slice(), shape, stride: contiguous_stride(shape) },
            t: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            dy: TensorRef { data: dev_dy.as_slice(), shape: [1, 1], stride: [1, 1] },
            dx1: TensorMut { data: dev_dx1.as_slice_mut(), shape, stride: contiguous_stride(shape) },
            dx2: TensorMut { data: dev_dx2.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got1 = vec![0f32; numel];
    let mut got2 = vec![0f32; numel];
    dev_dx1.copy_to_host(&mut got1).unwrap();
    dev_dx2.copy_to_host(&mut got2).unwrap();
    for i in 0..numel {
        let want1 = e1[i] as f32;
        let want2 = e2[i] as f32;
        let tol = want1.abs().max(want2.abs()) * 8.0 * f32::EPSILON + 1e-6;
        assert!((got1[i] - want1).abs() <= tol, "f32 MR BW dx1 @{i}: got={} want={}", got1[i], want1);
        assert!((got2[i] - want2).abs() <= tol, "f32 MR BW dx2 @{i}: got={} want={}", got2[i], want2);
    }
}

#[test]
#[ignore]
fn loss_margin_ranking_backward_f64_mean() {
    let (ctx, stream) = setup();
    let shape = [2i32, 5];
    let numel = 10usize;
    let margin = 1.0f32;
    let host_x1: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.08 - 0.4).collect();
    let host_x2: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.05).collect();
    let host_t: Vec<f64> = (0..numel)
        .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
        .collect();
    let dy_host = [2.0f64];
    let (e1, e2) = host_mr_bw_f64(&host_x1, &host_x2, &host_t, margin as f64, 2.0);
    let dev_x1 = DeviceBuffer::from_slice(&ctx, &host_x1).unwrap();
    let dev_x2 = DeviceBuffer::from_slice(&ctx, &host_x2).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_dx1: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let mut dev_dx2: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let desc = MarginRankingLossBackwardDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        margin,
        element: ElementKind::F64,
    };
    let plan = MarginRankingLossBackwardPlan::<f64, 2>::select(
        &stream,
        &desc,
        PlanPreference::default(),
    )
    .unwrap();
    plan.run(
        &stream,
        Workspace::None,
        MarginRankingLossBackwardArgs {
            x1: TensorRef { data: dev_x1.as_slice(), shape, stride: contiguous_stride(shape) },
            x2: TensorRef { data: dev_x2.as_slice(), shape, stride: contiguous_stride(shape) },
            t: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            dy: TensorRef { data: dev_dy.as_slice(), shape: [1, 1], stride: [1, 1] },
            dx1: TensorMut { data: dev_dx1.as_slice_mut(), shape, stride: contiguous_stride(shape) },
            dx2: TensorMut { data: dev_dx2.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got1 = vec![0f64; numel];
    let mut got2 = vec![0f64; numel];
    dev_dx1.copy_to_host(&mut got1).unwrap();
    dev_dx2.copy_to_host(&mut got2).unwrap();
    for i in 0..numel {
        let tol = e1[i].abs().max(e2[i].abs()) * 8.0 * f64::EPSILON + 1e-12;
        assert!((got1[i] - e1[i]).abs() <= tol);
        assert!((got2[i] - e2[i]).abs() <= tol);
    }
}

#[test]
#[ignore]
fn loss_margin_ranking_backward_f16_mean() {
    let (ctx, stream) = setup();
    let shape = [3i32, 4];
    let numel = 12usize;
    let margin = 0.5f32;
    let h_x1_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05 - 0.3).collect();
    let h_x2_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.04).collect();
    let h_t_f32: Vec<f32> = (0..numel)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let host_x1: Vec<f16> = h_x1_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_x2: Vec<f16> = h_x2_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_t: Vec<f16> = h_t_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let dy_host = [f16::from_f32(1.0)];
    let x1_64: Vec<f64> = host_x1.iter().map(|&v| v.to_f32() as f64).collect();
    let x2_64: Vec<f64> = host_x2.iter().map(|&v| v.to_f32() as f64).collect();
    let t_64: Vec<f64> = host_t.iter().map(|&v| v.to_f32() as f64).collect();
    let (e1, e2) = host_mr_bw_f64(&x1_64, &x2_64, &t_64, margin as f64, 1.0);
    let dev_x1 = DeviceBuffer::from_slice(&ctx, &host_x1).unwrap();
    let dev_x2 = DeviceBuffer::from_slice(&ctx, &host_x2).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_dx1: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let mut dev_dx2: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let desc = MarginRankingLossBackwardDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        margin,
        element: ElementKind::F16,
    };
    let plan = MarginRankingLossBackwardPlan::<f16, 2>::select(
        &stream,
        &desc,
        PlanPreference::default(),
    )
    .unwrap();
    plan.run(
        &stream,
        Workspace::None,
        MarginRankingLossBackwardArgs {
            x1: TensorRef { data: dev_x1.as_slice(), shape, stride: contiguous_stride(shape) },
            x2: TensorRef { data: dev_x2.as_slice(), shape, stride: contiguous_stride(shape) },
            t: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            dy: TensorRef { data: dev_dy.as_slice(), shape: [1, 1], stride: [1, 1] },
            dx1: TensorMut { data: dev_dx1.as_slice_mut(), shape, stride: contiguous_stride(shape) },
            dx2: TensorMut { data: dev_dx2.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got1 = vec![f16::ZERO; numel];
    let mut got2 = vec![f16::ZERO; numel];
    dev_dx1.copy_to_host(&mut got1).unwrap();
    dev_dx2.copy_to_host(&mut got2).unwrap();
    for i in 0..numel {
        let want1 = e1[i] as f32;
        let want2 = e2[i] as f32;
        let g1 = got1[i].to_f32();
        let g2 = got2[i].to_f32();
        let tol = want1.abs().max(want2.abs()).max(1.0) * 16.0 * 9.77e-4_f32 + 5e-3;
        assert!((g1 - want1).abs() <= tol, "f16 MR BW dx1 @{i}: got={} want={}", g1, want1);
        assert!((g2 - want2).abs() <= tol, "f16 MR BW dx2 @{i}: got={} want={}", g2, want2);
    }
}

#[test]
#[ignore]
fn loss_margin_ranking_backward_bf16_mean() {
    let (ctx, stream) = setup();
    let shape = [3i32, 4];
    let numel = 12usize;
    let margin = 0.5f32;
    let h_x1_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05 - 0.3).collect();
    let h_x2_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.04).collect();
    let h_t_f32: Vec<f32> = (0..numel)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let host_x1: Vec<bf16> = h_x1_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_x2: Vec<bf16> = h_x2_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_t: Vec<bf16> = h_t_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let dy_host = [bf16::from_f32(1.0)];
    let x1_64: Vec<f64> = host_x1.iter().map(|&v| v.to_f32() as f64).collect();
    let x2_64: Vec<f64> = host_x2.iter().map(|&v| v.to_f32() as f64).collect();
    let t_64: Vec<f64> = host_t.iter().map(|&v| v.to_f32() as f64).collect();
    let (e1, e2) = host_mr_bw_f64(&x1_64, &x2_64, &t_64, margin as f64, 1.0);
    let dev_x1 = DeviceBuffer::from_slice(&ctx, &host_x1).unwrap();
    let dev_x2 = DeviceBuffer::from_slice(&ctx, &host_x2).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_dx1: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let mut dev_dx2: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let desc = MarginRankingLossBackwardDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        margin,
        element: ElementKind::Bf16,
    };
    let plan = MarginRankingLossBackwardPlan::<bf16, 2>::select(
        &stream,
        &desc,
        PlanPreference::default(),
    )
    .unwrap();
    plan.run(
        &stream,
        Workspace::None,
        MarginRankingLossBackwardArgs {
            x1: TensorRef { data: dev_x1.as_slice(), shape, stride: contiguous_stride(shape) },
            x2: TensorRef { data: dev_x2.as_slice(), shape, stride: contiguous_stride(shape) },
            t: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            dy: TensorRef { data: dev_dy.as_slice(), shape: [1, 1], stride: [1, 1] },
            dx1: TensorMut { data: dev_dx1.as_slice_mut(), shape, stride: contiguous_stride(shape) },
            dx2: TensorMut { data: dev_dx2.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got1 = vec![bf16::ZERO; numel];
    let mut got2 = vec![bf16::ZERO; numel];
    dev_dx1.copy_to_host(&mut got1).unwrap();
    dev_dx2.copy_to_host(&mut got2).unwrap();
    for i in 0..numel {
        let want1 = e1[i] as f32;
        let want2 = e2[i] as f32;
        let g1 = got1[i].to_f32();
        let g2 = got2[i].to_f32();
        let tol = want1.abs().max(want2.abs()).max(1.0) * 16.0 * 7.81e-3_f32 + 2e-2;
        assert!((g1 - want1).abs() <= tol, "bf16 MR BW dx1 @{i}: got={} want={}", g1, want1);
        assert!((g2 - want2).abs() <= tol, "bf16 MR BW dx2 @{i}: got={} want={}", g2, want2);
    }
}
