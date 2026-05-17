//! Real-GPU smoke test for `CosineEmbeddingLossBackwardPlan`. BW × 4 dtypes.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    CosineEmbeddingLossBackwardArgs, CosineEmbeddingLossBackwardDescriptor,
    CosineEmbeddingLossBackwardPlan, ElementKind, LossReduction, PlanPreference, TensorMut,
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

fn host_cos_bw_f64(
    x1: &[f64], x2: &[f64], t: &[f64],
    n: usize, d: usize, margin: f64, dy: f64,
) -> (Vec<f64>, Vec<f64>) {
    let scale = dy / (n as f64);
    let mut d1 = vec![0.0; n * d];
    let mut d2 = vec![0.0; n * d];
    for r in 0..n {
        let mut dot = 0.0;
        let mut n1 = 0.0;
        let mut n2 = 0.0;
        for j in 0..d {
            dot += x1[r * d + j] * x2[r * d + j];
            n1 += x1[r * d + j].powi(2);
            n2 += x2[r * d + j].powi(2);
        }
        let denom = (n1.sqrt() * n2.sqrt()).max(1e-30);
        let cs = dot / denom;
        let dcs = if t[r] > 0.0 { -1.0 } else if cs > margin { 1.0 } else { 0.0 };
        let coef = dcs * scale;
        let inv_n1n2 = 1.0 / denom;
        let inv_n1sq = if n1 > 1e-60 { 1.0 / n1 } else { 0.0 };
        let inv_n2sq = if n2 > 1e-60 { 1.0 / n2 } else { 0.0 };
        for j in 0..d {
            d1[r * d + j] = coef * (x2[r * d + j] * inv_n1n2 - cs * x1[r * d + j] * inv_n1sq);
            d2[r * d + j] = coef * (x1[r * d + j] * inv_n1n2 - cs * x2[r * d + j] * inv_n2sq);
        }
    }
    (d1, d2)
}

#[test]
#[ignore]
fn loss_cosine_embedding_backward_f32_mean() {
    let (ctx, stream) = setup();
    let n = 3usize;
    let d = 4usize;
    let margin = 0.5f32;
    let numel = n * d;
    let h_x1: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.1 - 0.5).collect();
    let h_x2: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.07 + 0.1).collect();
    let h_t: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let dy_host = [1.0f32];
    let x1_64: Vec<f64> = h_x1.iter().map(|&v| v as f64).collect();
    let x2_64: Vec<f64> = h_x2.iter().map(|&v| v as f64).collect();
    let t_64: Vec<f64> = h_t.iter().map(|&v| v as f64).collect();
    let (e1, e2) = host_cos_bw_f64(&x1_64, &x2_64, &t_64, n, d, margin as f64, 1.0);
    let dev_x1 = DeviceBuffer::from_slice(&ctx, &h_x1).unwrap();
    let dev_x2 = DeviceBuffer::from_slice(&ctx, &h_x2).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &h_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_dx1: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let mut dev_dx2: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let desc = CosineEmbeddingLossBackwardDescriptor {
        n_rows: n as i32,
        d_extent: d as i32,
        reduction: LossReduction::Mean,
        margin,
        element: ElementKind::F32,
    };
    let plan = CosineEmbeddingLossBackwardPlan::<f32>::select(
        &stream,
        &desc,
        PlanPreference::default(),
    )
    .unwrap();
    plan.run(
        &stream,
        Workspace::None,
        CosineEmbeddingLossBackwardArgs {
            x1: TensorRef { data: dev_x1.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            x2: TensorRef { data: dev_x2.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            t: TensorRef { data: dev_t.as_slice(), shape: [n as i32, 1], stride: [1, 1] },
            dy: TensorRef { data: dev_dy.as_slice(), shape: [1, 1], stride: [1, 1] },
            dx1: TensorMut { data: dev_dx1.as_slice_mut(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            dx2: TensorMut { data: dev_dx2.as_slice_mut(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got1 = vec![0f32; numel];
    let mut got2 = vec![0f32; numel];
    dev_dx1.copy_to_host(&mut got1).unwrap();
    dev_dx2.copy_to_host(&mut got2).unwrap();
    for i in 0..numel {
        let w1 = e1[i] as f32;
        let w2 = e2[i] as f32;
        let tol = (w1.abs().max(w2.abs())).max(1.0) * 32.0 * f32::EPSILON + 1e-5;
        assert!((got1[i] - w1).abs() <= tol, "f32 Cos BW dx1 @{i}: got={} want={}", got1[i], w1);
        assert!((got2[i] - w2).abs() <= tol, "f32 Cos BW dx2 @{i}: got={} want={}", got2[i], w2);
    }
}

#[test]
#[ignore]
fn loss_cosine_embedding_backward_f64_mean() {
    let (ctx, stream) = setup();
    let n = 2usize;
    let d = 4usize;
    let margin = 0.5f32;
    let numel = n * d;
    let h_x1: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.1 - 0.5).collect();
    let h_x2: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.07 + 0.1).collect();
    let h_t: Vec<f64> = vec![1.0, -1.0];
    let dy_host = [2.0f64];
    let (e1, e2) = host_cos_bw_f64(&h_x1, &h_x2, &h_t, n, d, margin as f64, 2.0);
    let dev_x1 = DeviceBuffer::from_slice(&ctx, &h_x1).unwrap();
    let dev_x2 = DeviceBuffer::from_slice(&ctx, &h_x2).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &h_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_dx1: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let mut dev_dx2: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let desc = CosineEmbeddingLossBackwardDescriptor {
        n_rows: n as i32,
        d_extent: d as i32,
        reduction: LossReduction::Mean,
        margin,
        element: ElementKind::F64,
    };
    let plan = CosineEmbeddingLossBackwardPlan::<f64>::select(
        &stream,
        &desc,
        PlanPreference::default(),
    )
    .unwrap();
    plan.run(
        &stream,
        Workspace::None,
        CosineEmbeddingLossBackwardArgs {
            x1: TensorRef { data: dev_x1.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            x2: TensorRef { data: dev_x2.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            t: TensorRef { data: dev_t.as_slice(), shape: [n as i32, 1], stride: [1, 1] },
            dy: TensorRef { data: dev_dy.as_slice(), shape: [1, 1], stride: [1, 1] },
            dx1: TensorMut { data: dev_dx1.as_slice_mut(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            dx2: TensorMut { data: dev_dx2.as_slice_mut(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got1 = vec![0f64; numel];
    let mut got2 = vec![0f64; numel];
    dev_dx1.copy_to_host(&mut got1).unwrap();
    dev_dx2.copy_to_host(&mut got2).unwrap();
    for i in 0..numel {
        let tol = e1[i].abs().max(e2[i].abs()).max(1.0) * 32.0 * f64::EPSILON + 1e-11;
        assert!((got1[i] - e1[i]).abs() <= tol);
        assert!((got2[i] - e2[i]).abs() <= tol);
    }
}

#[test]
#[ignore]
fn loss_cosine_embedding_backward_f16_mean() {
    let (ctx, stream) = setup();
    let n = 3usize;
    let d = 4usize;
    let margin = 0.5f32;
    let numel = n * d;
    let x1_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05 + 0.05).collect();
    let x2_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.04 + 0.1).collect();
    let t_f32: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let h_x1: Vec<f16> = x1_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let h_x2: Vec<f16> = x2_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let h_t: Vec<f16> = t_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let dy_host = [f16::from_f32(1.0)];
    let x1_64: Vec<f64> = h_x1.iter().map(|&v| v.to_f32() as f64).collect();
    let x2_64: Vec<f64> = h_x2.iter().map(|&v| v.to_f32() as f64).collect();
    let t_64: Vec<f64> = h_t.iter().map(|&v| v.to_f32() as f64).collect();
    let (e1, e2) = host_cos_bw_f64(&x1_64, &x2_64, &t_64, n, d, margin as f64, 1.0);
    let dev_x1 = DeviceBuffer::from_slice(&ctx, &h_x1).unwrap();
    let dev_x2 = DeviceBuffer::from_slice(&ctx, &h_x2).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &h_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_dx1: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let mut dev_dx2: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let desc = CosineEmbeddingLossBackwardDescriptor {
        n_rows: n as i32,
        d_extent: d as i32,
        reduction: LossReduction::Mean,
        margin,
        element: ElementKind::F16,
    };
    let plan = CosineEmbeddingLossBackwardPlan::<f16>::select(
        &stream,
        &desc,
        PlanPreference::default(),
    )
    .unwrap();
    plan.run(
        &stream,
        Workspace::None,
        CosineEmbeddingLossBackwardArgs {
            x1: TensorRef { data: dev_x1.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            x2: TensorRef { data: dev_x2.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            t: TensorRef { data: dev_t.as_slice(), shape: [n as i32, 1], stride: [1, 1] },
            dy: TensorRef { data: dev_dy.as_slice(), shape: [1, 1], stride: [1, 1] },
            dx1: TensorMut { data: dev_dx1.as_slice_mut(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            dx2: TensorMut { data: dev_dx2.as_slice_mut(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got1 = vec![f16::ZERO; numel];
    let mut got2 = vec![f16::ZERO; numel];
    dev_dx1.copy_to_host(&mut got1).unwrap();
    dev_dx2.copy_to_host(&mut got2).unwrap();
    for i in 0..numel {
        let w1 = e1[i] as f32;
        let w2 = e2[i] as f32;
        let g1 = got1[i].to_f32();
        let g2 = got2[i].to_f32();
        let tol = (w1.abs().max(w2.abs())).max(1.0) * 32.0 * 9.77e-4_f32 + 1e-2;
        assert!((g1 - w1).abs() <= tol, "f16 Cos BW dx1 @{i}: got={} want={}", g1, w1);
        assert!((g2 - w2).abs() <= tol, "f16 Cos BW dx2 @{i}: got={} want={}", g2, w2);
    }
}

#[test]
#[ignore]
fn loss_cosine_embedding_backward_bf16_mean() {
    let (ctx, stream) = setup();
    let n = 3usize;
    let d = 4usize;
    let margin = 0.5f32;
    let numel = n * d;
    let x1_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05 + 0.05).collect();
    let x2_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.04 + 0.1).collect();
    let t_f32: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let h_x1: Vec<bf16> = x1_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let h_x2: Vec<bf16> = x2_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let h_t: Vec<bf16> = t_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let dy_host = [bf16::from_f32(1.0)];
    let x1_64: Vec<f64> = h_x1.iter().map(|&v| v.to_f32() as f64).collect();
    let x2_64: Vec<f64> = h_x2.iter().map(|&v| v.to_f32() as f64).collect();
    let t_64: Vec<f64> = h_t.iter().map(|&v| v.to_f32() as f64).collect();
    let (e1, e2) = host_cos_bw_f64(&x1_64, &x2_64, &t_64, n, d, margin as f64, 1.0);
    let dev_x1 = DeviceBuffer::from_slice(&ctx, &h_x1).unwrap();
    let dev_x2 = DeviceBuffer::from_slice(&ctx, &h_x2).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &h_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_dx1: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let mut dev_dx2: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let desc = CosineEmbeddingLossBackwardDescriptor {
        n_rows: n as i32,
        d_extent: d as i32,
        reduction: LossReduction::Mean,
        margin,
        element: ElementKind::Bf16,
    };
    let plan = CosineEmbeddingLossBackwardPlan::<bf16>::select(
        &stream,
        &desc,
        PlanPreference::default(),
    )
    .unwrap();
    plan.run(
        &stream,
        Workspace::None,
        CosineEmbeddingLossBackwardArgs {
            x1: TensorRef { data: dev_x1.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            x2: TensorRef { data: dev_x2.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            t: TensorRef { data: dev_t.as_slice(), shape: [n as i32, 1], stride: [1, 1] },
            dy: TensorRef { data: dev_dy.as_slice(), shape: [1, 1], stride: [1, 1] },
            dx1: TensorMut { data: dev_dx1.as_slice_mut(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            dx2: TensorMut { data: dev_dx2.as_slice_mut(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got1 = vec![bf16::ZERO; numel];
    let mut got2 = vec![bf16::ZERO; numel];
    dev_dx1.copy_to_host(&mut got1).unwrap();
    dev_dx2.copy_to_host(&mut got2).unwrap();
    for i in 0..numel {
        let w1 = e1[i] as f32;
        let w2 = e2[i] as f32;
        let g1 = got1[i].to_f32();
        let g2 = got2[i].to_f32();
        let tol = (w1.abs().max(w2.abs())).max(1.0) * 32.0 * 7.81e-3_f32 + 5e-2;
        assert!((g1 - w1).abs() <= tol, "bf16 Cos BW dx1 @{i}: got={} want={}", g1, w1);
        assert!((g2 - w2).abs() <= tol, "bf16 Cos BW dx2 @{i}: got={} want={}", g2, w2);
    }
}
