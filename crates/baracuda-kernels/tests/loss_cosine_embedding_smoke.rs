//! Real-GPU smoke test for `CosineEmbeddingLossPlan`. FW × 4 dtypes.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    CosineEmbeddingLossArgs, CosineEmbeddingLossDescriptor, CosineEmbeddingLossPlan, ElementKind,
    LossReduction, PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn host_cos_mean_f64(x1: &[f64], x2: &[f64], t: &[f64], n: usize, d: usize, margin: f64) -> f64 {
    let mut s = 0.0;
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
        let term = if t[r] > 0.0 {
            1.0 - cs
        } else {
            let h = cs - margin;
            if h > 0.0 { h } else { 0.0 }
        };
        s += term;
    }
    s / (n as f64)
}

#[test]
#[ignore]
fn loss_cosine_embedding_f32_mean() {
    let (ctx, stream) = setup();
    let n = 3usize;
    let d = 4usize;
    let margin = 0.5f32;
    let numel = n * d;
    let h_x1: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.1 - 0.5).collect();
    let h_x2: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.07 + 0.1).collect();
    let h_t: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let x1_64: Vec<f64> = h_x1.iter().map(|&v| v as f64).collect();
    let x2_64: Vec<f64> = h_x2.iter().map(|&v| v as f64).collect();
    let t_64: Vec<f64> = h_t.iter().map(|&v| v as f64).collect();
    let expected = host_cos_mean_f64(&x1_64, &x2_64, &t_64, n, d, margin as f64) as f32;
    let dev_x1 = DeviceBuffer::from_slice(&ctx, &h_x1).unwrap();
    let dev_x2 = DeviceBuffer::from_slice(&ctx, &h_x2).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &h_t).unwrap();
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, n * 4).unwrap();
    let desc = CosineEmbeddingLossDescriptor {
        n_rows: n as i32,
        d_extent: d as i32,
        reduction: LossReduction::Mean,
        margin,
        element: ElementKind::F32,
    };
    let plan =
        CosineEmbeddingLossPlan::<f32>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        CosineEmbeddingLossArgs {
            x1: TensorRef { data: dev_x1.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            x2: TensorRef { data: dev_x2.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            t: TensorRef { data: dev_t.as_slice(), shape: [n as i32, 1], stride: [1, 1] },
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1, 1], stride: [1, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = [0f32; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let tol = expected.abs() * 16.0 * f32::EPSILON + 1e-5;
    assert!((got[0] - expected).abs() <= tol, "f32 Cos: got={} want={}", got[0], expected);
}

#[test]
#[ignore]
fn loss_cosine_embedding_f64_mean() {
    let (ctx, stream) = setup();
    let n = 2usize;
    let d = 4usize;
    let margin = 0.5f32;
    let numel = n * d;
    let h_x1: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.1 - 0.5).collect();
    let h_x2: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.07 + 0.1).collect();
    let h_t: Vec<f64> = vec![1.0, -1.0];
    let expected = host_cos_mean_f64(&h_x1, &h_x2, &h_t, n, d, margin as f64);
    let dev_x1 = DeviceBuffer::from_slice(&ctx, &h_x1).unwrap();
    let dev_x2 = DeviceBuffer::from_slice(&ctx, &h_x2).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &h_t).unwrap();
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, n * 8).unwrap();
    let desc = CosineEmbeddingLossDescriptor {
        n_rows: n as i32,
        d_extent: d as i32,
        reduction: LossReduction::Mean,
        margin,
        element: ElementKind::F64,
    };
    let plan =
        CosineEmbeddingLossPlan::<f64>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        CosineEmbeddingLossArgs {
            x1: TensorRef { data: dev_x1.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            x2: TensorRef { data: dev_x2.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            t: TensorRef { data: dev_t.as_slice(), shape: [n as i32, 1], stride: [1, 1] },
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1, 1], stride: [1, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = [0f64; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let tol = expected.abs() * 16.0 * f64::EPSILON + 1e-12;
    assert!((got[0] - expected).abs() <= tol);
}

#[test]
#[ignore]
fn loss_cosine_embedding_f16_mean() {
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
    let x1_64: Vec<f64> = h_x1.iter().map(|&v| v.to_f32() as f64).collect();
    let x2_64: Vec<f64> = h_x2.iter().map(|&v| v.to_f32() as f64).collect();
    let t_64: Vec<f64> = h_t.iter().map(|&v| v.to_f32() as f64).collect();
    let expected = host_cos_mean_f64(&x1_64, &x2_64, &t_64, n, d, margin as f64) as f32;
    let dev_x1 = DeviceBuffer::from_slice(&ctx, &h_x1).unwrap();
    let dev_x2 = DeviceBuffer::from_slice(&ctx, &h_x2).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &h_t).unwrap();
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, n * 2).unwrap();
    let desc = CosineEmbeddingLossDescriptor {
        n_rows: n as i32,
        d_extent: d as i32,
        reduction: LossReduction::Mean,
        margin,
        element: ElementKind::F16,
    };
    let plan =
        CosineEmbeddingLossPlan::<f16>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        CosineEmbeddingLossArgs {
            x1: TensorRef { data: dev_x1.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            x2: TensorRef { data: dev_x2.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            t: TensorRef { data: dev_t.as_slice(), shape: [n as i32, 1], stride: [1, 1] },
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1, 1], stride: [1, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = [f16::ZERO; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let got_f32 = got[0].to_f32();
    // Cosine involves dot/sqrt/division — wider tolerance at half precision.
    let tol = expected.abs() * 32.0 * 9.77e-4_f32 + 1e-2;
    assert!((got_f32 - expected).abs() <= tol, "f16 Cos: got={} want={}", got_f32, expected);
}

#[test]
#[ignore]
fn loss_cosine_embedding_bf16_mean() {
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
    let x1_64: Vec<f64> = h_x1.iter().map(|&v| v.to_f32() as f64).collect();
    let x2_64: Vec<f64> = h_x2.iter().map(|&v| v.to_f32() as f64).collect();
    let t_64: Vec<f64> = h_t.iter().map(|&v| v.to_f32() as f64).collect();
    let expected = host_cos_mean_f64(&x1_64, &x2_64, &t_64, n, d, margin as f64) as f32;
    let dev_x1 = DeviceBuffer::from_slice(&ctx, &h_x1).unwrap();
    let dev_x2 = DeviceBuffer::from_slice(&ctx, &h_x2).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &h_t).unwrap();
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, n * 2).unwrap();
    let desc = CosineEmbeddingLossDescriptor {
        n_rows: n as i32,
        d_extent: d as i32,
        reduction: LossReduction::Mean,
        margin,
        element: ElementKind::Bf16,
    };
    let plan =
        CosineEmbeddingLossPlan::<bf16>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        CosineEmbeddingLossArgs {
            x1: TensorRef { data: dev_x1.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            x2: TensorRef { data: dev_x2.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            t: TensorRef { data: dev_t.as_slice(), shape: [n as i32, 1], stride: [1, 1] },
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1, 1], stride: [1, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = [bf16::ZERO; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let got_f32 = got[0].to_f32();
    let tol = expected.abs() * 32.0 * 7.81e-3_f32 + 5e-2;
    assert!((got_f32 - expected).abs() <= tol, "bf16 Cos: got={} want={}", got_f32, expected);
}
