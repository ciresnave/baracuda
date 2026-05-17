//! Real-GPU smoke test for `MarginRankingLossPlan`. FW × 4 dtypes × Mean.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, LossReduction, MarginRankingLossArgs,
    MarginRankingLossDescriptor, MarginRankingLossPlan, PlanPreference, TensorMut, TensorRef,
    Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn host_mr_mean_f64(x1: &[f64], x2: &[f64], t: &[f64], margin: f64) -> f64 {
    let n = x1.len();
    let mut s = 0.0;
    for i in 0..n {
        let v = -t[i] * (x1[i] - x2[i]) + margin;
        s += if v > 0.0 { v } else { 0.0 };
    }
    s / (n as f64)
}

#[test]
#[ignore]
fn loss_margin_ranking_f32_mean() {
    let (ctx, stream) = setup();
    let shape = [3i32, 5];
    let numel = 15usize;
    let margin = 0.5f32;
    let host_x1: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.1 - 0.7).collect();
    let host_x2: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.08).collect();
    let host_t: Vec<f32> = (0..numel)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let expected = host_mr_mean_f64(
        &host_x1.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &host_x2.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &host_t.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        margin as f64,
    ) as f32;
    let dev_x1 = DeviceBuffer::from_slice(&ctx, &host_x1).unwrap();
    let dev_x2 = DeviceBuffer::from_slice(&ctx, &host_x2).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel * 4).unwrap();
    let desc = MarginRankingLossDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        margin,
        element: ElementKind::F32,
    };
    let plan =
        MarginRankingLossPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        MarginRankingLossArgs {
            x1: TensorRef { data: dev_x1.as_slice(), shape, stride: contiguous_stride(shape) },
            x2: TensorRef { data: dev_x2.as_slice(), shape, stride: contiguous_stride(shape) },
            t: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1, 1], stride: [1, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = [0f32; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let tol = expected.abs() * 8.0 * f32::EPSILON + 1e-6;
    assert!((got[0] - expected).abs() <= tol, "f32 MR: got={} want={}", got[0], expected);
}

#[test]
#[ignore]
fn loss_margin_ranking_f64_mean() {
    let (ctx, stream) = setup();
    let shape = [2i32, 5];
    let numel = 10usize;
    let margin = 1.0f32;
    let host_x1: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.08 - 0.4).collect();
    let host_x2: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.05).collect();
    let host_t: Vec<f64> = (0..numel)
        .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
        .collect();
    let expected = host_mr_mean_f64(&host_x1, &host_x2, &host_t, margin as f64);
    let dev_x1 = DeviceBuffer::from_slice(&ctx, &host_x1).unwrap();
    let dev_x2 = DeviceBuffer::from_slice(&ctx, &host_x2).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel * 8).unwrap();
    let desc = MarginRankingLossDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        margin,
        element: ElementKind::F64,
    };
    let plan =
        MarginRankingLossPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        MarginRankingLossArgs {
            x1: TensorRef { data: dev_x1.as_slice(), shape, stride: contiguous_stride(shape) },
            x2: TensorRef { data: dev_x2.as_slice(), shape, stride: contiguous_stride(shape) },
            t: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1, 1], stride: [1, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = [0f64; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let tol = expected.abs() * 8.0 * f64::EPSILON + 1e-12;
    assert!((got[0] - expected).abs() <= tol);
}

#[test]
#[ignore]
fn loss_margin_ranking_f16_mean() {
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
    let x1_64: Vec<f64> = host_x1.iter().map(|&v| v.to_f32() as f64).collect();
    let x2_64: Vec<f64> = host_x2.iter().map(|&v| v.to_f32() as f64).collect();
    let t_64: Vec<f64> = host_t.iter().map(|&v| v.to_f32() as f64).collect();
    let expected = host_mr_mean_f64(&x1_64, &x2_64, &t_64, margin as f64) as f32;
    let dev_x1 = DeviceBuffer::from_slice(&ctx, &host_x1).unwrap();
    let dev_x2 = DeviceBuffer::from_slice(&ctx, &host_x2).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel * 2).unwrap();
    let desc = MarginRankingLossDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        margin,
        element: ElementKind::F16,
    };
    let plan =
        MarginRankingLossPlan::<f16, 2>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        MarginRankingLossArgs {
            x1: TensorRef { data: dev_x1.as_slice(), shape, stride: contiguous_stride(shape) },
            x2: TensorRef { data: dev_x2.as_slice(), shape, stride: contiguous_stride(shape) },
            t: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1, 1], stride: [1, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = [f16::ZERO; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let got_f32 = got[0].to_f32();
    let tol = expected.abs() * 16.0 * 9.77e-4_f32 + 5e-3;
    assert!((got_f32 - expected).abs() <= tol);
}

#[test]
#[ignore]
fn loss_margin_ranking_bf16_mean() {
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
    let x1_64: Vec<f64> = host_x1.iter().map(|&v| v.to_f32() as f64).collect();
    let x2_64: Vec<f64> = host_x2.iter().map(|&v| v.to_f32() as f64).collect();
    let t_64: Vec<f64> = host_t.iter().map(|&v| v.to_f32() as f64).collect();
    let expected = host_mr_mean_f64(&x1_64, &x2_64, &t_64, margin as f64) as f32;
    let dev_x1 = DeviceBuffer::from_slice(&ctx, &host_x1).unwrap();
    let dev_x2 = DeviceBuffer::from_slice(&ctx, &host_x2).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &host_t).unwrap();
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel * 2).unwrap();
    let desc = MarginRankingLossDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        margin,
        element: ElementKind::Bf16,
    };
    let plan =
        MarginRankingLossPlan::<bf16, 2>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        MarginRankingLossArgs {
            x1: TensorRef { data: dev_x1.as_slice(), shape, stride: contiguous_stride(shape) },
            x2: TensorRef { data: dev_x2.as_slice(), shape, stride: contiguous_stride(shape) },
            t: TensorRef { data: dev_t.as_slice(), shape, stride: contiguous_stride(shape) },
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1, 1], stride: [1, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = [bf16::ZERO; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let got_f32 = got[0].to_f32();
    let tol = expected.abs() * 16.0 * 7.81e-3_f32 + 2e-2;
    assert!((got_f32 - expected).abs() <= tol);
}
