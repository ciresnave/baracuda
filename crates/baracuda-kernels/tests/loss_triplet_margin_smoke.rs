//! Real-GPU smoke test for `TripletMarginLossPlan`. FW × 4 dtypes.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    ElementKind, LossReduction, PlanPreference, TensorMut, TensorRef, TripletMarginLossArgs,
    TripletMarginLossDescriptor, TripletMarginLossPlan, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn host_triplet_mean_f64(a: &[f64], p: &[f64], n: &[f64], rows: usize, d: usize, margin: f64, pn: f64) -> f64 {
    let mut s = 0.0;
    for r in 0..rows {
        let mut sp = 0.0;
        let mut sn = 0.0;
        for j in 0..d {
            sp += (a[r * d + j] - p[r * d + j]).abs().powf(pn);
            sn += (a[r * d + j] - n[r * d + j]).abs().powf(pn);
        }
        let pd = sp.powf(1.0 / pn);
        let nd = sn.powf(1.0 / pn);
        let v = pd - nd + margin;
        s += if v > 0.0 { v } else { 0.0 };
    }
    s / (rows as f64)
}

#[test]
#[ignore]
fn loss_triplet_margin_f32_mean() {
    let (ctx, stream) = setup();
    let n = 3usize;
    let d = 4usize;
    let margin = 1.0f32;
    let p_norm = 2.0f32;
    let numel = n * d;
    let h_a: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05).collect();
    let h_p: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05 + 0.02).collect();
    let h_n: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05 - 0.3).collect();
    let expected = host_triplet_mean_f64(
        &h_a.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &h_p.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &h_n.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        n, d, margin as f64, p_norm as f64,
    ) as f32;
    let dev_a = DeviceBuffer::from_slice(&ctx, &h_a).unwrap();
    let dev_p = DeviceBuffer::from_slice(&ctx, &h_p).unwrap();
    let dev_n = DeviceBuffer::from_slice(&ctx, &h_n).unwrap();
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, n * 4).unwrap();
    let desc = TripletMarginLossDescriptor {
        n_rows: n as i32,
        d_extent: d as i32,
        reduction: LossReduction::Mean,
        margin,
        p_norm,
        element: ElementKind::F32,
    };
    let plan =
        TripletMarginLossPlan::<f32>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        TripletMarginLossArgs {
            anchor: TensorRef { data: dev_a.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            positive: TensorRef { data: dev_p.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            negative: TensorRef { data: dev_n.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1, 1], stride: [1, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = [0f32; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let tol = expected.abs() * 16.0 * f32::EPSILON + 1e-5;
    assert!((got[0] - expected).abs() <= tol, "f32 Triplet: got={} want={}", got[0], expected);
}

#[test]
#[ignore]
fn loss_triplet_margin_f64_mean() {
    let (ctx, stream) = setup();
    let n = 2usize;
    let d = 4usize;
    let margin = 1.0f32;
    let p_norm = 2.0f32;
    let numel = n * d;
    let h_a: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.05).collect();
    let h_p: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.05 + 0.02).collect();
    let h_n: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.05 - 0.3).collect();
    let expected = host_triplet_mean_f64(&h_a, &h_p, &h_n, n, d, margin as f64, p_norm as f64);
    let dev_a = DeviceBuffer::from_slice(&ctx, &h_a).unwrap();
    let dev_p = DeviceBuffer::from_slice(&ctx, &h_p).unwrap();
    let dev_n = DeviceBuffer::from_slice(&ctx, &h_n).unwrap();
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, n * 8).unwrap();
    let desc = TripletMarginLossDescriptor {
        n_rows: n as i32,
        d_extent: d as i32,
        reduction: LossReduction::Mean,
        margin,
        p_norm,
        element: ElementKind::F64,
    };
    let plan =
        TripletMarginLossPlan::<f64>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        TripletMarginLossArgs {
            anchor: TensorRef { data: dev_a.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            positive: TensorRef { data: dev_p.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            negative: TensorRef { data: dev_n.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
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
fn loss_triplet_margin_f16_mean() {
    let (ctx, stream) = setup();
    let n = 3usize;
    let d = 4usize;
    let margin = 1.0f32;
    let p_norm = 2.0f32;
    let numel = n * d;
    let h_a_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05).collect();
    let h_p_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05 + 0.02).collect();
    let h_n_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05 - 0.3).collect();
    let h_a: Vec<f16> = h_a_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let h_p: Vec<f16> = h_p_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let h_n: Vec<f16> = h_n_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let a64: Vec<f64> = h_a.iter().map(|&v| v.to_f32() as f64).collect();
    let p64: Vec<f64> = h_p.iter().map(|&v| v.to_f32() as f64).collect();
    let n64: Vec<f64> = h_n.iter().map(|&v| v.to_f32() as f64).collect();
    let expected = host_triplet_mean_f64(&a64, &p64, &n64, n, d, margin as f64, p_norm as f64) as f32;
    let dev_a = DeviceBuffer::from_slice(&ctx, &h_a).unwrap();
    let dev_p = DeviceBuffer::from_slice(&ctx, &h_p).unwrap();
    let dev_n = DeviceBuffer::from_slice(&ctx, &h_n).unwrap();
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, n * 2).unwrap();
    let desc = TripletMarginLossDescriptor {
        n_rows: n as i32,
        d_extent: d as i32,
        reduction: LossReduction::Mean,
        margin,
        p_norm,
        element: ElementKind::F16,
    };
    let plan =
        TripletMarginLossPlan::<f16>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        TripletMarginLossArgs {
            anchor: TensorRef { data: dev_a.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            positive: TensorRef { data: dev_p.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            negative: TensorRef { data: dev_n.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1, 1], stride: [1, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = [f16::ZERO; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let got_f32 = got[0].to_f32();
    // Triplet involves sqrt/pow — wider tolerance at half precision.
    let tol = expected.abs() * 32.0 * 9.77e-4_f32 + 1e-2;
    assert!((got_f32 - expected).abs() <= tol, "f16 Triplet: got={} want={}", got_f32, expected);
}

#[test]
#[ignore]
fn loss_triplet_margin_bf16_mean() {
    let (ctx, stream) = setup();
    let n = 3usize;
    let d = 4usize;
    let margin = 1.0f32;
    let p_norm = 2.0f32;
    let numel = n * d;
    let h_a_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05).collect();
    let h_p_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05 + 0.02).collect();
    let h_n_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05 - 0.3).collect();
    let h_a: Vec<bf16> = h_a_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let h_p: Vec<bf16> = h_p_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let h_n: Vec<bf16> = h_n_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let a64: Vec<f64> = h_a.iter().map(|&v| v.to_f32() as f64).collect();
    let p64: Vec<f64> = h_p.iter().map(|&v| v.to_f32() as f64).collect();
    let n64: Vec<f64> = h_n.iter().map(|&v| v.to_f32() as f64).collect();
    let expected = host_triplet_mean_f64(&a64, &p64, &n64, n, d, margin as f64, p_norm as f64) as f32;
    let dev_a = DeviceBuffer::from_slice(&ctx, &h_a).unwrap();
    let dev_p = DeviceBuffer::from_slice(&ctx, &h_p).unwrap();
    let dev_n = DeviceBuffer::from_slice(&ctx, &h_n).unwrap();
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, n * 2).unwrap();
    let desc = TripletMarginLossDescriptor {
        n_rows: n as i32,
        d_extent: d as i32,
        reduction: LossReduction::Mean,
        margin,
        p_norm,
        element: ElementKind::Bf16,
    };
    let plan =
        TripletMarginLossPlan::<bf16>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        TripletMarginLossArgs {
            anchor: TensorRef { data: dev_a.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            positive: TensorRef { data: dev_p.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            negative: TensorRef { data: dev_n.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1, 1], stride: [1, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = [bf16::ZERO; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let got_f32 = got[0].to_f32();
    let tol = expected.abs() * 32.0 * 7.81e-3_f32 + 5e-2;
    assert!((got_f32 - expected).abs() <= tol, "bf16 Triplet: got={} want={}", got_f32, expected);
}
