//! Real-GPU smoke test for `MultilabelSoftMarginLossPlan`. FW × 4 dtypes.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    ElementKind, LossReduction, MultilabelSoftMarginLossArgs,
    MultilabelSoftMarginLossDescriptor, MultilabelSoftMarginLossPlan, PlanPreference, TensorMut,
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

fn host_mlsm_mean_f64(input: &[f64], tgt: &[f64], n: usize, c: usize) -> f64 {
    let mut s = 0.0;
    for r in 0..n {
        let mut acc = 0.0;
        for j in 0..c {
            let x = input[r * c + j];
            let y = tgt[r * c + j];
            let ax = x.abs();
            let mx = if x > 0.0 { x } else { 0.0 };
            acc += mx - x * y + (1.0 + (-ax).exp()).ln();
        }
        s += acc / (c as f64);
    }
    s / (n as f64)
}

#[test]
#[ignore]
fn loss_multilabel_soft_margin_f32_mean() {
    let (ctx, stream) = setup();
    let n = 2usize;
    let c = 4usize;
    let h_in: Vec<f32> = (0..n * c).map(|i| (i as f32) * 0.1 - 0.4).collect();
    let h_t: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
    let expected = host_mlsm_mean_f64(
        &h_in.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &h_t.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        n, c,
    ) as f32;
    let dev_in = DeviceBuffer::from_slice(&ctx, &h_in).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &h_t).unwrap();
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, n * 4).unwrap();
    let desc = MultilabelSoftMarginLossDescriptor {
        n_rows: n as i32,
        class_extent: c as i32,
        reduction: LossReduction::Mean,
        element: ElementKind::F32,
    };
    let plan = MultilabelSoftMarginLossPlan::<f32>::select(
        &stream,
        &desc,
        PlanPreference::default(),
    )
    .unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        MultilabelSoftMarginLossArgs {
            input: TensorRef { data: dev_in.as_slice(), shape: [n as i32, c as i32], stride: [c as i64, 1] },
            target: TensorRef { data: dev_t.as_slice(), shape: [n as i32, c as i32], stride: [c as i64, 1] },
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1, 1], stride: [1, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = [0f32; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let tol = expected.abs() * 16.0 * f32::EPSILON + 1e-5;
    assert!((got[0] - expected).abs() <= tol);
}

#[test]
#[ignore]
fn loss_multilabel_soft_margin_f64_mean() {
    let (ctx, stream) = setup();
    let n = 2usize;
    let c = 4usize;
    let h_in: Vec<f64> = (0..n * c).map(|i| (i as f64) * 0.1 - 0.4).collect();
    let h_t: Vec<f64> = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
    let expected = host_mlsm_mean_f64(&h_in, &h_t, n, c);
    let dev_in = DeviceBuffer::from_slice(&ctx, &h_in).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &h_t).unwrap();
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, n * 8).unwrap();
    let desc = MultilabelSoftMarginLossDescriptor {
        n_rows: n as i32,
        class_extent: c as i32,
        reduction: LossReduction::Mean,
        element: ElementKind::F64,
    };
    let plan = MultilabelSoftMarginLossPlan::<f64>::select(
        &stream,
        &desc,
        PlanPreference::default(),
    )
    .unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        MultilabelSoftMarginLossArgs {
            input: TensorRef { data: dev_in.as_slice(), shape: [n as i32, c as i32], stride: [c as i64, 1] },
            target: TensorRef { data: dev_t.as_slice(), shape: [n as i32, c as i32], stride: [c as i64, 1] },
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
fn loss_multilabel_soft_margin_f16_mean() {
    let (ctx, stream) = setup();
    let n = 2usize;
    let c = 4usize;
    let h_in_f32: Vec<f32> = (0..n * c).map(|i| (i as f32) * 0.1 - 0.4).collect();
    let h_t_f32: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
    let h_in: Vec<f16> = h_in_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let h_t: Vec<f16> = h_t_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let in64: Vec<f64> = h_in.iter().map(|&v| v.to_f32() as f64).collect();
    let t64: Vec<f64> = h_t.iter().map(|&v| v.to_f32() as f64).collect();
    let expected = host_mlsm_mean_f64(&in64, &t64, n, c) as f32;
    let dev_in = DeviceBuffer::from_slice(&ctx, &h_in).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &h_t).unwrap();
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, n * 2).unwrap();
    let desc = MultilabelSoftMarginLossDescriptor {
        n_rows: n as i32,
        class_extent: c as i32,
        reduction: LossReduction::Mean,
        element: ElementKind::F16,
    };
    let plan = MultilabelSoftMarginLossPlan::<f16>::select(
        &stream,
        &desc,
        PlanPreference::default(),
    )
    .unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        MultilabelSoftMarginLossArgs {
            input: TensorRef { data: dev_in.as_slice(), shape: [n as i32, c as i32], stride: [c as i64, 1] },
            target: TensorRef { data: dev_t.as_slice(), shape: [n as i32, c as i32], stride: [c as i64, 1] },
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1, 1], stride: [1, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = [f16::ZERO; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let got_f32 = got[0].to_f32();
    // log/exp pipeline — wider tolerance at half precision.
    let tol = expected.abs() * 32.0 * 9.77e-4_f32 + 1e-2;
    assert!((got_f32 - expected).abs() <= tol, "f16 MLSM: got={} want={}", got_f32, expected);
}

#[test]
#[ignore]
fn loss_multilabel_soft_margin_bf16_mean() {
    let (ctx, stream) = setup();
    let n = 2usize;
    let c = 4usize;
    let h_in_f32: Vec<f32> = (0..n * c).map(|i| (i as f32) * 0.1 - 0.4).collect();
    let h_t_f32: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
    let h_in: Vec<bf16> = h_in_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let h_t: Vec<bf16> = h_t_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let in64: Vec<f64> = h_in.iter().map(|&v| v.to_f32() as f64).collect();
    let t64: Vec<f64> = h_t.iter().map(|&v| v.to_f32() as f64).collect();
    let expected = host_mlsm_mean_f64(&in64, &t64, n, c) as f32;
    let dev_in = DeviceBuffer::from_slice(&ctx, &h_in).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &h_t).unwrap();
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, n * 2).unwrap();
    let desc = MultilabelSoftMarginLossDescriptor {
        n_rows: n as i32,
        class_extent: c as i32,
        reduction: LossReduction::Mean,
        element: ElementKind::Bf16,
    };
    let plan = MultilabelSoftMarginLossPlan::<bf16>::select(
        &stream,
        &desc,
        PlanPreference::default(),
    )
    .unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        MultilabelSoftMarginLossArgs {
            input: TensorRef { data: dev_in.as_slice(), shape: [n as i32, c as i32], stride: [c as i64, 1] },
            target: TensorRef { data: dev_t.as_slice(), shape: [n as i32, c as i32], stride: [c as i64, 1] },
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1, 1], stride: [1, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = [bf16::ZERO; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let got_f32 = got[0].to_f32();
    let tol = expected.abs() * 32.0 * 7.81e-3_f32 + 5e-2;
    assert!((got_f32 - expected).abs() <= tol, "bf16 MLSM: got={} want={}", got_f32, expected);
}
