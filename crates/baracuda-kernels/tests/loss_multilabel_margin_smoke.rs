//! Real-GPU smoke test for `MultilabelMarginLossPlan`. FW × 4 dtypes.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    ElementKind, LossReduction, MultilabelMarginLossArgs, MultilabelMarginLossDescriptor,
    MultilabelMarginLossPlan, PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn host_mlm_mean_f64(input: &[f64], tgt: &[i64], n: usize, c: usize) -> f64 {
    let mut total = 0.0;
    for r in 0..n {
        let row = &input[r * c..(r + 1) * c];
        let trow = &tgt[r * c..(r + 1) * c];
        let mut acc = 0.0;
        for k in 0..c {
            let j = trow[k];
            if j < 0 { break; }
            if j as usize >= c { continue; }
            let xj = row[j as usize];
            for i in 0..c {
                let mut in_pos = false;
                for kk in 0..c {
                    let pp_ = trow[kk];
                    if pp_ < 0 { break; }
                    if pp_ as usize == i { in_pos = true; break; }
                }
                if in_pos { continue; }
                let h = 1.0 - xj + row[i];
                if h > 0.0 { acc += h; }
            }
        }
        total += acc / (c as f64);
    }
    total / (n as f64)
}

#[test]
#[ignore]
fn loss_multilabel_margin_f32_mean() {
    let (ctx, stream) = setup();
    let n = 2usize;
    let c = 4usize;
    // row 0 has positives [0, 2], row 1 has positive [1]
    let h_in: Vec<f32> = vec![
        0.1, 0.5, 0.3, 0.7,
        -0.2, 0.4, 0.1, 0.9,
    ];
    let h_t: Vec<i64> = vec![0, 2, -1, -1, 1, -1, -1, -1];
    let expected = host_mlm_mean_f64(
        &h_in.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &h_t, n, c,
    ) as f32;
    let dev_in = DeviceBuffer::from_slice(&ctx, &h_in).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &h_t).unwrap();
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, n * 4).unwrap();
    let desc = MultilabelMarginLossDescriptor {
        n_rows: n as i32,
        class_extent: c as i32,
        reduction: LossReduction::Mean,
        element: ElementKind::F32,
    };
    let plan = MultilabelMarginLossPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        MultilabelMarginLossArgs {
            input: TensorRef { data: dev_in.as_slice(), shape: [n as i32, c as i32], stride: [c as i64, 1] },
            target: TensorRef { data: dev_t.as_slice(), shape: [n as i32, c as i32], stride: [c as i64, 1] },
            out: TensorMut { data: dev_y.as_slice_mut(), shape: [1, 1], stride: [1, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = [0f32; 1];
    dev_y.copy_to_host(&mut got).unwrap();
    let tol = expected.abs() * 16.0 * f32::EPSILON + 1e-6;
    assert!((got[0] - expected).abs() <= tol, "f32 MLM: got={} want={}", got[0], expected);
}

#[test]
#[ignore]
fn loss_multilabel_margin_f64_mean() {
    let (ctx, stream) = setup();
    let n = 2usize;
    let c = 4usize;
    let h_in: Vec<f64> = vec![
        0.1, 0.5, 0.3, 0.7,
        -0.2, 0.4, 0.1, 0.9,
    ];
    let h_t: Vec<i64> = vec![0, 2, -1, -1, 1, -1, -1, -1];
    let expected = host_mlm_mean_f64(&h_in, &h_t, n, c);
    let dev_in = DeviceBuffer::from_slice(&ctx, &h_in).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &h_t).unwrap();
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, n * 8).unwrap();
    let desc = MultilabelMarginLossDescriptor {
        n_rows: n as i32,
        class_extent: c as i32,
        reduction: LossReduction::Mean,
        element: ElementKind::F64,
    };
    let plan = MultilabelMarginLossPlan::<f64>::select(&stream, &desc, PlanPreference::default())
        .unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        MultilabelMarginLossArgs {
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
fn loss_multilabel_margin_f16_mean() {
    let (ctx, stream) = setup();
    let n = 2usize;
    let c = 4usize;
    let h_in_f32: Vec<f32> = vec![
        0.1, 0.5, 0.3, 0.7,
        -0.2, 0.4, 0.1, 0.9,
    ];
    let h_in: Vec<f16> = h_in_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let h_t: Vec<i64> = vec![0, 2, -1, -1, 1, -1, -1, -1];
    let in64: Vec<f64> = h_in.iter().map(|&v| v.to_f32() as f64).collect();
    let expected = host_mlm_mean_f64(&in64, &h_t, n, c) as f32;
    let dev_in = DeviceBuffer::from_slice(&ctx, &h_in).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &h_t).unwrap();
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, n * 2).unwrap();
    let desc = MultilabelMarginLossDescriptor {
        n_rows: n as i32,
        class_extent: c as i32,
        reduction: LossReduction::Mean,
        element: ElementKind::F16,
    };
    let plan = MultilabelMarginLossPlan::<f16>::select(&stream, &desc, PlanPreference::default())
        .unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        MultilabelMarginLossArgs {
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
    let tol = expected.abs() * 16.0 * 9.77e-4_f32 + 5e-3;
    assert!((got_f32 - expected).abs() <= tol, "f16 MLM: got={} want={}", got_f32, expected);
}

#[test]
#[ignore]
fn loss_multilabel_margin_bf16_mean() {
    let (ctx, stream) = setup();
    let n = 2usize;
    let c = 4usize;
    let h_in_f32: Vec<f32> = vec![
        0.1, 0.5, 0.3, 0.7,
        -0.2, 0.4, 0.1, 0.9,
    ];
    let h_in: Vec<bf16> = h_in_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let h_t: Vec<i64> = vec![0, 2, -1, -1, 1, -1, -1, -1];
    let in64: Vec<f64> = h_in.iter().map(|&v| v.to_f32() as f64).collect();
    let expected = host_mlm_mean_f64(&in64, &h_t, n, c) as f32;
    let dev_in = DeviceBuffer::from_slice(&ctx, &h_in).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &h_t).unwrap();
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, n * 2).unwrap();
    let desc = MultilabelMarginLossDescriptor {
        n_rows: n as i32,
        class_extent: c as i32,
        reduction: LossReduction::Mean,
        element: ElementKind::Bf16,
    };
    let plan = MultilabelMarginLossPlan::<bf16>::select(&stream, &desc, PlanPreference::default())
        .unwrap();
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        MultilabelMarginLossArgs {
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
    let tol = expected.abs() * 16.0 * 7.81e-3_f32 + 2e-2;
    assert!((got_f32 - expected).abs() <= tol, "bf16 MLM: got={} want={}", got_f32, expected);
}
