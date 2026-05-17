//! Real-GPU smoke test for `MultilabelMarginLossBackwardPlan`. BW × 4 dtypes.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    ElementKind, LossReduction, MultilabelMarginLossBackwardArgs,
    MultilabelMarginLossBackwardDescriptor, MultilabelMarginLossBackwardPlan, PlanPreference,
    TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn host_mlm_bw_f64(input: &[f64], tgt: &[i64], n: usize, c: usize, dy: f64) -> Vec<f64> {
    let scale = dy / (n as f64);
    let mut din = vec![0.0; n * c];
    for r in 0..n {
        let row = &input[r * c..(r + 1) * c];
        let trow = &tgt[r * c..(r + 1) * c];
        let drow = &mut din[r * c..(r + 1) * c];
        let coef = scale / (c as f64);
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
                if h > 0.0 {
                    drow[j as usize] -= coef;
                    drow[i] += coef;
                }
            }
        }
    }
    din
}

#[test]
#[ignore]
fn loss_multilabel_margin_backward_f32_mean() {
    let (ctx, stream) = setup();
    let n = 2usize;
    let c = 4usize;
    let h_in: Vec<f32> = vec![
        0.1, 0.5, 0.3, 0.7,
        -0.2, 0.4, 0.1, 0.9,
    ];
    let h_t: Vec<i64> = vec![0, 2, -1, -1, 1, -1, -1, -1];
    let dy_host = [1.0f32];
    let expected: Vec<f32> = host_mlm_bw_f64(
        &h_in.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &h_t, n, c, 1.0,
    )
    .into_iter()
    .map(|v| v as f32)
    .collect();
    let dev_in = DeviceBuffer::from_slice(&ctx, &h_in).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &h_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, n * c).unwrap();
    let desc = MultilabelMarginLossBackwardDescriptor {
        n_rows: n as i32,
        class_extent: c as i32,
        reduction: LossReduction::Mean,
        element: ElementKind::F32,
    };
    let plan = MultilabelMarginLossBackwardPlan::<f32>::select(
        &stream,
        &desc,
        PlanPreference::default(),
    )
    .unwrap();
    plan.run(
        &stream,
        Workspace::None,
        MultilabelMarginLossBackwardArgs {
            input: TensorRef { data: dev_in.as_slice(), shape: [n as i32, c as i32], stride: [c as i64, 1] },
            target: TensorRef { data: dev_t.as_slice(), shape: [n as i32, c as i32], stride: [c as i64, 1] },
            dy: TensorRef { data: dev_dy.as_slice(), shape: [1, 1], stride: [1, 1] },
            dinput: TensorMut { data: dev_dx.as_slice_mut(), shape: [n as i32, c as i32], stride: [c as i64, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = vec![0f32; n * c];
    dev_dx.copy_to_host(&mut got).unwrap();
    for i in 0..(n * c) {
        let tol = expected[i].abs().max(1.0) * 16.0 * f32::EPSILON + 1e-6;
        assert!((got[i] - expected[i]).abs() <= tol, "f32 MLM BW @{i}: got={} want={}", got[i], expected[i]);
    }
}

#[test]
#[ignore]
fn loss_multilabel_margin_backward_f64_mean() {
    let (ctx, stream) = setup();
    let n = 2usize;
    let c = 4usize;
    let h_in: Vec<f64> = vec![
        0.1, 0.5, 0.3, 0.7,
        -0.2, 0.4, 0.1, 0.9,
    ];
    let h_t: Vec<i64> = vec![0, 2, -1, -1, 1, -1, -1, -1];
    let dy_host = [2.0f64];
    let expected = host_mlm_bw_f64(&h_in, &h_t, n, c, 2.0);
    let dev_in = DeviceBuffer::from_slice(&ctx, &h_in).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &h_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_dx: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, n * c).unwrap();
    let desc = MultilabelMarginLossBackwardDescriptor {
        n_rows: n as i32,
        class_extent: c as i32,
        reduction: LossReduction::Mean,
        element: ElementKind::F64,
    };
    let plan = MultilabelMarginLossBackwardPlan::<f64>::select(
        &stream,
        &desc,
        PlanPreference::default(),
    )
    .unwrap();
    plan.run(
        &stream,
        Workspace::None,
        MultilabelMarginLossBackwardArgs {
            input: TensorRef { data: dev_in.as_slice(), shape: [n as i32, c as i32], stride: [c as i64, 1] },
            target: TensorRef { data: dev_t.as_slice(), shape: [n as i32, c as i32], stride: [c as i64, 1] },
            dy: TensorRef { data: dev_dy.as_slice(), shape: [1, 1], stride: [1, 1] },
            dinput: TensorMut { data: dev_dx.as_slice_mut(), shape: [n as i32, c as i32], stride: [c as i64, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = vec![0f64; n * c];
    dev_dx.copy_to_host(&mut got).unwrap();
    for i in 0..(n * c) {
        let tol = expected[i].abs().max(1.0) * 16.0 * f64::EPSILON + 1e-12;
        assert!((got[i] - expected[i]).abs() <= tol);
    }
}

#[test]
#[ignore]
fn loss_multilabel_margin_backward_f16_mean() {
    let (ctx, stream) = setup();
    let n = 2usize;
    let c = 4usize;
    let h_in_f32: Vec<f32> = vec![
        0.1, 0.5, 0.3, 0.7,
        -0.2, 0.4, 0.1, 0.9,
    ];
    let h_in: Vec<f16> = h_in_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let h_t: Vec<i64> = vec![0, 2, -1, -1, 1, -1, -1, -1];
    let dy_host = [f16::from_f32(1.0)];
    let in64: Vec<f64> = h_in.iter().map(|&v| v.to_f32() as f64).collect();
    let expected = host_mlm_bw_f64(&in64, &h_t, n, c, 1.0);
    let dev_in = DeviceBuffer::from_slice(&ctx, &h_in).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &h_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_dx: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, n * c).unwrap();
    let desc = MultilabelMarginLossBackwardDescriptor {
        n_rows: n as i32,
        class_extent: c as i32,
        reduction: LossReduction::Mean,
        element: ElementKind::F16,
    };
    let plan = MultilabelMarginLossBackwardPlan::<f16>::select(
        &stream,
        &desc,
        PlanPreference::default(),
    )
    .unwrap();
    plan.run(
        &stream,
        Workspace::None,
        MultilabelMarginLossBackwardArgs {
            input: TensorRef { data: dev_in.as_slice(), shape: [n as i32, c as i32], stride: [c as i64, 1] },
            target: TensorRef { data: dev_t.as_slice(), shape: [n as i32, c as i32], stride: [c as i64, 1] },
            dy: TensorRef { data: dev_dy.as_slice(), shape: [1, 1], stride: [1, 1] },
            dinput: TensorMut { data: dev_dx.as_slice_mut(), shape: [n as i32, c as i32], stride: [c as i64, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = vec![f16::ZERO; n * c];
    dev_dx.copy_to_host(&mut got).unwrap();
    for i in 0..(n * c) {
        let want = expected[i] as f32;
        let g = got[i].to_f32();
        let tol = want.abs().max(1.0) * 16.0 * 9.77e-4_f32 + 5e-3;
        assert!((g - want).abs() <= tol, "f16 MLM BW @{i}: got={} want={}", g, want);
    }
}

#[test]
#[ignore]
fn loss_multilabel_margin_backward_bf16_mean() {
    let (ctx, stream) = setup();
    let n = 2usize;
    let c = 4usize;
    let h_in_f32: Vec<f32> = vec![
        0.1, 0.5, 0.3, 0.7,
        -0.2, 0.4, 0.1, 0.9,
    ];
    let h_in: Vec<bf16> = h_in_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let h_t: Vec<i64> = vec![0, 2, -1, -1, 1, -1, -1, -1];
    let dy_host = [bf16::from_f32(1.0)];
    let in64: Vec<f64> = h_in.iter().map(|&v| v.to_f32() as f64).collect();
    let expected = host_mlm_bw_f64(&in64, &h_t, n, c, 1.0);
    let dev_in = DeviceBuffer::from_slice(&ctx, &h_in).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &h_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_dx: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, n * c).unwrap();
    let desc = MultilabelMarginLossBackwardDescriptor {
        n_rows: n as i32,
        class_extent: c as i32,
        reduction: LossReduction::Mean,
        element: ElementKind::Bf16,
    };
    let plan = MultilabelMarginLossBackwardPlan::<bf16>::select(
        &stream,
        &desc,
        PlanPreference::default(),
    )
    .unwrap();
    plan.run(
        &stream,
        Workspace::None,
        MultilabelMarginLossBackwardArgs {
            input: TensorRef { data: dev_in.as_slice(), shape: [n as i32, c as i32], stride: [c as i64, 1] },
            target: TensorRef { data: dev_t.as_slice(), shape: [n as i32, c as i32], stride: [c as i64, 1] },
            dy: TensorRef { data: dev_dy.as_slice(), shape: [1, 1], stride: [1, 1] },
            dinput: TensorMut { data: dev_dx.as_slice_mut(), shape: [n as i32, c as i32], stride: [c as i64, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got = vec![bf16::ZERO; n * c];
    dev_dx.copy_to_host(&mut got).unwrap();
    for i in 0..(n * c) {
        let want = expected[i] as f32;
        let g = got[i].to_f32();
        let tol = want.abs().max(1.0) * 16.0 * 7.81e-3_f32 + 2e-2;
        assert!((g - want).abs() <= tol, "bf16 MLM BW @{i}: got={} want={}", g, want);
    }
}
