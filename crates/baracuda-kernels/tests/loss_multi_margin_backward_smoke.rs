//! Real-GPU smoke test for `MultiMarginLossBackwardPlan`. BW × 4 dtypes.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    ElementKind, LossReduction, MultiMarginLossBackwardArgs,
    MultiMarginLossBackwardDescriptor, MultiMarginLossBackwardPlan, PlanPreference, TensorMut,
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

fn host_mm_bw_f64(
    input: &[f64], t: &[i64], n: usize, c: usize, margin: f64, pn: f64, dy: f64,
) -> Vec<f64> {
    let scale = dy / (n as f64);
    let mut din = vec![0.0; n * c];
    for r in 0..n {
        let ti = t[r] as usize;
        let xt = input[r * c + ti];
        let coef = scale / (c as f64);
        let mut acc_t = 0.0;
        for j in 0..c {
            if j == ti { continue; }
            let h = margin - xt + input[r * c + j];
            if h > 0.0 {
                let grad_h = if pn == 1.0 { 1.0 } else { pn * h.powf(pn - 1.0) };
                din[r * c + j] = grad_h * coef;
                acc_t += grad_h;
            }
        }
        din[r * c + ti] = -acc_t * coef;
    }
    din
}

#[test]
#[ignore]
fn loss_multi_margin_backward_f32_mean() {
    let (ctx, stream) = setup();
    let n = 3usize;
    let c = 4usize;
    let margin = 1.0f32;
    let p_norm = 1.0f32;
    let h_in: Vec<f32> = (0..n * c).map(|i| (i as f32) * 0.1 - 0.5).collect();
    let h_t: Vec<i64> = vec![0, 2, 1];
    let dy_host = [1.0f32];
    let expected: Vec<f32> = host_mm_bw_f64(
        &h_in.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &h_t, n, c, margin as f64, p_norm as f64, 1.0,
    )
    .into_iter()
    .map(|v| v as f32)
    .collect();
    let dev_in = DeviceBuffer::from_slice(&ctx, &h_in).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &h_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, n * c).unwrap();
    let desc = MultiMarginLossBackwardDescriptor {
        n_rows: n as i32,
        class_extent: c as i32,
        reduction: LossReduction::Mean,
        margin,
        p_norm,
        element: ElementKind::F32,
    };
    let plan = MultiMarginLossBackwardPlan::<f32>::select(
        &stream,
        &desc,
        PlanPreference::default(),
    )
    .unwrap();
    plan.run(
        &stream,
        Workspace::None,
        MultiMarginLossBackwardArgs {
            input: TensorRef { data: dev_in.as_slice(), shape: [n as i32, c as i32], stride: [c as i64, 1] },
            target: TensorRef { data: dev_t.as_slice(), shape: [n as i32, 1], stride: [1, 1] },
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
        assert!((got[i] - expected[i]).abs() <= tol, "f32 MM BW @{i}: got={} want={}", got[i], expected[i]);
    }
}

#[test]
#[ignore]
fn loss_multi_margin_backward_f64_mean() {
    let (ctx, stream) = setup();
    let n = 2usize;
    let c = 5usize;
    let margin = 1.0f32;
    let p_norm = 1.0f32;
    let h_in: Vec<f64> = (0..n * c).map(|i| (i as f64) * 0.05 - 0.2).collect();
    let h_t: Vec<i64> = vec![1, 3];
    let dy_host = [2.0f64];
    let expected = host_mm_bw_f64(&h_in, &h_t, n, c, margin as f64, p_norm as f64, 2.0);
    let dev_in = DeviceBuffer::from_slice(&ctx, &h_in).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &h_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_dx: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, n * c).unwrap();
    let desc = MultiMarginLossBackwardDescriptor {
        n_rows: n as i32,
        class_extent: c as i32,
        reduction: LossReduction::Mean,
        margin,
        p_norm,
        element: ElementKind::F64,
    };
    let plan = MultiMarginLossBackwardPlan::<f64>::select(
        &stream,
        &desc,
        PlanPreference::default(),
    )
    .unwrap();
    plan.run(
        &stream,
        Workspace::None,
        MultiMarginLossBackwardArgs {
            input: TensorRef { data: dev_in.as_slice(), shape: [n as i32, c as i32], stride: [c as i64, 1] },
            target: TensorRef { data: dev_t.as_slice(), shape: [n as i32, 1], stride: [1, 1] },
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
fn loss_multi_margin_backward_f16_mean() {
    let (ctx, stream) = setup();
    let n = 3usize;
    let c = 4usize;
    let margin = 1.0f32;
    let p_norm = 1.0f32;
    let h_in_f32: Vec<f32> = (0..n * c).map(|i| (i as f32) * 0.1 - 0.5).collect();
    let h_in: Vec<f16> = h_in_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let h_t: Vec<i64> = vec![0, 2, 1];
    let dy_host = [f16::from_f32(1.0)];
    let in64: Vec<f64> = h_in.iter().map(|&v| v.to_f32() as f64).collect();
    let expected = host_mm_bw_f64(&in64, &h_t, n, c, margin as f64, p_norm as f64, 1.0);
    let dev_in = DeviceBuffer::from_slice(&ctx, &h_in).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &h_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_dx: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, n * c).unwrap();
    let desc = MultiMarginLossBackwardDescriptor {
        n_rows: n as i32,
        class_extent: c as i32,
        reduction: LossReduction::Mean,
        margin,
        p_norm,
        element: ElementKind::F16,
    };
    let plan = MultiMarginLossBackwardPlan::<f16>::select(
        &stream,
        &desc,
        PlanPreference::default(),
    )
    .unwrap();
    plan.run(
        &stream,
        Workspace::None,
        MultiMarginLossBackwardArgs {
            input: TensorRef { data: dev_in.as_slice(), shape: [n as i32, c as i32], stride: [c as i64, 1] },
            target: TensorRef { data: dev_t.as_slice(), shape: [n as i32, 1], stride: [1, 1] },
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
        assert!((g - want).abs() <= tol, "f16 MM BW @{i}: got={} want={}", g, want);
    }
}

#[test]
#[ignore]
fn loss_multi_margin_backward_bf16_mean() {
    let (ctx, stream) = setup();
    let n = 3usize;
    let c = 4usize;
    let margin = 1.0f32;
    let p_norm = 1.0f32;
    let h_in_f32: Vec<f32> = (0..n * c).map(|i| (i as f32) * 0.1 - 0.5).collect();
    let h_in: Vec<bf16> = h_in_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let h_t: Vec<i64> = vec![0, 2, 1];
    let dy_host = [bf16::from_f32(1.0)];
    let in64: Vec<f64> = h_in.iter().map(|&v| v.to_f32() as f64).collect();
    let expected = host_mm_bw_f64(&in64, &h_t, n, c, margin as f64, p_norm as f64, 1.0);
    let dev_in = DeviceBuffer::from_slice(&ctx, &h_in).unwrap();
    let dev_t = DeviceBuffer::from_slice(&ctx, &h_t).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_dx: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, n * c).unwrap();
    let desc = MultiMarginLossBackwardDescriptor {
        n_rows: n as i32,
        class_extent: c as i32,
        reduction: LossReduction::Mean,
        margin,
        p_norm,
        element: ElementKind::Bf16,
    };
    let plan = MultiMarginLossBackwardPlan::<bf16>::select(
        &stream,
        &desc,
        PlanPreference::default(),
    )
    .unwrap();
    plan.run(
        &stream,
        Workspace::None,
        MultiMarginLossBackwardArgs {
            input: TensorRef { data: dev_in.as_slice(), shape: [n as i32, c as i32], stride: [c as i64, 1] },
            target: TensorRef { data: dev_t.as_slice(), shape: [n as i32, 1], stride: [1, 1] },
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
        assert!((g - want).abs() <= tol, "bf16 MM BW @{i}: got={} want={}", g, want);
    }
}
