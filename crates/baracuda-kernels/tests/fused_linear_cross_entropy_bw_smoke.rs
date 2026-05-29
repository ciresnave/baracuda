//! Phase 47 — Fused Linear Cross-Entropy BW smoke test.
//!
//! The FW pass already produces `grad_input` and `grad_weight` for
//! `dy = 1.0`. Verifies via finite-difference around a small problem
//! that the gradients are within the analytical expectation for the
//! combined Linear + CrossEntropy operation.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, FusedLinearCrossEntropyArgs,
    FusedLinearCrossEntropyBackwardArgs, FusedLinearCrossEntropyBackwardDescriptor,
    FusedLinearCrossEntropyBackwardPlan, FusedLinearCrossEntropyDescriptor,
    FusedLinearCrossEntropyPlan, LossReduction, PlanPreference, TensorMut, TensorRef, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Reference: scalar mean CE loss given (input, weight, target).
fn host_loss_mean(
    input: &[f64],
    weight: &[f64],
    target: &[i64],
    bt: usize,
    h: usize,
    v: usize,
) -> f64 {
    let mut total = 0.0;
    let mut n = 0usize;
    for i in 0..bt {
        let t = target[i];
        if t < 0 || t >= v as i64 {
            continue;
        }
        let mut logits = vec![0.0f64; v];
        for j in 0..v {
            let mut acc = 0.0;
            for k in 0..h {
                acc += input[i * h + k] * weight[j * h + k];
            }
            logits[j] = acc;
        }
        let mut max = f64::NEG_INFINITY;
        for &l in &logits {
            if l > max {
                max = l;
            }
        }
        let mut se = 0.0;
        for &l in &logits {
            se += (l - max).exp();
        }
        let log_z = max + se.ln();
        total += -(logits[t as usize] - log_z);
        n += 1;
    }
    if n == 0 {
        0.0
    } else {
        total / (n as f64)
    }
}

#[test]
#[ignore]
fn flce_bw_f32_grad_input_finite_diff() {
    let (ctx, stream) = setup();
    let bt = 4i32;
    let h = 8i32;
    let v = 12i32;

    // Small, well-conditioned fixture so finite-difference noise is low.
    let mut host_input = vec![0.0f32; (bt * h) as usize];
    let mut host_weight = vec![0.0f32; (v * h) as usize];
    for i in 0..host_input.len() {
        host_input[i] = ((i as f32) * 0.07 + 0.3).cos() * 0.4;
    }
    for i in 0..host_weight.len() {
        host_weight[i] = ((i as f32) * 0.11 + 0.1).sin() * 0.3;
    }
    let host_target: Vec<i64> = vec![1, 5, 7, 11];

    let dev_input = DeviceBuffer::from_slice(&ctx, &host_input).unwrap();
    let dev_weight = DeviceBuffer::from_slice(&ctx, &host_weight).unwrap();
    let dev_target = DeviceBuffer::from_slice(&ctx, &host_target).unwrap();
    let mut dev_out: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_grad_input: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, (bt * h) as usize).unwrap();
    let mut dev_grad_weight: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, (v * h) as usize).unwrap();

    let desc = FusedLinearCrossEntropyDescriptor::new(bt, h, v, ElementKind::F32)
        .with_reduction(LossReduction::Mean);
    let plan = FusedLinearCrossEntropyPlan::<f32>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        FusedLinearCrossEntropyArgs {
            input: TensorRef {
                data: dev_input.as_slice(),
                shape: [bt, h],
                stride: contiguous_stride([bt, h]),
            },
            weight: TensorRef {
                data: dev_weight.as_slice(),
                shape: [v, h],
                stride: contiguous_stride([v, h]),
            },
            target: TensorRef {
                data: dev_target.as_slice(),
                shape: [bt],
                stride: contiguous_stride([bt]),
            },
            out: TensorMut {
                data: dev_out.as_slice_mut(),
                shape: [1],
                stride: [1],
            },
            grad_input: Some(TensorMut {
                data: dev_grad_input.as_slice_mut(),
                shape: [bt, h],
                stride: contiguous_stride([bt, h]),
            }),
            grad_weight: Some(TensorMut {
                data: dev_grad_weight.as_slice_mut(),
                shape: [v, h],
                stride: contiguous_stride([v, h]),
            }),
        },
    )
    .unwrap();
    stream.synchronize().unwrap();

    // BW pass with dy=1.0 (the typical "CE is the last layer" case).
    let bw_desc = FusedLinearCrossEntropyBackwardDescriptor::new(bt, h, v, ElementKind::F32);
    let bw_plan = FusedLinearCrossEntropyBackwardPlan::<f32>::select(&stream, &bw_desc, PlanPreference::default()).unwrap();
    bw_plan.run(
        &stream,
        Workspace::None,
        FusedLinearCrossEntropyBackwardArgs {
            dy_scalar: 1.0,
            grad_input: Some(TensorMut {
                data: dev_grad_input.as_slice_mut(),
                shape: [bt, h],
                stride: contiguous_stride([bt, h]),
            }),
            grad_weight: Some(TensorMut {
                data: dev_grad_weight.as_slice_mut(),
                shape: [v, h],
                stride: contiguous_stride([v, h]),
            }),
        },
    )
    .unwrap();
    stream.synchronize().unwrap();

    let mut got_grad_input = vec![0.0f32; (bt * h) as usize];
    let mut got_grad_weight = vec![0.0f32; (v * h) as usize];
    dev_grad_input.copy_to_host(&mut got_grad_input).unwrap();
    dev_grad_weight.copy_to_host(&mut got_grad_weight).unwrap();

    // Finite-difference check: pick a handful of (i, k) input cells
    // and a handful of (j, k) weight cells; check that
    // (loss(x + eps) - loss(x - eps)) / (2·eps) ≈ grad.
    let host_input_f64: Vec<f64> = host_input.iter().map(|&v| v as f64).collect();
    let host_weight_f64: Vec<f64> = host_weight.iter().map(|&v| v as f64).collect();
    let eps = 1e-3f64;

    // Sample 6 random-ish (i, k) positions for grad_input.
    let input_samples: [(usize, usize); 6] = [
        (0, 0), (0, 5), (1, 2), (2, 4), (3, 0), (3, 7),
    ];
    for &(i, k) in input_samples.iter() {
        let idx = i * (h as usize) + k;
        let mut input_p = host_input_f64.clone();
        let mut input_m = host_input_f64.clone();
        input_p[idx] += eps;
        input_m[idx] -= eps;
        let loss_p = host_loss_mean(
            &input_p, &host_weight_f64, &host_target,
            bt as usize, h as usize, v as usize,
        );
        let loss_m = host_loss_mean(
            &input_m, &host_weight_f64, &host_target,
            bt as usize, h as usize, v as usize,
        );
        let fd_grad = (loss_p - loss_m) / (2.0 * eps);
        let got = got_grad_input[idx] as f64;
        let tol = 1e-3 + 0.05 * fd_grad.abs();
        assert!(
            (got - fd_grad).abs() <= tol,
            "grad_input[{}, {}]: got={} fd={} tol={}",
            i, k, got, fd_grad, tol
        );
    }

    // Sample 6 (j, k) positions for grad_weight.
    let weight_samples: [(usize, usize); 6] = [
        (1, 0), (3, 4), (5, 2), (7, 7), (9, 1), (11, 5),
    ];
    for &(j, k) in weight_samples.iter() {
        let idx = j * (h as usize) + k;
        let mut weight_p = host_weight_f64.clone();
        let mut weight_m = host_weight_f64.clone();
        weight_p[idx] += eps;
        weight_m[idx] -= eps;
        let loss_p = host_loss_mean(
            &host_input_f64, &weight_p, &host_target,
            bt as usize, h as usize, v as usize,
        );
        let loss_m = host_loss_mean(
            &host_input_f64, &weight_m, &host_target,
            bt as usize, h as usize, v as usize,
        );
        let fd_grad = (loss_p - loss_m) / (2.0 * eps);
        let got = got_grad_weight[idx] as f64;
        let tol = 1e-3 + 0.05 * fd_grad.abs();
        assert!(
            (got - fd_grad).abs() <= tol,
            "grad_weight[{}, {}]: got={} fd={} tol={}",
            j, k, got, fd_grad, tol
        );
    }
}

#[test]
#[ignore]
fn flce_bw_f32_dy_scalar_2_doubles_gradients() {
    // Run FW twice; second time apply BW with dy_scalar=2.0; verify the
    // gradient is 2× the first.
    let (ctx, stream) = setup();
    let bt = 4i32;
    let h = 8i32;
    let v = 12i32;

    let mut host_input = vec![0.0f32; (bt * h) as usize];
    let mut host_weight = vec![0.0f32; (v * h) as usize];
    for i in 0..host_input.len() {
        host_input[i] = ((i as f32) * 0.05).sin() * 0.3;
    }
    for i in 0..host_weight.len() {
        host_weight[i] = ((i as f32) * 0.07).cos() * 0.3;
    }
    let host_target: Vec<i64> = vec![1, 5, 7, 11];

    let dev_input = DeviceBuffer::from_slice(&ctx, &host_input).unwrap();
    let dev_weight = DeviceBuffer::from_slice(&ctx, &host_weight).unwrap();
    let dev_target = DeviceBuffer::from_slice(&ctx, &host_target).unwrap();

    // Pass 1: FW, capture grad_input1.
    let mut dev_out_1: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_grad_input_1: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, (bt * h) as usize).unwrap();
    let mut dev_grad_weight_1: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, (v * h) as usize).unwrap();

    let desc = FusedLinearCrossEntropyDescriptor::new(bt, h, v, ElementKind::F32)
        .with_reduction(LossReduction::Mean);
    let plan = FusedLinearCrossEntropyPlan::<f32>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        FusedLinearCrossEntropyArgs {
            input: TensorRef {
                data: dev_input.as_slice(),
                shape: [bt, h],
                stride: contiguous_stride([bt, h]),
            },
            weight: TensorRef {
                data: dev_weight.as_slice(),
                shape: [v, h],
                stride: contiguous_stride([v, h]),
            },
            target: TensorRef {
                data: dev_target.as_slice(),
                shape: [bt],
                stride: contiguous_stride([bt]),
            },
            out: TensorMut {
                data: dev_out_1.as_slice_mut(),
                shape: [1],
                stride: [1],
            },
            grad_input: Some(TensorMut {
                data: dev_grad_input_1.as_slice_mut(),
                shape: [bt, h],
                stride: contiguous_stride([bt, h]),
            }),
            grad_weight: Some(TensorMut {
                data: dev_grad_weight_1.as_slice_mut(),
                shape: [v, h],
                stride: contiguous_stride([v, h]),
            }),
        },
    )
    .unwrap();
    stream.synchronize().unwrap();

    let mut grad_input_1 = vec![0.0f32; (bt * h) as usize];
    let mut grad_weight_1 = vec![0.0f32; (v * h) as usize];
    dev_grad_input_1.copy_to_host(&mut grad_input_1).unwrap();
    dev_grad_weight_1.copy_to_host(&mut grad_weight_1).unwrap();

    // Pass 2: same FW, then BW with dy=2.0.
    let mut dev_out_2: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut dev_grad_input_2: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, (bt * h) as usize).unwrap();
    let mut dev_grad_weight_2: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, (v * h) as usize).unwrap();
    let plan2 = FusedLinearCrossEntropyPlan::<f32>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan2.run(
        &stream,
        Workspace::None,
        FusedLinearCrossEntropyArgs {
            input: TensorRef {
                data: dev_input.as_slice(),
                shape: [bt, h],
                stride: contiguous_stride([bt, h]),
            },
            weight: TensorRef {
                data: dev_weight.as_slice(),
                shape: [v, h],
                stride: contiguous_stride([v, h]),
            },
            target: TensorRef {
                data: dev_target.as_slice(),
                shape: [bt],
                stride: contiguous_stride([bt]),
            },
            out: TensorMut {
                data: dev_out_2.as_slice_mut(),
                shape: [1],
                stride: [1],
            },
            grad_input: Some(TensorMut {
                data: dev_grad_input_2.as_slice_mut(),
                shape: [bt, h],
                stride: contiguous_stride([bt, h]),
            }),
            grad_weight: Some(TensorMut {
                data: dev_grad_weight_2.as_slice_mut(),
                shape: [v, h],
                stride: contiguous_stride([v, h]),
            }),
        },
    )
    .unwrap();
    let bw_desc = FusedLinearCrossEntropyBackwardDescriptor::new(bt, h, v, ElementKind::F32);
    let bw_plan = FusedLinearCrossEntropyBackwardPlan::<f32>::select(&stream, &bw_desc, PlanPreference::default()).unwrap();
    bw_plan.run(
        &stream,
        Workspace::None,
        FusedLinearCrossEntropyBackwardArgs {
            dy_scalar: 2.0,
            grad_input: Some(TensorMut {
                data: dev_grad_input_2.as_slice_mut(),
                shape: [bt, h],
                stride: contiguous_stride([bt, h]),
            }),
            grad_weight: Some(TensorMut {
                data: dev_grad_weight_2.as_slice_mut(),
                shape: [v, h],
                stride: contiguous_stride([v, h]),
            }),
        },
    )
    .unwrap();
    stream.synchronize().unwrap();

    let mut grad_input_2 = vec![0.0f32; (bt * h) as usize];
    let mut grad_weight_2 = vec![0.0f32; (v * h) as usize];
    dev_grad_input_2.copy_to_host(&mut grad_input_2).unwrap();
    dev_grad_weight_2.copy_to_host(&mut grad_weight_2).unwrap();

    // Verify grad_input_2 ≈ 2 · grad_input_1 within tight FP tol.
    for i in 0..grad_input_1.len() {
        let want = grad_input_1[i] * 2.0;
        let tol = want.abs() * 1e-5 + 1e-6;
        assert!(
            (grad_input_2[i] - want).abs() <= tol,
            "grad_input scaling [{}]: got={} want={}",
            i, grad_input_2[i], want
        );
    }
    for j in 0..grad_weight_1.len() {
        let want = grad_weight_1[j] * 2.0;
        let tol = want.abs() * 1e-5 + 1e-6;
        assert!(
            (grad_weight_2[j] - want).abs() <= tol,
            "grad_weight scaling [{}]: got={} want={}",
            j, grad_weight_2[j], want
        );
    }
}
