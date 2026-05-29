//! Phase 47 — Fused Linear Cross-Entropy FW smoke test.
//!
//! Verifies the FLCE plan against an unfused PyTorch-style reference:
//!   1. `logits = input @ weight.T`
//!   2. `loss = mean(CrossEntropy(logits, target))`
//!
//! Tested dtypes: f32 / f16 / bf16. f64 spot-check only (extreme
//! precision isn't the bottleneck in LLM training; the plan supports it
//! for API completeness).

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, FusedLinearCrossEntropyArgs,
    FusedLinearCrossEntropyDescriptor, FusedLinearCrossEntropyPlan, LossReduction,
    PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Host reference (f64) — `mean(-log_softmax(input @ weight.T)[target])`.
fn host_flce_mean_f64(
    input: &[f64], // [bt, h] row-major
    weight: &[f64], // [v, h] row-major
    target: &[i64],
    bt: usize,
    h: usize,
    v: usize,
    ignore_index: i64,
) -> f64 {
    let mut total = 0.0;
    let mut n_valid = 0usize;
    for i in 0..bt {
        let t = target[i];
        if t == ignore_index {
            continue;
        }
        if t < 0 || t >= v as i64 {
            continue;
        }
        // Compute logits[i, :] = input[i, :] @ weight.T
        let mut logits = vec![0.0f64; v];
        for j in 0..v {
            let mut acc = 0.0;
            for k in 0..h {
                acc += input[i * h + k] * weight[j * h + k];
            }
            logits[j] = acc;
        }
        // log_softmax
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
        n_valid += 1;
    }
    if n_valid == 0 {
        0.0
    } else {
        total / (n_valid as f64)
    }
}

#[test]
#[ignore]
fn flce_fw_f32_mean_small() {
    let (ctx, stream) = setup();
    let bt = 8i32;
    let h = 16i32;
    let v = 32i32;

    let mut host_input = vec![0.0f32; (bt * h) as usize];
    let mut host_weight = vec![0.0f32; (v * h) as usize];
    for i in 0..host_input.len() {
        host_input[i] = ((i as f32) * 0.013).sin() * 0.5;
    }
    for i in 0..host_weight.len() {
        host_weight[i] = (((i as f32) * 0.027) + 0.1).cos() * 0.3;
    }
    let host_target: Vec<i64> = vec![0, 5, 10, 15, 20, 25, 30, 3];

    let dev_input = DeviceBuffer::from_slice(&ctx, &host_input).unwrap();
    let dev_weight = DeviceBuffer::from_slice(&ctx, &host_weight).unwrap();
    let dev_target = DeviceBuffer::from_slice(&ctx, &host_target).unwrap();
    let mut dev_out: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).unwrap();

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
            grad_input: None,
            grad_weight: None,
        },
    )
    .unwrap();
    stream.synchronize().unwrap();

    let mut got = [0f32; 1];
    dev_out.copy_to_host(&mut got).unwrap();

    let expected = host_flce_mean_f64(
        &host_input.iter().map(|&x| x as f64).collect::<Vec<_>>(),
        &host_weight.iter().map(|&x| x as f64).collect::<Vec<_>>(),
        &host_target,
        bt as usize,
        h as usize,
        v as usize,
        -100,
    ) as f32;

    // Tolerance: K · eps · |loss| (K = h for the GEMM reduction).
    let tol = expected.abs() * (h as f32) * f32::EPSILON * 16.0 + 1e-4;
    assert!(
        (got[0] - expected).abs() <= tol,
        "f32 FLCE mean: got={} want={} tol={}",
        got[0],
        expected,
        tol
    );
}

#[test]
#[ignore]
fn flce_fw_f16_mean_small() {
    let (ctx, stream) = setup();
    let bt = 8i32;
    let h = 16i32;
    let v = 32i32;

    let mut host_input_f32 = vec![0.0f32; (bt * h) as usize];
    let mut host_weight_f32 = vec![0.0f32; (v * h) as usize];
    for i in 0..host_input_f32.len() {
        host_input_f32[i] = ((i as f32) * 0.013).sin() * 0.5;
    }
    for i in 0..host_weight_f32.len() {
        host_weight_f32[i] = (((i as f32) * 0.027) + 0.1).cos() * 0.3;
    }
    let host_target: Vec<i64> = vec![0, 5, 10, 15, 20, 25, 30, 3];

    let host_input: Vec<f16> = host_input_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_weight: Vec<f16> = host_weight_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let dev_input = DeviceBuffer::from_slice(&ctx, &host_input).unwrap();
    let dev_weight = DeviceBuffer::from_slice(&ctx, &host_weight).unwrap();
    let dev_target = DeviceBuffer::from_slice(&ctx, &host_target).unwrap();
    let mut dev_out: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, 1).unwrap();

    let desc = FusedLinearCrossEntropyDescriptor::new(bt, h, v, ElementKind::F16)
        .with_reduction(LossReduction::Mean);
    let plan = FusedLinearCrossEntropyPlan::<f16>::select(&stream, &desc, PlanPreference::default()).unwrap();
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
            grad_input: None,
            grad_weight: None,
        },
    )
    .unwrap();
    stream.synchronize().unwrap();

    let mut got = [f16::ZERO; 1];
    dev_out.copy_to_host(&mut got).unwrap();
    let g = got[0].to_f32();

    let expected = host_flce_mean_f64(
        &host_input_f32.iter().map(|&x| x as f64).collect::<Vec<_>>(),
        &host_weight_f32.iter().map(|&x| x as f64).collect::<Vec<_>>(),
        &host_target,
        bt as usize,
        h as usize,
        v as usize,
        -100,
    ) as f32;

    let tol = expected.abs() * 0.05 + 5e-2;
    assert!(
        (g - expected).abs() <= tol,
        "f16 FLCE mean: got={} want={} tol={}",
        g,
        expected,
        tol
    );
}

#[test]
#[ignore]
fn flce_fw_bf16_mean_small() {
    let (ctx, stream) = setup();
    let bt = 8i32;
    let h = 16i32;
    let v = 32i32;

    let mut host_input_f32 = vec![0.0f32; (bt * h) as usize];
    let mut host_weight_f32 = vec![0.0f32; (v * h) as usize];
    for i in 0..host_input_f32.len() {
        host_input_f32[i] = ((i as f32) * 0.013).sin() * 0.5;
    }
    for i in 0..host_weight_f32.len() {
        host_weight_f32[i] = (((i as f32) * 0.027) + 0.1).cos() * 0.3;
    }
    let host_target: Vec<i64> = vec![0, 5, 10, 15, 20, 25, 30, 3];

    let host_input: Vec<bf16> = host_input_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_weight: Vec<bf16> = host_weight_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let dev_input = DeviceBuffer::from_slice(&ctx, &host_input).unwrap();
    let dev_weight = DeviceBuffer::from_slice(&ctx, &host_weight).unwrap();
    let dev_target = DeviceBuffer::from_slice(&ctx, &host_target).unwrap();
    let mut dev_out: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, 1).unwrap();

    let desc = FusedLinearCrossEntropyDescriptor::new(bt, h, v, ElementKind::Bf16)
        .with_reduction(LossReduction::Mean);
    let plan = FusedLinearCrossEntropyPlan::<bf16>::select(&stream, &desc, PlanPreference::default()).unwrap();
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
            grad_input: None,
            grad_weight: None,
        },
    )
    .unwrap();
    stream.synchronize().unwrap();

    let mut got = [bf16::ZERO; 1];
    dev_out.copy_to_host(&mut got).unwrap();
    let g = got[0].to_f32();

    let expected = host_flce_mean_f64(
        &host_input_f32.iter().map(|&x| x as f64).collect::<Vec<_>>(),
        &host_weight_f32.iter().map(|&x| x as f64).collect::<Vec<_>>(),
        &host_target,
        bt as usize,
        h as usize,
        v as usize,
        -100,
    ) as f32;

    let tol = expected.abs() * 0.08 + 8e-2;
    assert!(
        (g - expected).abs() <= tol,
        "bf16 FLCE mean: got={} want={} tol={}",
        g,
        expected,
        tol
    );
}

#[test]
#[ignore]
fn flce_fw_f32_sum_with_ignore_index() {
    // Verifies the ignore_index path: half the targets are -100,
    // the other half are valid. Sum mode means no divide.
    let (ctx, stream) = setup();
    let bt = 8i32;
    let h = 16i32;
    let v = 32i32;

    let mut host_input = vec![0.0f32; (bt * h) as usize];
    let mut host_weight = vec![0.0f32; (v * h) as usize];
    for i in 0..host_input.len() {
        host_input[i] = ((i as f32) * 0.013).sin() * 0.5;
    }
    for i in 0..host_weight.len() {
        host_weight[i] = (((i as f32) * 0.027) + 0.1).cos() * 0.3;
    }
    // Half ignored.
    let host_target: Vec<i64> = vec![0, -100, 10, -100, 20, -100, 30, -100];

    let dev_input = DeviceBuffer::from_slice(&ctx, &host_input).unwrap();
    let dev_weight = DeviceBuffer::from_slice(&ctx, &host_weight).unwrap();
    let dev_target = DeviceBuffer::from_slice(&ctx, &host_target).unwrap();
    let mut dev_out: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).unwrap();

    let desc = FusedLinearCrossEntropyDescriptor::new(bt, h, v, ElementKind::F32)
        .with_reduction(LossReduction::Sum);
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
            grad_input: None,
            grad_weight: None,
        },
    )
    .unwrap();
    stream.synchronize().unwrap();

    let mut got = [0f32; 1];
    dev_out.copy_to_host(&mut got).unwrap();

    // Host reference: sum, not mean — multiply mean by n_valid.
    let host_mean = host_flce_mean_f64(
        &host_input.iter().map(|&x| x as f64).collect::<Vec<_>>(),
        &host_weight.iter().map(|&x| x as f64).collect::<Vec<_>>(),
        &host_target,
        bt as usize,
        h as usize,
        v as usize,
        -100,
    );
    let expected = (host_mean * 4.0) as f32; // 4 valid out of 8

    let tol = expected.abs() * (h as f32) * f32::EPSILON * 16.0 + 1e-4;
    assert!(
        (got[0] - expected).abs() <= tol,
        "f32 FLCE sum w/ ignore: got={} want={} tol={}",
        got[0],
        expected,
        tol
    );
}
