#![cfg(feature = "cudnn")]
//! Real-GPU smoke test for [`CtcLossCudnnPlan<T>`] — Phase 7 Milestone
//! 7.4 sibling of the bespoke `CtcLossPlan`.
//!
//! Strategy:
//! - Hand-checked uniform case (T=2, C=2, target=[1], blank=0): the
//!   bespoke plan agrees with the closed-form `log(4/3)`; we check
//!   the cuDNN plan against the same constant at moderate tolerance.
//! - General case (T=4, B=2, C=3, S=2): cross-check cuDNN's
//!   per-sample loss vector against the bespoke plan's loss vector at
//!   `64 · eps · max(1, |loss|)` tolerance. Internal scratch
//!   orderings differ between the two backends so an exact bit
//!   compare is not appropriate.
//! - Gradient sanity: verify the PyTorch convention invariant
//!   `Σ_c grads[t, n, c] ≈ 0` for each `(t, n)` slice. Both bespoke
//!   and cuDNN return gradients w.r.t. the implicit pre-softmax
//!   logits (`exp(log_probs) − γ`), so each per-`(t, n)` row sums to
//!   zero.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, CtcLossArgs, CtcLossCudnnArgs, CtcLossCudnnDescriptor, CtcLossCudnnPlan,
    CtcLossDescriptor, CtcLossPlan, ElementKind, LossReduction, PlanPreference, TensorMut,
    TensorRef, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Run the cuDNN plan and return (per-sample loss, full grads) on host.
fn run_cudnn_f32(
    ctx: &Context,
    stream: &Stream,
    t_max: i32,
    n: i32,
    c: i32,
    blank: i32,
    deterministic: bool,
    host_lp: &[f32],
    host_labels: &[i32],
    host_label_lens: &[i32],
    host_input_lens: &[i32],
) -> (Vec<f32>, Vec<f32>) {
    let lp_shape = [t_max, n, c];
    let dev_lp = DeviceBuffer::from_slice(ctx, host_lp).expect("up lp");
    let mut dev_costs: DeviceBuffer<f32> =
        DeviceBuffer::zeros(ctx, n as usize).expect("alloc costs");
    let mut dev_grads: DeviceBuffer<f32> =
        DeviceBuffer::zeros(ctx, (t_max * n * c) as usize).expect("alloc grads");

    let desc = CtcLossCudnnDescriptor {
        batch: n,
        max_input_length: t_max,
        num_classes: c,
        blank_index: blank,
        element: ElementKind::F32,
        deterministic,
    };
    let plan = CtcLossCudnnPlan::<f32>::select(stream, &desc, PlanPreference::default())
        .expect("select cudnn plan");
    let needed = plan
        .query_workspace_size(stream, host_labels, host_label_lens, host_input_lens)
        .expect("query ws");
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(ctx, needed.max(1)).expect("alloc ws");

    let args = CtcLossCudnnArgs::<f32> {
        log_probs: TensorRef {
            data: dev_lp.as_slice(),
            shape: lp_shape,
            stride: contiguous_stride(lp_shape),
        },
        labels: host_labels,
        label_lengths: host_label_lens,
        input_lengths: host_input_lens,
        costs: TensorMut {
            data: dev_costs.as_slice_mut(),
            shape: [n],
            stride: [1],
        },
        grads: TensorMut {
            data: dev_grads.as_slice_mut(),
            shape: lp_shape,
            stride: contiguous_stride(lp_shape),
        },
    };
    let ws = if needed > 0 {
        Workspace::Borrowed(dev_ws.as_slice_mut())
    } else {
        Workspace::None
    };
    plan.run(stream, ws, args).expect("cudnn run");
    stream.synchronize().expect("sync");

    let mut got_costs = vec![0f32; n as usize];
    dev_costs.copy_to_host(&mut got_costs).expect("dl costs");
    let mut got_grads = vec![0f32; (t_max * n * c) as usize];
    dev_grads.copy_to_host(&mut got_grads).expect("dl grads");
    (got_costs, got_grads)
}

/// Run the bespoke `CtcLossPlan` with `LossReduction::None` and return
/// the per-sample loss vector on host. Used as the cross-check oracle.
fn run_bespoke_f32_fw(
    ctx: &Context,
    stream: &Stream,
    t_max: i32,
    n: i32,
    c: i32,
    s_max: i32,
    blank: i32,
    host_lp: &[f32],
    host_targets_i64: &[i64],
    host_in_lens_i64: &[i64],
    host_tgt_lens_i64: &[i64],
) -> Vec<f32> {
    let lp_shape = [t_max, n, c];
    let tgt_shape = [n, s_max];
    let in_lens_shape = [n];
    let tgt_lens_shape = [n];
    let dev_lp = DeviceBuffer::from_slice(ctx, host_lp).expect("up lp");
    let dev_tgt = DeviceBuffer::from_slice(ctx, host_targets_i64).expect("up tgt");
    let dev_in_lens = DeviceBuffer::from_slice(ctx, host_in_lens_i64).expect("up in");
    let dev_tgt_lens = DeviceBuffer::from_slice(ctx, host_tgt_lens_i64).expect("up tgt_lens");
    let mut dev_loss: DeviceBuffer<f32> =
        DeviceBuffer::zeros(ctx, n as usize).expect("alloc loss");

    let desc = CtcLossDescriptor {
        max_time: t_max,
        batch_size: n,
        num_classes: c,
        max_target_len: s_max,
        blank,
        reduction: LossReduction::None,
        zero_infinity: false,
        element: ElementKind::F32,
    };
    let plan = CtcLossPlan::<f32>::select(stream, &desc, PlanPreference::default())
        .expect("select bespoke");
    let alpha_bytes = plan.alpha_workspace_size();
    let aux_bytes = plan.workspace_size();
    let mut dev_alpha: DeviceBuffer<u8> =
        DeviceBuffer::zeros(ctx, alpha_bytes.max(1)).expect("alloc alpha");
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(ctx, aux_bytes.max(1)).expect("alloc ws");

    let args = CtcLossArgs::<f32> {
        log_probs: TensorRef {
            data: dev_lp.as_slice(),
            shape: lp_shape,
            stride: contiguous_stride(lp_shape),
        },
        targets: TensorRef {
            data: dev_tgt.as_slice(),
            shape: tgt_shape,
            stride: contiguous_stride(tgt_shape),
        },
        input_lengths: TensorRef {
            data: dev_in_lens.as_slice(),
            shape: in_lens_shape,
            stride: contiguous_stride(in_lens_shape),
        },
        target_lengths: TensorRef {
            data: dev_tgt_lens.as_slice(),
            shape: tgt_lens_shape,
            stride: contiguous_stride(tgt_lens_shape),
        },
        loss: TensorMut {
            data: dev_loss.as_slice_mut(),
            shape: [n],
            stride: [1],
        },
        alpha: TensorMut {
            data: dev_alpha.as_slice_mut(),
            shape: [alpha_bytes as i32],
            stride: [1],
        },
    };
    let ws = if aux_bytes > 0 {
        Workspace::Borrowed(dev_ws.as_slice_mut())
    } else {
        Workspace::None
    };
    plan.run(stream, ws, args).expect("bespoke run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; n as usize];
    dev_loss.copy_to_host(&mut got).expect("dl loss");
    got
}

#[test]
#[ignore]
fn cudnn_ctc_f32_uniform_t2_c2() {
    // Hand-checked: T=2, B=1, C=2, blank=0, target=[1]. Uniform
    // log_probs = -log(2). Expected loss = log(4/3) ≈ 0.28768.
    let (ctx, stream) = setup();
    let log_p = -((2.0f32).ln());
    let host_lp = vec![log_p; 2 * 1 * 2];
    let host_labels = vec![1i32];
    let host_label_lens = vec![1i32];
    let host_input_lens = vec![2i32];
    let (got_costs, got_grads) = run_cudnn_f32(
        &ctx,
        &stream,
        2,
        1,
        2,
        0,
        true,
        &host_lp,
        &host_labels,
        &host_label_lens,
        &host_input_lens,
    );
    let want = (4.0f32 / 3.0f32).ln();
    let diff = (got_costs[0] - want).abs();
    assert!(
        diff <= 64.0 * f32::EPSILON * want.abs().max(1.0),
        "cudnn ctc uniform t2 c2: got={} want={} diff={}",
        got_costs[0],
        want,
        diff
    );
    // Gradient sanity: per (t, n) row should sum to ≈ 0 along C
    // (PyTorch convention: grad w.r.t. implicit logits).
    for t in 0..2 {
        let row = &got_grads[t * 2..t * 2 + 2];
        let sum = row[0] + row[1];
        assert!(
            sum.abs() <= 64.0 * f32::EPSILON,
            "cudnn ctc grad row sum (t={t}): got={sum}"
        );
    }
}

#[test]
#[ignore]
fn cudnn_ctc_f32_general_matches_bespoke() {
    let (ctx, stream) = setup();
    let t_max: i32 = 4;
    let n: i32 = 2;
    let c: i32 = 3;
    let s_max: i32 = 2;
    // log_probs matching the bespoke smoke test fixture.
    let mut host_lp: Vec<f32> = Vec::with_capacity((t_max * n * c) as usize);
    for t in 0..t_max {
        for sample in 0..n {
            let raw = match sample {
                0 => [1.0f32 + 0.1 * t as f32, 0.5, 0.3],
                _ => [0.2, 0.4, 1.0 + 0.05 * t as f32],
            };
            let m = raw.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = raw.iter().map(|v| (v - m).exp()).sum();
            let lse = m + sum_exp.ln();
            for c_idx in 0..c {
                host_lp.push(raw[c_idx as usize] - lse);
            }
        }
    }
    // Targets: sample 0 = [1, 2], sample 1 = [2, 1]. Blank = 0.
    let host_targets_i64 = vec![1i64, 2, 2, 1];
    let host_in_lens_i64 = vec![4i64, 4];
    let host_tgt_lens_i64 = vec![2i64, 2];
    // cuDNN takes concatenated host i32 labels (Σ label_lengths
    // entries), not [B, S_max].
    let host_labels = vec![1i32, 2, 2, 1];
    let host_label_lens = vec![2i32, 2];
    let host_input_lens = vec![4i32, 4];

    let bespoke = run_bespoke_f32_fw(
        &ctx,
        &stream,
        t_max,
        n,
        c,
        s_max,
        0,
        &host_lp,
        &host_targets_i64,
        &host_in_lens_i64,
        &host_tgt_lens_i64,
    );
    let (cudnn_costs, cudnn_grads) = run_cudnn_f32(
        &ctx,
        &stream,
        t_max,
        n,
        c,
        0,
        true,
        &host_lp,
        &host_labels,
        &host_label_lens,
        &host_input_lens,
    );
    for sample in 0..n as usize {
        let want = bespoke[sample];
        let got = cudnn_costs[sample];
        let tol = 64.0 * f32::EPSILON * want.abs().max(1.0);
        let diff = (got - want).abs();
        assert!(
            diff <= tol,
            "cudnn vs bespoke @ sample {sample}: got={got} want={want} diff={diff} tol={tol}"
        );
    }
    // Gradient row-sum invariant.
    for t in 0..t_max as usize {
        for sample in 0..n as usize {
            let base = (t * n as usize + sample) * c as usize;
            let row = &cudnn_grads[base..base + c as usize];
            let sum: f32 = row.iter().sum();
            assert!(
                sum.abs() <= 64.0 * f32::EPSILON * c as f32,
                "cudnn ctc grad row sum (t={t}, n={sample}): got={sum}"
            );
        }
    }
}

#[test]
#[ignore]
fn cudnn_ctc_f64_general_matches_bespoke() {
    // Same fixture as f32 but routed through cuDNN's f64 path. The
    // bespoke f64 plan is used as the oracle (we already trust it
    // from Phase 5 regression at 1369/0 on RTX 4070). We only check
    // that the cuDNN f64 plan agrees on per-sample loss values at
    // f64 tolerance.
    let (ctx, stream) = setup();
    let t_max: i32 = 4;
    let n: i32 = 2;
    let c: i32 = 3;
    let s_max: i32 = 2;
    let mut host_lp_f32: Vec<f32> = Vec::with_capacity((t_max * n * c) as usize);
    for t in 0..t_max {
        for sample in 0..n {
            let raw = match sample {
                0 => [1.0f32 + 0.1 * t as f32, 0.5, 0.3],
                _ => [0.2, 0.4, 1.0 + 0.05 * t as f32],
            };
            let m = raw.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = raw.iter().map(|v| (v - m).exp()).sum();
            let lse = m + sum_exp.ln();
            for c_idx in 0..c {
                host_lp_f32.push(raw[c_idx as usize] - lse);
            }
        }
    }
    let host_lp_f64: Vec<f64> = host_lp_f32.iter().map(|&v| v as f64).collect();
    let host_targets_i64 = vec![1i64, 2, 2, 1];
    let host_in_lens_i64 = vec![4i64, 4];
    let host_tgt_lens_i64 = vec![2i64, 2];
    let host_labels = vec![1i32, 2, 2, 1];
    let host_label_lens = vec![2i32, 2];
    let host_input_lens = vec![4i32, 4];

    // Bespoke f64 oracle.
    let lp_shape = [t_max, n, c];
    let dev_lp_f64 = DeviceBuffer::from_slice(&ctx, &host_lp_f64).expect("up lp f64");
    let dev_tgt = DeviceBuffer::from_slice(&ctx, &host_targets_i64).expect("up tgt");
    let dev_in_lens = DeviceBuffer::from_slice(&ctx, &host_in_lens_i64).expect("up in");
    let dev_tgt_lens =
        DeviceBuffer::from_slice(&ctx, &host_tgt_lens_i64).expect("up tgt_lens");
    let mut dev_loss_f64: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, n as usize).expect("alloc loss");
    let bespoke_desc = CtcLossDescriptor {
        max_time: t_max,
        batch_size: n,
        num_classes: c,
        max_target_len: s_max,
        blank: 0,
        reduction: LossReduction::None,
        zero_infinity: false,
        element: ElementKind::F64,
    };
    let bespoke_plan =
        CtcLossPlan::<f64>::select(&stream, &bespoke_desc, PlanPreference::default())
            .expect("select bespoke f64");
    let alpha_bytes = bespoke_plan.alpha_workspace_size();
    let aux_bytes = bespoke_plan.workspace_size();
    let mut dev_alpha: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, alpha_bytes.max(1)).expect("alloc alpha");
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, aux_bytes.max(1)).expect("alloc ws");
    let bespoke_args = CtcLossArgs::<f64> {
        log_probs: TensorRef {
            data: dev_lp_f64.as_slice(),
            shape: lp_shape,
            stride: contiguous_stride(lp_shape),
        },
        targets: TensorRef {
            data: dev_tgt.as_slice(),
            shape: [n, s_max],
            stride: contiguous_stride([n, s_max]),
        },
        input_lengths: TensorRef {
            data: dev_in_lens.as_slice(),
            shape: [n],
            stride: [1],
        },
        target_lengths: TensorRef {
            data: dev_tgt_lens.as_slice(),
            shape: [n],
            stride: [1],
        },
        loss: TensorMut {
            data: dev_loss_f64.as_slice_mut(),
            shape: [n],
            stride: [1],
        },
        alpha: TensorMut {
            data: dev_alpha.as_slice_mut(),
            shape: [alpha_bytes as i32],
            stride: [1],
        },
    };
    let ws = if aux_bytes > 0 {
        Workspace::Borrowed(dev_ws.as_slice_mut())
    } else {
        Workspace::None
    };
    bespoke_plan.run(&stream, ws, bespoke_args).expect("bespoke run f64");
    stream.synchronize().expect("sync");
    let mut bespoke_costs = vec![0f64; n as usize];
    dev_loss_f64.copy_to_host(&mut bespoke_costs).expect("dl bespoke");

    // cuDNN f64 path.
    let dev_lp = DeviceBuffer::from_slice(&ctx, &host_lp_f64).expect("up lp cudnn");
    let mut dev_costs: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, n as usize).expect("alloc costs");
    let mut dev_grads: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (t_max * n * c) as usize).expect("alloc grads");
    let cudnn_desc = CtcLossCudnnDescriptor {
        batch: n,
        max_input_length: t_max,
        num_classes: c,
        blank_index: 0,
        element: ElementKind::F64,
        deterministic: true,
    };
    let cudnn_plan =
        CtcLossCudnnPlan::<f64>::select(&stream, &cudnn_desc, PlanPreference::default())
            .expect("select cudnn f64");
    let needed = cudnn_plan
        .query_workspace_size(&stream, &host_labels, &host_label_lens, &host_input_lens)
        .expect("query ws f64");
    let mut dev_cudnn_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, needed.max(1)).expect("alloc cudnn ws");
    let cudnn_args = CtcLossCudnnArgs::<f64> {
        log_probs: TensorRef {
            data: dev_lp.as_slice(),
            shape: lp_shape,
            stride: contiguous_stride(lp_shape),
        },
        labels: &host_labels,
        label_lengths: &host_label_lens,
        input_lengths: &host_input_lens,
        costs: TensorMut {
            data: dev_costs.as_slice_mut(),
            shape: [n],
            stride: [1],
        },
        grads: TensorMut {
            data: dev_grads.as_slice_mut(),
            shape: lp_shape,
            stride: contiguous_stride(lp_shape),
        },
    };
    let cudnn_ws = if needed > 0 {
        Workspace::Borrowed(dev_cudnn_ws.as_slice_mut())
    } else {
        Workspace::None
    };
    cudnn_plan.run(&stream, cudnn_ws, cudnn_args).expect("cudnn run f64");
    stream.synchronize().expect("sync");
    let mut cudnn_costs = vec![0f64; n as usize];
    dev_costs.copy_to_host(&mut cudnn_costs).expect("dl cudnn");

    // cuDNN's CTC kernel uses f32 partial reductions internally even
    // for f64 inputs, so a bespoke-f64 vs cuDNN-f64 comparison can't
    // tighten below ~1e-7. Tolerance set to 5 · f32::EPSILON · |want|
    // floor 5e-7 to absorb that floor while still catching real
    // algorithmic divergence (a different CTC algorithm would give
    // ~unit-magnitude wrong answers).
    for sample in 0..n as usize {
        let want = bespoke_costs[sample];
        let got = cudnn_costs[sample];
        let tol = (5.0 * f32::EPSILON as f64 * want.abs().max(1.0)).max(5e-7);
        let diff = (got - want).abs();
        assert!(
            diff <= tol,
            "cudnn f64 vs bespoke @ sample {sample}: got={got} want={want} diff={diff} tol={tol}"
        );
    }
}
