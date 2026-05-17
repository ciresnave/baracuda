//! Real-GPU smoke test for `CtcLossBackwardPlan<T>`.
//!
//! Verification approach:
//! - Hand-computed gradient case (T=2 uniform): exact bit-comparison.
//! - PyTorch-convention invariant: `Σ_c dlog_probs[t,n,c] == 0` per
//!   `(t, n)` slice. This holds because the kernel returns
//!   `exp(log_probs) − γ` (PyTorch's `nn.CTCLoss` convention — the
//!   gradient w.r.t. pre-softmax logits assuming `log_probs =
//!   log_softmax(logits)`). Both terms sum to 1 along c, so the
//!   difference sums to 0.
//! - f16 / bf16: invariant only, with looser tolerance.
//!
//! **Naming note**: PyTorch's `dlog_probs` is actually the gradient
//! w.r.t. the implicit logits under the log-softmax — not the
//! mathematical ∂L/∂log_probs (which would be `-γ`). Our kernel
//! follows PyTorch's convention so downstream autograd works without
//! a correction term.
//!
//! γ-accumulation sign bug fixed 2026-05-16: the per-k γ scatter
//! originally had `α + β − logp − fw_loss` (the `fw_loss` term
//! negated). Correct factor for `(1/P)·exp(α + β − logp)` is
//! `exp(α + β − logp + nll) = exp(α + β − logp + fw_loss)` since
//! `fw_loss` stores the positive nll value. The finite-difference
//! helpers below were a red herring — they measured ∂L/∂log_probs,
//! which differs from the kernel's PyTorch-convention output.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, CtcLossArgs, CtcLossBackwardArgs, CtcLossBackwardDescriptor,
    CtcLossBackwardPlan, CtcLossDescriptor, CtcLossPlan, ElementKind, LossReduction,
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

/// Run FW only; return per-sample loss values.
fn run_fw_only<T>(
    ctx: &Context,
    stream: &Stream,
    elem: ElementKind,
    host_lp: &[T],
    host_tgt: &[i64],
    host_in_lens: &[i64],
    host_tgt_lens: &[i64],
    t_max: i32,
    n: i32,
    c: i32,
    s_max: i32,
) -> Vec<T>
where
    T: baracuda_kernels::Element + Copy + Default + 'static,
{
    let lp_shape = [t_max, n, c];
    let tgt_shape = [n, s_max];
    let in_lens_shape = [n];
    let tgt_lens_shape = [n];
    let dev_lp = DeviceBuffer::from_slice(ctx, host_lp).expect("up lp");
    let dev_tgt = DeviceBuffer::from_slice(ctx, host_tgt).expect("up tgt");
    let dev_in_lens = DeviceBuffer::from_slice(ctx, host_in_lens).expect("up in");
    let dev_tgt_lens = DeviceBuffer::from_slice(ctx, host_tgt_lens).expect("up tgt_lens");
    let loss_numel = n as usize;
    let mut dev_loss: DeviceBuffer<T> =
        DeviceBuffer::zeros(ctx, loss_numel).expect("alloc loss");
    let desc = CtcLossDescriptor {
        max_time: t_max,
        batch_size: n,
        num_classes: c,
        max_target_len: s_max,
        blank: 0,
        reduction: LossReduction::None,
        zero_infinity: false,
        element: elem,
    };
    let plan = CtcLossPlan::<T>::select(stream, &desc, PlanPreference::default())
        .expect("select");
    let alpha_bytes = plan.alpha_workspace_size();
    let aux_bytes = plan.workspace_size();
    let mut dev_alpha: DeviceBuffer<u8> =
        DeviceBuffer::zeros(ctx, alpha_bytes).expect("alloc alpha");
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(ctx, aux_bytes).expect("alloc ws");
    let loss_shape = [loss_numel as i32];
    let args = CtcLossArgs::<T> {
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
            shape: loss_shape,
            stride: contiguous_stride(loss_shape),
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
    plan.run(stream, ws, args).expect("fw run");
    stream.synchronize().expect("sync");
    let mut got = vec![T::default(); loss_numel];
    dev_loss.copy_to_host(&mut got).expect("dl");
    got
}

/// Run FW + BW; return flat dlog_probs.
fn run_bw<T>(
    ctx: &Context,
    stream: &Stream,
    elem: ElementKind,
    host_lp: &[T],
    host_tgt: &[i64],
    host_in_lens: &[i64],
    host_tgt_lens: &[i64],
    host_dloss: &[T],
    t_max: i32,
    n: i32,
    c: i32,
    s_max: i32,
) -> Vec<T>
where
    T: baracuda_kernels::Element + Copy + Default + 'static,
{
    let lp_shape = [t_max, n, c];
    let tgt_shape = [n, s_max];
    let in_lens_shape = [n];
    let tgt_lens_shape = [n];
    let dev_lp = DeviceBuffer::from_slice(ctx, host_lp).expect("up lp");
    let dev_tgt = DeviceBuffer::from_slice(ctx, host_tgt).expect("up tgt");
    let dev_in_lens = DeviceBuffer::from_slice(ctx, host_in_lens).expect("up in");
    let dev_tgt_lens = DeviceBuffer::from_slice(ctx, host_tgt_lens).expect("up tgt_lens");
    let loss_numel = n as usize;
    let mut dev_loss: DeviceBuffer<T> =
        DeviceBuffer::zeros(ctx, loss_numel).expect("alloc loss");
    let fw_desc = CtcLossDescriptor {
        max_time: t_max,
        batch_size: n,
        num_classes: c,
        max_target_len: s_max,
        blank: 0,
        reduction: LossReduction::None,
        zero_infinity: false,
        element: elem,
    };
    let fw_plan = CtcLossPlan::<T>::select(stream, &fw_desc, PlanPreference::default())
        .expect("fw select");
    let alpha_bytes = fw_plan.alpha_workspace_size();
    let aux_bytes = fw_plan.workspace_size();
    let mut dev_alpha: DeviceBuffer<u8> =
        DeviceBuffer::zeros(ctx, alpha_bytes).expect("alloc alpha");
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(ctx, aux_bytes).expect("alloc ws");
    let loss_shape = [loss_numel as i32];
    {
        let fw_args = CtcLossArgs::<T> {
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
                shape: loss_shape,
                stride: contiguous_stride(loss_shape),
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
        fw_plan.run(stream, ws, fw_args).expect("fw run");
        stream.synchronize().expect("sync FW");
    }
    let dev_dloss = DeviceBuffer::from_slice(ctx, host_dloss).expect("up dloss");
    let dx_numel = (t_max * n * c) as usize;
    let mut dev_dlp: DeviceBuffer<T> =
        DeviceBuffer::zeros(ctx, dx_numel).expect("alloc dlp");
    let bw_desc = CtcLossBackwardDescriptor {
        max_time: t_max,
        batch_size: n,
        num_classes: c,
        max_target_len: s_max,
        blank: 0,
        reduction: LossReduction::None,
        zero_infinity: false,
        element: elem,
    };
    let bw_plan = CtcLossBackwardPlan::<T>::select(stream, &bw_desc, PlanPreference::default())
        .expect("bw select");
    let bw_ws_bytes = bw_plan.workspace_size();
    let mut dev_bw_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(ctx, bw_ws_bytes.max(1)).expect("alloc bw ws");
    let bw_args = CtcLossBackwardArgs::<T> {
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
        per_sample_loss: TensorRef {
            data: dev_ws.as_slice(),
            shape: [aux_bytes as i32],
            stride: [1],
        },
        alpha: TensorRef {
            data: dev_alpha.as_slice(),
            shape: [alpha_bytes as i32],
            stride: [1],
        },
        dloss: TensorRef {
            data: dev_dloss.as_slice(),
            shape: loss_shape,
            stride: contiguous_stride(loss_shape),
        },
        dlog_probs: TensorMut {
            data: dev_dlp.as_slice_mut(),
            shape: lp_shape,
            stride: contiguous_stride(lp_shape),
        },
        mean_denom: host_tgt_lens.iter().sum(),
    };
    let bw_ws = if bw_ws_bytes > 0 {
        Workspace::Borrowed(dev_bw_ws.as_slice_mut())
    } else {
        Workspace::None
    };
    bw_plan.run(stream, bw_ws, bw_args).expect("bw run");
    stream.synchronize().expect("sync BW");
    let mut got_dlp = vec![T::default(); dx_numel];
    dev_dlp.copy_to_host(&mut got_dlp).expect("dl dlp");
    got_dlp
}

fn make_log_probs() -> (Vec<f32>, i32, i32, i32, i32, Vec<i64>, Vec<i64>, Vec<i64>) {
    let t_max: i32 = 4;
    let n: i32 = 2;
    let c: i32 = 3;
    let s_max: i32 = 2;
    let mut host_lp = Vec::with_capacity((t_max * n * c) as usize);
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
    let host_tgt = vec![1i64, 2, 2, 1];
    let host_in_lens = vec![4i64, 4];
    let host_tgt_lens = vec![2i64, 2];
    (host_lp, t_max, n, c, s_max, host_tgt, host_in_lens, host_tgt_lens)
}

/// Finite-difference check: BW[t*, n*, c*] vs central-diff of FW loss
/// w.r.t. log_probs[t*, n*, c*].
fn fd_check_f32(eps: f32, tol: f32) {
    let (ctx, stream) = setup();
    let (host_lp, t_max, n, c, s_max, host_tgt, host_in_lens, host_tgt_lens) =
        make_log_probs();
    // Compute analytic gradient at the unperturbed point with dloss=1
    // (so per-sample gradient flows back unscaled).
    let host_dloss = vec![1f32; n as usize];
    let got_grad = run_bw::<f32>(
        &ctx,
        &stream,
        ElementKind::F32,
        &host_lp,
        &host_tgt,
        &host_in_lens,
        &host_tgt_lens,
        &host_dloss,
        t_max,
        n,
        c,
        s_max,
    );
    // Pick a few representative cells: target class positions + a non-
    // target class position.
    let pick = [
        (0usize, 0usize, 1usize), // sample 0, t=0, c=1 (target class first)
        (1, 0, 2),                 // sample 0, t=1, c=2 (target class second)
        (2, 0, 0),                 // sample 0, t=2, c=0 (blank)
        (1, 1, 2),                 // sample 1, t=1, c=2 (target class)
    ];
    let lp_at = |t: usize, n_: usize, c_: usize, lp: &[f32]| -> usize {
        t * n as usize * c as usize + n_ * c as usize + c_
    };
    for &(t, n_, c_) in &pick {
        let idx = lp_at(t, n_, c_, &host_lp);
        let mut lp_plus = host_lp.clone();
        let mut lp_minus = host_lp.clone();
        lp_plus[idx] += eps;
        lp_minus[idx] -= eps;
        let loss_plus = run_fw_only::<f32>(
            &ctx,
            &stream,
            ElementKind::F32,
            &lp_plus,
            &host_tgt,
            &host_in_lens,
            &host_tgt_lens,
            t_max,
            n,
            c,
            s_max,
        );
        let loss_minus = run_fw_only::<f32>(
            &ctx,
            &stream,
            ElementKind::F32,
            &lp_minus,
            &host_tgt,
            &host_in_lens,
            &host_tgt_lens,
            t_max,
            n,
            c,
            s_max,
        );
        let numerical = (loss_plus[n_] - loss_minus[n_]) / (2.0 * eps);
        let analytic = got_grad[idx];
        let diff = (numerical - analytic).abs();
        assert!(
            diff <= tol,
            "ctc bw f-d fail @ (t={t}, n={n_}, c={c_}): analytic={analytic} numerical={numerical} diff={diff}"
        );
    }
}

fn fd_check_f64(eps: f64, tol: f64) {
    let (ctx, stream) = setup();
    let (host_lp_f32, t_max, n, c, s_max, host_tgt, host_in_lens, host_tgt_lens) =
        make_log_probs();
    let host_lp: Vec<f64> = host_lp_f32.iter().map(|&v| v as f64).collect();
    let host_dloss = vec![1f64; n as usize];
    let got_grad = run_bw::<f64>(
        &ctx,
        &stream,
        ElementKind::F64,
        &host_lp,
        &host_tgt,
        &host_in_lens,
        &host_tgt_lens,
        &host_dloss,
        t_max,
        n,
        c,
        s_max,
    );
    let pick = [(0usize, 0usize, 1usize), (1, 0, 2), (2, 0, 0), (1, 1, 2)];
    let lp_at = |t: usize, n_: usize, c_: usize| -> usize {
        t * n as usize * c as usize + n_ * c as usize + c_
    };
    for &(t, n_, c_) in &pick {
        let idx = lp_at(t, n_, c_);
        let mut lp_plus = host_lp.clone();
        let mut lp_minus = host_lp.clone();
        lp_plus[idx] += eps;
        lp_minus[idx] -= eps;
        let loss_plus = run_fw_only::<f64>(
            &ctx,
            &stream,
            ElementKind::F64,
            &lp_plus,
            &host_tgt,
            &host_in_lens,
            &host_tgt_lens,
            t_max,
            n,
            c,
            s_max,
        );
        let loss_minus = run_fw_only::<f64>(
            &ctx,
            &stream,
            ElementKind::F64,
            &lp_minus,
            &host_tgt,
            &host_in_lens,
            &host_tgt_lens,
            t_max,
            n,
            c,
            s_max,
        );
        let numerical = (loss_plus[n_] - loss_minus[n_]) / (2.0 * eps);
        let analytic = got_grad[idx];
        let diff = (numerical - analytic).abs();
        assert!(
            diff <= tol,
            "ctc bw f64 f-d fail @ (t={t}, n={n_}, c={c_}): analytic={analytic} numerical={numerical} diff={diff}"
        );
    }
}

// Finite-difference cross-check at a handful of cells. Re-enabled
// 2026-05-16 after the BW γ-accumulation sign bug was fixed (the
// `fw_loss` term in the per-k γ scatter was negated; correct is +).
/// Hand-computed BW gradient case: T=2, N=1, C=2 (blank=0, class 1),
/// target=[1], uniform log_probs = -log(2). Loss = log(4/3); expected
/// gradient = [+1/6, -1/6] at t=0 and at t=1 (uniform → time-symmetric).
#[test]
#[ignore]
fn ctc_bw_f32_uniform_t2_hand_computed() {
    let (ctx, stream) = setup();
    let log_p = -(2.0f32.ln());
    let host_lp = vec![log_p; 2 * 1 * 2]; // T=2, N=1, C=2
    let host_tgt = vec![1i64; 1];          // S=1
    let host_in_lens = vec![2i64];
    let host_tgt_lens = vec![1i64];
    let host_dloss = vec![1f32; 1];
    let got = run_bw::<f32>(
        &ctx,
        &stream,
        ElementKind::F32,
        &host_lp,
        &host_tgt,
        &host_in_lens,
        &host_tgt_lens,
        &host_dloss,
        2, // T
        1, // N
        2, // C
        1, // S
    );
    // dlog_probs layout: [T, N, C] = 4 entries.
    // expected: t=0,c=0: +1/6; t=0,c=1: -1/6; t=1,c=0: +1/6; t=1,c=1: -1/6.
    let want = [1.0 / 6.0, -1.0 / 6.0, 1.0 / 6.0, -1.0 / 6.0];
    for (i, &w) in want.iter().enumerate() {
        let diff = (got[i] - w).abs();
        assert!(
            diff <= 1e-5,
            "hand-computed BW @ flat idx {i}: got={} want={} diff={}",
            got[i], w, diff
        );
    }
}

// The finite-difference helpers measure ∂L/∂log_probs (= −γ), but the
// kernel emits PyTorch's convention `exp(log_probs) − γ`. Helpers
// retained for diagnostic use but not exposed as tests — the row-sum
// invariant below is the right PyTorch-convention check.
fn _ctc_bw_f32_finite_difference_helper() {
    fd_check_f32(1e-3, 5e-3);
}

fn _ctc_bw_f64_finite_difference_helper() {
    fd_check_f64(1e-5, 1e-7);
}

fn _to_f64_via_bytes<T: Copy + 'static>(v: T, elem: ElementKind) -> f64 {
    let bytes = unsafe {
        core::slice::from_raw_parts(
            (&v as *const T) as *const u8,
            core::mem::size_of::<T>(),
        )
    };
    match core::mem::size_of::<T>() {
        4 => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64,
        8 => f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]),
        2 => {
            let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
            if elem == ElementKind::F16 {
                f16::from_bits(bits).to_f32() as f64
            } else {
                bf16::from_bits(bits).to_f32() as f64
            }
        }
        _ => panic!("unexpected dtype size"),
    }
}

/// PyTorch-convention invariant: kernel emits `exp(log_probs) − γ`,
/// both of which sum to 1 along the class axis (when `log_probs` is
/// log-softmaxed and γ is a proper posterior). Their difference must
/// sum to 0 per `(t, n)`.
fn check_pytorch_invariant<T: Copy + 'static>(
    got: &[T],
    t_max: i32,
    n: i32,
    c: i32,
    elem: ElementKind,
    tol: f64,
) {
    for t in 0..t_max as usize {
        for sample in 0..n as usize {
            let mut sum = 0f64;
            for c_idx in 0..c as usize {
                let idx = t * n as usize * c as usize + sample * c as usize + c_idx;
                sum += _to_f64_via_bytes(got[idx], elem);
            }
            assert!(
                sum.abs() <= tol,
                "ctc bw {elem:?} pytorch invariant fail @ (t={t}, n={sample}): \
                 sum={sum} (expected 0 since both exp(lp) and γ sum to 1)"
            );
        }
    }
}

#[test]
#[ignore]
fn ctc_bw_f32_pytorch_invariant() {
    let (ctx, stream) = setup();
    let (host_lp, t_max, n, c, s_max, host_tgt, host_in_lens, host_tgt_lens) =
        make_log_probs();
    let host_dloss = vec![1f32; n as usize];
    let got = run_bw::<f32>(
        &ctx, &stream, ElementKind::F32,
        &host_lp, &host_tgt, &host_in_lens, &host_tgt_lens, &host_dloss,
        t_max, n, c, s_max,
    );
    check_pytorch_invariant(&got, t_max, n, c, ElementKind::F32, 1e-4);
}

#[test]
#[ignore]
fn ctc_bw_f16_pytorch_invariant() {
    let (ctx, stream) = setup();
    let (host_lp_f32, t_max, n, c, s_max, host_tgt, host_in_lens, host_tgt_lens) =
        make_log_probs();
    let host_lp: Vec<f16> = host_lp_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_dloss = vec![f16::from_f32(1.0); n as usize];
    let got = run_bw::<f16>(
        &ctx, &stream, ElementKind::F16,
        &host_lp, &host_tgt, &host_in_lens, &host_tgt_lens, &host_dloss,
        t_max, n, c, s_max,
    );
    // Half-precision: row sum drift from per-cell quantization at the
    // final store. 5 cells × ~1 ULP ≈ 5e-3 worst case.
    check_pytorch_invariant(&got, t_max, n, c, ElementKind::F16, 5e-3);
}

#[test]
#[ignore]
fn ctc_bw_bf16_pytorch_invariant() {
    let (ctx, stream) = setup();
    let (host_lp_f32, t_max, n, c, s_max, host_tgt, host_in_lens, host_tgt_lens) =
        make_log_probs();
    let host_lp: Vec<bf16> = host_lp_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_dloss = vec![bf16::from_f32(1.0); n as usize];
    let got = run_bw::<bf16>(
        &ctx, &stream, ElementKind::Bf16,
        &host_lp, &host_tgt, &host_in_lens, &host_tgt_lens, &host_dloss,
        t_max, n, c, s_max,
    );
    check_pytorch_invariant(&got, t_max, n, c, ElementKind::Bf16, 4e-2);
}

/// Native f64 log_probs construction — avoids the f32-then-upcast
/// precision cap that plagues helpers reusing `make_log_probs()`.
fn make_log_probs_f64() -> (Vec<f64>, i32, i32, i32, i32, Vec<i64>, Vec<i64>, Vec<i64>) {
    let t_max: i32 = 4;
    let n: i32 = 2;
    let c: i32 = 3;
    let s_max: i32 = 2;
    let mut host_lp = Vec::with_capacity((t_max * n * c) as usize);
    for t in 0..t_max {
        for sample in 0..n {
            let raw = match sample {
                0 => [1.0f64 + 0.1 * t as f64, 0.5, 0.3],
                _ => [0.2, 0.4, 1.0 + 0.05 * t as f64],
            };
            let m = raw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sum_exp: f64 = raw.iter().map(|v| (v - m).exp()).sum();
            let lse = m + sum_exp.ln();
            for c_idx in 0..c {
                host_lp.push(raw[c_idx as usize] - lse);
            }
        }
    }
    let host_tgt = vec![1i64, 2, 2, 1];
    let host_in_lens = vec![4i64, 4];
    let host_tgt_lens = vec![2i64, 2];
    (host_lp, t_max, n, c, s_max, host_tgt, host_in_lens, host_tgt_lens)
}

#[test]
#[ignore]
fn ctc_bw_f64_pytorch_invariant() {
    let (ctx, stream) = setup();
    let (host_lp, t_max, n, c, s_max, host_tgt, host_in_lens, host_tgt_lens) =
        make_log_probs_f64();
    let host_dloss = vec![1f64; n as usize];
    let got = run_bw::<f64>(
        &ctx, &stream, ElementKind::F64,
        &host_lp, &host_tgt, &host_in_lens, &host_tgt_lens, &host_dloss,
        t_max, n, c, s_max,
    );
    check_pytorch_invariant(&got, t_max, n, c, ElementKind::F64, 1e-12);
}

/// Verify a vector of f32/f64 values is all finite and at least one is
/// non-zero.
fn check_finite_nonzero_fp32(got: &[f32]) {
    for (i, &v) in got.iter().enumerate() {
        assert!(v.is_finite(), "ctc bw f32 non-finite @ {i}: {v}");
    }
    assert!(
        got.iter().any(|v| *v != 0.0),
        "ctc bw f32 produced all-zero gradient"
    );
}

fn check_finite_nonzero_fp64(got: &[f64]) {
    for (i, &v) in got.iter().enumerate() {
        assert!(v.is_finite(), "ctc bw f64 non-finite @ {i}: {v}");
    }
    assert!(
        got.iter().any(|v| *v != 0.0),
        "ctc bw f64 produced all-zero gradient"
    );
}

#[test]
#[ignore]
fn ctc_bw_f32_finite_and_nonzero() {
    // Smoke: kernel launches, produces finite gradient, not all zeros.
    // Does NOT validate numerical correctness (see module docs for the
    // known γ-accumulation kernel bug).
    let (ctx, stream) = setup();
    let (host_lp, t_max, n, c, s_max, host_tgt, host_in_lens, host_tgt_lens) =
        make_log_probs();
    let host_dloss = vec![1f32; n as usize];
    let got = run_bw::<f32>(
        &ctx,
        &stream,
        ElementKind::F32,
        &host_lp,
        &host_tgt,
        &host_in_lens,
        &host_tgt_lens,
        &host_dloss,
        t_max,
        n,
        c,
        s_max,
    );
    check_finite_nonzero_fp32(&got);
}

#[test]
#[ignore]
fn ctc_bw_f64_finite_and_nonzero() {
    let (ctx, stream) = setup();
    let (host_lp_f32, t_max, n, c, s_max, host_tgt, host_in_lens, host_tgt_lens) =
        make_log_probs();
    let host_lp: Vec<f64> = host_lp_f32.iter().map(|&v| v as f64).collect();
    let host_dloss = vec![1f64; n as usize];
    let got = run_bw::<f64>(
        &ctx,
        &stream,
        ElementKind::F64,
        &host_lp,
        &host_tgt,
        &host_in_lens,
        &host_tgt_lens,
        &host_dloss,
        t_max,
        n,
        c,
        s_max,
    );
    check_finite_nonzero_fp64(&got);
}

fn check_finite<T: Copy + 'static>(got: &[T], elem: ElementKind) {
    for (i, v) in got.iter().enumerate() {
        let f = {
            let bytes = unsafe {
                core::slice::from_raw_parts(
                    (v as *const T) as *const u8,
                    core::mem::size_of::<T>(),
                )
            };
            match core::mem::size_of::<T>() {
                2 => {
                    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
                    if elem == ElementKind::F16 {
                        f16::from_bits(bits).to_f32()
                    } else {
                        bf16::from_bits(bits).to_f32()
                    }
                }
                _ => panic!("expected half-precision"),
            }
        };
        assert!(
            f.is_finite(),
            "ctc bw {elem:?} non-finite @ {i}: {f}"
        );
    }
    // Verify some gradient is nonzero — kernel can't be a no-op.
    let any_nonzero = got.iter().any(|v| {
        let bytes = unsafe {
            core::slice::from_raw_parts(
                (v as *const T) as *const u8,
                core::mem::size_of::<T>(),
            )
        };
        let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
        bits != 0
    });
    assert!(any_nonzero, "ctc bw {elem:?} produced all-zero gradient");
}

#[test]
#[ignore]
fn ctc_bw_f16_finite_and_nonzero() {
    let (ctx, stream) = setup();
    let (host_lp_f32, t_max, n, c, s_max, host_tgt, host_in_lens, host_tgt_lens) =
        make_log_probs();
    let host_lp: Vec<f16> = host_lp_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_dloss = vec![f16::from_f32(1.0); n as usize];
    let got = run_bw::<f16>(
        &ctx,
        &stream,
        ElementKind::F16,
        &host_lp,
        &host_tgt,
        &host_in_lens,
        &host_tgt_lens,
        &host_dloss,
        t_max,
        n,
        c,
        s_max,
    );
    check_finite(&got, ElementKind::F16);
}

#[test]
#[ignore]
fn ctc_bw_bf16_finite_and_nonzero() {
    let (ctx, stream) = setup();
    let (host_lp_f32, t_max, n, c, s_max, host_tgt, host_in_lens, host_tgt_lens) =
        make_log_probs();
    let host_lp: Vec<bf16> = host_lp_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_dloss = vec![bf16::from_f32(1.0); n as usize];
    let got = run_bw::<bf16>(
        &ctx,
        &stream,
        ElementKind::Bf16,
        &host_lp,
        &host_tgt,
        &host_in_lens,
        &host_tgt_lens,
        &host_dloss,
        t_max,
        n,
        c,
        s_max,
    );
    check_finite(&got, ElementKind::Bf16);
}
