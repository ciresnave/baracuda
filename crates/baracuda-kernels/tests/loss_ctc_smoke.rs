//! Real-GPU smoke test for `CtcLossPlan<T>` — CTC forward via lattice DP.
//!
//! Verified against a hand-computed reference plus a CPU re-implementation
//! of the standard forward DP recurrence. Reference: PyTorch's
//! `torch.nn.CTCLoss` semantics with `reduction='none'`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, CtcLossArgs, CtcLossDescriptor, CtcLossPlan, ElementKind, LossReduction,
    PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

const NEG_INF: f32 = f32::NEG_INFINITY;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[inline]
fn lse2(a: f32, b: f32) -> f32 {
    if a == NEG_INF {
        return b;
    }
    if b == NEG_INF {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

#[inline]
fn lse3(a: f32, b: f32, c: f32) -> f32 {
    lse2(lse2(a, b), c)
}

/// CPU CTC FW reference. Returns per-sample loss (length `batch_size`).
fn host_ctc_fw_f32(
    log_probs: &[f32], // [T, N, C] row-major
    targets: &[i64],   // [N, S]
    input_lengths: &[i64],
    target_lengths: &[i64],
    t_max: usize,
    n: usize,
    c: usize,
    s_max: usize,
    blank: usize,
) -> Vec<f32> {
    let mut losses = vec![0f32; n];
    for sample in 0..n {
        let t_n = input_lengths[sample] as usize;
        let s_n = target_lengths[sample] as usize;
        let l = 2 * s_n + 1;
        // Build extended target.
        let mut ext = vec![blank; l];
        for k in 0..s_n {
            ext[2 * k + 1] = targets[sample * s_max + k] as usize;
        }
        // α[t, k]; allocate one t at a time.
        let mut prev = vec![NEG_INF; l];
        let lp_at = |t: usize, cls: usize| -> f32 {
            log_probs[t * n * c + sample * c + cls]
        };
        prev[0] = lp_at(0, ext[0]);
        if l >= 2 {
            prev[1] = lp_at(0, ext[1]);
        }
        let mut next = vec![NEG_INF; l];
        for t in 1..t_n {
            for k in 0..l {
                let self_a = prev[k];
                let prev_a = if k >= 1 { prev[k - 1] } else { NEG_INF };
                let skip_a = if k >= 2
                    && ext[k] != blank
                    && ext[k] != ext[k - 2]
                {
                    prev[k - 2]
                } else {
                    NEG_INF
                };
                next[k] = lse3(self_a, prev_a, skip_a) + lp_at(t, ext[k]);
            }
            std::mem::swap(&mut prev, &mut next);
            for v in next.iter_mut() {
                *v = NEG_INF;
            }
        }
        // Terminal: LSE of α[T-1, L-1] and α[T-1, L-2].
        let last1 = prev[l - 1];
        let last2 = if l >= 2 { prev[l - 2] } else { NEG_INF };
        losses[sample] = -lse2(last1, last2);
    }
    losses
}

fn run_ctc_fw<T>(
    ctx: &Context,
    stream: &Stream,
    t_max: i32,
    n: i32,
    c: i32,
    s_max: i32,
    blank: i32,
    reduction: LossReduction,
    elem: ElementKind,
    host_log_probs: Vec<T>,
    host_targets: Vec<i64>,
    host_in_lens: Vec<i64>,
    host_tgt_lens: Vec<i64>,
) -> Vec<T>
where
    T: baracuda_kernels::Element + Copy + Default + 'static,
{
    let lp_shape = [t_max, n, c];
    let tgt_shape = [n, s_max];
    let in_lens_shape = [n];
    let tgt_lens_shape = [n];
    let dev_lp = DeviceBuffer::from_slice(ctx, &host_log_probs).expect("up lp");
    let dev_tgt = DeviceBuffer::from_slice(ctx, &host_targets).expect("up tgt");
    let dev_in_lens = DeviceBuffer::from_slice(ctx, &host_in_lens).expect("up in");
    let dev_tgt_lens = DeviceBuffer::from_slice(ctx, &host_tgt_lens).expect("up tgt_lens");

    let loss_numel = match reduction {
        LossReduction::None => n as usize,
        _ => 1,
    };
    let mut dev_loss: DeviceBuffer<T> =
        DeviceBuffer::zeros(ctx, loss_numel).expect("alloc loss");

    let desc = CtcLossDescriptor {
        max_time: t_max,
        batch_size: n,
        num_classes: c,
        max_target_len: s_max,
        blank,
        reduction,
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
    plan.run(stream, ws, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![T::default(); loss_numel];
    dev_loss.copy_to_host(&mut got).expect("download");
    got
}

#[test]
#[ignore]
fn ctc_fw_f32_uniform_t2_c2() {
    // Hand-checked case: T=2, N=1, C=2 (blank=0, class 1), target=[1].
    // Uniform log_probs = -log(2). Expected loss = log(4/3) ≈ 0.28768.
    let (ctx, stream) = setup();
    let log_p = -((2.0f32).ln());
    let host_lp = vec![log_p; 2 * 1 * 2];
    let host_tgt = vec![1i64; 1];
    let host_in_lens = vec![2i64];
    let host_tgt_lens = vec![1i64];
    let got = run_ctc_fw::<f32>(
        &ctx,
        &stream,
        2,
        1,
        2,
        1,
        0,
        LossReduction::None,
        ElementKind::F32,
        host_lp,
        host_tgt,
        host_in_lens,
        host_tgt_lens,
    );
    let want = (4.0f32 / 3.0f32).ln();
    let diff = (got[0] - want).abs();
    assert!(diff <= 1e-5, "f32 ctc fw: got={} want={} diff={}", got[0], want, diff);
}

fn run_general_case<T>(
    ctx: &Context,
    stream: &Stream,
    elem: ElementKind,
    convert_to_t: impl Fn(f32) -> T,
    tol: f32,
) where
    T: baracuda_kernels::Element + Copy + Default + 'static,
    T: std::fmt::Debug,
{
    let t_max: i32 = 4;
    let n: i32 = 2;
    let c: i32 = 3;
    let s_max: i32 = 2;
    // log_probs: sample 0 mostly favors blank, sample 1 mostly favors class 2.
    let mut host_lp_f32: Vec<f32> = Vec::with_capacity((t_max * n * c) as usize);
    for t in 0..t_max {
        for sample in 0..n {
            // Build a small distribution and log-normalize.
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
    let host_lp: Vec<T> = host_lp_f32.iter().map(|&v| convert_to_t(v)).collect();
    // Targets: sample 0 = [1, 2], sample 1 = [2, 1].
    let host_tgt = vec![1i64, 2, 2, 1];
    let host_in_lens = vec![4i64, 4];
    let host_tgt_lens = vec![2i64, 2];
    let host_ref = host_ctc_fw_f32(
        &host_lp_f32,
        &host_tgt,
        &host_in_lens,
        &host_tgt_lens,
        t_max as usize,
        n as usize,
        c as usize,
        s_max as usize,
        0,
    );
    let got = run_ctc_fw::<T>(
        ctx,
        stream,
        t_max,
        n,
        c,
        s_max,
        0,
        LossReduction::None,
        elem,
        host_lp,
        host_tgt,
        host_in_lens,
        host_tgt_lens,
    );
    for sample in 0..n as usize {
        // Approximate compare via roundtrip through f32.
        let got_f32 = {
            let bytes = unsafe {
                core::slice::from_raw_parts(
                    (&got[sample] as *const T) as *const u8,
                    core::mem::size_of::<T>(),
                )
            };
            match core::mem::size_of::<T>() {
                4 => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                8 => f64::from_le_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                    bytes[7],
                ]) as f32,
                2 => {
                    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
                    if elem == ElementKind::F16 {
                        f16::from_bits(bits).to_f32()
                    } else {
                        bf16::from_bits(bits).to_f32()
                    }
                }
                _ => panic!("unexpected dtype size"),
            }
        };
        let want = host_ref[sample];
        let diff = (got_f32 - want).abs();
        assert!(
            diff <= tol,
            "ctc fw {elem:?} @ sample {sample}: got={got_f32} want={want} diff={diff}"
        );
    }
}

#[test]
#[ignore]
fn ctc_fw_f32_general() {
    let (ctx, stream) = setup();
    run_general_case::<f32>(&ctx, &stream, ElementKind::F32, |v| v, 1e-4);
}

#[test]
#[ignore]
fn ctc_fw_f64_general() {
    let (ctx, stream) = setup();
    run_general_case::<f64>(&ctx, &stream, ElementKind::F64, |v| v as f64, 1e-5);
}

#[test]
#[ignore]
fn ctc_fw_f16_general() {
    let (ctx, stream) = setup();
    run_general_case::<f16>(
        &ctx,
        &stream,
        ElementKind::F16,
        |v| f16::from_f32(v),
        // Loss is computed in f32 acc; only the input log_probs and the
        // final loss store undergo half-precision rounding. Two rounding
        // hops + accumulated DP error → generous bound.
        2e-2,
    );
}

#[test]
#[ignore]
fn ctc_fw_bf16_general() {
    let (ctx, stream) = setup();
    run_general_case::<bf16>(
        &ctx,
        &stream,
        ElementKind::Bf16,
        |v| bf16::from_f32(v),
        4e-2,
    );
}
