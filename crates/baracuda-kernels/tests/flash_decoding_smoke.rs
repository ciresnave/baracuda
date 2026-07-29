//! Real-GPU smoke test for `FlashDecodingPlan` FW (Phase 73 follow-up).
//!
//! Validates the split-K decode kernel against a CPU fp32 reference.
//! Covers f16 + bf16 at several (B, H, K_len, D) shapes including:
//!   - The minimum nontrivial case (B=1, H=1, K=64, D=32).
//!   - A two-split case (K_len = 300 > CHUNK_K = 256).
//!   - The LLM-decode-shaped case (B=1, H=32, K=2048, D=128).
//!
//! `#[ignore]` by default — requires a real CUDA device.

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    ElementKind, FlashDecodingArgs, FlashDecodingDescriptor, FlashDecodingPlan, PlanPreference,
    TensorMut, TensorRef, Workspace, contiguous_stride,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// CPU fp32 reference for SDPA at seq_q=1 with optional GQA.
///
/// Q: [B, H_q, D], K/V: [B, H_kv, K_len, D]. For pure MHA pass
/// `h_kv == h_q`. For GQA: Q-head `q` reads K/V-head `q / group_size`.
fn sdpa_decode_cpu(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    b: usize,
    h_q: usize,
    h_kv: usize,
    k_len: usize,
    d: usize,
    scale: f32,
) -> (Vec<f32>, Vec<f32>) {
    assert!(h_q % h_kv == 0);
    let group_size = h_q / h_kv;
    let mut y = vec![0.0_f32; b * h_q * d];
    // Per-key attention weights a[b, h_q, k_len] = softmax(q·kᵀ·scale) over the
    // keys, for the single decode query — the expected FlashDecoding `a` output.
    let mut a = vec![0.0_f32; b * h_q * k_len];
    for bi in 0..b {
        for hi in 0..h_q {
            let h_k_idx = hi / group_size;
            // Scores: s[ki] = (Q[bi, hi] · K[bi, h_k_idx, ki]) * scale.
            let mut scores = vec![0.0_f32; k_len];
            for ki in 0..k_len {
                let q_off = (bi * h_q + hi) * d;
                let k_off = ((bi * h_kv + h_k_idx) * k_len + ki) * d;
                let mut dot = 0.0_f32;
                for di in 0..d {
                    dot += q[q_off + di] * k[k_off + di];
                }
                scores[ki] = dot * scale;
            }
            // Softmax across k.
            let mut max_s = f32::NEG_INFINITY;
            for &s in &scores {
                if s > max_s {
                    max_s = s;
                }
            }
            let mut sum = 0.0_f32;
            for s in &mut scores {
                *s = (*s - max_s).exp();
                sum += *s;
            }
            let inv = 1.0 / sum;
            for s in &mut scores {
                *s *= inv;
            }
            // `scores` is now the normalized per-key weight vector — this IS the
            // expected `a[bi, hi, :]`.
            let a_off = (bi * h_q + hi) * k_len;
            a[a_off..a_off + k_len].copy_from_slice(&scores);
            // Y[bi, hi] = Σ_ki scores[ki] * V[bi, h_k_idx, ki].
            let y_off = (bi * h_q + hi) * d;
            for di in 0..d {
                let mut acc = 0.0_f32;
                for ki in 0..k_len {
                    let v_off = ((bi * h_kv + h_k_idx) * k_len + ki) * d;
                    acc += scores[ki] * v[v_off + di];
                }
                y[y_off + di] = acc;
            }
        }
    }
    (y, a)
}

/// Compare an f32 device output against an f32 reference with a relative
/// tolerance and an absolute floor (attention weights are ≤1 and near 0 for
/// masked-out keys, so the floor matters).
fn assert_close_f32(actual: &[f32], expected: &[f32], tol: f32, abs_floor: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "len mismatch in {label}");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        let rel_bound = tol * e.abs().max(abs_floor);
        if diff > rel_bound {
            panic!(
                "{label}: idx={i} actual={a:.6e} expected={e:.6e} \
                 abs_diff={diff:.6e} bound={rel_bound:.6e}",
            );
        }
    }
}

fn deterministic_f32(n: usize, seed_a: f32, seed_b: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as f32) * seed_a + seed_b;
            x.sin() * 0.3
        })
        .collect()
}

fn assert_close_f16(actual: &[f16], expected: &[f32], tol: f32, label: &str) {
    assert_close_f16_floor(actual, expected, tol, 1e-3, label);
}

fn assert_close_f16_floor(actual: &[f16], expected: &[f32], tol: f32, abs_floor: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "len mismatch in {label}");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let a_f = a.to_f32();
        let diff = (a_f - e).abs();
        let rel_bound = tol * e.abs().max(abs_floor);
        if diff > rel_bound {
            panic!(
                "{label}: idx={i} actual={a_f:.6e} expected={e:.6e} \
                 abs_diff={diff:.6e} bound={rel_bound:.6e}",
            );
        }
    }
}

fn assert_close_bf16(actual: &[bf16], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "len mismatch in {label}");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let a_f = a.to_f32();
        let diff = (a_f - e).abs();
        let rel_bound = tol * e.abs().max(1e-3);
        if diff > rel_bound {
            panic!(
                "{label}: idx={i} actual={a_f:.6e} expected={e:.6e} \
                 abs_diff={diff:.6e} bound={rel_bound:.6e}",
            );
        }
    }
}

fn run_case_f16(b: i32, h: i32, k_len: i32, d: i32, tol: f32, label: &str) {
    let (ctx, stream) = setup();
    let scale = 1.0_f32 / (d as f32).sqrt();

    let q_f32 = deterministic_f32((b * h * d) as usize, 0.013, -0.5);
    let k_f32 = deterministic_f32((b * h * k_len * d) as usize, 0.017, 0.2);
    let v_f32 = deterministic_f32((b * h * k_len * d) as usize, 0.011, -0.1);

    let (expected, a_expected) = sdpa_decode_cpu(
        &q_f32,
        &k_f32,
        &v_f32,
        b as usize,
        h as usize,
        h as usize,
        k_len as usize,
        d as usize,
        scale,
    );

    let q_h: Vec<f16> = q_f32.iter().map(|&x| f16::from_f32(x)).collect();
    let k_h: Vec<f16> = k_f32.iter().map(|&x| f16::from_f32(x)).collect();
    let v_h: Vec<f16> = v_f32.iter().map(|&x| f16::from_f32(x)).collect();

    let dq = DeviceBuffer::from_slice(&ctx, &q_h).expect("up q");
    let dk = DeviceBuffer::from_slice(&ctx, &k_h).expect("up k");
    let dv = DeviceBuffer::from_slice(&ctx, &v_h).expect("up v");
    let mut dy: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (b * h * d) as usize).expect("alloc y");

    let desc = FlashDecodingDescriptor::new(b, h, k_len, d, ElementKind::F16);
    let plan = FlashDecodingPlan::<f16>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let mut ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, plan.workspace_size()).expect("alloc workspace");

    let sq = [b, h, d];
    let sk = [b, h, k_len, d];
    let sv = [b, h, k_len, d];
    let sy = [b, h, d];

    let mut da: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (sy[0] * sy[1] * k_len) as usize).expect("alloc a");

    let args = FlashDecodingArgs::<f16> {
        q: TensorRef {
            data: dq.as_slice(),
            shape: sq,
            stride: contiguous_stride(sq),
        },
        k: TensorRef {
            data: dk.as_slice(),
            shape: sk,
            stride: contiguous_stride(sk),
        },
        v: TensorRef {
            data: dv.as_slice(),
            shape: sv,
            stride: contiguous_stride(sv),
        },
        y: TensorMut {
            data: dy.as_slice_mut(),
            shape: sy,
            stride: contiguous_stride(sy),
        },
        a: Some(TensorMut {
            data: da.as_slice_mut(),
            shape: [sy[0], sy[1], k_len],
            stride: contiguous_stride([sy[0], sy[1], k_len]),
        }),
    };
    plan.run(&stream, Workspace::Borrowed(ws.as_slice_mut()), args)
        .expect("run");
    stream.synchronize().expect("sync");

    // Verify the per-key attention-weight output against the CPU reference.
    let mut a_host = vec![0.0_f32; (sy[0] * sy[1] * k_len) as usize];
    da.copy_to_host(&mut a_host).expect("dl a");
    assert_close_f32(&a_host, &a_expected, tol, 1e-3, label);

    let mut y_host = vec![f16::ZERO; (b * h * d) as usize];
    dy.copy_to_host(&mut y_host).expect("dl y");

    assert_close_f16(&y_host, &expected, tol, label);
}

fn run_case_bf16(b: i32, h: i32, k_len: i32, d: i32, tol: f32, label: &str) {
    let (ctx, stream) = setup();
    let scale = 1.0_f32 / (d as f32).sqrt();

    let q_f32 = deterministic_f32((b * h * d) as usize, 0.013, -0.5);
    let k_f32 = deterministic_f32((b * h * k_len * d) as usize, 0.017, 0.2);
    let v_f32 = deterministic_f32((b * h * k_len * d) as usize, 0.011, -0.1);

    let (expected, a_expected) = sdpa_decode_cpu(
        &q_f32,
        &k_f32,
        &v_f32,
        b as usize,
        h as usize,
        h as usize,
        k_len as usize,
        d as usize,
        scale,
    );

    let q_h: Vec<bf16> = q_f32.iter().map(|&x| bf16::from_f32(x)).collect();
    let k_h: Vec<bf16> = k_f32.iter().map(|&x| bf16::from_f32(x)).collect();
    let v_h: Vec<bf16> = v_f32.iter().map(|&x| bf16::from_f32(x)).collect();

    let dq = DeviceBuffer::from_slice(&ctx, &q_h).expect("up q");
    let dk = DeviceBuffer::from_slice(&ctx, &k_h).expect("up k");
    let dv = DeviceBuffer::from_slice(&ctx, &v_h).expect("up v");
    let mut dy: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, (b * h * d) as usize).expect("alloc y");

    let desc = FlashDecodingDescriptor::new(b, h, k_len, d, ElementKind::Bf16);
    let plan = FlashDecodingPlan::<bf16>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let mut ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, plan.workspace_size()).expect("alloc workspace");

    let sq = [b, h, d];
    let sk = [b, h, k_len, d];
    let sv = [b, h, k_len, d];
    let sy = [b, h, d];

    let mut da: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (sy[0] * sy[1] * k_len) as usize).expect("alloc a");

    let args = FlashDecodingArgs::<bf16> {
        q: TensorRef {
            data: dq.as_slice(),
            shape: sq,
            stride: contiguous_stride(sq),
        },
        k: TensorRef {
            data: dk.as_slice(),
            shape: sk,
            stride: contiguous_stride(sk),
        },
        v: TensorRef {
            data: dv.as_slice(),
            shape: sv,
            stride: contiguous_stride(sv),
        },
        y: TensorMut {
            data: dy.as_slice_mut(),
            shape: sy,
            stride: contiguous_stride(sy),
        },
        a: Some(TensorMut {
            data: da.as_slice_mut(),
            shape: [sy[0], sy[1], k_len],
            stride: contiguous_stride([sy[0], sy[1], k_len]),
        }),
    };
    plan.run(&stream, Workspace::Borrowed(ws.as_slice_mut()), args)
        .expect("run");
    stream.synchronize().expect("sync");

    // Verify the per-key attention-weight output against the CPU reference.
    let mut a_host = vec![0.0_f32; (sy[0] * sy[1] * k_len) as usize];
    da.copy_to_host(&mut a_host).expect("dl a");
    assert_close_f32(&a_host, &a_expected, tol, 1e-3, label);

    let mut y_host = vec![bf16::ZERO; (b * h * d) as usize];
    dy.copy_to_host(&mut y_host).expect("dl y");

    assert_close_bf16(&y_host, &expected, tol, label);
}

#[ignore]
#[test]
fn flash_decoding_f16_single_split() {
    // One split (K_len ≤ 256).
    run_case_f16(1, 1, 64, 32, 5e-2, "f16/1×1×64×32");
}

#[ignore]
#[test]
fn flash_decoding_f16_multi_split() {
    // Two splits (K_len = 300 > 256). Tail handling matters.
    run_case_f16(1, 2, 300, 64, 5e-2, "f16/1×2×300×64");
}

#[ignore]
#[test]
fn flash_decoding_f16_llm_decode() {
    // The bench-typical shape.
    run_case_f16(1, 32, 2048, 128, 7e-2, "f16/1×32×2048×128");
}

#[ignore]
#[test]
fn flash_decoding_bf16_single_split() {
    run_case_bf16(1, 1, 64, 32, 1e-1, "bf16/1×1×64×32");
}

#[ignore]
#[test]
fn flash_decoding_bf16_multi_split() {
    run_case_bf16(1, 2, 300, 64, 1e-1, "bf16/1×2×300×64");
}

#[ignore]
#[test]
fn flash_decoding_bf16_llm_decode() {
    run_case_bf16(1, 32, 2048, 128, 1.5e-1, "bf16/1×32×2048×128");
}

// ----------------------------------------------------------------------------
// GQA / MQA cases — Tier-2 WMMA path.
//
// The dispatch chooses the WMMA kernel when `group_size >= 4` and
// `head_dim % 16 == 0`. These cases exercise it:
//   - Llama-3-8B class: H_q=32, H_kv=8, group=4
//   - Llama-3-70B class: H_q=64, H_kv=8, group=8
//   - MQA: H_q=32, H_kv=2, group=16 (caps at WMMA M-tile = 16)
// ----------------------------------------------------------------------------

fn run_gqa_case_f16(b: i32, h_q: i32, h_kv: i32, k_len: i32, d: i32, tol: f32, label: &str) {
    let (ctx, stream) = setup();
    let scale = 1.0_f32 / (d as f32).sqrt();

    let q_f32 = deterministic_f32((b * h_q * d) as usize, 0.013, -0.5);
    let k_f32 = deterministic_f32((b * h_kv * k_len * d) as usize, 0.017, 0.2);
    let v_f32 = deterministic_f32((b * h_kv * k_len * d) as usize, 0.011, -0.1);

    let (expected, a_expected) = sdpa_decode_cpu(
        &q_f32,
        &k_f32,
        &v_f32,
        b as usize,
        h_q as usize,
        h_kv as usize,
        k_len as usize,
        d as usize,
        scale,
    );

    let q_h: Vec<f16> = q_f32.iter().map(|&x| f16::from_f32(x)).collect();
    let k_h: Vec<f16> = k_f32.iter().map(|&x| f16::from_f32(x)).collect();
    let v_h: Vec<f16> = v_f32.iter().map(|&x| f16::from_f32(x)).collect();

    let dq = DeviceBuffer::from_slice(&ctx, &q_h).expect("up q");
    let dk = DeviceBuffer::from_slice(&ctx, &k_h).expect("up k");
    let dv = DeviceBuffer::from_slice(&ctx, &v_h).expect("up v");
    let mut dy: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (b * h_q * d) as usize).expect("alloc y");

    let desc = FlashDecodingDescriptor::new_gqa(b, h_q, h_kv, k_len, d, ElementKind::F16);
    let plan = FlashDecodingPlan::<f16>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let mut ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, plan.workspace_size()).expect("alloc workspace");

    let sq = [b, h_q, d];
    let sk = [b, h_kv, k_len, d];
    let sv = [b, h_kv, k_len, d];
    let sy = [b, h_q, d];

    let mut da: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (sy[0] * sy[1] * k_len) as usize).expect("alloc a");

    let args = FlashDecodingArgs::<f16> {
        q: TensorRef {
            data: dq.as_slice(),
            shape: sq,
            stride: contiguous_stride(sq),
        },
        k: TensorRef {
            data: dk.as_slice(),
            shape: sk,
            stride: contiguous_stride(sk),
        },
        v: TensorRef {
            data: dv.as_slice(),
            shape: sv,
            stride: contiguous_stride(sv),
        },
        y: TensorMut {
            data: dy.as_slice_mut(),
            shape: sy,
            stride: contiguous_stride(sy),
        },
        a: Some(TensorMut {
            data: da.as_slice_mut(),
            shape: [sy[0], sy[1], k_len],
            stride: contiguous_stride([sy[0], sy[1], k_len]),
        }),
    };
    plan.run(&stream, Workspace::Borrowed(ws.as_slice_mut()), args)
        .expect("run");
    stream.synchronize().expect("sync");

    // Verify the per-key attention-weight output against the CPU reference.
    let mut a_host = vec![0.0_f32; (sy[0] * sy[1] * k_len) as usize];
    da.copy_to_host(&mut a_host).expect("dl a");
    assert_close_f32(&a_host, &a_expected, tol, 1e-3, label);

    let mut y_host = vec![f16::ZERO; (b * h_q * d) as usize];
    dy.copy_to_host(&mut y_host).expect("dl y");

    // GQA TC path: bump the per-cell absolute floor from 1e-3 to 3e-3
    // because the WMMA path adds an fp32→fp16→fp32 round-trip
    // (sScores fp32 → sP fp16 → mma → sO fp32) that the MHA SIMT
    // path doesn't have. Near-zero cells (~1e-4 expected magnitude)
    // pick up noise of similar magnitude through that round-trip.
    assert_close_f16_floor(&y_host, &expected, tol, 3e-3, label);
}

// GQA tolerance is looser than MHA because the WMMA path adds a
// fp32→fp16→fp32 round-trip in the PV step (sScores stored fp32,
// converted to fp16 sP before the mma, accumulated back to fp32 sO).
// The MHA SIMT path keeps everything in fp32 from sQ load to sO write,
// so it's tighter. ~1.5× of the MHA tolerance is empirically enough
// across the bench-shape sweep.
const GQA_TC_TOL: f32 = 1.5e-1;

#[ignore]
#[test]
fn flash_decoding_gqa_llama3_8b() {
    // Llama 3 8B class — H_q=32, H_kv=8, group=4.
    run_gqa_case_f16(
        1,
        32,
        8,
        2048,
        128,
        GQA_TC_TOL,
        "f16/gqa-group4 (Llama-3-8B)",
    );
}

#[ignore]
#[test]
fn flash_decoding_gqa_llama3_70b() {
    // Llama 3 70B class — H_q=64, H_kv=8, group=8.
    run_gqa_case_f16(
        1,
        64,
        8,
        2048,
        128,
        GQA_TC_TOL,
        "f16/gqa-group8 (Llama-3-70B)",
    );
}

#[ignore]
#[test]
fn flash_decoding_gqa_mqa_full() {
    // MQA — H_q=16, H_kv=1, group=16 (caps at WMMA M-tile).
    run_gqa_case_f16(1, 16, 1, 2048, 128, GQA_TC_TOL, "f16/mqa-group16");
}

#[ignore]
#[test]
fn flash_decoding_gqa_small_shapes() {
    // Sanity: small shapes through the GQA TC path.
    run_gqa_case_f16(2, 8, 2, 300, 64, 1e-1, "f16/gqa-group4-multisplit");
}

/// Cost of the OPTIONAL per-key attention-weight output: `a=None` (standard
/// decode) vs `a=Some` (H2O/R-KV), amortized end-to-end wall-clock over the
/// decode K sweep. This is the real per-step cost the consumer pays (launch +
/// GPU). `#[ignore]` — run with `--ignored --nocapture`.
#[ignore]
#[test]
fn flash_decoding_a_cost_sweep_f16() {
    use std::time::Instant;
    const B: i32 = 1;
    const H: i32 = 32;
    const D: i32 = 128;
    const WARMUP: usize = 30;
    const ITERS: usize = 400;

    let (ctx, stream) = setup();
    println!(
        "\nflash_decoding a-cost (f16, B=1 H=32 D=128; amortized wall-clock, best of 3 x {ITERS}):"
    );
    for &k_len in &[1024_i32, 2048, 4096, 8192] {
        let q_f32 = deterministic_f32((B * H * D) as usize, 0.013, -0.5);
        let kv_f32 = deterministic_f32((B * H * k_len * D) as usize, 0.017, 0.2);
        let q_h: Vec<f16> = q_f32.iter().map(|&x| f16::from_f32(x)).collect();
        let kv_h: Vec<f16> = kv_f32.iter().map(|&x| f16::from_f32(x)).collect();
        let dq = DeviceBuffer::from_slice(&ctx, &q_h).expect("q");
        let dk = DeviceBuffer::from_slice(&ctx, &kv_h).expect("k");
        let dv = DeviceBuffer::from_slice(&ctx, &kv_h).expect("v");
        let mut dy: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, (B * H * D) as usize).expect("y");
        let mut da: DeviceBuffer<f32> =
            DeviceBuffer::zeros(&ctx, (B * H * k_len) as usize).expect("a");

        let desc = FlashDecodingDescriptor::new(B, H, k_len, D, ElementKind::F16);
        let plan = FlashDecodingPlan::<f16>::select(&stream, &desc, PlanPreference::default())
            .expect("select");
        let mut ws: DeviceBuffer<u8> =
            DeviceBuffer::zeros(&ctx, plan.workspace_size()).expect("ws");

        let sq = [B, H, D];
        let sk = [B, H, k_len, D];
        let sv = sk;
        let sy = sq;
        let sa = [B, H, k_len];

        let mut result = [0.0_f64; 2];
        for (idx, &with_a) in [false, true].iter().enumerate() {
            for _ in 0..WARMUP {
                let a = if with_a {
                    Some(TensorMut {
                        data: da.as_slice_mut(),
                        shape: sa,
                        stride: contiguous_stride(sa),
                    })
                } else {
                    None
                };
                let args = FlashDecodingArgs::<f16> {
                    a,
                    q: TensorRef {
                        data: dq.as_slice(),
                        shape: sq,
                        stride: contiguous_stride(sq),
                    },
                    k: TensorRef {
                        data: dk.as_slice(),
                        shape: sk,
                        stride: contiguous_stride(sk),
                    },
                    v: TensorRef {
                        data: dv.as_slice(),
                        shape: sv,
                        stride: contiguous_stride(sv),
                    },
                    y: TensorMut {
                        data: dy.as_slice_mut(),
                        shape: sy,
                        stride: contiguous_stride(sy),
                    },
                };
                plan.run(&stream, Workspace::Borrowed(ws.as_slice_mut()), args)
                    .expect("run");
            }
            stream.synchronize().expect("sync");
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let t0 = Instant::now();
                for _ in 0..ITERS {
                    let a = if with_a {
                        Some(TensorMut {
                            data: da.as_slice_mut(),
                            shape: sa,
                            stride: contiguous_stride(sa),
                        })
                    } else {
                        None
                    };
                    let args = FlashDecodingArgs::<f16> {
                        a,
                        q: TensorRef {
                            data: dq.as_slice(),
                            shape: sq,
                            stride: contiguous_stride(sq),
                        },
                        k: TensorRef {
                            data: dk.as_slice(),
                            shape: sk,
                            stride: contiguous_stride(sk),
                        },
                        v: TensorRef {
                            data: dv.as_slice(),
                            shape: sv,
                            stride: contiguous_stride(sv),
                        },
                        y: TensorMut {
                            data: dy.as_slice_mut(),
                            shape: sy,
                            stride: contiguous_stride(sy),
                        },
                    };
                    plan.run(&stream, Workspace::Borrowed(ws.as_slice_mut()), args)
                        .expect("run");
                }
                stream.synchronize().expect("sync");
                let ns = t0.elapsed().as_nanos() as f64 / ITERS as f64;
                if ns < best {
                    best = ns;
                }
            }
            result[idx] = best;
        }
        let (base, wa) = (result[0], result[1]);
        println!(
            "  K={k_len:>4}: baseline {base:>7.0} ns   +a {wa:>7.0} ns   delta {:+5.1}% ({:+.0} ns)",
            100.0 * (wa - base) / base,
            wa - base
        );
    }
}
