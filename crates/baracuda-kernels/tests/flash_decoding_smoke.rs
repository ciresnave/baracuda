//! Real-GPU smoke test for `FlashDecodingPlan` FW (Phase 73 follow-up).
//!
//! Validates the split-K decode kernel against a CPU fp32 reference.
//! Covers f16 + bf16 at several (B, H, K_len, D) shapes including:
//!   - The minimum nontrivial case (B=1, H=1, K=64, D=32).
//!   - A two-split case (K_len = 300 > CHUNK_K = 256).
//!   - The LLM-decode-shaped case (B=1, H=32, K=2048, D=128).
//!
//! `#[ignore]` by default — requires a real CUDA device.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, FlashDecodingArgs, FlashDecodingDescriptor, FlashDecodingPlan,
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

/// CPU fp32 reference for SDPA at seq_q=1.
///
/// Reads Q[B, H, D], K[B, H, K_len, D], V[B, H, K_len, D] (all
/// row-major contiguous) and writes Y[B, H, D].
fn sdpa_decode_cpu(
    q: &[f32], k: &[f32], v: &[f32],
    b: usize, h: usize, k_len: usize, d: usize, scale: f32,
) -> Vec<f32> {
    let mut y = vec![0.0_f32; b * h * d];
    for bi in 0..b {
        for hi in 0..h {
            // Scores: s[ki] = (Q[bi, hi] · K[bi, hi, ki]) * scale.
            let mut scores = vec![0.0_f32; k_len];
            for ki in 0..k_len {
                let q_off = (bi * h + hi) * d;
                let k_off = ((bi * h + hi) * k_len + ki) * d;
                let mut dot = 0.0_f32;
                for di in 0..d {
                    dot += q[q_off + di] * k[k_off + di];
                }
                scores[ki] = dot * scale;
            }
            // Softmax across k.
            let mut max_s = f32::NEG_INFINITY;
            for &s in &scores {
                if s > max_s { max_s = s; }
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
            // Y[bi, hi] = Σ_ki scores[ki] * V[bi, hi, ki].
            let y_off = (bi * h + hi) * d;
            for di in 0..d {
                let mut acc = 0.0_f32;
                for ki in 0..k_len {
                    let v_off = ((bi * h + hi) * k_len + ki) * d;
                    acc += scores[ki] * v[v_off + di];
                }
                y[y_off + di] = acc;
            }
        }
    }
    y
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

    let expected = sdpa_decode_cpu(
        &q_f32, &k_f32, &v_f32,
        b as usize, h as usize, k_len as usize, d as usize, scale,
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
    let mut ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, plan.workspace_size())
        .expect("alloc workspace");

    let sq = [b, h, d];
    let sk = [b, h, k_len, d];
    let sv = [b, h, k_len, d];
    let sy = [b, h, d];

    let args = FlashDecodingArgs::<f16> {
        q: TensorRef { data: dq.as_slice(), shape: sq, stride: contiguous_stride(sq) },
        k: TensorRef { data: dk.as_slice(), shape: sk, stride: contiguous_stride(sk) },
        v: TensorRef { data: dv.as_slice(), shape: sv, stride: contiguous_stride(sv) },
        y: TensorMut { data: dy.as_slice_mut(), shape: sy, stride: contiguous_stride(sy) },
    };
    plan.run(&stream, Workspace::Borrowed(ws.as_slice_mut()), args)
        .expect("run");
    stream.synchronize().expect("sync");

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

    let expected = sdpa_decode_cpu(
        &q_f32, &k_f32, &v_f32,
        b as usize, h as usize, k_len as usize, d as usize, scale,
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
    let mut ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, plan.workspace_size())
        .expect("alloc workspace");

    let sq = [b, h, d];
    let sk = [b, h, k_len, d];
    let sv = [b, h, k_len, d];
    let sy = [b, h, d];

    let args = FlashDecodingArgs::<bf16> {
        q: TensorRef { data: dq.as_slice(), shape: sq, stride: contiguous_stride(sq) },
        k: TensorRef { data: dk.as_slice(), shape: sk, stride: contiguous_stride(sk) },
        v: TensorRef { data: dv.as_slice(), shape: sv, stride: contiguous_stride(sv) },
        y: TensorMut { data: dy.as_slice_mut(), shape: sy, stride: contiguous_stride(sy) },
    };
    plan.run(&stream, Workspace::Borrowed(ws.as_slice_mut()), args)
        .expect("run");
    stream.synchronize().expect("sync");

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
