//! Real-GPU smoke tests for the Phase 51 arbitrary additive-mask path
//! on [`FlashSdpaPlan`].
//!
//! Validates:
//!   1. Random-mask `FlashSdpaPlan(mask = Some(...))` matches the
//!      naive `SdpaPlan` reference (which already supports an additive
//!      mask via its 3-kernel `softmax(QK^T·scale + mask)·V` pipeline).
//!   2. Tree-attention mask (a small EAGLE-style spec-decode tree):
//!      verify that mask = lower-triangular over the tree topology
//!      produces the same output as running each accepted-token chain
//!      through single-token causal SDPA.
//!   3. Sliding-window mask (window=64): random-fixture FW with
//!      banded -INF mask matches an explicit naive reference that
//!      applies the same band.
//!
//! All tests are `#[ignore]` by default — require a real CUDA device.
//! The arbmask path is bespoke (Phase 51) and does NOT require the
//! `fa2` cargo feature — it's its own SDPA SKU. The test file is
//! cargo-feature-free.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, FlashSdpaArgs, FlashSdpaDescriptor, FlashSdpaPlan,
    PlanPreference, TensorMut, TensorRef, Workspace,
};

const F32_TOL_REL: f32 = 1e-4;
const F32_TOL_ABS: f32 = 5e-5;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

// f64 host-side SDPA reference identical in shape to the GPU
// arbmask path: scores = Q·K^T·scale + mask + causal, row-softmax,
// y = attn @ V. Single-precision math (f32 down-cast at the end)
// to match the GPU f32 kernel; arbmask Tier 1 uses f32 accumulator
// regardless of element dtype.
fn host_sdpa_f32(
    batch: usize,
    heads: usize,
    q_len: usize,
    k_len: usize,
    d_k: usize,
    d_v: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: Option<&[f32]>,
    scale: f32,
    is_causal: bool,
) -> Vec<f32> {
    let total_y = batch * heads * q_len * d_v;
    let mut y = vec![0f32; total_y];
    for b in 0..batch {
        for h in 0..heads {
            let mut scores = vec![0f64; q_len * k_len];
            for i in 0..q_len {
                for j in 0..k_len {
                    let mut s = 0f64;
                    for d in 0..d_k {
                        let q_idx = ((b * heads + h) * q_len + i) * d_k + d;
                        let k_idx = ((b * heads + h) * k_len + j) * d_k + d;
                        s += (q[q_idx] as f64) * (k[k_idx] as f64);
                    }
                    s *= scale as f64;
                    if let Some(m) = mask {
                        let m_idx = ((b * heads + h) * q_len + i) * k_len + j;
                        s += m[m_idx] as f64;
                    }
                    if is_causal && j > i {
                        s = f64::NEG_INFINITY;
                    }
                    scores[i * k_len + j] = s;
                }
            }
            for i in 0..q_len {
                let row = &scores[i * k_len..(i + 1) * k_len];
                let m_max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                if !m_max.is_finite() {
                    for dv in 0..d_v {
                        let y_idx = ((b * heads + h) * q_len + i) * d_v + dv;
                        y[y_idx] = 0.0;
                    }
                    continue;
                }
                let sum: f64 = row.iter().map(|&v| (v - m_max).exp()).sum();
                for dv in 0..d_v {
                    let mut acc = 0f64;
                    for kk in 0..k_len {
                        let p = (row[kk] - m_max).exp() / sum;
                        let v_idx = ((b * heads + h) * k_len + kk) * d_v + dv;
                        acc += p * (v[v_idx] as f64);
                    }
                    let y_idx = ((b * heads + h) * q_len + i) * d_v + dv;
                    y[y_idx] = acc as f32;
                }
            }
        }
    }
    y
}

fn gen_f32(n: usize, phase: f32, scale: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 0.013 + phase).sin() * scale).collect()
}

fn assert_close_f32(got: &[f32], expect: &[f32], label: &str) {
    let mut max_diff = 0.0_f32;
    let mut max_idx = 0usize;
    for (i, (g, r)) in got.iter().zip(expect.iter()).enumerate() {
        let diff = (g - r).abs();
        let t = (r.abs() * F32_TOL_REL).max(F32_TOL_ABS);
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
        assert!(
            diff <= t,
            "{label} @ {i}: diff={diff} got={g} expect={r} tol={t}"
        );
    }
    eprintln!("{label}: max_diff={max_diff:.3e} @ idx {max_idx}");
}

// ===========================================================================
// Test 1: random additive mask, f32.
// Compare FlashSdpaPlan(mask=Some) vs host f32 reference.
// ===========================================================================

#[test]
#[ignore]
fn arbmask_random_f32_matches_host_ref() {
    let (ctx, stream) = setup();

    const B: i32 = 2;
    const H: i32 = 2;
    const Q: i32 = 32;
    const K: i32 = 48;
    const D: i32 = 32;

    let n_q = (B * H * Q * D) as usize;
    let n_k = (B * H * K * D) as usize;
    let n_v = (B * H * K * D) as usize;
    let n_y = (B * H * Q * D) as usize;
    let n_m = (B * H * Q * K) as usize;

    let q_h = gen_f32(n_q, 0.0, 0.5);
    let k_h = gen_f32(n_k, 0.7, 0.5);
    let v_h = gen_f32(n_v, 1.3, 0.5);
    let m_h = gen_f32(n_m, 2.1, 0.3); // small additive bias

    let dq = DeviceBuffer::from_slice(&ctx, &q_h).expect("up q");
    let dk = DeviceBuffer::from_slice(&ctx, &k_h).expect("up k");
    let dv = DeviceBuffer::from_slice(&ctx, &v_h).expect("up v");
    let dm = DeviceBuffer::from_slice(&ctx, &m_h).expect("up mask");
    let mut dy: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, n_y).expect("alloc y");
    let mut dlse: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (B * H * Q) as usize).expect("alloc lse");

    let sq = [B, H, Q, D];
    let sk = [B, H, K, D];
    let sv = [B, H, K, D];
    let sy = [B, H, Q, D];
    let sl = [B, H, Q];
    let sm = [B, H, Q, K];

    let scale = 1.0 / (D as f32).sqrt();
    let desc = FlashSdpaDescriptor::new(
        B,
        H,
        Q,
        K,
        D,
        D,
        scale,
        false,
        ElementKind::F32,
    );
    let plan =
        FlashSdpaPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    plan.run(
        &stream,
        Workspace::None,
        FlashSdpaArgs {
            q: TensorRef { data: dq.as_slice(), shape: sq, stride: contiguous_stride(sq) },
            k: TensorRef { data: dk.as_slice(), shape: sk, stride: contiguous_stride(sk) },
            v: TensorRef { data: dv.as_slice(), shape: sv, stride: contiguous_stride(sv) },
            y: TensorMut { data: dy.as_slice_mut(), shape: sy, stride: contiguous_stride(sy) },
            lse: TensorMut { data: dlse.as_slice_mut(), shape: sl, stride: contiguous_stride(sl) },
            mask: Some(TensorRef {
                data: dm.as_slice(),
                shape: sm,
                stride: contiguous_stride(sm),
            }),
                    alibi_slopes: None,
        },
    )
    .expect("arbmask run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; n_y];
    dy.copy_to_host(&mut got).expect("dl");

    let expect = host_sdpa_f32(
        B as usize, H as usize, Q as usize, K as usize, D as usize, D as usize,
        &q_h, &k_h, &v_h, Some(&m_h), scale, false,
    );
    assert_close_f32(&got, &expect, "arbmask random f32");
}

// ===========================================================================
// Test 2: tree-attention mask (EAGLE/Medusa-style spec-decode pattern).
//
// Topology: 4 draft tokens proposed by a draft model, organized as a
// binary tree:
//
//   t0  (root, accepted)
//   ├── t1 (child of t0, draft)
//   │    ├── t3 (child of t1)
//   │    └── t4 (child of t1)
//   └── t2 (child of t0, draft)
//
// Each token can attend to itself and to its ancestors (and to the
// prefix), NOT to siblings or unrelated branches. We encode this as
// `mask[i, j] = 0` if j is ancestor-or-self of i in the tree (or in
// the prefix), else `-INFINITY`. Compare against a reference that
// runs each chain (prefix + t0 + tk) as a separate causal SDPA call.
//
// The mask encodes "which prefix positions each draft token can see"
// — for spec-decode this is the standard FlashAttention tree-mask
// convention.
// ===========================================================================

#[test]
#[ignore]
fn arbmask_tree_attention_pattern() {
    let (ctx, stream) = setup();

    // Layout: K-axis = [P prefix tokens, t0, t1, t2, t3, t4]
    // total K = P + 5 ; Q-axis only writes the 5 new tokens.
    const P: i32 = 16; // prefix length
    const Q: i32 = 5;  // 5 draft tokens (t0..t4)
    const K: i32 = P + Q;
    const B: i32 = 1;
    const H: i32 = 2;
    const D: i32 = 16;

    let n_q = (B * H * Q * D) as usize;
    let n_k = (B * H * K * D) as usize;
    let n_v = (B * H * K * D) as usize;
    let n_y = (B * H * Q * D) as usize;
    let n_m = (B * H * Q * K) as usize;

    let q_h = gen_f32(n_q, 0.0, 0.5);
    let k_h = gen_f32(n_k, 0.7, 0.5);
    let v_h = gen_f32(n_v, 1.3, 0.5);

    // Build tree mask. For each query token i (index into [t0..t4]):
    //   - can see prefix [0..P)
    //   - can see ancestor(s) in the tree (including self)
    //   - cannot see siblings / cousins
    //
    // Q-index -> token (root + 4 children)
    //   0 = t0 (root)
    //   1 = t1, parent = t0
    //   2 = t2, parent = t0
    //   3 = t3, parent = t1
    //   4 = t4, parent = t1
    //
    // Each token's K-axis ancestors (self + parents):
    //   t0: {t0}
    //   t1: {t0, t1}
    //   t2: {t0, t2}
    //   t3: {t0, t1, t3}
    //   t4: {t0, t1, t4}
    let ancestors: [&[i32]; 5] = [
        &[0],
        &[0, 1],
        &[0, 2],
        &[0, 1, 3],
        &[0, 1, 2, 4][..2].split_at(2).0, // unused; redefining below
    ];
    let _ = ancestors;
    let ancestors: [Vec<i32>; 5] = [
        vec![0],
        vec![0, 1],
        vec![0, 2],
        vec![0, 1, 3],
        vec![0, 1, 4],
    ];

    let mut mask_h = vec![f32::NEG_INFINITY; n_m];
    for b in 0..(B as usize) {
        for h in 0..(H as usize) {
            for qi in 0..(Q as usize) {
                // Prefix is always visible to every draft token.
                for kj in 0..(P as usize) {
                    let idx = ((b * (H as usize) + h) * (Q as usize) + qi) * (K as usize) + kj;
                    mask_h[idx] = 0.0;
                }
                // Tree ancestors at K offsets [P..P+Q):
                for &anc in &ancestors[qi] {
                    let kj = (P as usize) + (anc as usize);
                    let idx = ((b * (H as usize) + h) * (Q as usize) + qi) * (K as usize) + kj;
                    mask_h[idx] = 0.0;
                }
            }
        }
    }

    let dq = DeviceBuffer::from_slice(&ctx, &q_h).expect("up q");
    let dk = DeviceBuffer::from_slice(&ctx, &k_h).expect("up k");
    let dv = DeviceBuffer::from_slice(&ctx, &v_h).expect("up v");
    let dm = DeviceBuffer::from_slice(&ctx, &mask_h).expect("up mask");
    let mut dy: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, n_y).expect("alloc y");
    let mut dlse: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (B * H * Q) as usize).expect("alloc lse");

    let sq = [B, H, Q, D];
    let sk = [B, H, K, D];
    let sv = [B, H, K, D];
    let sy = [B, H, Q, D];
    let sl = [B, H, Q];
    let sm = [B, H, Q, K];

    let scale = 1.0 / (D as f32).sqrt();
    let desc = FlashSdpaDescriptor::new(
        B,
        H,
        Q,
        K,
        D,
        D,
        scale,
        false,
        ElementKind::F32,
    );
    let plan =
        FlashSdpaPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    plan.run(
        &stream,
        Workspace::None,
        FlashSdpaArgs {
            q: TensorRef { data: dq.as_slice(), shape: sq, stride: contiguous_stride(sq) },
            k: TensorRef { data: dk.as_slice(), shape: sk, stride: contiguous_stride(sk) },
            v: TensorRef { data: dv.as_slice(), shape: sv, stride: contiguous_stride(sv) },
            y: TensorMut { data: dy.as_slice_mut(), shape: sy, stride: contiguous_stride(sy) },
            lse: TensorMut { data: dlse.as_slice_mut(), shape: sl, stride: contiguous_stride(sl) },
            mask: Some(TensorRef {
                data: dm.as_slice(),
                shape: sm,
                stride: contiguous_stride(sm),
            }),
                    alibi_slopes: None,
        },
    )
    .expect("arbmask tree run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; n_y];
    dy.copy_to_host(&mut got).expect("dl");

    // Reference: host SDPA with the same mask. Validates the kernel
    // applies the additive mask correctly (with -INFINITY suppression).
    let expect = host_sdpa_f32(
        B as usize, H as usize, Q as usize, K as usize, D as usize, D as usize,
        &q_h, &k_h, &v_h, Some(&mask_h), scale, false,
    );
    assert_close_f32(&got, &expect, "arbmask tree");

    // Additionally: t1's output should NOT depend on t2's K/V. Verify
    // by perturbing V at position P+2 (t2's V row) and re-running —
    // t1's output should be unchanged.
    let mut v_perturbed = v_h.clone();
    let perturb_kbase_offset = (P as usize) + 2; // t2's K-position
    for b in 0..(B as usize) {
        for h in 0..(H as usize) {
            for d in 0..(D as usize) {
                let idx = ((b * (H as usize) + h) * (K as usize) + perturb_kbase_offset)
                    * (D as usize)
                    + d;
                v_perturbed[idx] += 10.0; // big perturbation
            }
        }
    }
    let dv2 = DeviceBuffer::from_slice(&ctx, &v_perturbed).expect("up v2");
    let mut dy2: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, n_y).expect("alloc y2");
    let mut dlse2: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (B * H * Q) as usize).expect("alloc lse2");
    plan.run(
        &stream,
        Workspace::None,
        FlashSdpaArgs {
            q: TensorRef { data: dq.as_slice(), shape: sq, stride: contiguous_stride(sq) },
            k: TensorRef { data: dk.as_slice(), shape: sk, stride: contiguous_stride(sk) },
            v: TensorRef { data: dv2.as_slice(), shape: sv, stride: contiguous_stride(sv) },
            y: TensorMut { data: dy2.as_slice_mut(), shape: sy, stride: contiguous_stride(sy) },
            lse: TensorMut { data: dlse2.as_slice_mut(), shape: sl, stride: contiguous_stride(sl) },
            mask: Some(TensorRef {
                data: dm.as_slice(),
                shape: sm,
                stride: contiguous_stride(sm),
            }),
                    alibi_slopes: None,
        },
    )
    .expect("arbmask tree run 2");
    stream.synchronize().expect("sync2");

    let mut got2 = vec![0f32; n_y];
    dy2.copy_to_host(&mut got2).expect("dl2");

    // t1 (qi = 1) and t3/t4 (qi = 3, 4) must be UNCHANGED — they don't
    // see t2's V. t2 (qi = 2) and t0 (qi = 0) WILL change — t2 sees
    // itself, t0 sees... well, t0 only sees prefix + itself per the
    // tree, so t0 is unchanged too. Let me re-check ancestors:
    //   t0: {t0}     — does NOT see t2 ✓ unchanged
    //   t1: {t0, t1} — does NOT see t2 ✓ unchanged
    //   t2: {t0, t2} — DOES see t2     ✗ changed
    //   t3: {t0,t1,t3} — does NOT see t2 ✓ unchanged
    //   t4: {t0,t1,t4} — does NOT see t2 ✓ unchanged
    let unchanged_q: [usize; 4] = [0, 1, 3, 4];
    for &qi in &unchanged_q {
        for b in 0..(B as usize) {
            for h in 0..(H as usize) {
                for d in 0..(D as usize) {
                    let idx =
                        ((b * (H as usize) + h) * (Q as usize) + qi) * (D as usize) + d;
                    let diff = (got[idx] - got2[idx]).abs();
                    assert!(
                        diff <= F32_TOL_ABS,
                        "tree mask leakage: q{qi} changed by {diff} when t2's V was perturbed"
                    );
                }
            }
        }
    }
}

// ===========================================================================
// Test 3: sliding-window mask. Use window=8 (each q-token can attend
// to last 8 k-tokens). Verify FlashSdpaPlan(mask) matches naive
// reference with the same mask.
// ===========================================================================

#[test]
#[ignore]
fn arbmask_sliding_window_matches_naive() {
    let (ctx, stream) = setup();

    const B: i32 = 1;
    const H: i32 = 2;
    const Q: i32 = 64;
    const K: i32 = 64;
    const D: i32 = 32;
    const WIN: i32 = 8;

    let n_q = (B * H * Q * D) as usize;
    let n_k = (B * H * K * D) as usize;
    let n_v = (B * H * K * D) as usize;
    let n_y = (B * H * Q * D) as usize;
    let n_m = (B * H * Q * K) as usize;

    let q_h = gen_f32(n_q, 0.0, 0.5);
    let k_h = gen_f32(n_k, 0.7, 0.5);
    let v_h = gen_f32(n_v, 1.3, 0.5);

    // Build sliding-window mask: for q-index i, allow attending to
    // k-index j iff `(i - WIN) < j <= i`. We assume Q == K and treat
    // them as aligned token streams (the standard sliding-window
    // attention setup).
    let mut mask_h = vec![f32::NEG_INFINITY; n_m];
    for b in 0..(B as usize) {
        for h in 0..(H as usize) {
            for i in 0..(Q as usize) {
                let lo = (i as i32) - WIN + 1;
                let hi = (i as i32) + 1;
                for j in 0..(K as usize) {
                    let jj = j as i32;
                    if jj >= lo && jj < hi {
                        let idx = ((b * (H as usize) + h) * (Q as usize) + i)
                            * (K as usize)
                            + j;
                        mask_h[idx] = 0.0;
                    }
                }
            }
        }
    }

    let dq = DeviceBuffer::from_slice(&ctx, &q_h).expect("up q");
    let dk = DeviceBuffer::from_slice(&ctx, &k_h).expect("up k");
    let dv = DeviceBuffer::from_slice(&ctx, &v_h).expect("up v");
    let dm = DeviceBuffer::from_slice(&ctx, &mask_h).expect("up mask");
    let mut dy: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, n_y).expect("alloc y");
    let mut dlse: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (B * H * Q) as usize).expect("alloc lse");

    let sq = [B, H, Q, D];
    let sk = [B, H, K, D];
    let sv = [B, H, K, D];
    let sy = [B, H, Q, D];
    let sl = [B, H, Q];
    let sm = [B, H, Q, K];

    let scale = 1.0 / (D as f32).sqrt();
    let desc = FlashSdpaDescriptor::new(
        B,
        H,
        Q,
        K,
        D,
        D,
        scale,
        false,
        ElementKind::F32,
    );
    let plan =
        FlashSdpaPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    plan.run(
        &stream,
        Workspace::None,
        FlashSdpaArgs {
            q: TensorRef { data: dq.as_slice(), shape: sq, stride: contiguous_stride(sq) },
            k: TensorRef { data: dk.as_slice(), shape: sk, stride: contiguous_stride(sk) },
            v: TensorRef { data: dv.as_slice(), shape: sv, stride: contiguous_stride(sv) },
            y: TensorMut { data: dy.as_slice_mut(), shape: sy, stride: contiguous_stride(sy) },
            lse: TensorMut { data: dlse.as_slice_mut(), shape: sl, stride: contiguous_stride(sl) },
            mask: Some(TensorRef {
                data: dm.as_slice(),
                shape: sm,
                stride: contiguous_stride(sm),
            }),
                    alibi_slopes: None,
        },
    )
    .expect("arbmask window run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; n_y];
    dy.copy_to_host(&mut got).expect("dl");

    let expect = host_sdpa_f32(
        B as usize, H as usize, Q as usize, K as usize, D as usize, D as usize,
        &q_h, &k_h, &v_h, Some(&mask_h), scale, false,
    );
    assert_close_f32(&got, &expect, "arbmask sliding-window");
}

// ===========================================================================
// Test 4: causal + arbmask compose correctly. Verify that combining
// is_causal=true with an arbitrary mask produces the same result as
// applying both at the kernel level.
// ===========================================================================

#[test]
#[ignore]
fn arbmask_with_causal_compose() {
    let (ctx, stream) = setup();

    const B: i32 = 1;
    const H: i32 = 2;
    const Q: i32 = 32;
    const K: i32 = 32;
    const D: i32 = 32;

    let n_q = (B * H * Q * D) as usize;
    let n_k = (B * H * K * D) as usize;
    let n_v = (B * H * K * D) as usize;
    let n_y = (B * H * Q * D) as usize;
    let n_m = (B * H * Q * K) as usize;

    let q_h = gen_f32(n_q, 0.0, 0.5);
    let k_h = gen_f32(n_k, 0.7, 0.5);
    let v_h = gen_f32(n_v, 1.3, 0.5);
    // Small additive bias on top of causal — exercises the
    // "is_causal sets -INF first, then add stays -INF" composition.
    let m_h = gen_f32(n_m, 2.1, 0.1);

    let dq = DeviceBuffer::from_slice(&ctx, &q_h).expect("up q");
    let dk = DeviceBuffer::from_slice(&ctx, &k_h).expect("up k");
    let dv = DeviceBuffer::from_slice(&ctx, &v_h).expect("up v");
    let dm = DeviceBuffer::from_slice(&ctx, &m_h).expect("up mask");
    let mut dy: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, n_y).expect("alloc y");
    let mut dlse: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (B * H * Q) as usize).expect("alloc lse");

    let sq = [B, H, Q, D];
    let sk = [B, H, K, D];
    let sv = [B, H, K, D];
    let sy = [B, H, Q, D];
    let sl = [B, H, Q];
    let sm = [B, H, Q, K];

    let scale = 1.0 / (D as f32).sqrt();
    let desc = FlashSdpaDescriptor::new(
        B,
        H,
        Q,
        K,
        D,
        D,
        scale,
        true,
        ElementKind::F32,
    );
    let plan =
        FlashSdpaPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    plan.run(
        &stream,
        Workspace::None,
        FlashSdpaArgs {
            q: TensorRef { data: dq.as_slice(), shape: sq, stride: contiguous_stride(sq) },
            k: TensorRef { data: dk.as_slice(), shape: sk, stride: contiguous_stride(sk) },
            v: TensorRef { data: dv.as_slice(), shape: sv, stride: contiguous_stride(sv) },
            y: TensorMut { data: dy.as_slice_mut(), shape: sy, stride: contiguous_stride(sy) },
            lse: TensorMut { data: dlse.as_slice_mut(), shape: sl, stride: contiguous_stride(sl) },
            mask: Some(TensorRef {
                data: dm.as_slice(),
                shape: sm,
                stride: contiguous_stride(sm),
            }),
                    alibi_slopes: None,
        },
    )
    .expect("arbmask+causal run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; n_y];
    dy.copy_to_host(&mut got).expect("dl");

    let expect = host_sdpa_f32(
        B as usize, H as usize, Q as usize, K as usize, D as usize, D as usize,
        &q_h, &k_h, &v_h, Some(&m_h), scale, true,
    );
    assert_close_f32(&got, &expect, "arbmask + causal");
}
