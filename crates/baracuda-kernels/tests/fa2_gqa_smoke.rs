//! Phase 59a — GQA (grouped-query attention) smoke test for FA2.
//!
//! Phase 42 (Tier-1) rejected `num_heads_k != num_heads`. Phase 59a's
//! lift accepts any `num_heads_k` where `num_heads % num_heads_k == 0`
//! (FA2's `h_h_k_ratio` mechanism handles the K/V head broadcast
//! in-kernel).
//!
//! The reference path materializes the broadcast manually (replicating
//! each K/V head `num_heads / num_heads_k` times into a dense
//! `[B, num_heads, K, D]` tensor) and runs through the bespoke
//! `FlashSdpaPlan`. FA2 takes the original `[B, num_heads_k, K, D]`
//! tensors directly. Outputs should match within FA2 vs bespoke
//! tolerance.

#![cfg(feature = "fa2")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, BackendKind, ElementKind, FlashSdpaArgs, FlashSdpaDescriptor,
    FlashSdpaPlan, PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::f16;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

const B: i32 = 1;
const NH_Q: i32 = 32;     // query heads
const NH_K: i32 = 8;      // K/V heads — ratio = 4 (typical Llama GQA)
const Q_LEN: i32 = 1024;
const K_LEN: i32 = 1024;
const D: i32 = 128;

fn default_scale(d: i32) -> f32 {
    1.0 / (d as f32).sqrt()
}

fn gen_f16(n: usize, phase: f32) -> Vec<f16> {
    (0..n)
        .map(|i| f16::from_f32(((i as f32) * 0.013 + phase).sin() * 0.25))
        .collect()
}

/// Broadcast K/V from `[B, NH_K, K, D]` to `[B, NH_Q, K, D]` by
/// replicating each H_k head NH_Q / NH_K times. This is the
/// reference's manual GQA materialization.
fn broadcast_kv_f16(src: &[f16], _b: i32, nh_k: i32, k: i32, d: i32, nh_q: i32) -> Vec<f16> {
    let ratio = nh_q / nh_k;
    let mut out = Vec::with_capacity(((_b * nh_q * k * d) as usize).max(1));
    for bi in 0.._b {
        for hi in 0..nh_q {
            let hi_k = hi / ratio;
            let src_base = ((bi * nh_k + hi_k) * k * d) as usize;
            let len = (k * d) as usize;
            out.extend_from_slice(&src[src_base..src_base + len]);
        }
    }
    out
}

#[test]
#[ignore]
fn fa2_gqa_f16_vs_dense_bespoke() {
    let (ctx, stream) = setup();
    assert_eq!(
        NH_Q % NH_K,
        0,
        "test fixture invariant: NH_Q must be divisible by NH_K"
    );

    let n_q = (B * NH_Q * Q_LEN * D) as usize;
    let n_k_gqa = (B * NH_K * K_LEN * D) as usize;
    let n_k_dense = (B * NH_Q * K_LEN * D) as usize;
    let n_y = (B * NH_Q * Q_LEN * D) as usize;

    let q_h = gen_f16(n_q, 0.0);
    let k_h_gqa = gen_f16(n_k_gqa, 0.7);
    let v_h_gqa = gen_f16(n_k_gqa, 1.3);
    let k_h_dense = broadcast_kv_f16(&k_h_gqa, B, NH_K, K_LEN, D, NH_Q);
    let v_h_dense = broadcast_kv_f16(&v_h_gqa, B, NH_K, K_LEN, D, NH_Q);
    assert_eq!(k_h_dense.len(), n_k_dense);
    assert_eq!(v_h_dense.len(), n_k_dense);

    let dq = DeviceBuffer::from_slice(&ctx, &q_h).expect("up q");
    let dk_gqa = DeviceBuffer::from_slice(&ctx, &k_h_gqa).expect("up k gqa");
    let dv_gqa = DeviceBuffer::from_slice(&ctx, &v_h_gqa).expect("up v gqa");
    let dk_dense = DeviceBuffer::from_slice(&ctx, &k_h_dense).expect("up k dense");
    let dv_dense = DeviceBuffer::from_slice(&ctx, &v_h_dense).expect("up v dense");

    let sq = [B, NH_Q, Q_LEN, D];
    let sk_gqa = [B, NH_K, K_LEN, D];
    let sv_gqa = [B, NH_K, K_LEN, D];
    let sk_dense = [B, NH_Q, K_LEN, D];
    let sv_dense = [B, NH_Q, K_LEN, D];
    let sy = [B, NH_Q, Q_LEN, D];
    let sl = [B, NH_Q, Q_LEN];

    let desc = FlashSdpaDescriptor::new(
        B,
        NH_Q,
        Q_LEN,
        K_LEN,
        D,
        D,
        default_scale(D),
        false,
        ElementKind::F16,
    );

    // --- bespoke (dense reference, K/V manually broadcast) ---
    let mut dy_ref: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, n_y).expect("alloc y_ref");
    let mut dlse_ref: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (B * NH_Q * Q_LEN) as usize).expect("alloc lse_ref");
    let pref_b = PlanPreference {
        prefer_backend: Some(BackendKind::Bespoke),
        ..Default::default()
    };
    let plan_ref = FlashSdpaPlan::<f16>::select(&stream, &desc, pref_b).expect("sel bespoke");
    plan_ref
        .run(
            &stream,
            Workspace::None,
            FlashSdpaArgs {
                q: TensorRef { data: dq.as_slice(), shape: sq, stride: contiguous_stride(sq) },
                k: TensorRef { data: dk_dense.as_slice(), shape: sk_dense, stride: contiguous_stride(sk_dense) },
                v: TensorRef { data: dv_dense.as_slice(), shape: sv_dense, stride: contiguous_stride(sv_dense) },
                y: TensorMut { data: dy_ref.as_slice_mut(), shape: sy, stride: contiguous_stride(sy) },
                lse: TensorMut { data: dlse_ref.as_slice_mut(), shape: sl, stride: contiguous_stride(sl) },
                mask: None,
                alibi_slopes: None,
            },
        )
        .expect("bespoke run");
    stream.synchronize().expect("sync bespoke");

    // --- FA2 (native GQA, K/V have NH_K heads) ---
    let mut dy_fa2: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, n_y).expect("alloc y_fa2");
    let mut dlse_fa2: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (B * NH_Q * Q_LEN) as usize).expect("alloc lse_fa2");
    let pref_f = PlanPreference {
        prefer_backend: Some(BackendKind::FlashAttentionV2),
        ..Default::default()
    };
    let plan_fa2 = FlashSdpaPlan::<f16>::select(&stream, &desc, pref_f).expect("sel fa2");
    assert_eq!(plan_fa2.backend(), BackendKind::FlashAttentionV2);
    let ws_bytes = plan_fa2.workspace_size();
    let mut ws_buf: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc fa2 ws");
    plan_fa2
        .run(
            &stream,
            Workspace::Borrowed(ws_buf.as_slice_mut()),
            FlashSdpaArgs {
                q: TensorRef { data: dq.as_slice(), shape: sq, stride: contiguous_stride(sq) },
                k: TensorRef { data: dk_gqa.as_slice(), shape: sk_gqa, stride: contiguous_stride(sk_gqa) },
                v: TensorRef { data: dv_gqa.as_slice(), shape: sv_gqa, stride: contiguous_stride(sv_gqa) },
                y: TensorMut { data: dy_fa2.as_slice_mut(), shape: sy, stride: contiguous_stride(sy) },
                lse: TensorMut { data: dlse_fa2.as_slice_mut(), shape: sl, stride: contiguous_stride(sl) },
                mask: None,
                alibi_slopes: None,
            },
        )
        .expect("fa2 gqa run");
    stream.synchronize().expect("sync fa2");

    let mut got = vec![f16::ZERO; n_y];
    let mut refv = vec![f16::ZERO; n_y];
    dy_fa2.copy_to_host(&mut got).expect("dl fa2");
    dy_ref.copy_to_host(&mut refv).expect("dl ref");

    let tol = 64.0 * 9.77e-4_f32;
    for i in 0..got.len() {
        let r = refv[i].to_f32();
        let f = got[i].to_f32();
        let diff = (f - r).abs();
        let t = (r.abs() * tol).max(tol);
        assert!(
            diff <= t,
            "GQA fa2 vs dense-bespoke @ {i}: diff={diff} fa2={f} bespoke={r}"
        );
    }
}
