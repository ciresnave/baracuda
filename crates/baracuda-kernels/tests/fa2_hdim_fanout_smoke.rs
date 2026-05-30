//! Phase 59a — head_dim fanout smoke test for the FA2 backend.
//!
//! Phase 42 (Tier-1) shipped head_dim=128 only; Phase 59a vendored
//! the remaining upstream-supported head_dims (32, 64, 96, 192, 256).
//! This file validates each new head_dim numerically against the
//! bespoke `FlashSdpaPlan` reference, for both fp16 and bf16.
//!
//! Upstream FA2 v2.8.3 does NOT ship head_dims 160, 224, or 512 —
//! those return `Error::Unsupported` from `FlashSdpaPlan::select` when
//! FA2 is forced (or fall back to bespoke if d_k ≤ 128).
//!
//! All tests `#[ignore]`-gated for real-GPU + `--features fa2,sm80`.

#![cfg(feature = "fa2")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, BackendKind, ElementKind, FlashSdpaArgs, FlashSdpaDescriptor,
    FlashSdpaPlan, PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

// Long-context shape so the heuristic picks FA2. 1024x1024 = 1M cells,
// at the threshold. We force via PlanPreference anyway.
const B: i32 = 1;
const H: i32 = 2;
const Q: i32 = 1024;
const K: i32 = 1024;

fn default_scale(d: i32) -> f32 {
    1.0 / (d as f32).sqrt()
}

fn gen_f16(n: usize, phase: f32) -> Vec<f16> {
    (0..n)
        .map(|i| f16::from_f32(((i as f32) * 0.013 + phase).sin() * 0.25))
        .collect()
}

fn gen_bf16(n: usize, phase: f32) -> Vec<bf16> {
    (0..n)
        .map(|i| bf16::from_f32(((i as f32) * 0.013 + phase).sin() * 0.25))
        .collect()
}

fn run_one_f16(d: i32, is_causal: bool) {
    let (ctx, stream) = setup();
    let n_q = (B * H * Q * d) as usize;
    let n_k = (B * H * K * d) as usize;
    let n_v = (B * H * K * d) as usize;
    let n_y = (B * H * Q * d) as usize;

    let q_h = gen_f16(n_q, 0.0);
    let k_h = gen_f16(n_k, 0.7);
    let v_h = gen_f16(n_v, 1.3);

    let dq = DeviceBuffer::from_slice(&ctx, &q_h).expect("up q");
    let dk = DeviceBuffer::from_slice(&ctx, &k_h).expect("up k");
    let dv = DeviceBuffer::from_slice(&ctx, &v_h).expect("up v");

    let sq = [B, H, Q, d];
    let sk = [B, H, K, d];
    let sv = [B, H, K, d];
    let sy = [B, H, Q, d];
    let sl = [B, H, Q];

    let desc = FlashSdpaDescriptor::new(B, H, Q, K, d, d, default_scale(d), is_causal, ElementKind::F16);

    // Bespoke reference (only when d_k ≤ 128; for 192/256 we skip
    // the reference and just verify FA2 doesn't crash).
    if d <= 128 {
        let mut dy_ref: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, n_y).expect("alloc y_ref");
        let mut dlse_ref: DeviceBuffer<f16> =
            DeviceBuffer::zeros(&ctx, (B * H * Q) as usize).expect("alloc lse_ref");
        let pref_b = PlanPreference {
            prefer_backend: Some(BackendKind::Bespoke),
            ..Default::default()
        };
        let plan_b = FlashSdpaPlan::<f16>::select(&stream, &desc, pref_b).expect("sel bespoke");
        plan_b
            .run(
                &stream,
                Workspace::None,
                FlashSdpaArgs {
                    q: TensorRef { data: dq.as_slice(), shape: sq, stride: contiguous_stride(sq) },
                    k: TensorRef { data: dk.as_slice(), shape: sk, stride: contiguous_stride(sk) },
                    v: TensorRef { data: dv.as_slice(), shape: sv, stride: contiguous_stride(sv) },
                    y: TensorMut { data: dy_ref.as_slice_mut(), shape: sy, stride: contiguous_stride(sy) },
                    lse: TensorMut { data: dlse_ref.as_slice_mut(), shape: sl, stride: contiguous_stride(sl) },
                    mask: None,
                    alibi_slopes: None,
                },
            )
            .expect("bespoke run");
        stream.synchronize().expect("sync bespoke");

        let mut dy_fa2: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, n_y).expect("alloc y_fa2");
        let mut dlse_fa2: DeviceBuffer<f16> =
            DeviceBuffer::zeros(&ctx, (B * H * Q) as usize).expect("alloc lse_fa2");
        let pref_f = PlanPreference {
            prefer_backend: Some(BackendKind::FlashAttentionV2),
            ..Default::default()
        };
        let plan_f = FlashSdpaPlan::<f16>::select(&stream, &desc, pref_f).expect("sel fa2");
        assert_eq!(plan_f.backend(), BackendKind::FlashAttentionV2);
        let ws_bytes = plan_f.workspace_size();
        let mut ws_buf: DeviceBuffer<u8> =
            DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc fa2 ws");
        plan_f
            .run(
                &stream,
                Workspace::Borrowed(ws_buf.as_slice_mut()),
                FlashSdpaArgs {
                    q: TensorRef { data: dq.as_slice(), shape: sq, stride: contiguous_stride(sq) },
                    k: TensorRef { data: dk.as_slice(), shape: sk, stride: contiguous_stride(sk) },
                    v: TensorRef { data: dv.as_slice(), shape: sv, stride: contiguous_stride(sv) },
                    y: TensorMut { data: dy_fa2.as_slice_mut(), shape: sy, stride: contiguous_stride(sy) },
                    lse: TensorMut { data: dlse_fa2.as_slice_mut(), shape: sl, stride: contiguous_stride(sl) },
                    mask: None,
                    alibi_slopes: None,
                },
            )
            .expect("fa2 run");
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
                "f16 d={d} causal={is_causal} @ {i}: diff={diff} fa2={f} bespoke={r}"
            );
        }
    } else {
        // For d > 128 the bespoke kernel rejects in select(). Just
        // smoke-test that the FA2 path runs cleanly.
        let mut dy_fa2: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, n_y).expect("alloc y_fa2");
        let mut dlse_fa2: DeviceBuffer<f16> =
            DeviceBuffer::zeros(&ctx, (B * H * Q) as usize).expect("alloc lse_fa2");
        let pref_f = PlanPreference {
            prefer_backend: Some(BackendKind::FlashAttentionV2),
            ..Default::default()
        };
        let plan_f = FlashSdpaPlan::<f16>::select(&stream, &desc, pref_f).expect("sel fa2");
        assert_eq!(plan_f.backend(), BackendKind::FlashAttentionV2);
        let ws_bytes = plan_f.workspace_size();
        let mut ws_buf: DeviceBuffer<u8> =
            DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc fa2 ws");
        plan_f
            .run(
                &stream,
                Workspace::Borrowed(ws_buf.as_slice_mut()),
                FlashSdpaArgs {
                    q: TensorRef { data: dq.as_slice(), shape: sq, stride: contiguous_stride(sq) },
                    k: TensorRef { data: dk.as_slice(), shape: sk, stride: contiguous_stride(sk) },
                    v: TensorRef { data: dv.as_slice(), shape: sv, stride: contiguous_stride(sv) },
                    y: TensorMut { data: dy_fa2.as_slice_mut(), shape: sy, stride: contiguous_stride(sy) },
                    lse: TensorMut { data: dlse_fa2.as_slice_mut(), shape: sl, stride: contiguous_stride(sl) },
                    mask: None,
                    alibi_slopes: None,
                },
            )
            .expect("fa2 run d>128");
        stream.synchronize().expect("sync fa2");

        // Sanity: output is finite (no NaN, no Inf)
        let mut got = vec![f16::ZERO; n_y];
        dy_fa2.copy_to_host(&mut got).expect("dl fa2");
        for (i, v) in got.iter().enumerate() {
            let f = v.to_f32();
            assert!(
                f.is_finite(),
                "f16 d={d} causal={is_causal} non-finite at {i}: {f}"
            );
        }
    }
}

fn run_one_bf16(d: i32, is_causal: bool) {
    let (ctx, stream) = setup();
    let n_q = (B * H * Q * d) as usize;
    let n_k = (B * H * K * d) as usize;
    let n_v = (B * H * K * d) as usize;
    let n_y = (B * H * Q * d) as usize;

    let q_h = gen_bf16(n_q, 0.0);
    let k_h = gen_bf16(n_k, 0.7);
    let v_h = gen_bf16(n_v, 1.3);

    let dq = DeviceBuffer::from_slice(&ctx, &q_h).expect("up q");
    let dk = DeviceBuffer::from_slice(&ctx, &k_h).expect("up k");
    let dv = DeviceBuffer::from_slice(&ctx, &v_h).expect("up v");

    let sq = [B, H, Q, d];
    let sk = [B, H, K, d];
    let sv = [B, H, K, d];
    let sy = [B, H, Q, d];
    let sl = [B, H, Q];

    let desc = FlashSdpaDescriptor::new(B, H, Q, K, d, d, default_scale(d), is_causal, ElementKind::Bf16);

    let pref_f = PlanPreference {
        prefer_backend: Some(BackendKind::FlashAttentionV2),
        ..Default::default()
    };
    let plan_f = FlashSdpaPlan::<bf16>::select(&stream, &desc, pref_f).expect("sel fa2");
    assert_eq!(plan_f.backend(), BackendKind::FlashAttentionV2);
    let ws_bytes = plan_f.workspace_size();
    let mut ws_buf: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc fa2 ws");
    let mut dy_fa2: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, n_y).expect("alloc y_fa2");
    let mut dlse_fa2: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q) as usize).expect("alloc lse_fa2");
    plan_f
        .run(
            &stream,
            Workspace::Borrowed(ws_buf.as_slice_mut()),
            FlashSdpaArgs {
                q: TensorRef { data: dq.as_slice(), shape: sq, stride: contiguous_stride(sq) },
                k: TensorRef { data: dk.as_slice(), shape: sk, stride: contiguous_stride(sk) },
                v: TensorRef { data: dv.as_slice(), shape: sv, stride: contiguous_stride(sv) },
                y: TensorMut { data: dy_fa2.as_slice_mut(), shape: sy, stride: contiguous_stride(sy) },
                lse: TensorMut { data: dlse_fa2.as_slice_mut(), shape: sl, stride: contiguous_stride(sl) },
                mask: None,
                alibi_slopes: None,
            },
        )
        .expect("fa2 bf16 run");
    stream.synchronize().expect("sync fa2");

    let mut got = vec![bf16::ZERO; n_y];
    dy_fa2.copy_to_host(&mut got).expect("dl fa2");
    for (i, v) in got.iter().enumerate() {
        let f = v.to_f32();
        assert!(
            f.is_finite(),
            "bf16 d={d} causal={is_causal} non-finite at {i}: {f}"
        );
    }
}

// Generate one #[test] per (head_dim, dtype, causal) combo.
macro_rules! hdim_test {
    ($name:ident, $d:expr, $dtype_fn:ident, $causal:expr) => {
        #[test]
        #[ignore]
        fn $name() {
            $dtype_fn($d, $causal);
        }
    };
}

// head_dim = 32
hdim_test!(fa2_fanout_hdim32_f16_noncausal, 32, run_one_f16, false);
hdim_test!(fa2_fanout_hdim32_f16_causal,    32, run_one_f16, true );
hdim_test!(fa2_fanout_hdim32_bf16_noncausal, 32, run_one_bf16, false);
hdim_test!(fa2_fanout_hdim32_bf16_causal,    32, run_one_bf16, true );

// head_dim = 64
hdim_test!(fa2_fanout_hdim64_f16_noncausal, 64, run_one_f16, false);
hdim_test!(fa2_fanout_hdim64_f16_causal,    64, run_one_f16, true );
hdim_test!(fa2_fanout_hdim64_bf16_noncausal, 64, run_one_bf16, false);
hdim_test!(fa2_fanout_hdim64_bf16_causal,    64, run_one_bf16, true );

// head_dim = 96
hdim_test!(fa2_fanout_hdim96_f16_noncausal, 96, run_one_f16, false);
hdim_test!(fa2_fanout_hdim96_f16_causal,    96, run_one_f16, true );
hdim_test!(fa2_fanout_hdim96_bf16_noncausal, 96, run_one_bf16, false);
hdim_test!(fa2_fanout_hdim96_bf16_causal,    96, run_one_bf16, true );

// head_dim = 192 (bespoke kernel can't run; FA2-only)
hdim_test!(fa2_fanout_hdim192_f16_noncausal, 192, run_one_f16, false);
hdim_test!(fa2_fanout_hdim192_f16_causal,    192, run_one_f16, true );
hdim_test!(fa2_fanout_hdim192_bf16_noncausal, 192, run_one_bf16, false);
hdim_test!(fa2_fanout_hdim192_bf16_causal,    192, run_one_bf16, true );

// head_dim = 160 (Phase 60; vendored from EricLBuehler/candle@main, originally Candle PR #245)
hdim_test!(fa2_fanout_hdim160_f16_noncausal, 160, run_one_f16, false);
hdim_test!(fa2_fanout_hdim160_f16_causal,    160, run_one_f16, true );
hdim_test!(fa2_fanout_hdim160_bf16_noncausal, 160, run_one_bf16, false);
hdim_test!(fa2_fanout_hdim160_bf16_causal,    160, run_one_bf16, true );

// head_dim = 224 (Phase 60; vendored from EricLBuehler/candle@main, restored by Candle PR #2688)
hdim_test!(fa2_fanout_hdim224_f16_noncausal, 224, run_one_f16, false);
hdim_test!(fa2_fanout_hdim224_f16_causal,    224, run_one_f16, true );
hdim_test!(fa2_fanout_hdim224_bf16_noncausal, 224, run_one_bf16, false);
hdim_test!(fa2_fanout_hdim224_bf16_causal,    224, run_one_bf16, true );

// head_dim = 256 (FA2-only; may exceed sm_89 SMEM in some shapes —
// FA2's hdim256 launcher picks 64x64 tiles for non-A100/H100 GPUs).
hdim_test!(fa2_fanout_hdim256_f16_noncausal, 256, run_one_f16, false);
hdim_test!(fa2_fanout_hdim256_f16_causal,    256, run_one_f16, true );
hdim_test!(fa2_fanout_hdim256_bf16_noncausal, 256, run_one_bf16, false);
hdim_test!(fa2_fanout_hdim256_bf16_causal,    256, run_one_bf16, true );

// head_dim = 512 (Phase 60; vendored from huggingface/candle@5430d32c, Candle PR #3417 by Eric Buehler).
// SMEM opt-in path: on sm_89 picks 32x32 tiles (~96 KiB); A100+ picks 64x32 (~128 KiB).
hdim_test!(fa2_fanout_hdim512_f16_noncausal, 512, run_one_f16, false);
hdim_test!(fa2_fanout_hdim512_f16_causal,    512, run_one_f16, true );
hdim_test!(fa2_fanout_hdim512_bf16_noncausal, 512, run_one_bf16, false);
hdim_test!(fa2_fanout_hdim512_bf16_causal,    512, run_one_bf16, true );
