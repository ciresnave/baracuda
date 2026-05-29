//! Phase 59a — softcap smoke test for FA2.
//!
//! Softcap is a tanh-style score capping used by Gemma-2 to bound the
//! attention scores before softmax:
//!
//!     scores = softcap * tanh(scores / softcap)
//!
//! This bounds `|scores| < softcap`, which improves training stability.
//! `softcap = 0.0` disables (default). Positive values enable.
//!
//! Tests:
//!   1. softcap=30.0 (Gemma-2 default) runs cleanly + outputs finite.
//!   2. softcap interacts with causal masking.
//!   3. Bespoke backend rejects softcap with `Error::Unsupported`.
//!   4. Negative softcap is rejected at select time.

#![cfg(feature = "fa2")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, BackendKind, ElementKind, FlashSdpaArgs, FlashSdpaDescriptor,
    FlashSdpaPlan, PlanPreference, TensorMut, TensorRef, Workspace,
};
use baracuda_cutlass::Error;
use half::f16;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

const B: i32 = 1;
const H: i32 = 2;
const Q: i32 = 1024;
const K: i32 = 1024;
const D: i32 = 128;

fn default_scale() -> f32 {
    1.0 / (D as f32).sqrt()
}

fn gen_f16(n: usize, phase: f32) -> Vec<f16> {
    (0..n)
        .map(|i| f16::from_f32(((i as f32) * 0.013 + phase).sin() * 0.5))
        .collect()
}

fn run_with_softcap(softcap: f32, is_causal: bool) {
    let (ctx, stream) = setup();
    let n_q = (B * H * Q * D) as usize;
    let n_kv = (B * H * K * D) as usize;
    let n_y = (B * H * Q * D) as usize;

    let q_h = gen_f16(n_q, 0.0);
    let k_h = gen_f16(n_kv, 0.7);
    let v_h = gen_f16(n_kv, 1.3);

    let dq = DeviceBuffer::from_slice(&ctx, &q_h).expect("up q");
    let dk = DeviceBuffer::from_slice(&ctx, &k_h).expect("up k");
    let dv = DeviceBuffer::from_slice(&ctx, &v_h).expect("up v");

    let sq = [B, H, Q, D];
    let sk = [B, H, K, D];
    let sv = [B, H, K, D];
    let sy = [B, H, Q, D];
    let sl = [B, H, Q];

    let mut dy: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, n_y).expect("alloc y");
    let mut dlse: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q) as usize).expect("alloc lse");

    let desc = FlashSdpaDescriptor::new(
        B, H, Q, K, D, D, default_scale(), is_causal, ElementKind::F16,
    )
    .with_softcap(softcap);

    let pref = PlanPreference {
        prefer_backend: Some(BackendKind::FlashAttentionV2),
        ..Default::default()
    };
    let plan = FlashSdpaPlan::<f16>::select(&stream, &desc, pref).expect("sel fa2");
    let ws_bytes = plan.workspace_size();
    let mut ws_buf: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws");

    plan.run(
        &stream,
        Workspace::Borrowed(ws_buf.as_slice_mut()),
        FlashSdpaArgs {
            q: TensorRef { data: dq.as_slice(), shape: sq, stride: contiguous_stride(sq) },
            k: TensorRef { data: dk.as_slice(), shape: sk, stride: contiguous_stride(sk) },
            v: TensorRef { data: dv.as_slice(), shape: sv, stride: contiguous_stride(sv) },
            y: TensorMut { data: dy.as_slice_mut(), shape: sy, stride: contiguous_stride(sy) },
            lse: TensorMut { data: dlse.as_slice_mut(), shape: sl, stride: contiguous_stride(sl) },
            mask: None,
            alibi_slopes: None,
        },
    )
    .expect("softcap run");
    stream.synchronize().expect("sync");

    let mut got = vec![f16::ZERO; n_y];
    dy.copy_to_host(&mut got).expect("dl");
    for (i, v) in got.iter().enumerate() {
        let f = v.to_f32();
        assert!(
            f.is_finite(),
            "softcap={softcap} causal={is_causal} non-finite @ {i}: {f}"
        );
    }
}

#[test]
#[ignore]
fn fa2_softcap_gemma_default_noncausal() {
    run_with_softcap(30.0, false);
}

#[test]
#[ignore]
fn fa2_softcap_gemma_default_causal() {
    run_with_softcap(30.0, true);
}

#[test]
#[ignore]
fn fa2_softcap_rejected_on_bespoke() {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    let _ = ctx;

    let desc = FlashSdpaDescriptor::new(
        B, H, 128, 128, D, D, default_scale(), false, ElementKind::F16,
    )
    .with_softcap(30.0);
    let pref = PlanPreference {
        prefer_backend: Some(BackendKind::Bespoke),
        ..Default::default()
    };
    match FlashSdpaPlan::<f16>::select(&stream, &desc, pref) {
        Err(e) => assert!(matches!(e, Error::Unsupported(_)), "got: {e:?}"),
        Ok(_) => panic!("expected Error::Unsupported on bespoke + softcap"),
    }
}

#[test]
#[ignore]
fn fa2_softcap_negative_rejected() {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    let _ = ctx;

    let desc = FlashSdpaDescriptor::new(
        B, H, 1024, 1024, D, D, default_scale(), false, ElementKind::F16,
    )
    .with_softcap(-1.0);
    let pref = PlanPreference {
        prefer_backend: Some(BackendKind::FlashAttentionV2),
        ..Default::default()
    };
    match FlashSdpaPlan::<f16>::select(&stream, &desc, pref) {
        Err(e) => assert!(matches!(e, Error::InvalidProblem(_)), "got: {e:?}"),
        Ok(_) => panic!("expected Error::InvalidProblem on negative softcap"),
    }
}
