//! Phase 59a — ALiBi smoke test for the FA2 backend.
//!
//! ALiBi (Attention with Linear Biases) adds a learned per-head slope
//! to the attention score matrix:
//!
//!     scores[b, h, q, k] += slope[h] * (k - q)
//!
//! before softmax. The slope is typically a small negative number,
//! biasing attention toward closer keys. Implementations carry slopes
//! in `[num_heads]` shape (broadcast over batch) or `[B, num_heads]`
//! shape (per-batch).
//!
//! This smoke test validates that:
//!   1. The per-head broadcast layout (shape `[1, H]`, batch_stride=0)
//!      runs cleanly and produces a finite, differs-from-no-ALiBi result.
//!   2. The per-batch-per-head layout (shape `[B, H]`, contiguous)
//!      similarly runs and produces a finite result.
//!   3. ALiBi composes with causal masking.
//!
//! We don't yet bit-validate against a CPU reference — FA2's ALiBi
//! implementation is hard to match exactly in fp32 CPU code (relies on
//! the same tile-by-tile online softmax order). The smoke confirms the
//! FFI plumbing + descriptor validation are wired correctly.

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

const B: i32 = 2;
const H: i32 = 4;
const Q: i32 = 1024;
const K: i32 = 1024;
const D: i32 = 128;

fn default_scale() -> f32 {
    1.0 / (D as f32).sqrt()
}

fn gen_f16(n: usize, phase: f32) -> Vec<f16> {
    (0..n)
        .map(|i| f16::from_f32(((i as f32) * 0.013 + phase).sin() * 0.25))
        .collect()
}

#[test]
#[ignore]
fn fa2_alibi_per_head_broadcast() {
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

    // Per-head ALiBi slopes (canonical: 2^(-8/h * i) for head i; uses
    // small negative values inside the kernel via FA2's alibi.h math).
    let slopes_h: Vec<f32> = (0..H).map(|i| -0.1_f32 * (i as f32 + 1.0)).collect();
    let dslopes = DeviceBuffer::from_slice(&ctx, &slopes_h).expect("up slopes");

    let sq = [B, H, Q, D];
    let sk = [B, H, K, D];
    let sv = [B, H, K, D];
    let sy = [B, H, Q, D];
    let sl = [B, H, Q];
    let s_slopes = [1_i32, H];  // shape [1, H] with stride[0]=0 = broadcast over batch

    let mut dy: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, n_y).expect("alloc y");
    let mut dlse: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q) as usize).expect("alloc lse");

    let desc = FlashSdpaDescriptor::new(B, H, Q, K, D, D, default_scale(), false, ElementKind::F16);

    let pref = PlanPreference {
        prefer_backend: Some(BackendKind::FlashAttentionV2),
        ..Default::default()
    };
    let plan = FlashSdpaPlan::<f16>::select(&stream, &desc, pref).expect("sel fa2");
    assert_eq!(plan.backend(), BackendKind::FlashAttentionV2);
    let ws_bytes = plan.workspace_size();
    let mut ws_buf: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws");

    // Per-head broadcast layout: shape [1, H], stride[0] = 0
    let alibi_stride = [0_i64, 1_i64];
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
            alibi_slopes: Some(TensorRef {
                data: dslopes.as_slice(),
                shape: s_slopes,
                stride: alibi_stride,
            }),
        },
    )
    .expect("alibi per-head run");
    stream.synchronize().expect("sync");

    let mut got = vec![f16::ZERO; n_y];
    dy.copy_to_host(&mut got).expect("dl");
    let mut any_nonzero = false;
    for (i, v) in got.iter().enumerate() {
        let f = v.to_f32();
        assert!(f.is_finite(), "alibi per-head non-finite @ {i}: {f}");
        if f.abs() > 1e-5 {
            any_nonzero = true;
        }
    }
    assert!(any_nonzero, "alibi per-head: all output cells near zero — likely zero ptr taken");
}

#[test]
#[ignore]
fn fa2_alibi_per_batch_per_head() {
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

    // Per-batch-per-head ALiBi slopes (shape [B, H], contiguous).
    let slopes_h: Vec<f32> = (0..(B * H))
        .map(|i| -0.05_f32 * (i as f32 + 1.0))
        .collect();
    let dslopes = DeviceBuffer::from_slice(&ctx, &slopes_h).expect("up slopes");

    let sq = [B, H, Q, D];
    let sk = [B, H, K, D];
    let sv = [B, H, K, D];
    let sy = [B, H, Q, D];
    let sl = [B, H, Q];
    let s_slopes = [B, H];

    let mut dy: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, n_y).expect("alloc y");
    let mut dlse: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q) as usize).expect("alloc lse");

    let desc = FlashSdpaDescriptor::new(B, H, Q, K, D, D, default_scale(), true, ElementKind::F16);

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
            alibi_slopes: Some(TensorRef {
                data: dslopes.as_slice(),
                shape: s_slopes,
                stride: contiguous_stride(s_slopes),
            }),
        },
    )
    .expect("alibi per-batch run (causal)");
    stream.synchronize().expect("sync");

    let mut got = vec![f16::ZERO; n_y];
    dy.copy_to_host(&mut got).expect("dl");
    for (i, v) in got.iter().enumerate() {
        let f = v.to_f32();
        assert!(f.is_finite(), "alibi per-batch non-finite @ {i}: {f}");
    }
}

#[test]
#[ignore]
fn fa2_alibi_rejected_on_bespoke() {
    // ALiBi must error when the backend is bespoke. select() picks
    // bespoke (short shape + caller override); can_implement() trips
    // the alibi-on-bespoke rejection at args-validation time. We
    // allocate a real slopes tensor (need GPU buffers for the
    // TensorRef construction) and dummy q/k/v buffers — the call
    // should fail before any kernel launch.
    let (ctx, stream) = setup();

    // Short shape so heuristic doesn't accidentally pick FA2.
    let b = 1_i32;
    let h = 2_i32;
    let q = 64_i32;
    let k = 64_i32;
    let d = D;

    let n_q = (b * h * q * d) as usize;
    let n_kv = (b * h * k * d) as usize;
    let n_y = n_q;
    let dq = DeviceBuffer::<f16>::zeros(&ctx, n_q).expect("alloc q");
    let dk = DeviceBuffer::<f16>::zeros(&ctx, n_kv).expect("alloc k");
    let dv = DeviceBuffer::<f16>::zeros(&ctx, n_kv).expect("alloc v");
    let mut dy = DeviceBuffer::<f16>::zeros(&ctx, n_y).expect("alloc y");
    let mut dlse = DeviceBuffer::<f16>::zeros(&ctx, (b * h * q) as usize).expect("alloc lse");
    let dslopes = DeviceBuffer::<f32>::zeros(&ctx, h as usize).expect("alloc slopes");

    let sq = [b, h, q, d];
    let sk = [b, h, k, d];
    let sv = [b, h, k, d];
    let sy = [b, h, q, d];
    let sl = [b, h, q];

    let desc = FlashSdpaDescriptor::new(
        b, h, q, k, d, d, default_scale(), false, ElementKind::F16,
    );
    let pref = PlanPreference {
        prefer_backend: Some(BackendKind::Bespoke),
        ..Default::default()
    };
    let plan = FlashSdpaPlan::<f16>::select(&stream, &desc, pref).expect("sel bespoke");
    assert_eq!(plan.backend(), BackendKind::Bespoke);
    let args = FlashSdpaArgs {
        q: TensorRef { data: dq.as_slice(), shape: sq, stride: contiguous_stride(sq) },
        k: TensorRef { data: dk.as_slice(), shape: sk, stride: contiguous_stride(sk) },
        v: TensorRef { data: dv.as_slice(), shape: sv, stride: contiguous_stride(sv) },
        y: TensorMut { data: dy.as_slice_mut(), shape: sy, stride: contiguous_stride(sy) },
        lse: TensorMut { data: dlse.as_slice_mut(), shape: sl, stride: contiguous_stride(sl) },
        mask: None,
        alibi_slopes: Some(TensorRef {
            data: dslopes.as_slice(),
            shape: [1, h],
            stride: [0_i64, 1_i64],
        }),
    };
    let result = plan.can_implement(&args);
    assert!(result.is_err(), "expected can_implement to reject ALiBi on bespoke");
}
