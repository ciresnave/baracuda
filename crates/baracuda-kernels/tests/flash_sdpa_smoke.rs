//! Real-GPU smoke test for `FlashSdpaPlan + AttentionKind::FlashAttention` FW.
//!
//! Validates the tiled / fused online-softmax flash kernel against the
//! naive [`SdpaPlan`] (Milestone 6.2) as ground truth — both kernels
//! compute the same math, just in a different float order.
//!
//! Covers FW × 4 FP dtypes × 2 mask modes (none + causal).
//! `#[ignore]` by default — requires a real CUDA device.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, FlashSdpaArgs, FlashSdpaDescriptor, FlashSdpaPlan,
    PlanPreference, SdpaArgs, SdpaDescriptor, SdpaPlan, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

// f16 / bf16 single-step tolerances. Flash's tile-by-tile online softmax
// vs naive's row-pass softmax produce slightly different rounding chains,
// so we use 32x (vs the 16x baseline) on top of the dtype eps.
const F16_EPS: f32 = 9.77e-4;
const BF16_EPS: f32 = 7.81e-3;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

// FlashAttention trailblazer requires d_k = d_v ≤ 128. Pick two q-blocks
// (Q = 2 × 64) and two k-blocks (K = 2 × 64) so the tiled pipeline is
// exercised non-trivially. Head dim 32 fits the 128 cap with room.
const B: i32 = 2;
const H: i32 = 4;
const Q: i32 = 128;
const K: i32 = 128;
const DK: i32 = 32;
const DV: i32 = 32;

fn default_scale() -> f32 {
    1.0 / (DK as f32).sqrt()
}

fn gen_q_f64() -> Vec<f64> {
    let n = (B * H * Q * DK) as usize;
    (0..n)
        .map(|i| ((i as f64) * 0.013 - 0.5).sin() * 0.5)
        .collect()
}
fn gen_k_f64() -> Vec<f64> {
    let n = (B * H * K * DK) as usize;
    (0..n)
        .map(|i| ((i as f64) * 0.017 + 0.2).cos() * 0.5)
        .collect()
}
fn gen_v_f64() -> Vec<f64> {
    let n = (B * H * K * DV) as usize;
    (0..n)
        .map(|i| ((i as f64) * 0.011 - 0.1).sin() * 0.7)
        .collect()
}

// ----------------------------------------------------------------------------
// f32 (covers no-mask + causal)
// ----------------------------------------------------------------------------

fn run_f32(is_causal: bool) {
    let (ctx, stream) = setup();
    let q_f64 = gen_q_f64();
    let k_f64 = gen_k_f64();
    let v_f64 = gen_v_f64();
    let scale = default_scale();

    let q_h: Vec<f32> = q_f64.iter().map(|&x| x as f32).collect();
    let k_h: Vec<f32> = k_f64.iter().map(|&x| x as f32).collect();
    let v_h: Vec<f32> = v_f64.iter().map(|&x| x as f32).collect();

    let dq = DeviceBuffer::from_slice(&ctx, &q_h).expect("up q");
    let dk = DeviceBuffer::from_slice(&ctx, &k_h).expect("up k");
    let dv = DeviceBuffer::from_slice(&ctx, &v_h).expect("up v");

    // Reference: naive SdpaPlan.
    let mut dattn_ref: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * K) as usize).expect("alloc attn");
    let mut dy_ref: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * DV) as usize).expect("alloc y_ref");

    let sq = [B, H, Q, DK];
    let sk = [B, H, K, DK];
    let sv = [B, H, K, DV];
    let sa = [B, H, Q, K];
    let sy = [B, H, Q, DV];
    let sl = [B, H, Q];

    let ref_desc = SdpaDescriptor {
        batch_size: B,
        num_heads: H,
        query_len: Q,
        key_len: K,
        d_k: DK,
        d_v: DV,
        scale,
        is_causal,
        has_mask: false,
        element: ElementKind::F32,
    };
    let ref_plan =
        SdpaPlan::<f32>::select(&stream, &ref_desc, PlanPreference::default()).expect("ref sel");
    ref_plan
        .run(
            &stream,
            Workspace::None,
            SdpaArgs {
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
                mask: None,
                y: TensorMut {
                    data: dy_ref.as_slice_mut(),
                    shape: sy,
                    stride: contiguous_stride(sy),
                },
                attn: TensorMut {
                    data: dattn_ref.as_slice_mut(),
                    shape: sa,
                    stride: contiguous_stride(sa),
                },
            },
        )
        .expect("ref run");
    stream.synchronize().expect("sync ref");

    // Flash.
    let mut dy_flash: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * DV) as usize).expect("alloc y_flash");
    let mut dlse: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (B * H * Q) as usize).expect("alloc lse");

    let desc = FlashSdpaDescriptor::new(
        B,
        H,
        Q,
        K,
        DK,
        DV,
        scale,
        is_causal,
        ElementKind::F32,
    );
    let plan =
        FlashSdpaPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("flash sel");
    plan.run(
        &stream,
        Workspace::None,
        FlashSdpaArgs {
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
                data: dy_flash.as_slice_mut(),
                shape: sy,
                stride: contiguous_stride(sy),
            },
            lse: TensorMut {
                data: dlse.as_slice_mut(),
                shape: sl,
                stride: contiguous_stride(sl),
            },
                mask: None,
                    alibi_slopes: None,
        },
    )
    .expect("flash run");
    stream.synchronize().expect("sync flash");

    let mut got = vec![0f32; (B * H * Q * DV) as usize];
    let mut refv = vec![0f32; (B * H * Q * DV) as usize];
    dy_flash.copy_to_host(&mut got).expect("dl flash");
    dy_ref.copy_to_host(&mut refv).expect("dl ref");

    // Flash vs naive: 32 * eps. ~10x naive's 16x is the documented
    // acceptable range (different float-order, same math).
    let tol = 32.0 * f32::EPSILON;
    for i in 0..got.len() {
        let r = refv[i];
        let diff = (got[i] - r).abs();
        let t = (r.abs() * tol).max(tol);
        assert!(
            diff <= t,
            "f32 flash-sdpa y @ {i}: diff={diff} flash={f} ref={r}",
            f = got[i]
        );
    }
}

#[test]
#[ignore]
fn flash_sdpa_f32_no_mask() {
    run_f32(false);
}

#[test]
#[ignore]
fn flash_sdpa_f32_causal() {
    run_f32(true);
}

// ----------------------------------------------------------------------------
// f64
// ----------------------------------------------------------------------------

#[test]
#[ignore]
fn flash_sdpa_f64_basic() {
    let (ctx, stream) = setup();
    let q_h = gen_q_f64();
    let k_h = gen_k_f64();
    let v_h = gen_v_f64();
    let scale = default_scale();

    let dq = DeviceBuffer::from_slice(&ctx, &q_h).expect("up q");
    let dk = DeviceBuffer::from_slice(&ctx, &k_h).expect("up k");
    let dv = DeviceBuffer::from_slice(&ctx, &v_h).expect("up v");

    let sq = [B, H, Q, DK];
    let sk = [B, H, K, DK];
    let sv = [B, H, K, DV];
    let sa = [B, H, Q, K];
    let sy = [B, H, Q, DV];
    let sl = [B, H, Q];

    let mut dattn_ref: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * K) as usize).expect("alloc attn");
    let mut dy_ref: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * DV) as usize).expect("alloc y_ref");

    let ref_desc = SdpaDescriptor {
        batch_size: B,
        num_heads: H,
        query_len: Q,
        key_len: K,
        d_k: DK,
        d_v: DV,
        scale,
        is_causal: false,
        has_mask: false,
        element: ElementKind::F64,
    };
    let ref_plan =
        SdpaPlan::<f64>::select(&stream, &ref_desc, PlanPreference::default()).expect("ref sel");
    ref_plan
        .run(
            &stream,
            Workspace::None,
            SdpaArgs {
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
                mask: None,
                y: TensorMut {
                    data: dy_ref.as_slice_mut(),
                    shape: sy,
                    stride: contiguous_stride(sy),
                },
                attn: TensorMut {
                    data: dattn_ref.as_slice_mut(),
                    shape: sa,
                    stride: contiguous_stride(sa),
                },
            },
        )
        .expect("ref run");
    stream.synchronize().expect("");

    let mut dy_flash: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * DV) as usize).expect("alloc y_flash");
    let mut dlse: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (B * H * Q) as usize).expect("alloc lse");

    let desc = FlashSdpaDescriptor::new(
        B,
        H,
        Q,
        K,
        DK,
        DV,
        scale,
        false,
        ElementKind::F64,
    );
    let plan =
        FlashSdpaPlan::<f64>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    plan.run(
        &stream,
        Workspace::None,
        FlashSdpaArgs {
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
                data: dy_flash.as_slice_mut(),
                shape: sy,
                stride: contiguous_stride(sy),
            },
            lse: TensorMut {
                data: dlse.as_slice_mut(),
                shape: sl,
                stride: contiguous_stride(sl),
            },
                mask: None,
                    alibi_slopes: None,
        },
    )
    .expect("run");
    stream.synchronize().expect("");

    let mut got = vec![0f64; (B * H * Q * DV) as usize];
    let mut refv = vec![0f64; (B * H * Q * DV) as usize];
    dy_flash.copy_to_host(&mut got).expect("");
    dy_ref.copy_to_host(&mut refv).expect("");

    let tol = 32.0 * f64::EPSILON;
    for i in 0..got.len() {
        let diff = (got[i] - refv[i]).abs();
        let t = (refv[i].abs() * tol).max(tol);
        assert!(diff <= t, "f64 flash-sdpa y @ {i}: diff={diff}");
    }
}

// ----------------------------------------------------------------------------
// f16
// ----------------------------------------------------------------------------

#[test]
#[ignore]
fn flash_sdpa_f16_basic() {
    let (ctx, stream) = setup();
    let q_f64 = gen_q_f64();
    let k_f64 = gen_k_f64();
    let v_f64 = gen_v_f64();
    let scale = default_scale();

    let q_h: Vec<f16> = q_f64.iter().map(|&v| f16::from_f64(v)).collect();
    let k_h: Vec<f16> = k_f64.iter().map(|&v| f16::from_f64(v)).collect();
    let v_h: Vec<f16> = v_f64.iter().map(|&v| f16::from_f64(v)).collect();
    let dq = DeviceBuffer::from_slice(&ctx, &q_h).expect("up q");
    let dk = DeviceBuffer::from_slice(&ctx, &k_h).expect("up k");
    let dv = DeviceBuffer::from_slice(&ctx, &v_h).expect("up v");

    let sq = [B, H, Q, DK];
    let sk = [B, H, K, DK];
    let sv = [B, H, K, DV];
    let sa = [B, H, Q, K];
    let sy = [B, H, Q, DV];
    let sl = [B, H, Q];

    let mut dattn_ref: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * K) as usize).expect("alloc attn");
    let mut dy_ref: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * DV) as usize).expect("alloc y_ref");

    let ref_desc = SdpaDescriptor {
        batch_size: B,
        num_heads: H,
        query_len: Q,
        key_len: K,
        d_k: DK,
        d_v: DV,
        scale,
        is_causal: false,
        has_mask: false,
        element: ElementKind::F16,
    };
    let ref_plan =
        SdpaPlan::<f16>::select(&stream, &ref_desc, PlanPreference::default()).expect("ref sel");
    ref_plan
        .run(
            &stream,
            Workspace::None,
            SdpaArgs {
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
                mask: None,
                y: TensorMut {
                    data: dy_ref.as_slice_mut(),
                    shape: sy,
                    stride: contiguous_stride(sy),
                },
                attn: TensorMut {
                    data: dattn_ref.as_slice_mut(),
                    shape: sa,
                    stride: contiguous_stride(sa),
                },
            },
        )
        .expect("ref run");
    stream.synchronize().expect("");

    let mut dy_flash: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * DV) as usize).expect("alloc y_flash");
    let mut dlse: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q) as usize).expect("alloc lse");

    let desc = FlashSdpaDescriptor::new(
        B,
        H,
        Q,
        K,
        DK,
        DV,
        scale,
        false,
        ElementKind::F16,
    );
    let plan =
        FlashSdpaPlan::<f16>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    plan.run(
        &stream,
        Workspace::None,
        FlashSdpaArgs {
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
                data: dy_flash.as_slice_mut(),
                shape: sy,
                stride: contiguous_stride(sy),
            },
            lse: TensorMut {
                data: dlse.as_slice_mut(),
                shape: sl,
                stride: contiguous_stride(sl),
            },
                mask: None,
                    alibi_slopes: None,
        },
    )
    .expect("run");
    stream.synchronize().expect("");

    let mut got = vec![f16::ZERO; (B * H * Q * DV) as usize];
    let mut refv = vec![f16::ZERO; (B * H * Q * DV) as usize];
    dy_flash.copy_to_host(&mut got).expect("");
    dy_ref.copy_to_host(&mut refv).expect("");

    // Per-storage difference: same dtype, same f32-accum math, slightly
    // different float-order. 32 ULP envelope.
    let tol = 32.0 * F16_EPS;
    for i in 0..got.len() {
        let r = refv[i].to_f32();
        let g = got[i].to_f32();
        let diff = (g - r).abs();
        let t = (r.abs() * tol).max(tol);
        assert!(
            diff <= t,
            "f16 flash-sdpa y @ {i}: diff={diff} flash={g} ref={r}"
        );
    }
}

// ----------------------------------------------------------------------------
// bf16
// ----------------------------------------------------------------------------

#[test]
#[ignore]
fn flash_sdpa_bf16_basic() {
    let (ctx, stream) = setup();
    let q_f64 = gen_q_f64();
    let k_f64 = gen_k_f64();
    let v_f64 = gen_v_f64();
    let scale = default_scale();

    let q_h: Vec<bf16> = q_f64.iter().map(|&v| bf16::from_f64(v)).collect();
    let k_h: Vec<bf16> = k_f64.iter().map(|&v| bf16::from_f64(v)).collect();
    let v_h: Vec<bf16> = v_f64.iter().map(|&v| bf16::from_f64(v)).collect();
    let dq = DeviceBuffer::from_slice(&ctx, &q_h).expect("up q");
    let dk = DeviceBuffer::from_slice(&ctx, &k_h).expect("up k");
    let dv = DeviceBuffer::from_slice(&ctx, &v_h).expect("up v");

    let sq = [B, H, Q, DK];
    let sk = [B, H, K, DK];
    let sv = [B, H, K, DV];
    let sa = [B, H, Q, K];
    let sy = [B, H, Q, DV];
    let sl = [B, H, Q];

    let mut dattn_ref: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * K) as usize).expect("alloc attn");
    let mut dy_ref: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * DV) as usize).expect("alloc y_ref");

    let ref_desc = SdpaDescriptor {
        batch_size: B,
        num_heads: H,
        query_len: Q,
        key_len: K,
        d_k: DK,
        d_v: DV,
        scale,
        is_causal: false,
        has_mask: false,
        element: ElementKind::Bf16,
    };
    let ref_plan =
        SdpaPlan::<bf16>::select(&stream, &ref_desc, PlanPreference::default()).expect("ref sel");
    ref_plan
        .run(
            &stream,
            Workspace::None,
            SdpaArgs {
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
                mask: None,
                y: TensorMut {
                    data: dy_ref.as_slice_mut(),
                    shape: sy,
                    stride: contiguous_stride(sy),
                },
                attn: TensorMut {
                    data: dattn_ref.as_slice_mut(),
                    shape: sa,
                    stride: contiguous_stride(sa),
                },
            },
        )
        .expect("ref run");
    stream.synchronize().expect("");

    let mut dy_flash: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * DV) as usize).expect("alloc y_flash");
    let mut dlse: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q) as usize).expect("alloc lse");

    let desc = FlashSdpaDescriptor::new(
        B,
        H,
        Q,
        K,
        DK,
        DV,
        scale,
        false,
        ElementKind::Bf16,
    );
    let plan =
        FlashSdpaPlan::<bf16>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    plan.run(
        &stream,
        Workspace::None,
        FlashSdpaArgs {
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
                data: dy_flash.as_slice_mut(),
                shape: sy,
                stride: contiguous_stride(sy),
            },
            lse: TensorMut {
                data: dlse.as_slice_mut(),
                shape: sl,
                stride: contiguous_stride(sl),
            },
                mask: None,
                    alibi_slopes: None,
        },
    )
    .expect("run");
    stream.synchronize().expect("");

    let mut got = vec![bf16::ZERO; (B * H * Q * DV) as usize];
    let mut refv = vec![bf16::ZERO; (B * H * Q * DV) as usize];
    dy_flash.copy_to_host(&mut got).expect("");
    dy_ref.copy_to_host(&mut refv).expect("");

    let tol = 32.0 * BF16_EPS;
    for i in 0..got.len() {
        let r = refv[i].to_f32();
        let g = got[i].to_f32();
        let diff = (g - r).abs();
        let t = (r.abs() * tol).max(tol);
        assert!(
            diff <= t,
            "bf16 flash-sdpa y @ {i}: diff={diff} flash={g} ref={r}"
        );
    }
}

// =============================================================================
// Phase 73 follow-up — full-MQA-broadcast routing.
//
// Validates that FlashSdpaPlan accepts K/V with stride[1] = 0 (the
// broadcast convention for full MQA) and routes to FlashSdpaSm89Plan
// under the hood. Only meaningful when the `sm89` cargo feature is on.
// =============================================================================

/// Parameterized broadcast-route check. Runs the same plan twice — once
/// with the stride-0 broadcast K/V view and once with a physically
/// replicated `[B, H, K, D]` K/V — and asserts the two outputs match.
///
/// The (B, H, Q, K, D) shape selects which routing the broadcast view
/// hits inside `FlashSdpaPlan::run`:
///   - small `seq_q × seq_k` (< 1M) → bespoke selected → sm89 sibling
///     (head_dim ≤ 64).
///   - large `seq_q × seq_k` (≥ 1M) → FA2 selected → physical
///     [B, 1, K, D] reinterpretation + FA2 MQA (any FA2 head_dim).
#[cfg(feature = "sm89")]
fn run_broadcast_route_case(b: i32, h: i32, q: i32, k: i32, d: i32, tol_mul: f32, label: &str) {
    let (ctx, stream) = setup();
    let scale = 1.0_f32 / (d as f32).sqrt();

    let q_f64 = (0..(b * h * q * d) as usize)
        .map(|i| ((i as f64) * 0.0007 - 0.5).sin() * 0.5)
        .collect::<Vec<_>>();
    // Single physical KV head: [B, 1, K, D].
    let k_one_head_f64 = (0..(b * k * d) as usize)
        .map(|i| ((i as f64) * 0.0011 + 0.2).cos() * 0.5)
        .collect::<Vec<_>>();
    let v_one_head_f64 = (0..(b * k * d) as usize)
        .map(|i| ((i as f64) * 0.0009 - 0.1).sin() * 0.7)
        .collect::<Vec<_>>();

    let q_h: Vec<f16> = q_f64.iter().map(|&x| f16::from_f32(x as f32)).collect();
    let k_h: Vec<f16> = k_one_head_f64.iter().map(|&x| f16::from_f32(x as f32)).collect();
    let v_h: Vec<f16> = v_one_head_f64.iter().map(|&x| f16::from_f32(x as f32)).collect();

    let dq = DeviceBuffer::from_slice(&ctx, &q_h).expect("up q");
    let dk = DeviceBuffer::from_slice(&ctx, &k_h).expect("up k");
    let dv = DeviceBuffer::from_slice(&ctx, &v_h).expect("up v");
    let mut dy: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (b * h * q * d) as usize).expect("alloc y");
    let mut dlse: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (b * h * q) as usize).expect("alloc lse");

    // Broadcast view: shape advertises H heads, stride[1] = 0; the
    // remaining strides describe the physical [B, 1, K, D] buffer
    // (stride[0] = K*D, stride[2] = D, stride[3] = 1).
    let sq = [b, h, q, d];
    let stq = contiguous_stride(sq);
    let sk = [b, h, k, d];
    let stk = [(k as i64) * (d as i64), 0, d as i64, 1];
    let sy = sq;
    let sty = stq;
    let sl = [b, h, q];
    let stl = contiguous_stride(sl);

    let desc = FlashSdpaDescriptor::new(b, h, q, k, d, d, scale, false, ElementKind::F16);
    let plan = FlashSdpaPlan::<f16>::select(&stream, &desc, PlanPreference::default())
        .expect("select");

    // Workspace sized from the plan (FA2 backend needs LSE scratch;
    // bespoke/sm89 reports 0).
    let ws_bytes = plan.workspace_size();
    let mut ws: Option<DeviceBuffer<u8>> = if ws_bytes > 0 {
        Some(DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws"))
    } else {
        None
    };

    {
        let wsp = match ws.as_mut() {
            Some(b) => Workspace::Borrowed(b.as_slice_mut()),
            None => Workspace::None,
        };
        let args = FlashSdpaArgs::<f16> {
            q: TensorRef { data: dq.as_slice(), shape: sq, stride: stq },
            k: TensorRef { data: dk.as_slice(), shape: sk, stride: stk },
            v: TensorRef { data: dv.as_slice(), shape: sk, stride: stk },
            y: TensorMut { data: dy.as_slice_mut(), shape: sy, stride: sty },
            lse: TensorMut { data: dlse.as_slice_mut(), shape: sl, stride: stl },
            mask: None,
            alibi_slopes: None,
        };
        plan.run(&stream, wsp, args).unwrap_or_else(|e| panic!("{label}: broadcast run: {e:?}"));
    }
    stream.synchronize().expect("sync");

    // Reference: physically replicate the single KV head to [B, H, K, D]
    // and run the same plan with contiguous K/V.
    let hk_kd = (h as usize) * (k as usize) * (d as usize);
    let kd = (k as usize) * (d as usize);
    let k_rep: Vec<f16> = (0..(b * h * k * d) as usize)
        .map(|i| k_h[(i / hk_kd) * kd + (i % hk_kd) % kd])
        .collect();
    let v_rep: Vec<f16> = (0..(b * h * k * d) as usize)
        .map(|i| v_h[(i / hk_kd) * kd + (i % hk_kd) % kd])
        .collect();
    let dk_rep = DeviceBuffer::from_slice(&ctx, &k_rep).expect("up k_rep");
    let dv_rep = DeviceBuffer::from_slice(&ctx, &v_rep).expect("up v_rep");
    let mut dy_rep: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (b * h * q * d) as usize).expect("alloc y_rep");
    let mut dlse_rep: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (b * h * q) as usize).expect("alloc lse_rep");

    {
        let wsp = match ws.as_mut() {
            Some(b) => Workspace::Borrowed(b.as_slice_mut()),
            None => Workspace::None,
        };
        let args_rep = FlashSdpaArgs::<f16> {
            q: TensorRef { data: dq.as_slice(), shape: sq, stride: stq },
            k: TensorRef { data: dk_rep.as_slice(), shape: sk, stride: contiguous_stride(sk) },
            v: TensorRef { data: dv_rep.as_slice(), shape: sk, stride: contiguous_stride(sk) },
            y: TensorMut { data: dy_rep.as_slice_mut(), shape: sy, stride: sty },
            lse: TensorMut { data: dlse_rep.as_slice_mut(), shape: sl, stride: stl },
            mask: None,
            alibi_slopes: None,
        };
        plan.run(&stream, wsp, args_rep).unwrap_or_else(|e| panic!("{label}: rep run: {e:?}"));
    }
    stream.synchronize().expect("sync rep");

    let mut got = vec![f16::ZERO; (b * h * q * d) as usize];
    let mut rep = vec![f16::ZERO; (b * h * q * d) as usize];
    dy.copy_to_host(&mut got).expect("dl");
    dy_rep.copy_to_host(&mut rep).expect("dl rep");

    let tol = tol_mul * F16_EPS;
    for i in 0..got.len() {
        let g = got[i].to_f32();
        let r = rep[i].to_f32();
        let diff = (g - r).abs();
        let t = (r.abs() * tol).max(tol);
        assert!(
            diff <= t,
            "{label}: broadcast vs replicated f16 @ {i}: diff={diff} broadcast={g} replicated={r}"
        );
    }
}

// =============================================================================
// Phase 73 follow-up — full-MQA-broadcast routing.
//
// Validates that FlashSdpaPlan accepts K/V with stride[1] = 0 (the
// broadcast convention for full MQA) and produces the same result as a
// physically-replicated K/V. Two cases exercise the two routing paths:
//   - D=32 small shape → bespoke selected → sm89 strided sibling.
//   - D=128 large shape → FA2 selected → physical [B,1,K,D] + FA2 MQA.
// Only meaningful when the `sm89` cargo feature is on (and, for the
// large case, `fa2` — which is a default feature).
// =============================================================================

#[cfg(feature = "sm89")]
#[test]
#[ignore]
fn flash_sdpa_f16_full_mqa_broadcast_route_small_sm89() {
    // seq_q × seq_k = 64 × 128 = 8192 < 1M → bespoke → sm89 (D=32 ≤ 64).
    run_broadcast_route_case(2, 4, 64, 128, 32, 32.0, "broadcast/small-sm89 D32");
}

#[cfg(all(feature = "sm89", feature = "fa2"))]
#[test]
#[ignore]
fn flash_sdpa_f16_full_mqa_broadcast_route_large_fa2() {
    // seq_q × seq_k = 2048 × 2048 = 4M ≥ 1M → FA2 → physical [B,1,K,D]
    // reinterpretation + FA2 MQA at head_dim 128 (the case the sm89
    // sibling can't fit in SMEM). Looser tolerance: FA2's tensor-core
    // accumulation differs in float-order from the naive replicated
    // reference more than the bespoke-vs-bespoke comparison does.
    run_broadcast_route_case(1, 32, 2048, 2048, 128, 96.0, "broadcast/large-fa2 D128");
}
