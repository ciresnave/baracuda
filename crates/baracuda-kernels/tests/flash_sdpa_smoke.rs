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

    let desc = FlashSdpaDescriptor {
        batch_size: B,
        num_heads: H,
        query_len: Q,
        key_len: K,
        d_k: DK,
        d_v: DV,
        scale,
        is_causal,
        element: ElementKind::F32,
    };
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

    let desc = FlashSdpaDescriptor {
        batch_size: B,
        num_heads: H,
        query_len: Q,
        key_len: K,
        d_k: DK,
        d_v: DV,
        scale,
        is_causal: false,
        element: ElementKind::F64,
    };
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

    let desc = FlashSdpaDescriptor {
        batch_size: B,
        num_heads: H,
        query_len: Q,
        key_len: K,
        d_k: DK,
        d_v: DV,
        scale,
        is_causal: false,
        element: ElementKind::F16,
    };
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

    let desc = FlashSdpaDescriptor {
        batch_size: B,
        num_heads: H,
        query_len: Q,
        key_len: K,
        d_k: DK,
        d_v: DV,
        scale,
        is_causal: false,
        element: ElementKind::Bf16,
    };
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
