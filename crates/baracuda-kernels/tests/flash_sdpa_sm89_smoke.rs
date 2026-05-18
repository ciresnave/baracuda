//! Real-GPU smoke test for `FlashSdpaSm89Plan` — Phase 10 Milestone 10.3.
//!
//! Cross-checks the sm_89 (Ada Lovelace) Flash Attention FW sibling
//! against the sm_80 baseline `FlashSdpaPlan`. Both kernels implement
//! the same Tri Dao 2022 algorithm with identical f32 inner-loop math;
//! the only difference is the data-movement strategy (`cp.async` double-
//! buffered K/V loads + 256-thread block on sm_89). Therefore the
//! outputs should match to a few ULPs of float-order noise.
//!
//! Gated on the `sm89` cargo feature. `#[ignore]` by default — requires
//! a real Ada-class CUDA device. Builds (but skips at runtime) on Ampere
//! since the kernel uses `cp.async` which IS available on sm_80, but
//! we still mark this as sm_89-targeted because the wider-block heuristic
//! is tuned for Ada.

#![cfg(feature = "sm89")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, FlashSdpaArgs, FlashSdpaDescriptor, FlashSdpaPlan,
    FlashSdpaSm89Args, FlashSdpaSm89Descriptor, FlashSdpaSm89Plan, PlanPreference, TensorMut,
    TensorRef, Workspace,
};
use half::{bf16, f16};

// f16 / bf16 single-step tolerances (dtype eps).
const F16_EPS: f32 = 9.77e-4;
const BF16_EPS: f32 = 7.81e-3;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

// Fixture matches `flash_sdpa_smoke.rs` so both tests share generators.
// B=2, H=4, Q=128 (two q-blocks), K=128 (two k-blocks → exercises the
// double-buffer pipeline non-trivially), D=32.
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
// f16
// ----------------------------------------------------------------------------

#[test]
#[ignore]
fn flash_sdpa_sm89_f16_basic() {
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
    let sy = [B, H, Q, DV];
    let sl = [B, H, Q];

    // Reference: sm_80 baseline FlashSdpaPlan.
    let mut dy_ref: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * DV) as usize).expect("alloc y_ref");
    let mut dlse_ref: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q) as usize).expect("alloc lse_ref");

    let ref_desc = FlashSdpaDescriptor {
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
    let ref_plan = FlashSdpaPlan::<f16>::select(&stream, &ref_desc, PlanPreference::default())
        .expect("ref sel");
    ref_plan
        .run(
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
                    data: dy_ref.as_slice_mut(),
                    shape: sy,
                    stride: contiguous_stride(sy),
                },
                lse: TensorMut {
                    data: dlse_ref.as_slice_mut(),
                    shape: sl,
                    stride: contiguous_stride(sl),
                },
            },
        )
        .expect("ref run");
    stream.synchronize().expect("sync ref");

    // sm_89 variant.
    let mut dy_sm89: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * DV) as usize).expect("alloc y_sm89");
    let mut dlse_sm89: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q) as usize).expect("alloc lse_sm89");

    let desc = FlashSdpaSm89Descriptor {
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
    let plan = FlashSdpaSm89Plan::<f16>::select(&stream, &desc, PlanPreference::default())
        .expect("sm89 sel");
    plan.run(
        &stream,
        Workspace::None,
        FlashSdpaSm89Args {
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
                data: dy_sm89.as_slice_mut(),
                shape: sy,
                stride: contiguous_stride(sy),
            },
            lse: TensorMut {
                data: dlse_sm89.as_slice_mut(),
                shape: sl,
                stride: contiguous_stride(sl),
            },
        },
    )
    .expect("sm89 run");
    stream.synchronize().expect("sync sm89");

    let mut got = vec![f16::ZERO; (B * H * Q * DV) as usize];
    let mut refv = vec![f16::ZERO; (B * H * Q * DV) as usize];
    dy_sm89.copy_to_host(&mut got).expect("dl sm89");
    dy_ref.copy_to_host(&mut refv).expect("dl ref");

    // Same algo, same float-order in the inner loops — 64·eps envelope
    // (vs the 32·eps the baseline uses against naive SDPA, which has a
    // genuinely different float-order).
    let tol = 64.0 * F16_EPS;
    for i in 0..got.len() {
        let r = refv[i].to_f32();
        let g = got[i].to_f32();
        let diff = (g - r).abs();
        let t = (r.abs() * tol).max(tol);
        assert!(
            diff <= t,
            "f16 flash-sdpa-sm89 y @ {i}: diff={diff} sm89={g} ref={r}"
        );
    }
}

// ----------------------------------------------------------------------------
// bf16
// ----------------------------------------------------------------------------

#[test]
#[ignore]
fn flash_sdpa_sm89_bf16_basic() {
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
    let sy = [B, H, Q, DV];
    let sl = [B, H, Q];

    let mut dy_ref: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * DV) as usize).expect("alloc y_ref");
    let mut dlse_ref: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q) as usize).expect("alloc lse_ref");

    let ref_desc = FlashSdpaDescriptor {
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
    let ref_plan = FlashSdpaPlan::<bf16>::select(&stream, &ref_desc, PlanPreference::default())
        .expect("ref sel");
    ref_plan
        .run(
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
                    data: dy_ref.as_slice_mut(),
                    shape: sy,
                    stride: contiguous_stride(sy),
                },
                lse: TensorMut {
                    data: dlse_ref.as_slice_mut(),
                    shape: sl,
                    stride: contiguous_stride(sl),
                },
            },
        )
        .expect("ref run");
    stream.synchronize().expect("sync ref");

    let mut dy_sm89: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * DV) as usize).expect("alloc y_sm89");
    let mut dlse_sm89: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q) as usize).expect("alloc lse_sm89");

    let desc = FlashSdpaSm89Descriptor {
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
    let plan = FlashSdpaSm89Plan::<bf16>::select(&stream, &desc, PlanPreference::default())
        .expect("sm89 sel");
    plan.run(
        &stream,
        Workspace::None,
        FlashSdpaSm89Args {
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
                data: dy_sm89.as_slice_mut(),
                shape: sy,
                stride: contiguous_stride(sy),
            },
            lse: TensorMut {
                data: dlse_sm89.as_slice_mut(),
                shape: sl,
                stride: contiguous_stride(sl),
            },
        },
    )
    .expect("sm89 run");
    stream.synchronize().expect("sync sm89");

    let mut got = vec![bf16::ZERO; (B * H * Q * DV) as usize];
    let mut refv = vec![bf16::ZERO; (B * H * Q * DV) as usize];
    dy_sm89.copy_to_host(&mut got).expect("dl sm89");
    dy_ref.copy_to_host(&mut refv).expect("dl ref");

    let tol = 64.0 * BF16_EPS;
    for i in 0..got.len() {
        let r = refv[i].to_f32();
        let g = got[i].to_f32();
        let diff = (g - r).abs();
        let t = (r.abs() * tol).max(tol);
        assert!(
            diff <= t,
            "bf16 flash-sdpa-sm89 y @ {i}: diff={diff} sm89={g} ref={r}"
        );
    }
}

// ----------------------------------------------------------------------------
// Causal mask coverage (f16) — exercises the early-out path in the kernel.
// ----------------------------------------------------------------------------

#[test]
#[ignore]
fn flash_sdpa_sm89_f16_causal() {
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
    let sy = [B, H, Q, DV];
    let sl = [B, H, Q];

    let mut dy_ref: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * DV) as usize).expect("alloc y_ref");
    let mut dlse_ref: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q) as usize).expect("alloc lse_ref");

    let ref_desc = FlashSdpaDescriptor {
        batch_size: B,
        num_heads: H,
        query_len: Q,
        key_len: K,
        d_k: DK,
        d_v: DV,
        scale,
        is_causal: true,
        element: ElementKind::F16,
    };
    let ref_plan = FlashSdpaPlan::<f16>::select(&stream, &ref_desc, PlanPreference::default())
        .expect("ref sel");
    ref_plan
        .run(
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
                    data: dy_ref.as_slice_mut(),
                    shape: sy,
                    stride: contiguous_stride(sy),
                },
                lse: TensorMut {
                    data: dlse_ref.as_slice_mut(),
                    shape: sl,
                    stride: contiguous_stride(sl),
                },
            },
        )
        .expect("ref run");
    stream.synchronize().expect("sync ref");

    let mut dy_sm89: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * DV) as usize).expect("alloc y_sm89");
    let mut dlse_sm89: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q) as usize).expect("alloc lse_sm89");

    let desc = FlashSdpaSm89Descriptor {
        batch_size: B,
        num_heads: H,
        query_len: Q,
        key_len: K,
        d_k: DK,
        d_v: DV,
        scale,
        is_causal: true,
        element: ElementKind::F16,
    };
    let plan = FlashSdpaSm89Plan::<f16>::select(&stream, &desc, PlanPreference::default())
        .expect("sm89 sel");
    plan.run(
        &stream,
        Workspace::None,
        FlashSdpaSm89Args {
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
                data: dy_sm89.as_slice_mut(),
                shape: sy,
                stride: contiguous_stride(sy),
            },
            lse: TensorMut {
                data: dlse_sm89.as_slice_mut(),
                shape: sl,
                stride: contiguous_stride(sl),
            },
        },
    )
    .expect("sm89 run");
    stream.synchronize().expect("sync sm89");

    let mut got = vec![f16::ZERO; (B * H * Q * DV) as usize];
    let mut refv = vec![f16::ZERO; (B * H * Q * DV) as usize];
    dy_sm89.copy_to_host(&mut got).expect("dl sm89");
    dy_ref.copy_to_host(&mut refv).expect("dl ref");

    let tol = 64.0 * F16_EPS;
    for i in 0..got.len() {
        let r = refv[i].to_f32();
        let g = got[i].to_f32();
        let diff = (g - r).abs();
        let t = (r.abs() * tol).max(tol);
        assert!(
            diff <= t,
            "f16 flash-sdpa-sm89 causal y @ {i}: diff={diff} sm89={g} ref={r}"
        );
    }
}
