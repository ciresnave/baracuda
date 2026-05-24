//! Real-GPU smoke test for `FlashSdpaSm89Plan` strided sibling — Phase 17.1.
//!
//! Validates that the Phase 17.1 strided FW kernel propagates per-tensor
//! outer-dim strides into the `cp.async` K/V tile loads correctly, and
//! that the GQA-broadcast path (`stride_k[1] == 0` / `stride_v[1] == 0`)
//! "just works" — multiple Q-heads in a kv-head group share the same K/V
//! row by reading the same gmem range.
//!
//! Cross-checks against the sm_80 baseline `FlashSdpaPlan` (which is
//! contig-only and was the Phase 10.3 reference). For the GQA case we
//! build an explicit physically-expanded contig K/V mirror and compare
//! against the baseline on the expanded tensors.
//!
//! Gated on the `sm89` cargo feature. `#[ignore]` by default — requires
//! a real Ada-class CUDA device.

#![cfg(feature = "sm89")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, FlashSdpaArgs, FlashSdpaDescriptor, FlashSdpaPlan,
    FlashSdpaSm89Args, FlashSdpaSm89Descriptor, FlashSdpaSm89Plan, PlanPreference, TensorMut,
    TensorRef, Workspace,
};
use half::{bf16, f16};

const F16_EPS: f32 = 9.77e-4;
const BF16_EPS: f32 = 7.81e-3;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

// Fixture matches `flash_sdpa_sm89_smoke.rs` so the contig fast-path
// case below shares generators with the reference smoke.
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
// 1. Contig f16 → fast path sanity. Routes through the existing
//    non-strided FFI (Phase 10.3 path); should match sm_80 baseline.
// ----------------------------------------------------------------------------

#[test]
#[ignore]
fn flash_sdpa_sm89_strided_f16_contig_fast_path() {
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

    // Reference: sm_80 baseline.
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
                q: TensorRef { data: dq.as_slice(), shape: sq, stride: contiguous_stride(sq) },
                k: TensorRef { data: dk.as_slice(), shape: sk, stride: contiguous_stride(sk) },
                v: TensorRef { data: dv.as_slice(), shape: sv, stride: contiguous_stride(sv) },
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

    // sm_89 plan with contig inputs — should pick the non-strided fast path.
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
            q: TensorRef { data: dq.as_slice(), shape: sq, stride: contiguous_stride(sq) },
            k: TensorRef { data: dk.as_slice(), shape: sk, stride: contiguous_stride(sk) },
            v: TensorRef { data: dv.as_slice(), shape: sv, stride: contiguous_stride(sv) },
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
            "f16 flash-sdpa-sm89 contig fast-path y @ {i}: diff={diff} sm89={g} ref={r}"
        );
    }
}

// ----------------------------------------------------------------------------
// 2. Transposed Q view → strided FFI fires.
//
// We physically lay out Q on device as `[B, Q, H, DK]` (the "head-after-seq"
// PyTorch view) and read it back into the kernel as a `[B, H, Q, DK]`
// logical tensor with a non-canonical stride. The plan's contig check
// fails on this shape, so the strided FFI is invoked. We compare against
// the sm_80 baseline running on the physically-contig `[B, H, Q, DK]`
// equivalent (built from the same f64 source).
// ----------------------------------------------------------------------------

#[test]
#[ignore]
fn flash_sdpa_sm89_strided_f16_transposed_q() {
    let (ctx, stream) = setup();
    let q_f64 = gen_q_f64();
    let k_f64 = gen_k_f64();
    let v_f64 = gen_v_f64();
    let scale = default_scale();

    // Logical [B, H, Q, DK] data → physical [B, Q, H, DK] storage on device.
    // Build the permuted host buffer first.
    let mut q_phys: Vec<f64> = vec![0.0; q_f64.len()];
    for b in 0..B as usize {
        for h in 0..H as usize {
            for q in 0..Q as usize {
                for d in 0..DK as usize {
                    let src = ((b * H as usize + h) * Q as usize + q) * DK as usize + d;
                    let dst = ((b * Q as usize + q) * H as usize + h) * DK as usize + d;
                    q_phys[dst] = q_f64[src];
                }
            }
        }
    }
    let q_h_phys: Vec<f16> = q_phys.iter().map(|&v| f16::from_f64(v)).collect();
    let q_h_contig: Vec<f16> = q_f64.iter().map(|&v| f16::from_f64(v)).collect();
    let k_h: Vec<f16> = k_f64.iter().map(|&v| f16::from_f64(v)).collect();
    let v_h: Vec<f16> = v_f64.iter().map(|&v| f16::from_f64(v)).collect();
    let dq_phys = DeviceBuffer::from_slice(&ctx, &q_h_phys).expect("up q phys");
    let dq_contig = DeviceBuffer::from_slice(&ctx, &q_h_contig).expect("up q contig");
    let dk = DeviceBuffer::from_slice(&ctx, &k_h).expect("up k");
    let dv = DeviceBuffer::from_slice(&ctx, &v_h).expect("up v");

    let sq = [B, H, Q, DK];
    let sk = [B, H, K, DK];
    let sv = [B, H, K, DV];
    let sy = [B, H, Q, DV];
    let sl = [B, H, Q];

    // Q strides over the permuted storage [B, Q, H, DK]:
    //   stride[b] = Q*H*DK, stride[h] = DK, stride[q] = H*DK, stride[d] = 1
    let stride_q: [i64; 4] = [
        (Q as i64) * (H as i64) * (DK as i64),
        DK as i64,
        (H as i64) * (DK as i64),
        1,
    ];

    // Reference: sm_80 baseline with contig Q.
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
                    data: dq_contig.as_slice(),
                    shape: sq,
                    stride: contiguous_stride(sq),
                },
                k: TensorRef { data: dk.as_slice(), shape: sk, stride: contiguous_stride(sk) },
                v: TensorRef { data: dv.as_slice(), shape: sv, stride: contiguous_stride(sv) },
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

    // sm_89 strided path with Q stored as [B, Q, H, DK] permuted.
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
                data: dq_phys.as_slice(),
                shape: sq,
                stride: stride_q,
            },
            k: TensorRef { data: dk.as_slice(), shape: sk, stride: contiguous_stride(sk) },
            v: TensorRef { data: dv.as_slice(), shape: sv, stride: contiguous_stride(sv) },
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
    .expect("sm89 strided run");
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
            "f16 flash-sdpa-sm89 transposed-Q y @ {i}: diff={diff} sm89={g} ref={r}"
        );
    }
}

// ----------------------------------------------------------------------------
// 3. GQA broadcast f16 — `stride_k[1] == 0` and `stride_v[1] == 0`.
//
// LOAD-BEARING test. K / V are stored as `[B, 1, K, DK]` (a single
// kv-head per batch), viewed as `[B, H, K, DK]` via stride[1] == 0.
// We compare against the sm_80 baseline running on a physically-
// expanded `[B, H, K, DK]` mirror where every q-head sees the same K/V.
// ----------------------------------------------------------------------------

#[test]
#[ignore]
fn flash_sdpa_sm89_strided_f16_gqa_broadcast() {
    let (ctx, stream) = setup();
    let q_f64 = gen_q_f64();
    let scale = default_scale();

    // For GQA we only want ONE kv-head per batch, so K/V have shape
    // [B, 1, K, DK]. Generate that smaller source and then physically
    // expand into [B, H, K, DK] for the reference.
    let kv_n_one_head = (B * 1 * K * DK) as usize;
    let kv_n_full = (B * H * K * DK) as usize;
    let k_one_f64: Vec<f64> = (0..kv_n_one_head)
        .map(|i| ((i as f64) * 0.019 + 0.3).cos() * 0.5)
        .collect();
    let v_one_f64: Vec<f64> = (0..kv_n_one_head)
        .map(|i| ((i as f64) * 0.013 - 0.15).sin() * 0.7)
        .collect();

    // Physically expanded K/V (every h replica is a copy of head-0).
    let mut k_full_f64: Vec<f64> = vec![0.0; kv_n_full];
    let mut v_full_f64: Vec<f64> = vec![0.0; kv_n_full];
    for b in 0..B as usize {
        for h in 0..H as usize {
            for kk in 0..K as usize {
                for d in 0..DK as usize {
                    let src = ((b * 1 + 0) * K as usize + kk) * DK as usize + d;
                    let dst = ((b * H as usize + h) * K as usize + kk) * DK as usize + d;
                    k_full_f64[dst] = k_one_f64[src];
                    v_full_f64[dst] = v_one_f64[src];
                }
            }
        }
    }

    let q_h: Vec<f16> = q_f64.iter().map(|&v| f16::from_f64(v)).collect();
    let k_one_h: Vec<f16> = k_one_f64.iter().map(|&v| f16::from_f64(v)).collect();
    let v_one_h: Vec<f16> = v_one_f64.iter().map(|&v| f16::from_f64(v)).collect();
    let k_full_h: Vec<f16> = k_full_f64.iter().map(|&v| f16::from_f64(v)).collect();
    let v_full_h: Vec<f16> = v_full_f64.iter().map(|&v| f16::from_f64(v)).collect();

    let dq = DeviceBuffer::from_slice(&ctx, &q_h).expect("up q");
    let dk_one = DeviceBuffer::from_slice(&ctx, &k_one_h).expect("up k1");
    let dv_one = DeviceBuffer::from_slice(&ctx, &v_one_h).expect("up v1");
    let dk_full = DeviceBuffer::from_slice(&ctx, &k_full_h).expect("up kF");
    let dv_full = DeviceBuffer::from_slice(&ctx, &v_full_h).expect("up vF");

    let sq = [B, H, Q, DK];
    let sk = [B, H, K, DK];
    let sv = [B, H, K, DV];
    let sy = [B, H, Q, DV];
    let sl = [B, H, Q];

    // Reference: sm_80 baseline with the physically-expanded K/V.
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
                    data: dk_full.as_slice(),
                    shape: sk,
                    stride: contiguous_stride(sk),
                },
                v: TensorRef {
                    data: dv_full.as_slice(),
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

    // sm_89 strided plan with GQA-broadcast K/V (stride[1] == 0).
    let mut dy_sm89: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * DV) as usize).expect("alloc y_sm89");
    let mut dlse_sm89: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q) as usize).expect("alloc lse_sm89");

    // Logical [B, H, K, DK] view backed by physical [B, 1, K, DK] storage.
    //   stride[b] = K*DK
    //   stride[h] = 0     ← broadcast
    //   stride[k] = DK
    //   stride[d] = 1
    let stride_k: [i64; 4] = [
        (K as i64) * (DK as i64),
        0,
        DK as i64,
        1,
    ];
    let stride_v: [i64; 4] = [
        (K as i64) * (DV as i64),
        0,
        DV as i64,
        1,
    ];

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
            q: TensorRef { data: dq.as_slice(), shape: sq, stride: contiguous_stride(sq) },
            k: TensorRef {
                data: dk_one.as_slice(),
                shape: sk,
                stride: stride_k,
            },
            v: TensorRef {
                data: dv_one.as_slice(),
                shape: sv,
                stride: stride_v,
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
    .expect("sm89 gqa run");
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
            "f16 flash-sdpa-sm89 GQA-broadcast y @ {i}: diff={diff} sm89={g} ref={r}"
        );
    }
}

// ----------------------------------------------------------------------------
// 4. Causal masked path bf16 — exercises the causal early-out + strided
//    K/V loads in one shot. Uses contig inputs so the strided path is
//    exercised only by virtue of a non-contig Q view (we transpose Q like
//    case 2 above).
// ----------------------------------------------------------------------------

#[test]
#[ignore]
fn flash_sdpa_sm89_strided_bf16_transposed_q_causal() {
    let (ctx, stream) = setup();
    let q_f64 = gen_q_f64();
    let k_f64 = gen_k_f64();
    let v_f64 = gen_v_f64();
    let scale = default_scale();

    let mut q_phys: Vec<f64> = vec![0.0; q_f64.len()];
    for b in 0..B as usize {
        for h in 0..H as usize {
            for q in 0..Q as usize {
                for d in 0..DK as usize {
                    let src = ((b * H as usize + h) * Q as usize + q) * DK as usize + d;
                    let dst = ((b * Q as usize + q) * H as usize + h) * DK as usize + d;
                    q_phys[dst] = q_f64[src];
                }
            }
        }
    }
    let q_h_phys: Vec<bf16> = q_phys.iter().map(|&v| bf16::from_f64(v)).collect();
    let q_h_contig: Vec<bf16> = q_f64.iter().map(|&v| bf16::from_f64(v)).collect();
    let k_h: Vec<bf16> = k_f64.iter().map(|&v| bf16::from_f64(v)).collect();
    let v_h: Vec<bf16> = v_f64.iter().map(|&v| bf16::from_f64(v)).collect();
    let dq_phys = DeviceBuffer::from_slice(&ctx, &q_h_phys).expect("up q phys");
    let dq_contig = DeviceBuffer::from_slice(&ctx, &q_h_contig).expect("up q contig");
    let dk = DeviceBuffer::from_slice(&ctx, &k_h).expect("up k");
    let dv = DeviceBuffer::from_slice(&ctx, &v_h).expect("up v");

    let sq = [B, H, Q, DK];
    let sk = [B, H, K, DK];
    let sv = [B, H, K, DV];
    let sy = [B, H, Q, DV];
    let sl = [B, H, Q];
    let stride_q: [i64; 4] = [
        (Q as i64) * (H as i64) * (DK as i64),
        DK as i64,
        (H as i64) * (DK as i64),
        1,
    ];

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
        is_causal: true,
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
                    data: dq_contig.as_slice(),
                    shape: sq,
                    stride: contiguous_stride(sq),
                },
                k: TensorRef { data: dk.as_slice(), shape: sk, stride: contiguous_stride(sk) },
                v: TensorRef { data: dv.as_slice(), shape: sv, stride: contiguous_stride(sv) },
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
        is_causal: true,
        element: ElementKind::Bf16,
    };
    let plan = FlashSdpaSm89Plan::<bf16>::select(&stream, &desc, PlanPreference::default())
        .expect("sm89 sel");
    plan.run(
        &stream,
        Workspace::None,
        FlashSdpaSm89Args {
            q: TensorRef {
                data: dq_phys.as_slice(),
                shape: sq,
                stride: stride_q,
            },
            k: TensorRef { data: dk.as_slice(), shape: sk, stride: contiguous_stride(sk) },
            v: TensorRef { data: dv.as_slice(), shape: sv, stride: contiguous_stride(sv) },
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
    .expect("sm89 causal run");
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
            "bf16 flash-sdpa-sm89 causal transposed-Q y @ {i}: diff={diff} sm89={g} ref={r}"
        );
    }
}

// ----------------------------------------------------------------------------
// 5. Negative test: reject non-unit head_dim stride at the plan layer.
// ----------------------------------------------------------------------------

#[test]
#[ignore]
fn flash_sdpa_sm89_strided_rejects_non_unit_head_dim_stride() {
    let (ctx, stream) = setup();
    let scale = default_scale();
    let n = (B * H * Q * DK) as usize;
    let host = vec![f16::ZERO; n];
    let dq = DeviceBuffer::from_slice(&ctx, &host).expect("up q");
    let dk_n = (B * H * K * DK) as usize;
    let dv_n = (B * H * K * DV) as usize;
    let dk = DeviceBuffer::from_slice(&ctx, &vec![f16::ZERO; dk_n]).expect("up k");
    let dv = DeviceBuffer::from_slice(&ctx, &vec![f16::ZERO; dv_n]).expect("up v");
    let mut dy: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * DV) as usize).expect("alloc y");
    let mut dlse: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q) as usize).expect("alloc lse");

    let sq = [B, H, Q, DK];
    let sk = [B, H, K, DK];
    let sv = [B, H, K, DV];
    let sy = [B, H, Q, DV];
    let sl = [B, H, Q];

    // Q has stride[3] = 2 — illegal (head_dim must be stride=1).
    let bad_stride_q: [i64; 4] = [
        2 * (H as i64) * (Q as i64) * (DK as i64),
        2 * (Q as i64) * (DK as i64),
        2 * (DK as i64),
        2,
    ];

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
    let err = plan.run(
        &stream,
        Workspace::None,
        FlashSdpaSm89Args {
            q: TensorRef {
                data: dq.as_slice(),
                shape: sq,
                stride: bad_stride_q,
            },
            k: TensorRef { data: dk.as_slice(), shape: sk, stride: contiguous_stride(sk) },
            v: TensorRef { data: dv.as_slice(), shape: sv, stride: contiguous_stride(sv) },
            y: TensorMut {
                data: dy.as_slice_mut(),
                shape: sy,
                stride: contiguous_stride(sy),
            },
            lse: TensorMut {
                data: dlse.as_slice_mut(),
                shape: sl,
                stride: contiguous_stride(sl),
            },
        },
    );
    assert!(
        err.is_err(),
        "expected Err for non-unit head_dim stride on Q, got Ok"
    );
}
