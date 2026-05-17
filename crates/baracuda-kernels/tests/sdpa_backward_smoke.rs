//! Real-GPU smoke test for `SdpaBackwardPlan + AttentionKind::Sdpa` BW.
//!
//! Validates dQ / dK / dV against an f64 host reference for the naive
//! SDPA pipeline:
//!   dV       = attn^T @ dy
//!   dattn    = dy @ V^T
//!   dscores  = softmax_bw(attn, dattn)
//!   dQ       = dscores @ K * scale
//!   dK       = dscores^T @ Q * scale
//!
//! `#[ignore]` by default — requires a real CUDA device.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, SdpaArgs, SdpaBackwardArgs,
    SdpaBackwardDescriptor, SdpaBackwardPlan, SdpaDescriptor, SdpaPlan, TensorMut, TensorRef,
    Workspace,
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

const B: i32 = 2;
const H: i32 = 4;
const Q: i32 = 8;
const K: i32 = 8;
const DK: i32 = 16;
const DV: i32 = 16;

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
fn gen_dy_f64() -> Vec<f64> {
    let n = (B * H * Q * DV) as usize;
    (0..n)
        .map(|i| ((i as f64) * 0.019 + 0.3).cos() * 0.4)
        .collect()
}

/// Host-side BW reference. Recomputes `attn` from Q/K/V (FW), then
/// closes the BW chain. Returns (dQ, dK, dV) in row-major flat layout.
fn host_sdpa_bw_f64(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    dy: &[f64],
    scale: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let batch = B as usize;
    let heads = H as usize;
    let q_len = Q as usize;
    let k_len = K as usize;
    let d_k = DK as usize;
    let d_v = DV as usize;
    let total_attn = batch * heads * q_len * k_len;
    let mut attn = vec![0f64; total_attn];
    // FW: build attn
    for b in 0..batch {
        for h in 0..heads {
            for i in 0..q_len {
                let mut row = vec![0f64; k_len];
                for j in 0..k_len {
                    let mut s = 0f64;
                    for d in 0..d_k {
                        let qi = ((b * heads + h) * q_len + i) * d_k + d;
                        let kj = ((b * heads + h) * k_len + j) * d_k + d;
                        s += q[qi] * k[kj];
                    }
                    row[j] = s * scale;
                }
                let m = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let sum: f64 = row.iter().map(|&x| (x - m).exp()).sum();
                for j in 0..k_len {
                    let a_idx = ((b * heads + h) * q_len + i) * k_len + j;
                    attn[a_idx] = (row[j] - m).exp() / sum;
                }
            }
        }
    }
    let mut dv_grad = vec![0f64; batch * heads * k_len * d_v];
    let mut dq = vec![0f64; batch * heads * q_len * d_k];
    let mut dk = vec![0f64; batch * heads * k_len * d_k];
    for b in 0..batch {
        for h in 0..heads {
            // dV[b, h, kk, dv] = Σ_i attn[i, kk] · dy[i, dv]
            for kk in 0..k_len {
                for dv in 0..d_v {
                    let mut acc = 0f64;
                    for i in 0..q_len {
                        let a = attn[((b * heads + h) * q_len + i) * k_len + kk];
                        let d = dy[((b * heads + h) * q_len + i) * d_v + dv];
                        acc += a * d;
                    }
                    dv_grad[((b * heads + h) * k_len + kk) * d_v + dv] = acc;
                }
            }
            // dattn[i, kk] = Σ_dv dy[i, dv] · V[kk, dv]
            let mut dattn = vec![0f64; q_len * k_len];
            for i in 0..q_len {
                for kk in 0..k_len {
                    let mut acc = 0f64;
                    for dv in 0..d_v {
                        let dyv = dy[((b * heads + h) * q_len + i) * d_v + dv];
                        let vv = v[((b * heads + h) * k_len + kk) * d_v + dv];
                        acc += dyv * vv;
                    }
                    dattn[i * k_len + kk] = acc;
                }
            }
            // dscores[i, j] = attn[i, j] · (dattn[i, j] − Σ_l attn[i, l]·dattn[i, l])
            let mut dscores = vec![0f64; q_len * k_len];
            for i in 0..q_len {
                let mut dot = 0f64;
                for j in 0..k_len {
                    let a = attn[((b * heads + h) * q_len + i) * k_len + j];
                    dot += a * dattn[i * k_len + j];
                }
                for j in 0..k_len {
                    let a = attn[((b * heads + h) * q_len + i) * k_len + j];
                    dscores[i * k_len + j] = a * (dattn[i * k_len + j] - dot);
                }
            }
            // dQ[i, d] = scale · Σ_kk dscores[i, kk] · K[kk, d]
            for i in 0..q_len {
                for d in 0..d_k {
                    let mut acc = 0f64;
                    for kk in 0..k_len {
                        let ds = dscores[i * k_len + kk];
                        let kv = k[((b * heads + h) * k_len + kk) * d_k + d];
                        acc += ds * kv;
                    }
                    dq[((b * heads + h) * q_len + i) * d_k + d] = acc * scale;
                }
            }
            // dK[kk, d] = scale · Σ_i dscores[i, kk] · Q[i, d]
            for kk in 0..k_len {
                for d in 0..d_k {
                    let mut acc = 0f64;
                    for i in 0..q_len {
                        let ds = dscores[i * k_len + kk];
                        let qv = q[((b * heads + h) * q_len + i) * d_k + d];
                        acc += ds * qv;
                    }
                    dk[((b * heads + h) * k_len + kk) * d_k + d] = acc * scale;
                }
            }
        }
    }
    (dq, dk, dv_grad)
}

// ----------------------------------------------------------------------------
// f32
// ----------------------------------------------------------------------------

#[test]
#[ignore]
fn sdpa_backward_f32_basic() {
    let (ctx, stream) = setup();
    let q_f64 = gen_q_f64();
    let k_f64 = gen_k_f64();
    let v_f64 = gen_v_f64();
    let dy_f64 = gen_dy_f64();
    let scale = default_scale();
    let (dq_ref, dk_ref, dv_ref) = host_sdpa_bw_f64(&q_f64, &k_f64, &v_f64, &dy_f64, scale as f64);

    let q_h: Vec<f32> = q_f64.iter().map(|&v| v as f32).collect();
    let k_h: Vec<f32> = k_f64.iter().map(|&v| v as f32).collect();
    let v_h: Vec<f32> = v_f64.iter().map(|&v| v as f32).collect();
    let dy_h: Vec<f32> = dy_f64.iter().map(|&v| v as f32).collect();

    let dq_dev = DeviceBuffer::from_slice(&ctx, &q_h).expect("up");
    let dk_dev = DeviceBuffer::from_slice(&ctx, &k_h).expect("up");
    let dv_dev = DeviceBuffer::from_slice(&ctx, &v_h).expect("up");
    let ddy_dev = DeviceBuffer::from_slice(&ctx, &dy_h).expect("up dy");
    let mut dattn: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * K) as usize).expect("alloc attn");
    let mut dy_out: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * DV) as usize).expect("alloc y");
    let mut dws: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * K) as usize).expect("alloc ws");
    let mut ddq: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * DK) as usize).expect("alloc dq");
    let mut ddk: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (B * H * K * DK) as usize).expect("alloc dk");
    let mut ddv: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (B * H * K * DV) as usize).expect("alloc dv");

    let sq = [B, H, Q, DK];
    let sk = [B, H, K, DK];
    let sv = [B, H, K, DV];
    let sa = [B, H, Q, K];
    let sy = [B, H, Q, DV];

    // FW first to populate the saved attn.
    let fw_desc = SdpaDescriptor {
        batch_size: B,
        num_heads: H,
        query_len: Q,
        key_len: K,
        d_k: DK,
        d_v: DV,
        scale,
        is_causal: false,
        has_mask: false,
        element: ElementKind::F32,
    };
    let fw_plan =
        SdpaPlan::<f32>::select(&stream, &fw_desc, PlanPreference::default()).expect("fw sel");
    fw_plan
        .run(
            &stream,
            Workspace::None,
            SdpaArgs {
                q: TensorRef {
                    data: dq_dev.as_slice(),
                    shape: sq,
                    stride: contiguous_stride(sq),
                },
                k: TensorRef {
                    data: dk_dev.as_slice(),
                    shape: sk,
                    stride: contiguous_stride(sk),
                },
                v: TensorRef {
                    data: dv_dev.as_slice(),
                    shape: sv,
                    stride: contiguous_stride(sv),
                },
                mask: None,
                y: TensorMut {
                    data: dy_out.as_slice_mut(),
                    shape: sy,
                    stride: contiguous_stride(sy),
                },
                attn: TensorMut {
                    data: dattn.as_slice_mut(),
                    shape: sa,
                    stride: contiguous_stride(sa),
                },
            },
        )
        .expect("fw run");
    stream.synchronize().expect("sync fw");

    // BW.
    let bw_desc = SdpaBackwardDescriptor {
        batch_size: B,
        num_heads: H,
        query_len: Q,
        key_len: K,
        d_k: DK,
        d_v: DV,
        scale,
        element: ElementKind::F32,
    };
    let bw_plan = SdpaBackwardPlan::<f32>::select(&stream, &bw_desc, PlanPreference::default())
        .expect("bw sel");
    bw_plan
        .run(
            &stream,
            Workspace::None,
            SdpaBackwardArgs {
                q: TensorRef {
                    data: dq_dev.as_slice(),
                    shape: sq,
                    stride: contiguous_stride(sq),
                },
                k: TensorRef {
                    data: dk_dev.as_slice(),
                    shape: sk,
                    stride: contiguous_stride(sk),
                },
                v: TensorRef {
                    data: dv_dev.as_slice(),
                    shape: sv,
                    stride: contiguous_stride(sv),
                },
                attn: TensorRef {
                    data: dattn.as_slice(),
                    shape: sa,
                    stride: contiguous_stride(sa),
                },
                dy: TensorRef {
                    data: ddy_dev.as_slice(),
                    shape: sy,
                    stride: contiguous_stride(sy),
                },
                dscores_ws: TensorMut {
                    data: dws.as_slice_mut(),
                    shape: sa,
                    stride: contiguous_stride(sa),
                },
                dq: TensorMut {
                    data: ddq.as_slice_mut(),
                    shape: sq,
                    stride: contiguous_stride(sq),
                },
                dk: TensorMut {
                    data: ddk.as_slice_mut(),
                    shape: sk,
                    stride: contiguous_stride(sk),
                },
                dv: TensorMut {
                    data: ddv.as_slice_mut(),
                    shape: sv,
                    stride: contiguous_stride(sv),
                },
            },
        )
        .expect("bw run");
    stream.synchronize().expect("sync bw");

    let mut got_dq = vec![0f32; (B * H * Q * DK) as usize];
    let mut got_dk = vec![0f32; (B * H * K * DK) as usize];
    let mut got_dv = vec![0f32; (B * H * K * DV) as usize];
    ddq.copy_to_host(&mut got_dq).expect("dl dq");
    ddk.copy_to_host(&mut got_dk).expect("dl dk");
    ddv.copy_to_host(&mut got_dv).expect("dl dv");

    let tol = 32.0 * f32::EPSILON;
    for (label, got, refv) in [
        ("dQ", &got_dq[..], &dq_ref[..]),
        ("dK", &got_dk[..], &dk_ref[..]),
        ("dV", &got_dv[..], &dv_ref[..]),
    ] {
        for i in 0..got.len() {
            let r = refv[i] as f32;
            let diff = (got[i] - r).abs();
            let t = (r.abs() * tol).max(tol);
            assert!(
                diff <= t,
                "f32 sdpa-bw {label} @ {i}: diff={diff} got={g} ref={r}",
                g = got[i]
            );
        }
    }
}

// ----------------------------------------------------------------------------
// f64
// ----------------------------------------------------------------------------

#[test]
#[ignore]
fn sdpa_backward_f64_basic() {
    let (ctx, stream) = setup();
    let q_h = gen_q_f64();
    let k_h = gen_k_f64();
    let v_h = gen_v_f64();
    let dy_h = gen_dy_f64();
    let scale = default_scale();
    let (dq_ref, dk_ref, dv_ref) = host_sdpa_bw_f64(&q_h, &k_h, &v_h, &dy_h, scale as f64);

    let dq_dev = DeviceBuffer::from_slice(&ctx, &q_h).expect("up");
    let dk_dev = DeviceBuffer::from_slice(&ctx, &k_h).expect("up");
    let dv_dev = DeviceBuffer::from_slice(&ctx, &v_h).expect("up");
    let ddy_dev = DeviceBuffer::from_slice(&ctx, &dy_h).expect("up dy");
    let mut dattn: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * K) as usize).expect("alloc attn");
    let mut dy_out: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * DV) as usize).expect("alloc y");
    let mut dws: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * K) as usize).expect("alloc ws");
    let mut ddq: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * DK) as usize).expect("alloc dq");
    let mut ddk: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (B * H * K * DK) as usize).expect("alloc dk");
    let mut ddv: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (B * H * K * DV) as usize).expect("alloc dv");

    let sq = [B, H, Q, DK];
    let sk = [B, H, K, DK];
    let sv = [B, H, K, DV];
    let sa = [B, H, Q, K];
    let sy = [B, H, Q, DV];

    let fw_desc = SdpaDescriptor {
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
    let fw_plan = SdpaPlan::<f64>::select(&stream, &fw_desc, PlanPreference::default()).expect("");
    fw_plan
        .run(
            &stream,
            Workspace::None,
            SdpaArgs {
                q: TensorRef {
                    data: dq_dev.as_slice(),
                    shape: sq,
                    stride: contiguous_stride(sq),
                },
                k: TensorRef {
                    data: dk_dev.as_slice(),
                    shape: sk,
                    stride: contiguous_stride(sk),
                },
                v: TensorRef {
                    data: dv_dev.as_slice(),
                    shape: sv,
                    stride: contiguous_stride(sv),
                },
                mask: None,
                y: TensorMut {
                    data: dy_out.as_slice_mut(),
                    shape: sy,
                    stride: contiguous_stride(sy),
                },
                attn: TensorMut {
                    data: dattn.as_slice_mut(),
                    shape: sa,
                    stride: contiguous_stride(sa),
                },
            },
        )
        .expect("fw");
    stream.synchronize().expect("");

    let bw_desc = SdpaBackwardDescriptor {
        batch_size: B,
        num_heads: H,
        query_len: Q,
        key_len: K,
        d_k: DK,
        d_v: DV,
        scale,
        element: ElementKind::F64,
    };
    let bw_plan =
        SdpaBackwardPlan::<f64>::select(&stream, &bw_desc, PlanPreference::default()).expect("");
    bw_plan
        .run(
            &stream,
            Workspace::None,
            SdpaBackwardArgs {
                q: TensorRef {
                    data: dq_dev.as_slice(),
                    shape: sq,
                    stride: contiguous_stride(sq),
                },
                k: TensorRef {
                    data: dk_dev.as_slice(),
                    shape: sk,
                    stride: contiguous_stride(sk),
                },
                v: TensorRef {
                    data: dv_dev.as_slice(),
                    shape: sv,
                    stride: contiguous_stride(sv),
                },
                attn: TensorRef {
                    data: dattn.as_slice(),
                    shape: sa,
                    stride: contiguous_stride(sa),
                },
                dy: TensorRef {
                    data: ddy_dev.as_slice(),
                    shape: sy,
                    stride: contiguous_stride(sy),
                },
                dscores_ws: TensorMut {
                    data: dws.as_slice_mut(),
                    shape: sa,
                    stride: contiguous_stride(sa),
                },
                dq: TensorMut {
                    data: ddq.as_slice_mut(),
                    shape: sq,
                    stride: contiguous_stride(sq),
                },
                dk: TensorMut {
                    data: ddk.as_slice_mut(),
                    shape: sk,
                    stride: contiguous_stride(sk),
                },
                dv: TensorMut {
                    data: ddv.as_slice_mut(),
                    shape: sv,
                    stride: contiguous_stride(sv),
                },
            },
        )
        .expect("bw");
    stream.synchronize().expect("");

    let mut got_dq = vec![0f64; (B * H * Q * DK) as usize];
    let mut got_dk = vec![0f64; (B * H * K * DK) as usize];
    let mut got_dv = vec![0f64; (B * H * K * DV) as usize];
    ddq.copy_to_host(&mut got_dq).expect("");
    ddk.copy_to_host(&mut got_dk).expect("");
    ddv.copy_to_host(&mut got_dv).expect("");

    let tol = 32.0 * f64::EPSILON;
    for (label, got, refv) in [
        ("dQ", &got_dq[..], &dq_ref[..]),
        ("dK", &got_dk[..], &dk_ref[..]),
        ("dV", &got_dv[..], &dv_ref[..]),
    ] {
        for i in 0..got.len() {
            let diff = (got[i] - refv[i]).abs();
            let t = (refv[i].abs() * tol).max(tol);
            assert!(diff <= t, "f64 sdpa-bw {label} @ {i}: diff={diff}");
        }
    }
}

// ----------------------------------------------------------------------------
// f16 / bf16 — single test each (no mask, no causal). Tolerance is the
// 32× dtype-eps product.
// ----------------------------------------------------------------------------

#[test]
#[ignore]
fn sdpa_backward_f16_basic() {
    let (ctx, stream) = setup();
    let q_f64 = gen_q_f64();
    let k_f64 = gen_k_f64();
    let v_f64 = gen_v_f64();
    let dy_f64 = gen_dy_f64();
    let scale = default_scale();
    let (dq_ref, dk_ref, dv_ref) =
        host_sdpa_bw_f64(&q_f64, &k_f64, &v_f64, &dy_f64, scale as f64);

    let q_h: Vec<f16> = q_f64.iter().map(|&v| f16::from_f64(v)).collect();
    let k_h: Vec<f16> = k_f64.iter().map(|&v| f16::from_f64(v)).collect();
    let v_h: Vec<f16> = v_f64.iter().map(|&v| f16::from_f64(v)).collect();
    let dy_h: Vec<f16> = dy_f64.iter().map(|&v| f16::from_f64(v)).collect();

    let dq_dev = DeviceBuffer::from_slice(&ctx, &q_h).expect("up");
    let dk_dev = DeviceBuffer::from_slice(&ctx, &k_h).expect("up");
    let dv_dev = DeviceBuffer::from_slice(&ctx, &v_h).expect("up");
    let ddy_dev = DeviceBuffer::from_slice(&ctx, &dy_h).expect("up dy");
    let mut dattn: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * K) as usize).expect("alloc attn");
    let mut dy_out: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * DV) as usize).expect("alloc y");
    let mut dws: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * K) as usize).expect("alloc ws");
    let mut ddq: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * DK) as usize).expect("alloc dq");
    let mut ddk: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (B * H * K * DK) as usize).expect("alloc dk");
    let mut ddv: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (B * H * K * DV) as usize).expect("alloc dv");

    let sq = [B, H, Q, DK];
    let sk = [B, H, K, DK];
    let sv = [B, H, K, DV];
    let sa = [B, H, Q, K];
    let sy = [B, H, Q, DV];

    let fw_desc = SdpaDescriptor {
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
    let fw_plan = SdpaPlan::<f16>::select(&stream, &fw_desc, PlanPreference::default()).expect("");
    fw_plan
        .run(
            &stream,
            Workspace::None,
            SdpaArgs {
                q: TensorRef {
                    data: dq_dev.as_slice(),
                    shape: sq,
                    stride: contiguous_stride(sq),
                },
                k: TensorRef {
                    data: dk_dev.as_slice(),
                    shape: sk,
                    stride: contiguous_stride(sk),
                },
                v: TensorRef {
                    data: dv_dev.as_slice(),
                    shape: sv,
                    stride: contiguous_stride(sv),
                },
                mask: None,
                y: TensorMut {
                    data: dy_out.as_slice_mut(),
                    shape: sy,
                    stride: contiguous_stride(sy),
                },
                attn: TensorMut {
                    data: dattn.as_slice_mut(),
                    shape: sa,
                    stride: contiguous_stride(sa),
                },
            },
        )
        .expect("fw");
    stream.synchronize().expect("");

    let bw_desc = SdpaBackwardDescriptor {
        batch_size: B,
        num_heads: H,
        query_len: Q,
        key_len: K,
        d_k: DK,
        d_v: DV,
        scale,
        element: ElementKind::F16,
    };
    let bw_plan =
        SdpaBackwardPlan::<f16>::select(&stream, &bw_desc, PlanPreference::default()).expect("");
    bw_plan
        .run(
            &stream,
            Workspace::None,
            SdpaBackwardArgs {
                q: TensorRef {
                    data: dq_dev.as_slice(),
                    shape: sq,
                    stride: contiguous_stride(sq),
                },
                k: TensorRef {
                    data: dk_dev.as_slice(),
                    shape: sk,
                    stride: contiguous_stride(sk),
                },
                v: TensorRef {
                    data: dv_dev.as_slice(),
                    shape: sv,
                    stride: contiguous_stride(sv),
                },
                attn: TensorRef {
                    data: dattn.as_slice(),
                    shape: sa,
                    stride: contiguous_stride(sa),
                },
                dy: TensorRef {
                    data: ddy_dev.as_slice(),
                    shape: sy,
                    stride: contiguous_stride(sy),
                },
                dscores_ws: TensorMut {
                    data: dws.as_slice_mut(),
                    shape: sa,
                    stride: contiguous_stride(sa),
                },
                dq: TensorMut {
                    data: ddq.as_slice_mut(),
                    shape: sq,
                    stride: contiguous_stride(sq),
                },
                dk: TensorMut {
                    data: ddk.as_slice_mut(),
                    shape: sk,
                    stride: contiguous_stride(sk),
                },
                dv: TensorMut {
                    data: ddv.as_slice_mut(),
                    shape: sv,
                    stride: contiguous_stride(sv),
                },
            },
        )
        .expect("bw");
    stream.synchronize().expect("");

    let mut got_dq = vec![f16::ZERO; (B * H * Q * DK) as usize];
    let mut got_dk = vec![f16::ZERO; (B * H * K * DK) as usize];
    let mut got_dv = vec![f16::ZERO; (B * H * K * DV) as usize];
    ddq.copy_to_host(&mut got_dq).expect("");
    ddk.copy_to_host(&mut got_dk).expect("");
    ddv.copy_to_host(&mut got_dv).expect("");

    let tol = 32.0 * F16_EPS;
    for (label, got, refv) in [
        ("dQ", &got_dq[..], &dq_ref[..]),
        ("dK", &got_dk[..], &dk_ref[..]),
        ("dV", &got_dv[..], &dv_ref[..]),
    ] {
        for i in 0..got.len() {
            let r = refv[i] as f32;
            let g = got[i].to_f32();
            let diff = (g - r).abs();
            let t = (r.abs() * tol).max(tol);
            assert!(
                diff <= t,
                "f16 sdpa-bw {label} @ {i}: diff={diff} got={g} ref={r}"
            );
        }
    }
}

#[test]
#[ignore]
fn sdpa_backward_bf16_basic() {
    let (ctx, stream) = setup();
    let q_f64 = gen_q_f64();
    let k_f64 = gen_k_f64();
    let v_f64 = gen_v_f64();
    let dy_f64 = gen_dy_f64();
    let scale = default_scale();
    let (dq_ref, dk_ref, dv_ref) =
        host_sdpa_bw_f64(&q_f64, &k_f64, &v_f64, &dy_f64, scale as f64);

    let q_h: Vec<bf16> = q_f64.iter().map(|&v| bf16::from_f64(v)).collect();
    let k_h: Vec<bf16> = k_f64.iter().map(|&v| bf16::from_f64(v)).collect();
    let v_h: Vec<bf16> = v_f64.iter().map(|&v| bf16::from_f64(v)).collect();
    let dy_h: Vec<bf16> = dy_f64.iter().map(|&v| bf16::from_f64(v)).collect();

    let dq_dev = DeviceBuffer::from_slice(&ctx, &q_h).expect("up");
    let dk_dev = DeviceBuffer::from_slice(&ctx, &k_h).expect("up");
    let dv_dev = DeviceBuffer::from_slice(&ctx, &v_h).expect("up");
    let ddy_dev = DeviceBuffer::from_slice(&ctx, &dy_h).expect("up dy");
    let mut dattn: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * K) as usize).expect("alloc attn");
    let mut dy_out: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * DV) as usize).expect("alloc y");
    let mut dws: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * K) as usize).expect("alloc ws");
    let mut ddq: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * DK) as usize).expect("alloc dq");
    let mut ddk: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, (B * H * K * DK) as usize).expect("alloc dk");
    let mut ddv: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, (B * H * K * DV) as usize).expect("alloc dv");

    let sq = [B, H, Q, DK];
    let sk = [B, H, K, DK];
    let sv = [B, H, K, DV];
    let sa = [B, H, Q, K];
    let sy = [B, H, Q, DV];

    let fw_desc = SdpaDescriptor {
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
    let fw_plan = SdpaPlan::<bf16>::select(&stream, &fw_desc, PlanPreference::default()).expect("");
    fw_plan
        .run(
            &stream,
            Workspace::None,
            SdpaArgs {
                q: TensorRef {
                    data: dq_dev.as_slice(),
                    shape: sq,
                    stride: contiguous_stride(sq),
                },
                k: TensorRef {
                    data: dk_dev.as_slice(),
                    shape: sk,
                    stride: contiguous_stride(sk),
                },
                v: TensorRef {
                    data: dv_dev.as_slice(),
                    shape: sv,
                    stride: contiguous_stride(sv),
                },
                mask: None,
                y: TensorMut {
                    data: dy_out.as_slice_mut(),
                    shape: sy,
                    stride: contiguous_stride(sy),
                },
                attn: TensorMut {
                    data: dattn.as_slice_mut(),
                    shape: sa,
                    stride: contiguous_stride(sa),
                },
            },
        )
        .expect("fw");
    stream.synchronize().expect("");

    let bw_desc = SdpaBackwardDescriptor {
        batch_size: B,
        num_heads: H,
        query_len: Q,
        key_len: K,
        d_k: DK,
        d_v: DV,
        scale,
        element: ElementKind::Bf16,
    };
    let bw_plan =
        SdpaBackwardPlan::<bf16>::select(&stream, &bw_desc, PlanPreference::default()).expect("");
    bw_plan
        .run(
            &stream,
            Workspace::None,
            SdpaBackwardArgs {
                q: TensorRef {
                    data: dq_dev.as_slice(),
                    shape: sq,
                    stride: contiguous_stride(sq),
                },
                k: TensorRef {
                    data: dk_dev.as_slice(),
                    shape: sk,
                    stride: contiguous_stride(sk),
                },
                v: TensorRef {
                    data: dv_dev.as_slice(),
                    shape: sv,
                    stride: contiguous_stride(sv),
                },
                attn: TensorRef {
                    data: dattn.as_slice(),
                    shape: sa,
                    stride: contiguous_stride(sa),
                },
                dy: TensorRef {
                    data: ddy_dev.as_slice(),
                    shape: sy,
                    stride: contiguous_stride(sy),
                },
                dscores_ws: TensorMut {
                    data: dws.as_slice_mut(),
                    shape: sa,
                    stride: contiguous_stride(sa),
                },
                dq: TensorMut {
                    data: ddq.as_slice_mut(),
                    shape: sq,
                    stride: contiguous_stride(sq),
                },
                dk: TensorMut {
                    data: ddk.as_slice_mut(),
                    shape: sk,
                    stride: contiguous_stride(sk),
                },
                dv: TensorMut {
                    data: ddv.as_slice_mut(),
                    shape: sv,
                    stride: contiguous_stride(sv),
                },
            },
        )
        .expect("bw");
    stream.synchronize().expect("");

    let mut got_dq = vec![bf16::ZERO; (B * H * Q * DK) as usize];
    let mut got_dk = vec![bf16::ZERO; (B * H * K * DK) as usize];
    let mut got_dv = vec![bf16::ZERO; (B * H * K * DV) as usize];
    ddq.copy_to_host(&mut got_dq).expect("");
    ddk.copy_to_host(&mut got_dk).expect("");
    ddv.copy_to_host(&mut got_dv).expect("");

    let tol = 32.0 * BF16_EPS;
    for (label, got, refv) in [
        ("dQ", &got_dq[..], &dq_ref[..]),
        ("dK", &got_dk[..], &dk_ref[..]),
        ("dV", &got_dv[..], &dv_ref[..]),
    ] {
        for i in 0..got.len() {
            let r = refv[i] as f32;
            let g = got[i].to_f32();
            let diff = (g - r).abs();
            let t = (r.abs() * tol).max(tol);
            assert!(
                diff <= t,
                "bf16 sdpa-bw {label} @ {i}: diff={diff} got={g} ref={r}"
            );
        }
    }
}
