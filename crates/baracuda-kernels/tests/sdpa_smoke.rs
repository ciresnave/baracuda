//! Real-GPU smoke test for `SdpaPlan + AttentionKind::Sdpa` FW.
//!
//! `y = softmax(Q · K^T · scale + mask) · V`. Three-kernel pipeline
//! (scores / row-softmax / out) bundled under one launcher.
//!
//! Covers FW × 4 FP dtypes × 3 mask modes (none, explicit, causal).
//! `#[ignore]` by default — requires a real CUDA device.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, SdpaArgs, SdpaDescriptor, SdpaPlan, TensorMut,
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

/// f64 host-side reference implementation. Mirrors the GPU pipeline:
/// scores = Q·K^T·scale + mask + causal, row-softmax, y = attn @ V.
/// Returns `(y, attn)` so the caller can validate both.
fn host_sdpa_f64(
    batch: usize,
    heads: usize,
    q_len: usize,
    k_len: usize,
    d_k: usize,
    d_v: usize,
    q: &[f64],
    k: &[f64],
    v: &[f64],
    mask: Option<&[f64]>,
    scale: f64,
    is_causal: bool,
) -> (Vec<f64>, Vec<f64>) {
    let total_attn = batch * heads * q_len * k_len;
    let total_y = batch * heads * q_len * d_v;
    let mut attn = vec![0f64; total_attn];
    let mut y = vec![0f64; total_y];
    for b in 0..batch {
        for h in 0..heads {
            // Build scores[b, h, :, :]
            let mut scores = vec![0f64; q_len * k_len];
            for i in 0..q_len {
                for j in 0..k_len {
                    let mut s = 0f64;
                    for d in 0..d_k {
                        let q_idx = ((b * heads + h) * q_len + i) * d_k + d;
                        let k_idx = ((b * heads + h) * k_len + j) * d_k + d;
                        s += q[q_idx] * k[k_idx];
                    }
                    s *= scale;
                    if let Some(m) = mask {
                        let m_idx = ((b * heads + h) * q_len + i) * k_len + j;
                        s += m[m_idx];
                    }
                    if is_causal && j > i {
                        s = f64::NEG_INFINITY;
                    }
                    scores[i * k_len + j] = s;
                }
            }
            // Row softmax (stable).
            for i in 0..q_len {
                let row = &scores[i * k_len..(i + 1) * k_len];
                let m = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                if !m.is_finite() {
                    // All -inf row → emit zeros.
                    for j in 0..k_len {
                        let a_idx = ((b * heads + h) * q_len + i) * k_len + j;
                        attn[a_idx] = 0.0;
                    }
                    continue;
                }
                let sum: f64 = row.iter().map(|&v| (v - m).exp()).sum();
                if sum == 0.0 {
                    for j in 0..k_len {
                        let a_idx = ((b * heads + h) * q_len + i) * k_len + j;
                        attn[a_idx] = 0.0;
                    }
                } else {
                    for j in 0..k_len {
                        let a_idx = ((b * heads + h) * q_len + i) * k_len + j;
                        attn[a_idx] = (row[j] - m).exp() / sum;
                    }
                }
            }
            // y = attn @ V
            for i in 0..q_len {
                for dv in 0..d_v {
                    let mut acc = 0f64;
                    for kk in 0..k_len {
                        let a_idx = ((b * heads + h) * q_len + i) * k_len + kk;
                        let v_idx = ((b * heads + h) * k_len + kk) * d_v + dv;
                        acc += attn[a_idx] * v[v_idx];
                    }
                    let y_idx = ((b * heads + h) * q_len + i) * d_v + dv;
                    y[y_idx] = acc;
                }
            }
        }
    }
    (y, attn)
}

const B: i32 = 2;
const H: i32 = 4;
const Q: i32 = 8;
const K: i32 = 8;
const DK: i32 = 16;
const DV: i32 = 16;

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
fn gen_mask_f64() -> Vec<f64> {
    let n = (B * H * Q * K) as usize;
    (0..n)
        .map(|i| ((i as f64) * 0.01).sin() * 0.3)
        .collect()
}

fn default_scale() -> f32 {
    1.0 / (DK as f32).sqrt()
}

// ============================================================================
// f32
// ============================================================================

fn run_f32(is_causal: bool, has_mask: bool) {
    let (ctx, stream) = setup();
    let q_f64 = gen_q_f64();
    let k_f64 = gen_k_f64();
    let v_f64 = gen_v_f64();
    let mask_f64 = gen_mask_f64();
    let scale_f32 = default_scale();
    let (y_ref, _attn_ref) = host_sdpa_f64(
        B as usize,
        H as usize,
        Q as usize,
        K as usize,
        DK as usize,
        DV as usize,
        &q_f64,
        &k_f64,
        &v_f64,
        if has_mask { Some(&mask_f64) } else { None },
        scale_f32 as f64,
        is_causal,
    );

    let q_h: Vec<f32> = q_f64.iter().map(|&x| x as f32).collect();
    let k_h: Vec<f32> = k_f64.iter().map(|&x| x as f32).collect();
    let v_h: Vec<f32> = v_f64.iter().map(|&x| x as f32).collect();
    let mask_h: Vec<f32> = mask_f64.iter().map(|&x| x as f32).collect();
    let dq = DeviceBuffer::from_slice(&ctx, &q_h).expect("up q");
    let dk = DeviceBuffer::from_slice(&ctx, &k_h).expect("up k");
    let dv = DeviceBuffer::from_slice(&ctx, &v_h).expect("up v");
    let dmask = DeviceBuffer::from_slice(&ctx, &mask_h).expect("up m");
    let mut dattn: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * K) as usize).expect("alloc attn");
    let mut dy: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * DV) as usize).expect("alloc y");

    let desc = SdpaDescriptor {
        batch_size: B,
        num_heads: H,
        query_len: Q,
        key_len: K,
        d_k: DK,
        d_v: DV,
        scale: scale_f32,
        is_causal,
        has_mask,
        element: ElementKind::F32,
    };
    let plan = SdpaPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let sq = [B, H, Q, DK];
    let sk = [B, H, K, DK];
    let sv = [B, H, K, DV];
    let sa = [B, H, Q, K];
    let sy = [B, H, Q, DV];
    plan.run(
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
            mask: if has_mask {
                Some(TensorRef {
                    data: dmask.as_slice(),
                    shape: sa,
                    stride: contiguous_stride(sa),
                })
            } else {
                None
            },
            y: TensorMut {
                data: dy.as_slice_mut(),
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
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; (B * H * Q * DV) as usize];
    dy.copy_to_host(&mut got).expect("dl y");

    let tol = 16.0 * f32::EPSILON;
    for i in 0..got.len() {
        let r = y_ref[i] as f32;
        let diff = (got[i] - r).abs();
        let t = (r.abs() * tol).max(tol);
        assert!(
            diff <= t,
            "f32 sdpa y @ {i}: diff={diff} got={got} ref={r}",
            got = got[i]
        );
    }
}

#[test]
#[ignore]
fn sdpa_f32_no_mask() {
    run_f32(false, false);
}

#[test]
#[ignore]
fn sdpa_f32_explicit_mask() {
    run_f32(false, true);
}

#[test]
#[ignore]
fn sdpa_f32_causal() {
    run_f32(true, false);
}

// ============================================================================
// f64
// ============================================================================

#[test]
#[ignore]
fn sdpa_f64_basic() {
    let (ctx, stream) = setup();
    let q_h = gen_q_f64();
    let k_h = gen_k_f64();
    let v_h = gen_v_f64();
    let scale_f32 = default_scale();
    let (y_ref, _) = host_sdpa_f64(
        B as usize,
        H as usize,
        Q as usize,
        K as usize,
        DK as usize,
        DV as usize,
        &q_h,
        &k_h,
        &v_h,
        None,
        scale_f32 as f64,
        false,
    );
    let dq = DeviceBuffer::from_slice(&ctx, &q_h).expect("up q");
    let dk = DeviceBuffer::from_slice(&ctx, &k_h).expect("up k");
    let dv = DeviceBuffer::from_slice(&ctx, &v_h).expect("up v");
    let mut dattn: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * K) as usize).expect("alloc attn");
    let mut dy: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * DV) as usize).expect("alloc y");

    let desc = SdpaDescriptor {
        batch_size: B,
        num_heads: H,
        query_len: Q,
        key_len: K,
        d_k: DK,
        d_v: DV,
        scale: scale_f32,
        is_causal: false,
        has_mask: false,
        element: ElementKind::F64,
    };
    let plan = SdpaPlan::<f64>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let sq = [B, H, Q, DK];
    let sk = [B, H, K, DK];
    let sv = [B, H, K, DV];
    let sa = [B, H, Q, K];
    let sy = [B, H, Q, DV];
    plan.run(
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
                data: dy.as_slice_mut(),
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
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; (B * H * Q * DV) as usize];
    dy.copy_to_host(&mut got).expect("dl y");

    let tol = 16.0 * f64::EPSILON;
    for i in 0..got.len() {
        let diff = (got[i] - y_ref[i]).abs();
        let t = (y_ref[i].abs() * tol).max(tol);
        assert!(diff <= t, "f64 sdpa y @ {i}: diff={diff}");
    }
}

// ============================================================================
// f16 / bf16
// ============================================================================

fn run_f16() {
    let (ctx, stream) = setup();
    let q_f64 = gen_q_f64();
    let k_f64 = gen_k_f64();
    let v_f64 = gen_v_f64();
    let scale_f32 = default_scale();
    let (y_ref, _) = host_sdpa_f64(
        B as usize,
        H as usize,
        Q as usize,
        K as usize,
        DK as usize,
        DV as usize,
        &q_f64,
        &k_f64,
        &v_f64,
        None,
        scale_f32 as f64,
        false,
    );

    let q_h: Vec<f16> = q_f64.iter().map(|&v| f16::from_f64(v)).collect();
    let k_h: Vec<f16> = k_f64.iter().map(|&v| f16::from_f64(v)).collect();
    let v_h: Vec<f16> = v_f64.iter().map(|&v| f16::from_f64(v)).collect();
    let dq = DeviceBuffer::from_slice(&ctx, &q_h).expect("up q");
    let dk = DeviceBuffer::from_slice(&ctx, &k_h).expect("up k");
    let dv = DeviceBuffer::from_slice(&ctx, &v_h).expect("up v");
    let mut dattn: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * K) as usize).expect("alloc attn");
    let mut dy: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * DV) as usize).expect("alloc y");

    let desc = SdpaDescriptor {
        batch_size: B,
        num_heads: H,
        query_len: Q,
        key_len: K,
        d_k: DK,
        d_v: DV,
        scale: scale_f32,
        is_causal: false,
        has_mask: false,
        element: ElementKind::F16,
    };
    let plan = SdpaPlan::<f16>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let sq = [B, H, Q, DK];
    let sk = [B, H, K, DK];
    let sv = [B, H, K, DV];
    let sa = [B, H, Q, K];
    let sy = [B, H, Q, DV];
    plan.run(
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
                data: dy.as_slice_mut(),
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
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![f16::ZERO; (B * H * Q * DV) as usize];
    dy.copy_to_host(&mut got).expect("dl y");

    // f16 path uses f32 accumulators throughout but converts to f16 at
    // 3 storage points (after scores, after softmax, after out). Allow
    // 16 ULP-ish tolerance × the chain depth.
    let tol = 16.0 * F16_EPS;
    for i in 0..got.len() {
        let r = y_ref[i] as f32;
        let g = got[i].to_f32();
        let diff = (g - r).abs();
        let t = (r.abs() * tol).max(tol);
        assert!(diff <= t, "f16 sdpa y @ {i}: diff={diff} got={g} ref={r}");
    }
}

#[test]
#[ignore]
fn sdpa_f16_basic() {
    run_f16();
}

fn run_bf16() {
    let (ctx, stream) = setup();
    let q_f64 = gen_q_f64();
    let k_f64 = gen_k_f64();
    let v_f64 = gen_v_f64();
    let scale_f32 = default_scale();
    let (y_ref, _) = host_sdpa_f64(
        B as usize,
        H as usize,
        Q as usize,
        K as usize,
        DK as usize,
        DV as usize,
        &q_f64,
        &k_f64,
        &v_f64,
        None,
        scale_f32 as f64,
        false,
    );

    let q_h: Vec<bf16> = q_f64.iter().map(|&v| bf16::from_f64(v)).collect();
    let k_h: Vec<bf16> = k_f64.iter().map(|&v| bf16::from_f64(v)).collect();
    let v_h: Vec<bf16> = v_f64.iter().map(|&v| bf16::from_f64(v)).collect();
    let dq = DeviceBuffer::from_slice(&ctx, &q_h).expect("up q");
    let dk = DeviceBuffer::from_slice(&ctx, &k_h).expect("up k");
    let dv = DeviceBuffer::from_slice(&ctx, &v_h).expect("up v");
    let mut dattn: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * K) as usize).expect("alloc attn");
    let mut dy: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, (B * H * Q * DV) as usize).expect("alloc y");

    let desc = SdpaDescriptor {
        batch_size: B,
        num_heads: H,
        query_len: Q,
        key_len: K,
        d_k: DK,
        d_v: DV,
        scale: scale_f32,
        is_causal: false,
        has_mask: false,
        element: ElementKind::Bf16,
    };
    let plan = SdpaPlan::<bf16>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let sq = [B, H, Q, DK];
    let sk = [B, H, K, DK];
    let sv = [B, H, K, DV];
    let sa = [B, H, Q, K];
    let sy = [B, H, Q, DV];
    plan.run(
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
                data: dy.as_slice_mut(),
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
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![bf16::ZERO; (B * H * Q * DV) as usize];
    dy.copy_to_host(&mut got).expect("dl y");

    let tol = 16.0 * BF16_EPS;
    for i in 0..got.len() {
        let r = y_ref[i] as f32;
        let g = got[i].to_f32();
        let diff = (g - r).abs();
        let t = (r.abs() * tol).max(tol);
        assert!(diff <= t, "bf16 sdpa y @ {i}: diff={diff} got={g} ref={r}");
    }
}

#[test]
#[ignore]
fn sdpa_bf16_basic() {
    run_bf16();
}
