//! Real-GPU smoke tests for the Phase 14.4 strided SDPA siblings.
//!
//! Coverage:
//!   * Contiguous Q/K/V → fast path (sanity).
//!   * GQA broadcast case: `stride_k[1] == 0` and `stride_v[1] == 0`
//!     (K and V tensors with fewer unique heads than Q, broadcasting
//!     via zero stride). Validates output matches the equivalent
//!     contig-expanded reference.
//!   * Causal masked path.
//!   * f16 dtype.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test sdpa_strided_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, SdpaArgs, SdpaDescriptor, SdpaPlan, TensorMut,
    TensorRef, Workspace,
};
use half::f16;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// CPU reference, takes Q/K/V at the given outer strides; head_dim
/// (innermost) stride is implicit 1.
#[allow(clippy::too_many_arguments)]
fn host_sdpa_ref_f64(
    batch: usize,
    heads: usize,
    q_len: usize,
    k_len: usize,
    d_k: usize,
    d_v: usize,
    q: &[f64],
    stride_q: [i64; 4],
    k: &[f64],
    stride_k: [i64; 4],
    v: &[f64],
    stride_v: [i64; 4],
    mask: Option<&[f64]>,
    scale: f64,
    is_causal: bool,
) -> Vec<f64> {
    assert_eq!(stride_q[3], 1);
    assert_eq!(stride_k[3], 1);
    assert_eq!(stride_v[3], 1);
    let total_y = batch * heads * q_len * d_v;
    let mut y = vec![0f64; total_y];
    for b in 0..batch {
        for h in 0..heads {
            // Build scores[i, j]
            let mut scores = vec![0f64; q_len * k_len];
            for i in 0..q_len {
                for j in 0..k_len {
                    let mut s = 0f64;
                    for d in 0..d_k {
                        let q_off = (b as i64) * stride_q[0]
                            + (h as i64) * stride_q[1]
                            + (i as i64) * stride_q[2]
                            + d as i64;
                        let k_off = (b as i64) * stride_k[0]
                            + (h as i64) * stride_k[1]
                            + (j as i64) * stride_k[2]
                            + d as i64;
                        s += q[q_off as usize] * k[k_off as usize];
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
            // Row softmax
            let mut attn = vec![0f64; q_len * k_len];
            for i in 0..q_len {
                let row = &scores[i * k_len..(i + 1) * k_len];
                let m = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                if !m.is_finite() {
                    continue;
                }
                let sum: f64 = row.iter().map(|&v| (v - m).exp()).sum();
                if sum == 0.0 {
                    continue;
                }
                for j in 0..k_len {
                    attn[i * k_len + j] = (row[j] - m).exp() / sum;
                }
            }
            // y = attn @ V (output is canonical contig [B,H,Q,D_v])
            for i in 0..q_len {
                for dv in 0..d_v {
                    let mut acc = 0f64;
                    for kk in 0..k_len {
                        let v_off = (b as i64) * stride_v[0]
                            + (h as i64) * stride_v[1]
                            + (kk as i64) * stride_v[2]
                            + dv as i64;
                        acc += attn[i * k_len + kk] * v[v_off as usize];
                    }
                    let y_idx = ((b * heads + h) * q_len + i) * d_v + dv;
                    y[y_idx] = acc;
                }
            }
        }
    }
    y
}

/// 1. Contiguous case — fast path sanity check.
#[test]
#[ignore]
fn sdpa_strided_contig_f32() {
    let (ctx, stream) = setup();
    let batch = 2i32;
    let heads = 2i32;
    let q_len = 4i32;
    let k_len = 4i32;
    let d_k = 8i32;
    let d_v = 8i32;
    let q_n = (batch * heads * q_len * d_k) as usize;
    let k_n = (batch * heads * k_len * d_k) as usize;
    let v_n = (batch * heads * k_len * d_v) as usize;
    let attn_n = (batch * heads * q_len * k_len) as usize;
    let y_n = (batch * heads * q_len * d_v) as usize;
    let scale = 1.0 / (d_k as f32).sqrt();

    let host_q: Vec<f32> = (0..q_n).map(|i| ((i as f32) * 0.05 - 0.5).sin()).collect();
    let host_k: Vec<f32> = (0..k_n).map(|i| ((i as f32) * 0.07 + 0.1).cos()).collect();
    let host_v: Vec<f32> = (0..v_n).map(|i| ((i as f32) * 0.03 - 0.2).sin()).collect();

    let shape_q = [batch, heads, q_len, d_k];
    let shape_k = [batch, heads, k_len, d_k];
    let shape_v = [batch, heads, k_len, d_v];
    let shape_attn = [batch, heads, q_len, k_len];
    let shape_y = [batch, heads, q_len, d_v];

    let host_q_f64: Vec<f64> = host_q.iter().map(|&x| x as f64).collect();
    let host_k_f64: Vec<f64> = host_k.iter().map(|&x| x as f64).collect();
    let host_v_f64: Vec<f64> = host_v.iter().map(|&x| x as f64).collect();
    let expected = host_sdpa_ref_f64(
        batch as usize,
        heads as usize,
        q_len as usize,
        k_len as usize,
        d_k as usize,
        d_v as usize,
        &host_q_f64,
        contiguous_stride(shape_q),
        &host_k_f64,
        contiguous_stride(shape_k),
        &host_v_f64,
        contiguous_stride(shape_v),
        None,
        scale as f64,
        false,
    );

    let dev_q = DeviceBuffer::from_slice(&ctx, &host_q).expect("up q");
    let dev_k = DeviceBuffer::from_slice(&ctx, &host_k).expect("up k");
    let dev_v = DeviceBuffer::from_slice(&ctx, &host_v).expect("up v");
    let mut dev_attn: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, attn_n).expect("attn");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, y_n).expect("y");

    let desc = SdpaDescriptor {
        batch_size: batch,
        num_heads: heads,
        query_len: q_len,
        key_len: k_len,
        d_k,
        d_v,
        scale,
        is_causal: false,
        has_mask: false,
        element: ElementKind::F32,
    };
    let plan = SdpaPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    plan.run(
        &stream,
        Workspace::None,
        SdpaArgs {
            q: TensorRef {
                data: dev_q.as_slice(),
                shape: shape_q,
                stride: contiguous_stride(shape_q),
            },
            k: TensorRef {
                data: dev_k.as_slice(),
                shape: shape_k,
                stride: contiguous_stride(shape_k),
            },
            v: TensorRef {
                data: dev_v.as_slice(),
                shape: shape_v,
                stride: contiguous_stride(shape_v),
            },
            mask: None,
            y: TensorMut {
                data: dev_y.as_slice_mut(),
                shape: shape_y,
                stride: contiguous_stride(shape_y),
            },
            attn: TensorMut {
                data: dev_attn.as_slice_mut(),
                shape: shape_attn,
                stride: contiguous_stride(shape_attn),
            },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; y_n];
    dev_y.copy_to_host(&mut got).expect("dl");
    let tol = 1e-4;
    for i in 0..y_n {
        let t = (expected[i].abs() as f32 * tol).max(tol);
        let diff = (got[i] - expected[i] as f32).abs();
        assert!(diff <= t, "contig f32 y @ {i}: diff={diff}");
    }
}

/// 2. GQA broadcast: K and V have one unique head, Q has multiple. We
/// store K/V as `[B, 1, K, D]` and view them as `[B, heads, K, D]` with
/// `stride_h = 0`. Output should match the reference where K/V is
/// expanded across all Q heads.
#[test]
#[ignore]
fn sdpa_strided_gqa_broadcast_f32() {
    let (ctx, stream) = setup();
    let batch = 2i32;
    let heads = 4i32; // Q heads
    let kv_unique_heads = 1i32; // K/V groups
    let q_len = 4i32;
    let k_len = 4i32;
    let d_k = 8i32;
    let d_v = 8i32;
    let q_n = (batch * heads * q_len * d_k) as usize;
    // K/V physically stored at the smaller fanout
    let k_phys_n = (batch * kv_unique_heads * k_len * d_k) as usize;
    let v_phys_n = (batch * kv_unique_heads * k_len * d_v) as usize;
    let attn_n = (batch * heads * q_len * k_len) as usize;
    let y_n = (batch * heads * q_len * d_v) as usize;
    let scale = 1.0 / (d_k as f32).sqrt();

    let host_q: Vec<f32> = (0..q_n).map(|i| ((i as f32) * 0.04 - 0.4).sin()).collect();
    let host_k_phys: Vec<f32> = (0..k_phys_n)
        .map(|i| ((i as f32) * 0.06 + 0.3).cos())
        .collect();
    let host_v_phys: Vec<f32> = (0..v_phys_n)
        .map(|i| ((i as f32) * 0.02 - 0.1).sin())
        .collect();

    // Strides for K/V: head dim stride = 0 (broadcast).
    let stride_k: [i64; 4] = [
        (kv_unique_heads as i64) * (k_len as i64) * (d_k as i64),
        0, // broadcast over heads
        (d_k as i64),
        1,
    ];
    let stride_v: [i64; 4] = [
        (kv_unique_heads as i64) * (k_len as i64) * (d_v as i64),
        0, // broadcast over heads
        (d_v as i64),
        1,
    ];

    let shape_q = [batch, heads, q_len, d_k];
    let shape_k = [batch, heads, k_len, d_k];
    let shape_v = [batch, heads, k_len, d_v];
    let shape_attn = [batch, heads, q_len, k_len];
    let shape_y = [batch, heads, q_len, d_v];

    let host_q_f64: Vec<f64> = host_q.iter().map(|&x| x as f64).collect();
    let host_k_f64: Vec<f64> = host_k_phys.iter().map(|&x| x as f64).collect();
    let host_v_f64: Vec<f64> = host_v_phys.iter().map(|&x| x as f64).collect();

    let expected = host_sdpa_ref_f64(
        batch as usize,
        heads as usize,
        q_len as usize,
        k_len as usize,
        d_k as usize,
        d_v as usize,
        &host_q_f64,
        contiguous_stride(shape_q),
        &host_k_f64,
        stride_k,
        &host_v_f64,
        stride_v,
        None,
        scale as f64,
        false,
    );

    let dev_q = DeviceBuffer::from_slice(&ctx, &host_q).expect("up q");
    let dev_k = DeviceBuffer::from_slice(&ctx, &host_k_phys).expect("up k");
    let dev_v = DeviceBuffer::from_slice(&ctx, &host_v_phys).expect("up v");
    let mut dev_attn: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, attn_n).expect("attn");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, y_n).expect("y");

    let desc = SdpaDescriptor {
        batch_size: batch,
        num_heads: heads,
        query_len: q_len,
        key_len: k_len,
        d_k,
        d_v,
        scale,
        is_causal: false,
        has_mask: false,
        element: ElementKind::F32,
    };
    let plan = SdpaPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    plan.run(
        &stream,
        Workspace::None,
        SdpaArgs {
            q: TensorRef {
                data: dev_q.as_slice(),
                shape: shape_q,
                stride: contiguous_stride(shape_q),
            },
            k: TensorRef {
                data: dev_k.as_slice(),
                shape: shape_k,
                stride: stride_k,
            },
            v: TensorRef {
                data: dev_v.as_slice(),
                shape: shape_v,
                stride: stride_v,
            },
            mask: None,
            y: TensorMut {
                data: dev_y.as_slice_mut(),
                shape: shape_y,
                stride: contiguous_stride(shape_y),
            },
            attn: TensorMut {
                data: dev_attn.as_slice_mut(),
                shape: shape_attn,
                stride: contiguous_stride(shape_attn),
            },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; y_n];
    dev_y.copy_to_host(&mut got).expect("dl");
    let tol = 1e-4;
    for i in 0..y_n {
        let t = (expected[i].abs() as f32 * tol).max(tol);
        let diff = (got[i] - expected[i] as f32).abs();
        assert!(
            diff <= t,
            "gqa-broadcast f32 y @ {i}: diff={diff} expected={}",
            expected[i]
        );
    }
}

/// 3. Causal masked path with strided Q (transposed-batch view).
/// Physical buffer layout is `[H, B, Q, D_k]` (heads outermost), viewed
/// as `[B, H, Q, D_k]` with permuted strides.
#[test]
#[ignore]
fn sdpa_strided_causal_transposed_q_f32() {
    let (ctx, stream) = setup();
    let batch = 2i32;
    let heads = 2i32;
    let q_len = 4i32;
    let k_len = 4i32;
    let d_k = 8i32;
    let d_v = 8i32;
    let q_n = (batch * heads * q_len * d_k) as usize;
    let k_n = (batch * heads * k_len * d_k) as usize;
    let v_n = (batch * heads * k_len * d_v) as usize;
    let attn_n = (batch * heads * q_len * k_len) as usize;
    let y_n = (batch * heads * q_len * d_v) as usize;
    let scale = 1.0 / (d_k as f32).sqrt();

    // Q is stored as [H, B, Q, D_k] but viewed as [B, H, Q, D_k].
    let host_q: Vec<f32> = (0..q_n).map(|i| ((i as f32) * 0.05 - 0.5).sin()).collect();
    let host_k: Vec<f32> = (0..k_n).map(|i| ((i as f32) * 0.07 + 0.1).cos()).collect();
    let host_v: Vec<f32> = (0..v_n).map(|i| ((i as f32) * 0.03 - 0.2).sin()).collect();

    // Original [H, B, Q, D_k] contig strides: stride_h = B*Q*D, stride_b = Q*D
    // Viewed as [B, H, Q, D_k]: stride_b = Q*D, stride_h = B*Q*D, stride_q = D, stride_d = 1
    let stride_q_b: i64 = (q_len as i64) * (d_k as i64);
    let stride_q_h: i64 = (batch as i64) * (q_len as i64) * (d_k as i64);
    let stride_q_s: i64 = d_k as i64;
    let stride_q_view: [i64; 4] = [stride_q_b, stride_q_h, stride_q_s, 1];

    let shape_q = [batch, heads, q_len, d_k];
    let shape_k = [batch, heads, k_len, d_k];
    let shape_v = [batch, heads, k_len, d_v];
    let shape_attn = [batch, heads, q_len, k_len];
    let shape_y = [batch, heads, q_len, d_v];

    let host_q_f64: Vec<f64> = host_q.iter().map(|&x| x as f64).collect();
    let host_k_f64: Vec<f64> = host_k.iter().map(|&x| x as f64).collect();
    let host_v_f64: Vec<f64> = host_v.iter().map(|&x| x as f64).collect();

    let expected = host_sdpa_ref_f64(
        batch as usize,
        heads as usize,
        q_len as usize,
        k_len as usize,
        d_k as usize,
        d_v as usize,
        &host_q_f64,
        stride_q_view,
        &host_k_f64,
        contiguous_stride(shape_k),
        &host_v_f64,
        contiguous_stride(shape_v),
        None,
        scale as f64,
        true, // causal!
    );

    let dev_q = DeviceBuffer::from_slice(&ctx, &host_q).expect("up q");
    let dev_k = DeviceBuffer::from_slice(&ctx, &host_k).expect("up k");
    let dev_v = DeviceBuffer::from_slice(&ctx, &host_v).expect("up v");
    let mut dev_attn: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, attn_n).expect("attn");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, y_n).expect("y");

    let desc = SdpaDescriptor {
        batch_size: batch,
        num_heads: heads,
        query_len: q_len,
        key_len: k_len,
        d_k,
        d_v,
        scale,
        is_causal: true,
        has_mask: false,
        element: ElementKind::F32,
    };
    let plan = SdpaPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    plan.run(
        &stream,
        Workspace::None,
        SdpaArgs {
            q: TensorRef {
                data: dev_q.as_slice(),
                shape: shape_q,
                stride: stride_q_view,
            },
            k: TensorRef {
                data: dev_k.as_slice(),
                shape: shape_k,
                stride: contiguous_stride(shape_k),
            },
            v: TensorRef {
                data: dev_v.as_slice(),
                shape: shape_v,
                stride: contiguous_stride(shape_v),
            },
            mask: None,
            y: TensorMut {
                data: dev_y.as_slice_mut(),
                shape: shape_y,
                stride: contiguous_stride(shape_y),
            },
            attn: TensorMut {
                data: dev_attn.as_slice_mut(),
                shape: shape_attn,
                stride: contiguous_stride(shape_attn),
            },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; y_n];
    dev_y.copy_to_host(&mut got).expect("dl");
    let tol = 1e-4;
    for i in 0..y_n {
        let t = (expected[i].abs() as f32 * tol).max(tol);
        let diff = (got[i] - expected[i] as f32).abs();
        assert!(
            diff <= t,
            "causal strided f32 y @ {i}: diff={diff} expected={}",
            expected[i]
        );
    }
}

/// 4. f16 strided coverage — GQA broadcast.
#[test]
#[ignore]
fn sdpa_strided_gqa_broadcast_f16() {
    let (ctx, stream) = setup();
    let batch = 1i32;
    let heads = 2i32;
    let kv_unique_heads = 1i32;
    let q_len = 4i32;
    let k_len = 4i32;
    let d_k = 8i32;
    let d_v = 8i32;
    let q_n = (batch * heads * q_len * d_k) as usize;
    let k_phys_n = (batch * kv_unique_heads * k_len * d_k) as usize;
    let v_phys_n = (batch * kv_unique_heads * k_len * d_v) as usize;
    let attn_n = (batch * heads * q_len * k_len) as usize;
    let y_n = (batch * heads * q_len * d_v) as usize;
    let scale = 1.0 / (d_k as f32).sqrt();

    let host_q_f32: Vec<f32> = (0..q_n).map(|i| ((i as f32) * 0.04 - 0.3).sin()).collect();
    let host_k_f32: Vec<f32> = (0..k_phys_n)
        .map(|i| ((i as f32) * 0.06 + 0.2).cos())
        .collect();
    let host_v_f32: Vec<f32> = (0..v_phys_n)
        .map(|i| ((i as f32) * 0.02 - 0.05).sin())
        .collect();

    let stride_k: [i64; 4] = [
        (kv_unique_heads as i64) * (k_len as i64) * (d_k as i64),
        0,
        d_k as i64,
        1,
    ];
    let stride_v: [i64; 4] = [
        (kv_unique_heads as i64) * (k_len as i64) * (d_v as i64),
        0,
        d_v as i64,
        1,
    ];

    let shape_q = [batch, heads, q_len, d_k];
    let shape_k = [batch, heads, k_len, d_k];
    let shape_v = [batch, heads, k_len, d_v];
    let shape_attn = [batch, heads, q_len, k_len];
    let shape_y = [batch, heads, q_len, d_v];

    let host_q_f64: Vec<f64> = host_q_f32.iter().map(|&x| x as f64).collect();
    let host_k_f64: Vec<f64> = host_k_f32.iter().map(|&x| x as f64).collect();
    let host_v_f64: Vec<f64> = host_v_f32.iter().map(|&x| x as f64).collect();
    let expected = host_sdpa_ref_f64(
        batch as usize,
        heads as usize,
        q_len as usize,
        k_len as usize,
        d_k as usize,
        d_v as usize,
        &host_q_f64,
        contiguous_stride(shape_q),
        &host_k_f64,
        stride_k,
        &host_v_f64,
        stride_v,
        None,
        scale as f64,
        false,
    );

    let host_q: Vec<f16> = host_q_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_k: Vec<f16> = host_k_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_v: Vec<f16> = host_v_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let dev_q = DeviceBuffer::from_slice(&ctx, &host_q).expect("up q");
    let dev_k = DeviceBuffer::from_slice(&ctx, &host_k).expect("up k");
    let dev_v = DeviceBuffer::from_slice(&ctx, &host_v).expect("up v");
    let mut dev_attn: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, attn_n).expect("attn");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, y_n).expect("y");

    let desc = SdpaDescriptor {
        batch_size: batch,
        num_heads: heads,
        query_len: q_len,
        key_len: k_len,
        d_k,
        d_v,
        scale,
        is_causal: false,
        has_mask: false,
        element: ElementKind::F16,
    };
    let plan = SdpaPlan::<f16>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    plan.run(
        &stream,
        Workspace::None,
        SdpaArgs {
            q: TensorRef {
                data: dev_q.as_slice(),
                shape: shape_q,
                stride: contiguous_stride(shape_q),
            },
            k: TensorRef {
                data: dev_k.as_slice(),
                shape: shape_k,
                stride: stride_k,
            },
            v: TensorRef {
                data: dev_v.as_slice(),
                shape: shape_v,
                stride: stride_v,
            },
            mask: None,
            y: TensorMut {
                data: dev_y.as_slice_mut(),
                shape: shape_y,
                stride: contiguous_stride(shape_y),
            },
            attn: TensorMut {
                data: dev_attn.as_slice_mut(),
                shape: shape_attn,
                stride: contiguous_stride(shape_attn),
            },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![f16::ZERO; y_n];
    dev_y.copy_to_host(&mut got).expect("dl");
    // f16 / softmax / matmul stacking → loose tolerance
    let tol = 0.05f32;
    for i in 0..y_n {
        let t = (expected[i].abs() as f32 * tol).max(tol);
        let diff = (got[i].to_f32() - expected[i] as f32).abs();
        assert!(
            diff <= t,
            "gqa-broadcast f16 y @ {i}: diff={diff} expected={}",
            expected[i]
        );
    }
}
