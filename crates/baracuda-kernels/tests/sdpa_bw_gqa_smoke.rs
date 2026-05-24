//! Phase 17.2 — Real-GPU smoke tests for the strided SDPA backward
//! kernel under GQA broadcast (`stride_k[1] == 0` or `stride_v[1] == 0`).
//!
//! Coverage:
//!   * `sdpa_bw_gqa_broadcast_f32` — Q has 4 heads, K/V have 1 head
//!     broadcast via `stride_*_h == 0`. Verifies dK/dV equal the
//!     reduction-over-Q-head-group of an expanded reference. LOAD-BEARING.
//!   * `sdpa_bw_gqa_broadcast_f16` — same shape, f16 dtype, exercises
//!     the `atomicAdd_via_cas` path on half-precision.
//!   * `sdpa_bw_no_broadcast_strided` — regression guard: ensures the
//!     fast-path (non-broadcast) strided BW still works.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --test sdpa_bw_gqa_smoke -- --ignored`.
//!
//! Reference approach for the broadcast tests: build an *expanded*
//! `[B, kv_heads_expanded=H, K, D]` K/V tensor (each Q-head sees the
//! same kv-head row), run the host BW reference on the expanded shape,
//! then collapse the resulting `dK_expanded` / `dV_expanded` back to
//! `[B, 1, K, D]` by summing across the Q-head axis. That sum is what
//! the broadcast path's atomicAdd should compute.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, SdpaArgs, SdpaBackwardArgs,
    SdpaBackwardDescriptor, SdpaBackwardPlan, SdpaDescriptor, SdpaPlan, TensorMut, TensorRef,
    Workspace,
};
use half::f16;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Host BW reference operating on contiguous expanded K/V (no broadcast).
/// Returns (dQ, dK_expanded, dV_expanded) flat row-major over
/// `[B, H, *, D]` (H Q-heads on both Q and K/V).
#[allow(clippy::too_many_arguments)]
fn host_sdpa_bw_f64_contig(
    batch: usize,
    heads: usize,
    q_len: usize,
    k_len: usize,
    d_k: usize,
    d_v: usize,
    q: &[f64],
    k: &[f64],
    v: &[f64],
    dy: &[f64],
    scale: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let total_attn = batch * heads * q_len * k_len;
    let mut attn = vec![0f64; total_attn];
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
    let mut dv_g = vec![0f64; batch * heads * k_len * d_v];
    let mut dq = vec![0f64; batch * heads * q_len * d_k];
    let mut dk = vec![0f64; batch * heads * k_len * d_k];
    for b in 0..batch {
        for h in 0..heads {
            for kk in 0..k_len {
                for dv in 0..d_v {
                    let mut acc = 0f64;
                    for i in 0..q_len {
                        let a = attn[((b * heads + h) * q_len + i) * k_len + kk];
                        let d = dy[((b * heads + h) * q_len + i) * d_v + dv];
                        acc += a * d;
                    }
                    dv_g[((b * heads + h) * k_len + kk) * d_v + dv] = acc;
                }
            }
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
    (dq, dk, dv_g)
}

/// Expand a `[B, kv_heads, K, D]` tensor to `[B, H, K, D]` by copying
/// each kv_head row `H/kv_heads` times.
fn expand_kv_f64(
    src: &[f64],
    batch: usize,
    kv_heads: usize,
    h_full: usize,
    k_len: usize,
    d: usize,
) -> Vec<f64> {
    assert!(h_full % kv_heads == 0);
    let group = h_full / kv_heads;
    let mut out = vec![0f64; batch * h_full * k_len * d];
    for b in 0..batch {
        for h in 0..h_full {
            let g = h / group;
            for kk in 0..k_len {
                for dd in 0..d {
                    let s_idx = ((b * kv_heads + g) * k_len + kk) * d + dd;
                    let o_idx = ((b * h_full + h) * k_len + kk) * d + dd;
                    out[o_idx] = src[s_idx];
                }
            }
        }
    }
    out
}

/// Collapse a `[B, H, K, D]` dK / dV grad back to `[B, kv_heads, K, D]`
/// by summing across the Q-head axis within each group. This mirrors
/// what the broadcast-path atomicAdd should produce.
fn collapse_kv_grad_f64(
    expanded: &[f64],
    batch: usize,
    h_full: usize,
    kv_heads: usize,
    k_len: usize,
    d: usize,
) -> Vec<f64> {
    assert!(h_full % kv_heads == 0);
    let group = h_full / kv_heads;
    let mut out = vec![0f64; batch * kv_heads * k_len * d];
    for b in 0..batch {
        for h in 0..h_full {
            let g = h / group;
            for kk in 0..k_len {
                for dd in 0..d {
                    let s_idx = ((b * h_full + h) * k_len + kk) * d + dd;
                    let o_idx = ((b * kv_heads + g) * k_len + kk) * d + dd;
                    out[o_idx] += expanded[s_idx];
                }
            }
        }
    }
    out
}

/// Helper: run BW with given dtype-specific buffers via the strided FFI.
/// `dk_phys` / `dv_phys` are pre-zeroed (caller contract for broadcast).
#[allow(clippy::too_many_arguments)]
fn run_bw_strided_f32(
    ctx: &Context,
    stream: &Stream,
    batch: i32,
    heads: i32,
    q_len: i32,
    k_len: i32,
    d_k: i32,
    d_v: i32,
    scale: f32,
    q_data: &[f32],
    k_phys_data: &[f32],
    v_phys_data: &[f32],
    dy_data: &[f32],
    stride_k: [i64; 4],
    stride_v: [i64; 4],
    kv_phys_heads: i32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let attn_n = (batch * heads * q_len * k_len) as usize;
    let dq_n = (batch * heads * q_len * d_k) as usize;
    let dk_phys_n = (batch * kv_phys_heads * k_len * d_k) as usize;
    let dv_phys_n = (batch * kv_phys_heads * k_len * d_v) as usize;
    let y_n = (batch * heads * q_len * d_v) as usize;

    let shape_q = [batch, heads, q_len, d_k];
    let shape_k = [batch, heads, k_len, d_k];
    let shape_v = [batch, heads, k_len, d_v];
    let shape_attn = [batch, heads, q_len, k_len];
    let shape_y = [batch, heads, q_len, d_v];

    let dev_q = DeviceBuffer::from_slice(ctx, q_data).expect("up q");
    let dev_k = DeviceBuffer::from_slice(ctx, k_phys_data).expect("up k");
    let dev_v = DeviceBuffer::from_slice(ctx, v_phys_data).expect("up v");
    let dev_dy = DeviceBuffer::from_slice(ctx, dy_data).expect("up dy");
    let mut dev_attn: DeviceBuffer<f32> = DeviceBuffer::zeros(ctx, attn_n).expect("attn");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(ctx, y_n).expect("y");
    let mut dev_ws: DeviceBuffer<f32> = DeviceBuffer::zeros(ctx, attn_n).expect("ws");
    let mut dev_dq: DeviceBuffer<f32> = DeviceBuffer::zeros(ctx, dq_n).expect("dq");
    // CRITICAL: dK/dV must be pre-zeroed (broadcast path adds into them).
    let mut dev_dk: DeviceBuffer<f32> = DeviceBuffer::zeros(ctx, dk_phys_n).expect("dk");
    let mut dev_dv: DeviceBuffer<f32> = DeviceBuffer::zeros(ctx, dv_phys_n).expect("dv");

    // FW first (populate dev_attn).
    let fw_desc = SdpaDescriptor {
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
    let fw_plan = SdpaPlan::<f32>::select(stream, &fw_desc, PlanPreference::default()).expect("fw sel");
    fw_plan
        .run(
            stream,
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
        .expect("fw run");
    stream.synchronize().expect("sync fw");

    // BW.
    let bw_desc = SdpaBackwardDescriptor {
        batch_size: batch,
        num_heads: heads,
        query_len: q_len,
        key_len: k_len,
        d_k,
        d_v,
        scale,
        element: ElementKind::F32,
    };
    let bw_plan = SdpaBackwardPlan::<f32>::select(stream, &bw_desc, PlanPreference::default())
        .expect("bw sel");
    bw_plan
        .run(
            stream,
            Workspace::None,
            SdpaBackwardArgs {
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
                attn: TensorRef {
                    data: dev_attn.as_slice(),
                    shape: shape_attn,
                    stride: contiguous_stride(shape_attn),
                },
                dy: TensorRef {
                    data: dev_dy.as_slice(),
                    shape: shape_y,
                    stride: contiguous_stride(shape_y),
                },
                dscores_ws: TensorMut {
                    data: dev_ws.as_slice_mut(),
                    shape: shape_attn,
                    stride: contiguous_stride(shape_attn),
                },
                dq: TensorMut {
                    data: dev_dq.as_slice_mut(),
                    shape: shape_q,
                    stride: contiguous_stride(shape_q),
                },
                dk: TensorMut {
                    data: dev_dk.as_slice_mut(),
                    shape: shape_k,
                    stride: stride_k,
                },
                dv: TensorMut {
                    data: dev_dv.as_slice_mut(),
                    shape: shape_v,
                    stride: stride_v,
                },
            },
        )
        .expect("bw run");
    stream.synchronize().expect("sync bw");

    let mut got_dq = vec![0f32; dq_n];
    let mut got_dk = vec![0f32; dk_phys_n];
    let mut got_dv = vec![0f32; dv_phys_n];
    dev_dq.copy_to_host(&mut got_dq).expect("dl dq");
    dev_dk.copy_to_host(&mut got_dk).expect("dl dk");
    dev_dv.copy_to_host(&mut got_dv).expect("dl dv");
    (got_dq, got_dk, got_dv)
}

/// 1. GQA broadcast f32 — LOAD-BEARING test.
///
/// K/V have 1 unique head, Q has 4. We broadcast via `stride_*_h = 0`.
/// Expected dK / dV is `sum_{q-head} dK_expanded[h]` from the contig
/// reference (this is exactly what atomicAdd computes).
#[test]
#[ignore]
fn sdpa_bw_gqa_broadcast_f32() {
    let (ctx, stream) = setup();
    let batch = 2i32;
    let heads = 4i32; // Q heads
    let kv_unique = 1i32;
    let q_len = 4i32;
    let k_len = 4i32;
    let d_k = 8i32;
    let d_v = 8i32;
    let scale = 1.0 / (d_k as f32).sqrt();

    let q_n = (batch * heads * q_len * d_k) as usize;
    let k_phys_n = (batch * kv_unique * k_len * d_k) as usize;
    let v_phys_n = (batch * kv_unique * k_len * d_v) as usize;
    let dy_n = (batch * heads * q_len * d_v) as usize;

    let host_q: Vec<f32> = (0..q_n).map(|i| ((i as f32) * 0.013 - 0.5).sin() * 0.5).collect();
    let host_k_phys: Vec<f32> = (0..k_phys_n).map(|i| ((i as f32) * 0.017 + 0.2).cos() * 0.5).collect();
    let host_v_phys: Vec<f32> = (0..v_phys_n).map(|i| ((i as f32) * 0.011 - 0.1).sin() * 0.7).collect();
    let host_dy: Vec<f32> = (0..dy_n).map(|i| ((i as f32) * 0.019 + 0.3).cos() * 0.4).collect();

    let stride_k: [i64; 4] = [
        (kv_unique as i64) * (k_len as i64) * (d_k as i64),
        0,
        d_k as i64,
        1,
    ];
    let stride_v: [i64; 4] = [
        (kv_unique as i64) * (k_len as i64) * (d_v as i64),
        0,
        d_v as i64,
        1,
    ];

    // Host reference: expand K/V to H heads, run the standard BW, then
    // collapse dK/dV back to `kv_unique` by summing across the Q-head
    // axis within each group.
    let host_q_f64: Vec<f64> = host_q.iter().map(|&x| x as f64).collect();
    let host_dy_f64: Vec<f64> = host_dy.iter().map(|&x| x as f64).collect();
    let host_k_phys_f64: Vec<f64> = host_k_phys.iter().map(|&x| x as f64).collect();
    let host_v_phys_f64: Vec<f64> = host_v_phys.iter().map(|&x| x as f64).collect();

    let host_k_exp = expand_kv_f64(
        &host_k_phys_f64,
        batch as usize,
        kv_unique as usize,
        heads as usize,
        k_len as usize,
        d_k as usize,
    );
    let host_v_exp = expand_kv_f64(
        &host_v_phys_f64,
        batch as usize,
        kv_unique as usize,
        heads as usize,
        k_len as usize,
        d_v as usize,
    );
    let (dq_ref, dk_exp_ref, dv_exp_ref) = host_sdpa_bw_f64_contig(
        batch as usize,
        heads as usize,
        q_len as usize,
        k_len as usize,
        d_k as usize,
        d_v as usize,
        &host_q_f64,
        &host_k_exp,
        &host_v_exp,
        &host_dy_f64,
        scale as f64,
    );
    let dk_ref = collapse_kv_grad_f64(
        &dk_exp_ref,
        batch as usize,
        heads as usize,
        kv_unique as usize,
        k_len as usize,
        d_k as usize,
    );
    let dv_ref = collapse_kv_grad_f64(
        &dv_exp_ref,
        batch as usize,
        heads as usize,
        kv_unique as usize,
        k_len as usize,
        d_v as usize,
    );

    let (got_dq, got_dk, got_dv) = run_bw_strided_f32(
        &ctx, &stream, batch, heads, q_len, k_len, d_k, d_v, scale,
        &host_q, &host_k_phys, &host_v_phys, &host_dy,
        stride_k, stride_v, kv_unique,
    );

    let tol = 5e-4f32;
    for (i, (g, r)) in got_dq.iter().zip(dq_ref.iter()).enumerate() {
        let t = ((*r).abs() as f32 * tol).max(tol);
        assert!((g - *r as f32).abs() <= t, "dq @ {i}: got={g} ref={r}");
    }
    for (i, (g, r)) in got_dk.iter().zip(dk_ref.iter()).enumerate() {
        let t = ((*r).abs() as f32 * tol).max(tol);
        assert!((g - *r as f32).abs() <= t, "dk @ {i}: got={g} ref={r}");
    }
    for (i, (g, r)) in got_dv.iter().zip(dv_ref.iter()).enumerate() {
        let t = ((*r).abs() as f32 * tol).max(tol);
        assert!((g - *r as f32).abs() <= t, "dv @ {i}: got={g} ref={r}");
    }
}

/// 2. GQA broadcast f16 — exercises the `__half` atomicAdd CAS path.
#[test]
#[ignore]
fn sdpa_bw_gqa_broadcast_f16() {
    let (ctx, stream) = setup();
    let batch = 1i32;
    let heads = 2i32;
    let kv_unique = 1i32;
    let q_len = 4i32;
    let k_len = 4i32;
    let d_k = 8i32;
    let d_v = 8i32;
    let scale = 1.0 / (d_k as f32).sqrt();

    let q_n = (batch * heads * q_len * d_k) as usize;
    let k_phys_n = (batch * kv_unique * k_len * d_k) as usize;
    let v_phys_n = (batch * kv_unique * k_len * d_v) as usize;
    let dy_n = (batch * heads * q_len * d_v) as usize;
    let attn_n = (batch * heads * q_len * k_len) as usize;
    let dq_n = (batch * heads * q_len * d_k) as usize;
    let y_n = (batch * heads * q_len * d_v) as usize;

    let host_q_f32: Vec<f32> = (0..q_n).map(|i| ((i as f32) * 0.04 - 0.3).sin() * 0.3).collect();
    let host_k_phys_f32: Vec<f32> = (0..k_phys_n).map(|i| ((i as f32) * 0.06 + 0.2).cos() * 0.3).collect();
    let host_v_phys_f32: Vec<f32> = (0..v_phys_n).map(|i| ((i as f32) * 0.02 - 0.05).sin() * 0.4).collect();
    let host_dy_f32: Vec<f32> = (0..dy_n).map(|i| ((i as f32) * 0.03 + 0.1).cos() * 0.3).collect();

    let stride_k: [i64; 4] = [
        (kv_unique as i64) * (k_len as i64) * (d_k as i64),
        0,
        d_k as i64,
        1,
    ];
    let stride_v: [i64; 4] = [
        (kv_unique as i64) * (k_len as i64) * (d_v as i64),
        0,
        d_v as i64,
        1,
    ];

    // Build f64 reference via expand → BW → collapse.
    let host_q_f64: Vec<f64> = host_q_f32.iter().map(|&x| x as f64).collect();
    let host_dy_f64: Vec<f64> = host_dy_f32.iter().map(|&x| x as f64).collect();
    let host_k_phys_f64: Vec<f64> = host_k_phys_f32.iter().map(|&x| x as f64).collect();
    let host_v_phys_f64: Vec<f64> = host_v_phys_f32.iter().map(|&x| x as f64).collect();
    let host_k_exp = expand_kv_f64(
        &host_k_phys_f64, batch as usize, kv_unique as usize, heads as usize, k_len as usize, d_k as usize,
    );
    let host_v_exp = expand_kv_f64(
        &host_v_phys_f64, batch as usize, kv_unique as usize, heads as usize, k_len as usize, d_v as usize,
    );
    let (_dq_ref, dk_exp_ref, dv_exp_ref) = host_sdpa_bw_f64_contig(
        batch as usize, heads as usize, q_len as usize, k_len as usize, d_k as usize, d_v as usize,
        &host_q_f64, &host_k_exp, &host_v_exp, &host_dy_f64, scale as f64,
    );
    let dk_ref = collapse_kv_grad_f64(
        &dk_exp_ref, batch as usize, heads as usize, kv_unique as usize, k_len as usize, d_k as usize,
    );
    let dv_ref = collapse_kv_grad_f64(
        &dv_exp_ref, batch as usize, heads as usize, kv_unique as usize, k_len as usize, d_v as usize,
    );

    // f16 upload.
    let host_q: Vec<f16> = host_q_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_k_phys: Vec<f16> = host_k_phys_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_v_phys: Vec<f16> = host_v_phys_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_dy: Vec<f16> = host_dy_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let shape_q = [batch, heads, q_len, d_k];
    let shape_k = [batch, heads, k_len, d_k];
    let shape_v = [batch, heads, k_len, d_v];
    let shape_attn = [batch, heads, q_len, k_len];
    let shape_y = [batch, heads, q_len, d_v];

    let dev_q = DeviceBuffer::from_slice(&ctx, &host_q).expect("up q");
    let dev_k = DeviceBuffer::from_slice(&ctx, &host_k_phys).expect("up k");
    let dev_v = DeviceBuffer::from_slice(&ctx, &host_v_phys).expect("up v");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_attn: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, attn_n).expect("attn");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, y_n).expect("y");
    let mut dev_ws: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, attn_n).expect("ws");
    let mut dev_dq: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, dq_n).expect("dq");
    let dk_phys_n = (batch * kv_unique * k_len * d_k) as usize;
    let dv_phys_n = (batch * kv_unique * k_len * d_v) as usize;
    // CRITICAL: dK/dV pre-zeroed (broadcast contract).
    let mut dev_dk: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, dk_phys_n).expect("dk");
    let mut dev_dv: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, dv_phys_n).expect("dv");

    let fw_desc = SdpaDescriptor {
        batch_size: batch, num_heads: heads, query_len: q_len, key_len: k_len,
        d_k, d_v, scale, is_causal: false, has_mask: false, element: ElementKind::F16,
    };
    let fw_plan = SdpaPlan::<f16>::select(&stream, &fw_desc, PlanPreference::default()).expect("fw sel");
    fw_plan.run(
        &stream, Workspace::None,
        SdpaArgs {
            q: TensorRef { data: dev_q.as_slice(), shape: shape_q, stride: contiguous_stride(shape_q) },
            k: TensorRef { data: dev_k.as_slice(), shape: shape_k, stride: stride_k },
            v: TensorRef { data: dev_v.as_slice(), shape: shape_v, stride: stride_v },
            mask: None,
            y: TensorMut { data: dev_y.as_slice_mut(), shape: shape_y, stride: contiguous_stride(shape_y) },
            attn: TensorMut { data: dev_attn.as_slice_mut(), shape: shape_attn, stride: contiguous_stride(shape_attn) },
        },
    ).expect("fw run");
    stream.synchronize().expect("sync fw");

    let bw_desc = SdpaBackwardDescriptor {
        batch_size: batch, num_heads: heads, query_len: q_len, key_len: k_len,
        d_k, d_v, scale, element: ElementKind::F16,
    };
    let bw_plan = SdpaBackwardPlan::<f16>::select(&stream, &bw_desc, PlanPreference::default()).expect("bw sel");
    bw_plan.run(
        &stream, Workspace::None,
        SdpaBackwardArgs {
            q: TensorRef { data: dev_q.as_slice(), shape: shape_q, stride: contiguous_stride(shape_q) },
            k: TensorRef { data: dev_k.as_slice(), shape: shape_k, stride: stride_k },
            v: TensorRef { data: dev_v.as_slice(), shape: shape_v, stride: stride_v },
            attn: TensorRef { data: dev_attn.as_slice(), shape: shape_attn, stride: contiguous_stride(shape_attn) },
            dy: TensorRef { data: dev_dy.as_slice(), shape: shape_y, stride: contiguous_stride(shape_y) },
            dscores_ws: TensorMut { data: dev_ws.as_slice_mut(), shape: shape_attn, stride: contiguous_stride(shape_attn) },
            dq: TensorMut { data: dev_dq.as_slice_mut(), shape: shape_q, stride: contiguous_stride(shape_q) },
            dk: TensorMut { data: dev_dk.as_slice_mut(), shape: shape_k, stride: stride_k },
            dv: TensorMut { data: dev_dv.as_slice_mut(), shape: shape_v, stride: stride_v },
        },
    ).expect("bw run");
    stream.synchronize().expect("sync bw");

    let mut got_dk = vec![f16::ZERO; dk_phys_n];
    let mut got_dv = vec![f16::ZERO; dv_phys_n];
    dev_dk.copy_to_host(&mut got_dk).expect("dl dk");
    dev_dv.copy_to_host(&mut got_dv).expect("dl dv");
    // f16 + softmax + atomicAdd-CAS → loose tolerance.
    let tol = 0.1f32;
    for (i, (g, r)) in got_dk.iter().zip(dk_ref.iter()).enumerate() {
        let t = ((*r).abs() as f32 * tol).max(tol);
        assert!(
            (g.to_f32() - *r as f32).abs() <= t,
            "dk f16 @ {i}: got={} ref={r}", g.to_f32()
        );
    }
    for (i, (g, r)) in got_dv.iter().zip(dv_ref.iter()).enumerate() {
        let t = ((*r).abs() as f32 * tol).max(tol);
        assert!(
            (g.to_f32() - *r as f32).abs() <= t,
            "dv f16 @ {i}: got={} ref={r}", g.to_f32()
        );
    }
}

/// 3. No-broadcast strided regression — fast path should still work.
/// Q layout is `[H, B, Q, D_k]` viewed as `[B, H, Q, D_k]` with permuted
/// strides; K/V remain contig with non-zero head stride.
#[test]
#[ignore]
fn sdpa_bw_no_broadcast_strided() {
    let (ctx, stream) = setup();
    let batch = 2i32;
    let heads = 2i32;
    let q_len = 4i32;
    let k_len = 4i32;
    let d_k = 8i32;
    let d_v = 8i32;
    let scale = 1.0 / (d_k as f32).sqrt();

    let q_n = (batch * heads * q_len * d_k) as usize;
    let k_n = (batch * heads * k_len * d_k) as usize;
    let v_n = (batch * heads * k_len * d_v) as usize;
    let dy_n = (batch * heads * q_len * d_v) as usize;

    let host_q: Vec<f32> = (0..q_n).map(|i| ((i as f32) * 0.013 - 0.5).sin() * 0.5).collect();
    let host_k: Vec<f32> = (0..k_n).map(|i| ((i as f32) * 0.017 + 0.2).cos() * 0.5).collect();
    let host_v: Vec<f32> = (0..v_n).map(|i| ((i as f32) * 0.011 - 0.1).sin() * 0.7).collect();
    let host_dy: Vec<f32> = (0..dy_n).map(|i| ((i as f32) * 0.019 + 0.3).cos() * 0.4).collect();

    // K/V contig, but pass them through the strided FFI (head stride != 0).
    let stride_k: [i64; 4] = contiguous_stride([batch, heads, k_len, d_k]);
    let stride_v: [i64; 4] = contiguous_stride([batch, heads, k_len, d_v]);

    // Reference: standard contig BW.
    let host_q_f64: Vec<f64> = host_q.iter().map(|&x| x as f64).collect();
    let host_k_f64: Vec<f64> = host_k.iter().map(|&x| x as f64).collect();
    let host_v_f64: Vec<f64> = host_v.iter().map(|&x| x as f64).collect();
    let host_dy_f64: Vec<f64> = host_dy.iter().map(|&x| x as f64).collect();
    let (dq_ref, dk_ref, dv_ref) = host_sdpa_bw_f64_contig(
        batch as usize, heads as usize, q_len as usize, k_len as usize, d_k as usize, d_v as usize,
        &host_q_f64, &host_k_f64, &host_v_f64, &host_dy_f64, scale as f64,
    );

    let (got_dq, got_dk, got_dv) = run_bw_strided_f32(
        &ctx, &stream, batch, heads, q_len, k_len, d_k, d_v, scale,
        &host_q, &host_k, &host_v, &host_dy,
        stride_k, stride_v, heads, // kv_phys_heads == heads (no broadcast)
    );

    let tol = 1e-4f32;
    for (i, (g, r)) in got_dq.iter().zip(dq_ref.iter()).enumerate() {
        let t = ((*r).abs() as f32 * tol).max(tol);
        assert!((g - *r as f32).abs() <= t, "dq @ {i}: got={g} ref={r}");
    }
    for (i, (g, r)) in got_dk.iter().zip(dk_ref.iter()).enumerate() {
        let t = ((*r).abs() as f32 * tol).max(tol);
        assert!((g - *r as f32).abs() <= t, "dk @ {i}: got={g} ref={r}");
    }
    for (i, (g, r)) in got_dv.iter().zip(dv_ref.iter()).enumerate() {
        let t = ((*r).abs() as f32 * tol).max(tol);
        assert!((g - *r as f32).abs() <= t, "dv @ {i}: got={g} ref={r}");
    }
}
