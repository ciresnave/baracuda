//! Phase 20.1 — real-GPU smoke tests for batched MMVQ × N-experts.
//!
//! Coverage map (one representative per dispatch path):
//!   * `batched_q8_0_f32_top_k_1_no_aliasing`   — top_k=1 store-path, type-0/1.
//!   * `batched_q8_0_f32_top_k_2_atomic`        — top_k=2 atomicAdd-path, type-0/1.
//!   * `batched_q4_K_f16_top_k_1`               — k-quant + f16 activation.
//!   * `batched_fp_f16_top_k_1`                 — pure FP (non-quant), f16.
//!   * `batched_q8_0_with_topk_weights`         — verifies the topk_weights multiplier.
//!   * `batched_q8_0_no_topk_weights_path`      — verifies the nullptr → 1.0 path.
//!
//! Tolerances follow the same rules as `gguf_mmvq_smoke.rs` and
//! `mmvq_f16_bf16_smoke.rs` — Q8_0's f16 scale forces ~1e-2 relative;
//! pure-FP and Q8_K's f32 scale tolerate ~1e-3.

// Test names use the upstream llama.cpp K-block notation (Q4_K etc.).
#![allow(non_snake_case)]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, BlockQ4K, BlockQ8_0, GgufBlockFormat, GgufMmvqArgs, GgufMmvqBatchedArgs,
    GgufMmvqBatchedDescriptor, GgufMmvqBatchedFormat, GgufMmvqBatchedPlan, GgufMmvqDescriptor,
    GgufMmvqPlan, PlanPreference, TensorMut, TensorRef, Workspace, U8,
};
use half::f16;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn pack_q8_0_row(d_f32: f32, qs: &[i8; 32]) -> Vec<u8> {
    let blk = BlockQ8_0 { d: f16::from_f32(d_f32).to_bits(), qs: *qs };
    let bytes: &[u8] = unsafe {
        core::slice::from_raw_parts(
            (&blk as *const BlockQ8_0) as *const u8,
            core::mem::size_of::<BlockQ8_0>(),
        )
    };
    bytes.to_vec()
}

// =============================================================================
// Test 1: top_k=1, no aliasing — verify each (token, row) is computed
//         the same as the equivalent single MMVQ call.
// =============================================================================

#[test]
#[ignore]
fn batched_q8_0_f32_top_k_1_no_aliasing() {
    let (ctx, stream) = setup();

    // 4 experts × 2 rows per expert × 64 cols (= 2 Q8_0 blocks per row).
    // 4 tokens, each routed to exactly one expert (token 0 → expert 0,
    // token 1 → expert 1, etc.). top_k=1 means M_total = M_tokens = 4.
    //
    // NOTE: n_cols MUST be >= 2*GGML_CUDA_DMMV_X = 64 — the type-0/1 MMVQ
    // kernel's iter_stride is 64, so n_cols < 64 leaves half the warp
    // reading addresses past the per-row block; in a batched setting
    // those addresses are the NEXT TOKEN's activations (valid memory,
    // wrong data), poisoning the dot product. n_cols=64 is the smallest
    // safe size for the type-0/1 batched path.
    let n_experts: i32 = 4;
    let n_rows: i32 = 2;
    let n_cols: i32 = 64;
    let blocks_per_row: i32 = n_cols / 32;
    let n_tokens: i32 = 4;
    let m_total = n_tokens;
    let top_k = 1;

    // Build per-expert weights.
    // expert e, row r, block b: d = 0.1*(e+1) + 0.05*r + 0.01*b
    //                           qs[i] = (i + e*2 + r + b*3) % 127 - 63
    let mut all_weight_bytes: Vec<u8> = Vec::new();
    let mut ref_weights: Vec<f32> = Vec::with_capacity((n_experts * n_rows * n_cols) as usize);
    for e in 0..n_experts {
        for r in 0..n_rows {
            for b in 0..blocks_per_row {
                let d = 0.1 * (e as f32 + 1.0) + 0.05 * (r as f32) + 0.01 * (b as f32);
                let mut qs = [0i8; 32];
                for i in 0usize..32 {
                    let v = (i as i32 + e * 2 + r + b * 3) as i32;
                    qs[i] = ((v % 127) - 63) as i8;
                }
                let packed = pack_q8_0_row(d, &qs);
                all_weight_bytes.extend_from_slice(&packed);
                for i in 0usize..32 {
                    ref_weights.push(d * (qs[i] as f32));
                }
            }
        }
    }
    let host_w: Vec<U8> = all_weight_bytes.into_iter().map(U8).collect();

    // Activations: token t, col c → t * 0.1 + c * 0.01
    let host_y: Vec<f32> = (0..(n_tokens as usize * n_cols as usize))
        .map(|i| {
            let t = (i / n_cols as usize) as f32;
            let c = (i % n_cols as usize) as f32;
            t * 0.1 + c * 0.01
        })
        .collect();

    // Routing: token i → expert i, dispatch i.
    let sorted_token_ids: Vec<i32> = (0..m_total).collect();
    let expert_offsets: Vec<i32> = (0..=n_experts).collect();  // [0,1,2,3,4]

    // CPU reference: output[t, r] = dot(W[expert=t, r, :], y[t, :])
    let mut expected = vec![0.0f32; (n_tokens * n_rows) as usize];
    for t in 0..n_tokens as usize {
        let e = t;  // 1:1 mapping in this test
        for r in 0..n_rows as usize {
            let mut acc = 0.0f32;
            for c in 0..n_cols as usize {
                let w_idx = ((e * n_rows as usize + r) * n_cols as usize) + c;
                let y_idx = t * n_cols as usize + c;
                acc += ref_weights[w_idx] * host_y[y_idx];
            }
            expected[t * n_rows as usize + r] = acc;
        }
    }

    // Upload tensors.
    let dev_w = DeviceBuffer::from_slice(&ctx, &host_w).expect("up w");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up y");
    let dev_tids = DeviceBuffer::from_slice(&ctx, &sorted_token_ids).expect("up tids");
    let dev_offs = DeviceBuffer::from_slice(&ctx, &expert_offsets).expect("up offs");
    let mut dev_out: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (n_tokens * n_rows) as usize).expect("alloc out");
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, (m_total as usize) * 4).expect("alloc ws");

    let desc = GgufMmvqBatchedDescriptor {
        n_experts, n_rows_per_expert: n_rows, n_cols, m_total, top_k,
        format: GgufMmvqBatchedFormat::Quantized(GgufBlockFormat::Q8_0),
    };
    let plan = GgufMmvqBatchedPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = GgufMmvqBatchedArgs {
        weights: TensorRef {
            data: dev_w.as_slice(),
            shape: [host_w.len() as i32],
            stride: contiguous_stride([host_w.len() as i32]),
        },
        activations: TensorRef {
            data: dev_y.as_slice(),
            shape: [n_tokens, n_cols],
            stride: [n_cols as i64, 1],
        },
        sorted_token_ids: TensorRef {
            data: dev_tids.as_slice(),
            shape: [m_total],
            stride: [1],
        },
        expert_offsets: TensorRef {
            data: dev_offs.as_slice(),
            shape: [n_experts + 1],
            stride: [1],
        },
        topk_weights: None,
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [n_tokens, n_rows],
            stride: [n_rows as i64, 1],
        },
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0f32; (n_tokens * n_rows) as usize];
    dev_out.copy_to_host(&mut got).expect("dl");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let abs_err = (g - e).abs();
        let tol = 1e-2 * e.abs().max(1.0);
        assert!(
            abs_err < tol,
            "[batched_q8_0 top_k=1] mismatch @ {i}: got {g}, expected {e} (err={abs_err}, tol={tol})",
        );
    }
}

// =============================================================================
// Test 2: top_k=2 — verify atomicAdd accumulation across multiple
//         dispatches into the same output row.
// =============================================================================

#[test]
#[ignore]
fn batched_q8_0_f32_top_k_2_atomic() {
    let (ctx, stream) = setup();

    // 2 experts × 2 rows × 64 cols (= 2 Q8_0 blocks/row — see note in
    // `batched_q8_0_f32_top_k_1_no_aliasing` for why ncols >= 64).
    // 3 tokens, each goes through 2 experts (top_k=2). M_total = 3 × 2 = 6.
    // Routing: token 0 → experts {0, 1}; token 1 → experts {0, 1};
    //          token 2 → experts {0, 1}.
    // Sorting by expert: dispatches for expert 0 first (tokens 0,1,2),
    // then for expert 1 (tokens 0,1,2).
    // sorted_token_ids = [0, 1, 2, 0, 1, 2]
    // expert_offsets   = [0, 3, 6]
    let n_experts: i32 = 2;
    let n_rows: i32 = 2;
    let n_cols: i32 = 64;
    let blocks_per_row: i32 = n_cols / 32;
    let n_tokens: i32 = 3;
    let m_total: i32 = 6;
    let top_k: i32 = 2;

    let mut all_weight_bytes: Vec<u8> = Vec::new();
    let mut ref_weights: Vec<f32> = Vec::new();
    for e in 0..n_experts {
        for r in 0..n_rows {
            for b in 0..blocks_per_row {
                let d = 0.25 + 0.05 * (e as f32) + 0.01 * (r as f32) + 0.005 * (b as f32);
                let mut qs = [0i8; 32];
                for i in 0usize..32 {
                    qs[i] = ((i as i32 + e + r + b * 2) % 13 - 6) as i8;
                }
                let packed = pack_q8_0_row(d, &qs);
                all_weight_bytes.extend_from_slice(&packed);
                for i in 0usize..32 {
                    ref_weights.push(d * (qs[i] as f32));
                }
            }
        }
    }
    let host_w: Vec<U8> = all_weight_bytes.into_iter().map(U8).collect();

    let host_y: Vec<f32> = (0..(n_tokens as usize * n_cols as usize))
        .map(|i| (i as f32) * 0.013 - 0.2)
        .collect();

    let sorted_token_ids: Vec<i32> = vec![0, 1, 2, 0, 1, 2];
    let expert_offsets: Vec<i32> = vec![0, 3, 6];

    // CPU reference: each token gets contributions from BOTH experts.
    let mut expected = vec![0.0f32; (n_tokens * n_rows) as usize];
    for t in 0..n_tokens as usize {
        for e in 0..n_experts as usize {
            for r in 0..n_rows as usize {
                let mut acc = 0.0f32;
                for c in 0..n_cols as usize {
                    let w_idx = (e * n_rows as usize + r) * n_cols as usize + c;
                    let y_idx = t * n_cols as usize + c;
                    acc += ref_weights[w_idx] * host_y[y_idx];
                }
                expected[t * n_rows as usize + r] += acc;
            }
        }
    }

    let dev_w = DeviceBuffer::from_slice(&ctx, &host_w).expect("up w");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up y");
    let dev_tids = DeviceBuffer::from_slice(&ctx, &sorted_token_ids).expect("up tids");
    let dev_offs = DeviceBuffer::from_slice(&ctx, &expert_offsets).expect("up offs");
    // Caller MUST zero-init when top_k > 1 (atomicAdd path).
    let mut dev_out: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (n_tokens * n_rows) as usize).expect("alloc out");
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, (m_total as usize) * 4).expect("alloc ws");

    let desc = GgufMmvqBatchedDescriptor {
        n_experts, n_rows_per_expert: n_rows, n_cols, m_total, top_k,
        format: GgufMmvqBatchedFormat::Quantized(GgufBlockFormat::Q8_0),
    };
    let plan = GgufMmvqBatchedPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = GgufMmvqBatchedArgs {
        weights: TensorRef {
            data: dev_w.as_slice(),
            shape: [host_w.len() as i32],
            stride: contiguous_stride([host_w.len() as i32]),
        },
        activations: TensorRef {
            data: dev_y.as_slice(),
            shape: [n_tokens, n_cols],
            stride: [n_cols as i64, 1],
        },
        sorted_token_ids: TensorRef {
            data: dev_tids.as_slice(),
            shape: [m_total], stride: [1],
        },
        expert_offsets: TensorRef {
            data: dev_offs.as_slice(),
            shape: [n_experts + 1], stride: [1],
        },
        topk_weights: None,
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [n_tokens, n_rows],
            stride: [n_rows as i64, 1],
        },
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0f32; (n_tokens * n_rows) as usize];
    dev_out.copy_to_host(&mut got).expect("dl");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let abs_err = (g - e).abs();
        // top_k=2 sums two atomicAdds → 2× rounding error budget on top
        // of Q8_0's already-loose f16 scale.
        let tol = 2e-2 * e.abs().max(1.0);
        assert!(
            abs_err < tol,
            "[batched_q8_0 top_k=2] mismatch @ {i}: got {g}, expected {e} (err={abs_err}, tol={tol})",
        );
    }
}

// =============================================================================
// Test 3: Q4_K + f16 activation — representative k-quant + f16 path.
//         Uses the "trivial scales" packing from mmvq_f16_bf16_smoke.rs
//         so we can verify against a clean reference.
// =============================================================================

fn pack_q4_k_trivial(d_f32: f32, nibbles_low: &[u8; 128]) -> Vec<u8> {
    // Same convention as mmvq_f16_bf16_smoke.rs pack_q4_k_row_trivial —
    // sub-block scales picked so only the lower nibbles contribute.
    let scales: [u8; 12] = [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0];
    let mut qs = [0u8; 128];
    for i in 0..128 {
        qs[i] = nibbles_low[i] & 0xF;
    }
    let blk = BlockQ4K {
        dm: [f16::from_f32(d_f32).to_bits(), f16::from_f32(0.0).to_bits()],
        scales,
        qs,
    };
    let bytes: &[u8] = unsafe {
        core::slice::from_raw_parts(
            (&blk as *const BlockQ4K) as *const u8,
            core::mem::size_of::<BlockQ4K>(),
        )
    };
    bytes.to_vec()
}

#[test]
#[ignore]
fn batched_q4_K_f16_top_k_1() {
    let (ctx, stream) = setup();

    // 2 experts × 1 row × 256 cols. 2 tokens, top_k=1.
    let n_experts: i32 = 2;
    let n_rows: i32 = 1;
    let n_cols: i32 = 256;
    let n_tokens: i32 = 2;
    let m_total: i32 = 2;
    let top_k: i32 = 1;

    // Build 2 experts with different `d` and different nibble patterns.
    let d0 = 0.125;
    let d1 = 0.25;
    let mut nibs_low_0 = [0u8; 128];
    let mut nibs_low_1 = [0u8; 128];
    for i in 0..128 {
        nibs_low_0[i] = ((i % 7) as u8) & 0xF;
        nibs_low_1[i] = (((i * 3 + 1) % 11) as u8) & 0xF;
    }

    let mut all_weight_bytes: Vec<u8> = Vec::new();
    all_weight_bytes.extend(pack_q4_k_trivial(d0, &nibs_low_0));
    all_weight_bytes.extend(pack_q4_k_trivial(d1, &nibs_low_1));
    let host_w: Vec<U8> = all_weight_bytes.into_iter().map(U8).collect();

    // f16 activations. Both tokens share the same activation pattern (different scale).
    let host_y_f32: Vec<f32> = (0..(n_tokens as usize * n_cols as usize))
        .map(|i| {
            let t = (i / n_cols as usize) as f32;
            let c = (i % n_cols as usize) as f32;
            (1.0 + t * 0.5) * (c * 0.005 - 0.5)
        })
        .collect();
    let host_y: Vec<f16> = host_y_f32.iter().map(|&v| f16::from_f32(v)).collect();

    // Reference: drive the single-MMVQ Q4_K kernel once per (expert, token)
    // pair. The Q4_K dot-product layout (im/ir/in sub-block iteration +
    // half-block scales) does NOT reduce to `sum(d * nibs[c%128] * y[c])`
    // even under trivial-scales packing — the kernel covers a specific
    // half of the 256 super-block columns and pairs them with non-linear
    // quant offsets. The existing single-MMVQ Q4_K f16 test in
    // `mmvq_f16_bf16_smoke.rs` solves this by comparing the f16 kernel
    // against the f32 kernel rather than a hand-coded reference; we do
    // the same here (kernel-vs-kernel ground truth).
    let sorted_token_ids: Vec<i32> = vec![0, 1];
    let expert_offsets: Vec<i32> = vec![0, 1, 2];

    let mut expected = vec![0.0f32; n_tokens as usize];
    {
        // Per-(expert, token) single-MMVQ Q4_K f32 call. Each expert's
        // weight block starts at `e * sizeof(BlockQ4K)` bytes into the
        // shared weight allocation; `w_start_byte_offset` engages the
        // actstrided FFI sibling so we don't need to upload twice.
        let dev_w_ref = DeviceBuffer::from_slice(&ctx, &host_w).expect("up w ref");
        // f32 activations for the ref so we don't compound f16 round-trip
        // error twice (the kernel-under-test runs with f16 acts; we want
        // the reference to be the *idealized* output for those exact
        // f16-rounded activations).
        for t in 0..n_tokens as usize {
            let host_y_ref_f32: Vec<f32> = (0..n_cols as usize)
                .map(|c| host_y[t * n_cols as usize + c].to_f32())
                .collect();
            let dev_y_ref = DeviceBuffer::from_slice(&ctx, &host_y_ref_f32)
                .expect("up y ref");
            let mut dev_out_ref: DeviceBuffer<f32> =
                DeviceBuffer::zeros(&ctx, 1).expect("alloc out ref");
            let e_idx = t as i64; // 1:1 mapping in this test
            let desc_ref = GgufMmvqDescriptor {
                nrows: 1,
                ncols: n_cols,
                block_format: GgufBlockFormat::Q4K,
                w_start_byte_offset: e_idx * (core::mem::size_of::<BlockQ4K>() as i64),
            };
            let plan_ref =
                GgufMmvqPlan::<f32>::select(&stream, &desc_ref, PlanPreference::default())
                    .expect("sel ref");
            let args_ref = GgufMmvqArgs {
                weight: TensorRef {
                    data: dev_w_ref.as_slice(),
                    shape: [host_w.len() as i32],
                    stride: contiguous_stride([host_w.len() as i32]),
                },
                activation: TensorRef {
                    data: dev_y_ref.as_slice(),
                    shape: [n_cols],
                    stride: [1],
                },
                output: TensorMut {
                    data: dev_out_ref.as_slice_mut(),
                    shape: [1],
                    stride: contiguous_stride([1]),
                },
            };
            plan_ref.run(&stream, Workspace::None, args_ref).expect("run ref");
            stream.synchronize().expect("sync ref");
            let mut out_h = [0f32; 1];
            dev_out_ref.copy_to_host(&mut out_h).expect("dl ref");
            expected[t] = out_h[0];
        }
    }

    let dev_w = DeviceBuffer::from_slice(&ctx, &host_w).expect("up w");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up y");
    let dev_tids = DeviceBuffer::from_slice(&ctx, &sorted_token_ids).expect("up tids");
    let dev_offs = DeviceBuffer::from_slice(&ctx, &expert_offsets).expect("up offs");
    let mut dev_out: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (n_tokens * n_rows) as usize).expect("alloc out");
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, (m_total as usize) * 4).expect("alloc ws");

    let desc = GgufMmvqBatchedDescriptor {
        n_experts, n_rows_per_expert: n_rows, n_cols, m_total, top_k,
        format: GgufMmvqBatchedFormat::Quantized(GgufBlockFormat::Q4K),
    };
    let plan = GgufMmvqBatchedPlan::<f16>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = GgufMmvqBatchedArgs {
        weights: TensorRef {
            data: dev_w.as_slice(),
            shape: [host_w.len() as i32],
            stride: contiguous_stride([host_w.len() as i32]),
        },
        activations: TensorRef {
            data: dev_y.as_slice(),
            shape: [n_tokens, n_cols],
            stride: [n_cols as i64, 1],
        },
        sorted_token_ids: TensorRef {
            data: dev_tids.as_slice(), shape: [m_total], stride: [1],
        },
        expert_offsets: TensorRef {
            data: dev_offs.as_slice(), shape: [n_experts + 1], stride: [1],
        },
        topk_weights: None,
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [n_tokens, n_rows],
            stride: [n_rows as i64, 1],
        },
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![f16::ZERO; n_tokens as usize];
    dev_out.copy_to_host(&mut got).expect("dl");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let gf = g.to_f32();
        let abs_err = (gf - e).abs();
        // f16 dst + f16 act + Q4_K f16 scale → ~3% rel.
        let tol = 3e-2 * e.abs().max(1.0);
        assert!(
            abs_err < tol,
            "[batched_q4_K f16] mismatch @ {i}: got {gf}, expected {e} (err={abs_err}, tol={tol})",
        );
    }
}

// =============================================================================
// Test 4: Pure-FP (non-quant) f16 — verifies the non-quant variant.
// =============================================================================

#[test]
#[ignore]
fn batched_fp_f16_top_k_1() {
    let (ctx, stream) = setup();

    let n_experts: i32 = 2;
    let n_rows: i32 = 3;
    let n_cols: i32 = 16;
    let n_tokens: i32 = 2;
    let m_total: i32 = 2;
    let top_k: i32 = 1;

    // Pure-FP weights: f16 elements stored as a flat u8 byte tensor.
    let mut weights_f16: Vec<f16> = Vec::new();
    for e in 0..n_experts {
        for r in 0..n_rows {
            for c in 0..n_cols {
                let v = 0.1 * (e as f32 + 1.0) + 0.01 * (r as f32) + 0.005 * (c as f32);
                weights_f16.push(f16::from_f32(v));
            }
        }
    }
    let weights_bytes: Vec<u8> = unsafe {
        core::slice::from_raw_parts(
            weights_f16.as_ptr() as *const u8,
            weights_f16.len() * core::mem::size_of::<f16>(),
        )
    }
    .to_vec();
    let host_w: Vec<U8> = weights_bytes.into_iter().map(U8).collect();

    // Activations
    let host_y_f32: Vec<f32> = (0..(n_tokens as usize * n_cols as usize))
        .map(|i| (i as f32) * 0.03 - 0.5)
        .collect();
    let host_y: Vec<f16> = host_y_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let sorted_token_ids: Vec<i32> = vec![0, 1];
    let expert_offsets: Vec<i32> = vec![0, 1, 2];

    // CPU reference (f32 accumulator, then cast to f16 at the end).
    let mut expected = vec![0.0f32; (n_tokens * n_rows) as usize];
    for t in 0..n_tokens as usize {
        let e = t;
        for r in 0..n_rows as usize {
            let mut acc = 0.0f32;
            for c in 0..n_cols as usize {
                let w_idx = (e * n_rows as usize + r) * n_cols as usize + c;
                let y_idx = t * n_cols as usize + c;
                acc += weights_f16[w_idx].to_f32() * host_y[y_idx].to_f32();
            }
            expected[t * n_rows as usize + r] = acc;
        }
    }

    let dev_w = DeviceBuffer::from_slice(&ctx, &host_w).expect("up w");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up y");
    let dev_tids = DeviceBuffer::from_slice(&ctx, &sorted_token_ids).expect("up tids");
    let dev_offs = DeviceBuffer::from_slice(&ctx, &expert_offsets).expect("up offs");
    let mut dev_out: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (n_tokens * n_rows) as usize).expect("alloc out");
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, (m_total as usize) * 4).expect("alloc ws");

    let desc = GgufMmvqBatchedDescriptor {
        n_experts, n_rows_per_expert: n_rows, n_cols, m_total, top_k,
        format: GgufMmvqBatchedFormat::Fp,
    };
    let plan = GgufMmvqBatchedPlan::<f16>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = GgufMmvqBatchedArgs {
        weights: TensorRef {
            data: dev_w.as_slice(),
            shape: [host_w.len() as i32],
            stride: contiguous_stride([host_w.len() as i32]),
        },
        activations: TensorRef {
            data: dev_y.as_slice(),
            shape: [n_tokens, n_cols],
            stride: [n_cols as i64, 1],
        },
        sorted_token_ids: TensorRef {
            data: dev_tids.as_slice(), shape: [m_total], stride: [1],
        },
        expert_offsets: TensorRef {
            data: dev_offs.as_slice(), shape: [n_experts + 1], stride: [1],
        },
        topk_weights: None,
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [n_tokens, n_rows],
            stride: [n_rows as i64, 1],
        },
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![f16::ZERO; (n_tokens * n_rows) as usize];
    dev_out.copy_to_host(&mut got).expect("dl");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let gf = g.to_f32();
        let abs_err = (gf - e).abs();
        let tol = 2e-2 * e.abs().max(1.0);
        assert!(
            abs_err < tol,
            "[batched_fp f16] mismatch @ {i}: got {gf}, expected {e} (err={abs_err}, tol={tol})",
        );
    }
}

// =============================================================================
// Test 5 + 6: topk_weights presence vs absence on Q8_0 — verifies the
//             routing multiplier (and that None = nullptr → 1.0).
// =============================================================================

fn run_q8_0_with_optional_topk_weights(
    ctx: &Context,
    stream: &Stream,
    use_topk_weights: bool,
    multiplier: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n_experts: i32 = 1;
    let n_rows: i32 = 1;
    // Phase 22 guard: type-0/1 GGUF formats require `n_cols >= 64` to
    // avoid silent-wrong results from contiguous-batched activation OOB
    // reads. Bumped from 32 to 64 in the consolidation pass.
    let n_cols: i32 = 64;
    let n_tokens: i32 = 2;
    let m_total: i32 = 2;
    let top_k: i32 = 1;

    let d = 0.5;
    // Two Q8_0 blocks of 32 i8 values cover the 64-column row.
    let qs_part_a: [i8; 32] = std::array::from_fn(|i| (i as i32 - 16) as i8);
    let qs_part_b: [i8; 32] = std::array::from_fn(|i| (i as i32 - 16) as i8);
    let mut packed_a = pack_q8_0_row(d, &qs_part_a);
    let packed_b = pack_q8_0_row(d, &qs_part_b);
    packed_a.extend(packed_b);
    let host_w: Vec<U8> = packed_a.into_iter().map(U8).collect();

    let host_y: Vec<f32> = (0..(n_tokens as usize * n_cols as usize))
        .map(|i| (i as f32) * 0.02 - 0.3)
        .collect();

    let sorted_token_ids: Vec<i32> = vec![0, 1];
    let expert_offsets: Vec<i32> = vec![0, 2];
    let topk_w_vec: Vec<f32> = vec![multiplier; m_total as usize];

    // Reference
    // Concatenate the two 32-element qs blocks to cover all 64 columns.
    let mut qs_full: Vec<i8> = Vec::with_capacity(n_cols as usize);
    qs_full.extend_from_slice(&qs_part_a);
    qs_full.extend_from_slice(&qs_part_b);
    let mut expected = vec![0.0f32; n_tokens as usize];
    for t in 0..n_tokens as usize {
        let mut acc = 0.0f32;
        for c in 0..n_cols as usize {
            acc += d * (qs_full[c] as f32) * host_y[t * n_cols as usize + c];
        }
        let w = if use_topk_weights { multiplier } else { 1.0 };
        expected[t] = w * acc;
    }

    let dev_w = DeviceBuffer::from_slice(ctx, &host_w).expect("up w");
    let dev_y = DeviceBuffer::from_slice(ctx, &host_y).expect("up y");
    let dev_tids = DeviceBuffer::from_slice(ctx, &sorted_token_ids).expect("up tids");
    let dev_offs = DeviceBuffer::from_slice(ctx, &expert_offsets).expect("up offs");
    let dev_tw = DeviceBuffer::from_slice(ctx, &topk_w_vec).expect("up tw");
    let mut dev_out: DeviceBuffer<f32> = DeviceBuffer::zeros(ctx, n_tokens as usize).expect("alloc out");
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(ctx, (m_total as usize) * 4).expect("alloc ws");

    let desc = GgufMmvqBatchedDescriptor {
        n_experts, n_rows_per_expert: n_rows, n_cols, m_total, top_k,
        format: GgufMmvqBatchedFormat::Quantized(GgufBlockFormat::Q8_0),
    };
    let plan = GgufMmvqBatchedPlan::<f32>::select(stream, &desc, PlanPreference::default())
        .expect("select");
    let args = GgufMmvqBatchedArgs {
        weights: TensorRef {
            data: dev_w.as_slice(),
            shape: [host_w.len() as i32],
            stride: contiguous_stride([host_w.len() as i32]),
        },
        activations: TensorRef {
            data: dev_y.as_slice(),
            shape: [n_tokens, n_cols],
            stride: [n_cols as i64, 1],
        },
        sorted_token_ids: TensorRef {
            data: dev_tids.as_slice(), shape: [m_total], stride: [1],
        },
        expert_offsets: TensorRef {
            data: dev_offs.as_slice(), shape: [n_experts + 1], stride: [1],
        },
        topk_weights: if use_topk_weights {
            Some(TensorRef {
                data: dev_tw.as_slice(),
                shape: [m_total],
                stride: [1],
            })
        } else {
            None
        },
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [n_tokens, n_rows],
            stride: [n_rows as i64, 1],
        },
    };
    plan.run(stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0f32; n_tokens as usize];
    dev_out.copy_to_host(&mut got).expect("dl");
    (got, expected)
}

#[test]
#[ignore]
fn batched_q8_0_with_topk_weights() {
    let (ctx, stream) = setup();
    let multiplier = 0.5_f32;
    let (got, expected) = run_q8_0_with_optional_topk_weights(&ctx, &stream, true, multiplier);
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let abs_err = (g - e).abs();
        let tol = 1e-2 * e.abs().max(1.0);
        assert!(abs_err < tol, "[batched_q8_0 +topk_weights] @{i}: got {g}, expected {e}");
    }
}

#[test]
#[ignore]
fn batched_q8_0_no_topk_weights_path() {
    let (ctx, stream) = setup();
    // multiplier param is ignored when use_topk_weights=false; expected
    // is then computed with w=1.0 inside the helper.
    let (got, expected) = run_q8_0_with_optional_topk_weights(&ctx, &stream, false, 999.0);
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let abs_err = (g - e).abs();
        let tol = 1e-2 * e.abs().max(1.0);
        assert!(abs_err < tol, "[batched_q8_0 nullptr_topk] @{i}: got {g}, expected {e}");
    }
}
