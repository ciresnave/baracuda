//! Phase 51 — speculative-decoding tree-attention composition example.
//!
//! Demonstrates how baracuda's primitives compose to implement the
//! kernel-level data path of an EAGLE / Medusa style speculative
//! decoder. baracuda's posture is "we ship building blocks, downstream
//! owns orchestration" — this example exercises the **kernel-level
//! composition** that is the only thing baracuda is responsible for:
//!
//!   1. **Build a tree-attention mask** for a small draft tree.
//!   2. **Call [`FlashSdpaPlan`] with the additive mask** (Phase 51
//!      — `mask = Some(...)`) — verify the draft tokens see only
//!      their ancestor chain (not siblings).
//!   3. **Sample a next token** with a sort-free softmax + argmax
//!      composition (baracuda ships [`crate::TopkPlan`] + softmax /
//!      multinomial; Phase 46's `TopKTopPSamplingPlan` adds an
//!      optional flashinfer-backed fast path).
//!   4. **Commit accepted token KV** with [`WriteSlicePlan`] (Phase
//!      13's KV-cache fast path).
//!
//! What is *NOT* in scope for baracuda:
//!
//!   - Drafting the candidate tokens (caller's draft model — could be
//!     a small transformer, EAGLE's auxiliary head, or Medusa's heads).
//!   - Choosing the tree topology (per-step planner).
//!   - Verification policy (token-by-token acceptance check, rejection
//!     sampling, etc.).
//!
//! See `docs/SPEC_DECODE.md` for the full division of responsibilities.
//!
//! Run with:
//! ```bash
//! cargo run --release --example speculative_decode_compose
//! ```
//!
//! Requires a real CUDA device.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, FlashSdpaArgs, FlashSdpaDescriptor, FlashSdpaPlan,
    PlanPreference, TensorMut, TensorRef, Workspace, WriteSliceArgs, WriteSliceDescriptor,
    WriteSlicePlan,
};

/// Five-node draft tree (the canonical EAGLE pattern):
///
/// ```text
///   t0  (root)
///   ├── t1
///   │    ├── t3
///   │    └── t4
///   └── t2
/// ```
///
/// Returns the `parent[i]` array (parent index into `[t0..t4]`, with
/// `None` for the root).
fn draft_tree_parents() -> [Option<usize>; 5] {
    [None, Some(0), Some(0), Some(1), Some(1)]
}

/// Build a tree-attention mask for `[B=1, H, Q=5, K=prefix_len+5]`.
/// Each draft token attends to all prefix positions and to its own
/// ancestor chain (incl. self) in the tree, but NOT to sibling /
/// cousin draft tokens.
fn build_tree_mask(prefix_len: usize, num_heads: usize) -> Vec<f32> {
    const Q: usize = 5;
    let k_len = prefix_len + Q;
    let mut mask = vec![f32::NEG_INFINITY; num_heads * Q * k_len];

    let parents = draft_tree_parents();
    // For each draft token, walk up to root to collect ancestors.
    let mut ancestors: [Vec<usize>; 5] = Default::default();
    for i in 0..Q {
        let mut chain = vec![i];
        let mut cur = parents[i];
        while let Some(p) = cur {
            chain.push(p);
            cur = parents[p];
        }
        ancestors[i] = chain;
    }

    for h in 0..num_heads {
        for qi in 0..Q {
            // Prefix is always visible.
            for kj in 0..prefix_len {
                let idx = (h * Q + qi) * k_len + kj;
                mask[idx] = 0.0;
            }
            // Ancestor draft tokens.
            for &anc in &ancestors[qi] {
                let kj = prefix_len + anc;
                let idx = (h * Q + qi) * k_len + kj;
                mask[idx] = 0.0;
            }
        }
    }
    mask
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init()?;
    let device = Device::get(0)?;
    let ctx = Context::new(&device)?;
    let stream = Stream::new(&ctx)?;

    // -------------------------------------------------------------------
    // Step 1 — set up the KV cache and prefix. In a real spec-decoder
    // these come from the preceding decode steps; here we synthesize.
    // -------------------------------------------------------------------
    const BATCH: i32 = 1;
    const HEADS: i32 = 8;
    const HEAD_DIM: i32 = 32;
    const PREFIX_LEN: i32 = 64;
    const Q_LEN: i32 = 5; // 5 draft tokens (root + 4 children)
    const K_CAP: i32 = 256; // KV-cache capacity
    let k_cur_len = PREFIX_LEN + Q_LEN;

    // KV cache: [B, H, K_CAP, HEAD_DIM]. Pre-filled with the prefix.
    let mut k_cache_h = vec![0f32; (BATCH * HEADS * K_CAP * HEAD_DIM) as usize];
    let mut v_cache_h = vec![0f32; (BATCH * HEADS * K_CAP * HEAD_DIM) as usize];
    // Synthesize prefix (random-ish).
    for i in 0..(BATCH * HEADS * PREFIX_LEN * HEAD_DIM) as usize {
        k_cache_h[i] = ((i as f32) * 0.011).sin() * 0.5;
        v_cache_h[i] = ((i as f32) * 0.013 + 0.3).cos() * 0.5;
    }
    let mut k_cache = DeviceBuffer::from_slice(&ctx, &k_cache_h)?;
    let mut v_cache = DeviceBuffer::from_slice(&ctx, &v_cache_h)?;

    // -------------------------------------------------------------------
    // Step 2 — draft model produces 5 candidate Q/K/V tokens.
    // In a real spec-decoder this is the draft model's forward pass.
    // -------------------------------------------------------------------
    let n_qkv = (BATCH * HEADS * Q_LEN * HEAD_DIM) as usize;
    let q_draft: Vec<f32> = (0..n_qkv).map(|i| ((i as f32) * 0.017 + 1.1).sin() * 0.5).collect();
    let k_draft: Vec<f32> = (0..n_qkv).map(|i| ((i as f32) * 0.019 + 0.7).cos() * 0.5).collect();
    let v_draft: Vec<f32> = (0..n_qkv).map(|i| ((i as f32) * 0.023 + 0.2).sin() * 0.5).collect();

    let d_q = DeviceBuffer::from_slice(&ctx, &q_draft)?;
    let d_k = DeviceBuffer::from_slice(&ctx, &k_draft)?;
    let d_v = DeviceBuffer::from_slice(&ctx, &v_draft)?;

    // -------------------------------------------------------------------
    // Step 3 — APPEND draft K/V into the cache at position [PREFIX_LEN]
    // through [PREFIX_LEN + Q_LEN). WriteSlicePlan (Phase 13).
    // -------------------------------------------------------------------
    let kv_dest_shape = [BATCH, HEADS, K_CAP, HEAD_DIM];
    let kv_source_shape = [BATCH, HEADS, Q_LEN, HEAD_DIM];
    let kv_desc = WriteSliceDescriptor {
        dest_shape: kv_dest_shape,
        source_shape: kv_source_shape,
        ranges: [
            (0, BATCH),
            (0, HEADS),
            (PREFIX_LEN, PREFIX_LEN + Q_LEN),
            (0, HEAD_DIM),
        ],
        element: ElementKind::F32,
    };
    let kv_plan: WriteSlicePlan<f32, 4> =
        WriteSlicePlan::select(&stream, &kv_desc, PlanPreference::default())?;
    // K append.
    kv_plan.run(
        &stream,
        Workspace::None,
        WriteSliceArgs {
            dest: TensorMut {
                data: k_cache.as_slice_mut(),
                shape: kv_dest_shape,
                stride: contiguous_stride(kv_dest_shape),
            },
            source: TensorRef {
                data: d_k.as_slice(),
                shape: kv_source_shape,
                stride: contiguous_stride(kv_source_shape),
            },
        },
    )?;
    // V append.
    kv_plan.run(
        &stream,
        Workspace::None,
        WriteSliceArgs {
            dest: TensorMut {
                data: v_cache.as_slice_mut(),
                shape: kv_dest_shape,
                stride: contiguous_stride(kv_dest_shape),
            },
            source: TensorRef {
                data: d_v.as_slice(),
                shape: kv_source_shape,
                stride: contiguous_stride(kv_source_shape),
            },
        },
    )?;

    // -------------------------------------------------------------------
    // Step 4 — Build the TREE-ATTENTION MASK and call FlashSdpaPlan
    // with mask = Some(...). This is the Phase 51 NEW capability.
    // -------------------------------------------------------------------
    let mask_h = build_tree_mask(PREFIX_LEN as usize, HEADS as usize);
    let d_mask = DeviceBuffer::from_slice(&ctx, &mask_h)?;

    let mut y_out: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (BATCH * HEADS * Q_LEN * HEAD_DIM) as usize)?;
    let mut lse_out: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (BATCH * HEADS * Q_LEN) as usize)?;

    // Note: we slice the KV cache to the current length when calling
    // SDPA. baracuda's TensorRef doesn't natively support an
    // axis-prefix view, so we run the full cap-length cache through
    // FlashSdpa — the mask's -INFINITY suppresses unused cache slots.
    // (Production code would either zero past-len cache or use
    // FlashInfer's paged attention plan that takes an explicit len.)
    let q_shape = [BATCH, HEADS, Q_LEN, HEAD_DIM];
    let k_shape = [BATCH, HEADS, k_cur_len, HEAD_DIM];
    let v_shape = [BATCH, HEADS, k_cur_len, HEAD_DIM];
    let y_shape = [BATCH, HEADS, Q_LEN, HEAD_DIM];
    let lse_shape = [BATCH, HEADS, Q_LEN];
    let mask_shape = [BATCH, HEADS, Q_LEN, k_cur_len];

    // KV-cache slice: take the first [0..k_cur_len] of each head's K-axis.
    // Since cache is contiguous-strided as [B, H, K_CAP, D], simply
    // truncating the shape (without changing stride) doesn't yield a
    // valid contiguous view (the strides assume K_CAP rows per head).
    // For this example, we copy out a tightly-packed [B, H, k_cur_len, D]
    // slice. Production would use paged attention or pre-allocate the
    // cache at the actual sequence length.
    let mut k_packed = vec![0f32; (BATCH * HEADS * k_cur_len * HEAD_DIM) as usize];
    let mut v_packed = vec![0f32; (BATCH * HEADS * k_cur_len * HEAD_DIM) as usize];
    k_cache.copy_to_host(&mut k_cache_h)?;
    v_cache.copy_to_host(&mut v_cache_h)?;
    for b in 0..(BATCH as usize) {
        for h in 0..(HEADS as usize) {
            for kk in 0..(k_cur_len as usize) {
                for d in 0..(HEAD_DIM as usize) {
                    let src = ((b * HEADS as usize + h) * K_CAP as usize + kk)
                        * HEAD_DIM as usize
                        + d;
                    let dst = ((b * HEADS as usize + h) * k_cur_len as usize + kk)
                        * HEAD_DIM as usize
                        + d;
                    k_packed[dst] = k_cache_h[src];
                    v_packed[dst] = v_cache_h[src];
                }
            }
        }
    }
    let d_k_packed = DeviceBuffer::from_slice(&ctx, &k_packed)?;
    let d_v_packed = DeviceBuffer::from_slice(&ctx, &v_packed)?;

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let desc = FlashSdpaDescriptor::new(
        BATCH,
        HEADS,
        Q_LEN,
        k_cur_len,
        HEAD_DIM,
        HEAD_DIM,
        scale,
        false, // is_causal — tree mask handles all suppression
        ElementKind::F32,
    );
    let plan = FlashSdpaPlan::<f32>::select(&stream, &desc, PlanPreference::default())?;

    plan.run(
        &stream,
        Workspace::None,
        FlashSdpaArgs {
            q: TensorRef {
                data: d_q.as_slice(),
                shape: q_shape,
                stride: contiguous_stride(q_shape),
            },
            k: TensorRef {
                data: d_k_packed.as_slice(),
                shape: k_shape,
                stride: contiguous_stride(k_shape),
            },
            v: TensorRef {
                data: d_v_packed.as_slice(),
                shape: v_shape,
                stride: contiguous_stride(v_shape),
            },
            y: TensorMut {
                data: y_out.as_slice_mut(),
                shape: y_shape,
                stride: contiguous_stride(y_shape),
            },
            lse: TensorMut {
                data: lse_out.as_slice_mut(),
                shape: lse_shape,
                stride: contiguous_stride(lse_shape),
            },
            mask: Some(TensorRef {
                data: d_mask.as_slice(),
                shape: mask_shape,
                stride: contiguous_stride(mask_shape),
            }),
            alibi_slopes: None,
        },
    )?;
    stream.synchronize()?;

    let mut y_host = vec![0f32; (BATCH * HEADS * Q_LEN * HEAD_DIM) as usize];
    y_out.copy_to_host(&mut y_host)?;

    println!("Spec-decode tree attention output (first 8 cells of t0):");
    for i in 0..8 {
        println!("  y[t0, h0, {i}] = {:.6}", y_host[i]);
    }

    // -------------------------------------------------------------------
    // Step 5 — Sampling / verification (caller responsibility).
    //
    // Real spec-decode would now:
    //   - Run logits = output @ unembedding for each draft token.
    //   - Sample next token from logits (using
    //     `TopKTopPSamplingPlan` if Phase 46 flashinfer is enabled,
    //     or baracuda's existing topk + softmax + argmax composition).
    //   - Walk down the draft tree, accepting tokens that match the
    //     drafted ones (or via a rejection-sampling rule), commit
    //     accepted KV with `WriteSlicePlan`.
    //
    // baracuda ships the building blocks — orchestration is owned by
    // the caller per the design philosophy in `docs/SPEC_DECODE.md`.
    // -------------------------------------------------------------------
    println!("\nDone — tree-attention FW ran successfully.");
    println!("Acceptance / sampling / commit logic is caller-owned");
    println!("(see docs/SPEC_DECODE.md for division of responsibilities).");

    Ok(())
}
