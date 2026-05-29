# Speculative decoding with baracuda (Phase 51)

baracuda ships the **kernel-level building blocks** for speculative
decoding (EAGLE, Medusa, SMART, lookahead decoding, etc.). It does
**not** ship a bundled spec-decode orchestrator. This document
explains the division of responsibilities and how to compose the
primitives.

## The insight (and the gap Phase 51 closes)

The recurring observation across the spec-decode literature is that
the *kernel* picture is small:

- A **standard SDPA forward** that accepts an **arbitrary additive
  attention mask** instead of just the rectangular causal mask.
- A **KV cache append** that can write multiple tokens at once
  (the draft tree's accepted prefix).
- A **sampler** over the verifier model's logits.

EAGLE and Medusa, despite the model-architecture novelty, ship **zero
custom CUDA kernels** — the tree-attention mask is just an extra arg
to the existing SDPA call.

baracuda had the SDPA + KV-append + sampling primitives since earlier
phases. The one gap was that `FlashSdpaPlan` only supported the
rectangular causal mask (the FA2 `Mask` template covers
causal / local / alibi; arbitrary masks were out of scope through
Phase 50). **Phase 51 closes that gap** by adding an optional
`mask: TensorRef<f32, 4>` field to `FlashSdpaArgs`.

With that, every published spec-decode variant can be implemented
*entirely* in user code over baracuda's primitives — no new CUDA work
needed downstream.

## What baracuda provides

The kernel-level surface that a spec-decode orchestrator needs from
baracuda:

| Primitive | Plan | Phase |
|-----------|------|-------|
| Attention forward with arbitrary additive mask | `FlashSdpaPlan` (`mask: Some(...)`) | **51** |
| Attention forward with rectangular causal mask | `FlashSdpaPlan` (`mask: None`, `is_causal: true`) | 6.6, FA2 in 42 |
| Naive masked SDPA (when arbitrary mask + small shape) | `SdpaPlan` (always supports mask) | 6 |
| KV cache append (multi-token, contiguous) | `WriteSlicePlan` | 13 |
| Logits projection | `GemmPlan` | core |
| Softmax | `SoftmaxPlan` | 5 |
| Top-K / Top-P / Min-P sampling | `TopKTopPSamplingPlan` (FlashInfer) | 46 |
| Top-K + manual sampling | `TopkPlan` + `SoftmaxPlan` + multinomial | core |

The Phase 51 `mask` field on `FlashSdpaArgs` is **optional**; existing
callers pass `None` and get exact Phase 42 behaviour. The mask is
**always f32** regardless of QKV dtype (additive-bias precision is
decoupled from QKV precision; keeps the FFI surface from
combinatorially exploding across 4 dtype × 4 mask-dtype combinations).

## What you (the caller) provide

These are the pieces baracuda **does not** ship and that you own:

1. **Draft model.** A small auxiliary model (or extra heads on the
   verifier, as in Medusa) that produces candidate next tokens. baracuda
   doesn't bundle a draft model; you wire whatever fits — could be a
   smaller transformer from `baracuda-kernels` primitives, EAGLE's
   auxiliary projection layer, or Medusa's parallel heads.

2. **Tree topology.** Per decode step, decide the shape of the draft
   tree (which token spawns which children, depth, fan-out).
   baracuda's `FlashSdpaPlan(mask)` accepts any topology you encode
   into the `[B, H, Q, K]` mask.

3. **Verification policy.** Given the verifier model's logits over
   the draft tokens, decide which prefix of the draft tree to accept.
   The standard choices are:

   - **Greedy match**: accept tokens that match the verifier's argmax.
   - **Rejection sampling**: accept stochastically, with rejection
     probability tied to the ratio of draft and verifier probabilities
     (preserves the verifier model's exact sampling distribution).

4. **KV cache management.** baracuda's `WriteSlicePlan` does the
   actual cache write, but you decide what range to commit (the
   length of the accepted prefix) and when to rewind on rejection.

5. **Token bookkeeping.** Tracking which draft token corresponds to
   which leaf, decoding the accepted-prefix back to a token sequence.

## Building a tree-attention mask

For a tree with `Q` draft tokens (root + descendants) and a `P`-length
verified prefix, the mask shape is `[B, H, Q, P + Q]` (f32, contiguous,
row-major).

Per query token `i`:

- Cells `mask[..., i, 0..P]` are `0.0` (the prefix is always visible).
- Cell `mask[..., i, P + j]` is `0.0` iff `j` is an ancestor of `i`
  (or `i == j`).
- All other cells are `-INFINITY`.

The kernel adds the mask to `S = Q·K^T·scale` before softmax. The
`-INFINITY` cells get suppressed exactly (the softmax denominator
ignores them). Finite cells in the mask act as arbitrary additive
biases — useful for sliding-window attention sinks, alibi, learned
positional biases, prefix-LM, MoE expert masking, and so on.

```rust
fn build_tree_mask(parents: &[Option<usize>], prefix_len: usize, num_heads: usize) -> Vec<f32> {
    let q = parents.len();
    let k_len = prefix_len + q;
    let mut mask = vec![f32::NEG_INFINITY; num_heads * q * k_len];
    for h in 0..num_heads {
        for qi in 0..q {
            // Prefix.
            for kj in 0..prefix_len {
                mask[(h * q + qi) * k_len + kj] = 0.0;
            }
            // Ancestor chain (self + parents to root).
            let mut cur = Some(qi);
            while let Some(node) = cur {
                mask[(h * q + qi) * k_len + (prefix_len + node)] = 0.0;
                cur = parents[node];
            }
        }
    }
    mask
}
```

A runnable end-to-end example is at
[`crates/baracuda-kernels/examples/speculative_decode_compose.rs`](../../crates/baracuda-kernels/examples/speculative_decode_compose.rs).

## Composition pattern

```text
                    +--------------------+
                    | Caller-owned       |
                    | draft model        |   (small Transformer / Medusa heads / EAGLE projection)
                    +---------+----------+
                              |  candidate tokens [t0..tN]
                              v
                    +--------------------+
                    | Caller-owned       |
                    | tree planner       |   (decides parent[] array)
                    +---------+----------+
                              |  parent[] array
                              v
+-------------------+    +--------------------+    +-------------------+
| WriteSlicePlan    |--->| FlashSdpaPlan      |--->| GemmPlan          |
| (KV append)       |    | (mask = tree_mask) |    | (output @ unemb)  |
+-------------------+    +--------------------+    +---------+---------+
                                                              |
                                                              v
                                                  +---------------------+
                                                  | TopKTopPSampling    |
                                                  | (or topk + softmax) |
                                                  +---------+-----------+
                                                            |
                                                            v
                                                  +--------------------+
                                                  | Caller-owned       |
                                                  | verification logic |
                                                  | (rejection sample) |
                                                  +---------+----------+
                                                            |
                                                            v
                                                  accepted-prefix length
                                                  (rewind unused KV / commit)
```

baracuda provides every box marked `*Plan`. The two `Caller-owned`
boxes are the **architectural choice** of which spec-decode variant
you implement (EAGLE? Medusa? lookahead?) plus the **verification
policy** (greedy? rejection sampling?). baracuda intentionally does
not bundle these because:

- They differ across published variants and we'd be picking favourites.
- They're tiny — usually < 200 lines of host code each.
- Wrapping them in a baracuda `Plan` type adds API surface without
  reducing user code.

## When to choose Phase 51 arbmask vs alternatives

| Use case | Recommendation |
|----------|----------------|
| **Spec-decode tree attention** | `FlashSdpaPlan(mask = Some(tree_mask))` |
| **Sliding window with attention sinks** | `FlashSdpaPlan(mask = Some(band_with_sinks))` |
| **MoE expert masking at attention** | `FlashSdpaPlan(mask = Some(expert_route_mask))` |
| **Prefix-LM (bidirectional prefix, causal suffix)** | `FlashSdpaPlan(mask = Some(prefix_lm_mask))` |
| **Pure causal (no extra mask)** | `FlashSdpaPlan(mask = None, is_causal = true)` |
| **Pure causal + alibi** | bespoke alibi via `AlibiPlan` first, then `FlashSdpaPlan(mask = Some(alibi_bias))` |
| **Vanilla full attention** | `FlashSdpaPlan(mask = None, is_causal = false)` |
| **Paged KV cache + arbmask** | wait for Phase 46 FlashInfer integration to land paged + masked; Phase 51 is contiguous KV only |

## Implementation notes

- **Mask dtype is f32**, always. Phase 51 chose this single-dtype
  contract to avoid the FFI surface explosion (4 element dtype × 4
  mask dtype would be 16 SKUs; we ship 4). For mask-bias use cases
  where `-INFINITY` suppression is the only requirement, a 1-bit
  packed mask would be tighter but not enough to justify the SKU
  growth at this point.

- **Causal + arbitrary mask compose** safely: when both are present,
  the kernel applies the causal mask first (sets out-of-causal cells
  to `-INFINITY`), then adds the arbitrary mask. For finite mask
  values this is a no-op on the causal-suppressed cells (`a + -INF
  == -INF`). For `-INFINITY` mask cells, `-INF + -INF == -INF` — also
  suppressed. The combined behaviour is "AND of suppressions".

- **No GQA broadcast for arbmask Tier 1.** The arbmask kernel
  requires a fully materialized `[B, H, Q, K]` mask. If your
  attention has `num_heads_k < num_heads`, broadcast the mask to the
  query-head count before passing it in (cheap host-side op).

- **No backward** for arbmask Tier 1. Same deferral as the FA2
  vendor — training-time arbmask gradients land in Tier 2 of this
  effort.

- **No FA2 backend integration.** The arbmask path is bespoke
  (baracuda's own online-softmax FW with mask-load added). FA2
  v2.8.3's vendored `Mask` template doesn't have hooks for arbitrary
  per-cell biases; bolting one in would require modifying the
  vendored kernel template (vendor drift cost > benefit at decode
  shapes where the arbmask is most useful — small Q, modest K). The
  arbmask kernel handles Tier-1 head_dim ≤ 128 across `{f32, f16,
  bf16, f64}`; this covers every published spec-decode workload.

## See also

- [Phase 51 memory note](../../) — `project_phase51_complete.md` in
  the project's memory tree.
- [FlashSdpaPlan docstring](../../crates/baracuda-kernels/src/attention/flash_sdpa.rs)
  — full API reference for the mask field.
- [Phase 42 FA2 vendor note](../../crates/baracuda-kernels-sys/vendor/flash-attention/VENDOR.md)
  — the FA2 integration that this phase complements.
