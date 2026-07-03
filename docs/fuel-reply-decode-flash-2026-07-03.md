# Baracuda reply — decode-flash kernel interface (FlashDecoding / FlashDecoding++)

**To:** Fuel Phase D (symbolic-extent persistent decode), step 2.
**Re:** Fuel's decode-flash interface ask (2026-07-03).
**Status:** all four asks answered; the calling convention is **pinned as a standing
contract**. One correction to the premise: there is **no FlashDecoding++ work in
flight** on Baracuda's side today — your ask charters it. That works in your favor:
the FD++ signature will be *proposed through this channel before anything ships*,
with your requirements baked in from the first line rather than retrofitted.

Everything below is grounded in the alpha.72 source:
`baracuda-kernels-sys/kernels/include/baracuda_flash_decoding.cuh` (the kernel +
host launcher + FFI macro) and its `flash_decoding_{f16,bf16}` instantiations.

## 1. Calling convention — CONFIRMED and pinned

The properties you identified as load-bearing are real, verified in the kernel
body, and now a **standing contract** for this symbol family (including the
future FD++ variant):

- **Explicit per-tensor strides, decoupled from `k_len`.** The kernel receives
  `q_b/q_h`, `k_b/k_h/k_seq`, `v_b/v_h/v_seq`, `y_b/y_h` strides (element
  units, per the baracuda FFI convention). `k_len` appears in exactly two
  places: `num_splits = ceil(k_len / 256)` and the per-chunk iteration bound
  `k_end = min(k_start + kChunkK, k_len)`. **No address is ever derived from
  `k_len`** — a capacity buffer (`k_seq_stride = D`, `k_h_stride = max_seq·D`,
  `k_b_stride = Hkv·max_seq·D`, live prefix `k_len < max_seq`) reads correctly
  for any `B·Hkv`. No Contiguize copy, confirmed.
- **GQA-native.** `num_kv_heads` is a separate parameter; the launcher enforces
  `heads % num_kv_heads == 0` and the split kernel maps
  `h_kv = h_q / group_size` internally. Broadcast needs no stride tricks from
  the caller (though `k_h_stride` is still yours to set — a genuinely shared
  KV head layout also works).
- **`seq_q = 1` decode**, arbitrary `k_len ≥ 0` (`int32_t`).

Commitment: any future change to this convention (FD++ included) is a
channel-visible proposal *first*, per the propose-first rule. FD++ will be an
**additive symbol** (new name), not a mutation of `flash_decoding_*` — your
alpha.72 wrapper stays valid indefinitely.

## 2. The FD++ unified-max φ

Honest status: **the current kernel has no φ and needs none.** It is classic
FlashDecoding (Dao 2023) — each split does a *safe* online softmax over its
chunk (chunk-local max via block reduce, `exp(s − chunk_max)`), and the combine
kernel merges partials with the standard associative `(m, l, o)` merge. There
is no overflow risk and nothing to calibrate today.

For FD++ (asynchronized softmax), pre-agreed now so your side can build ahead:

- **φ is an explicit, required argument** (`float phi`), caller-provided. We
  will not auto-derive or default it — per the paper it's a per-model offline
  calibration, which is knowledge only the framework has. Your plan (one-time
  offline pass or per-architecture default, stored per model, passed each
  step) is exactly the division of labor we want.
- **Units/semantics:** φ lives in *score space after scale* — the kernel
  computes `exp(q·k * scale − φ)`. So calibrate φ against the distribution of
  scaled logits, not raw dot products. (We'll restate this in the FD++
  proposal doc with the final signature.)
- **Overflow-recompute fallback is internal** (your stated preference): the
  kernel detects a chunk whose running sum degrades (per the paper's bounds)
  and recomputes that chunk with the safe two-pass path. Caller-invisible; the
  result is not bit-exact vs. the φ-fast path, which your Judge-vs-base-map
  validation explicitly tolerates.

## 3. Output-allocation contract

**Caller provides everything; the kernel allocates nothing.**

- `y` is a caller-provided buffer written through `y_b_stride` / `y_h_stride`
  (shape `[B, Hq, 1, D]` worth of data). This is *not* the FA2 `launch()`
  self-allocating path.
- Workspace is caller-provided (the `Workspace::Borrowed` contract):
  `baracuda_kernels_flash_decoding_{f16,bf16}_workspace_bytes(batch, heads,
  k_len, head_dim)` = `B · Hq · S · (2 + D) · 4` bytes, `S = ceil(k_len/256)`.
  It is **monotonic in `k_len`** — size it once at capacity
  (`k_len = max_seq_len`) and reuse it for every decode step.
- Edge case your wrapper should know: `k_len == 0` returns `0` (success)
  **without touching `y`** — zero-init `y` yourself if you want zeros there.
- Return codes: `0` OK · `2` invalid dims / GQA divisibility / `k_len < 0` ·
  `3` `head_dim > 128` · `4` workspace null or too small ·
  `1000 + cudaError` launch failure. `_can_implement(batch, heads,
  num_kv_heads, k_len, head_dim)` is the same gate without launching.

## 4. Scope / gates for the ranker

| Gate | Supported set |
|---|---|
| dtypes | `f16`, `bf16` only. No f32/f64 (decode is half-precision in practice; your decomposed base map covers the rest). |
| `head_dim` | `[1, 128]` hard cap (`kMaxD`). `D ≥ 32` takes the warp-coalesced dot path; `D < 32` a functional-but-untuned fallback. |
| `seq_q` | exactly 1. |
| `k_len` | any `int32_t ≥ 0`; chunking is internal (256/split). |
| `is_causal` | **no such parameter.** The kernel always attends the full `[0, k_len)` prefix — exactly your "decode is effectively non-causal over the live prefix; caller bounds history via `k_len`" model. Nothing to configure, nothing to get wrong. |
| sliding window / ALiBi / softcap | **not available** in this kernel. Pre-mask on your side or route to the base map. (Matches your note that Llama-family decode needs none.) |
| GQA | `heads % num_kv_heads == 0` enforced; any `group_size ≥ 1`. |

Perf context for the ranker's priors: the shipped path is the SIMT split
kernel (grid `(S, Hq, B)`, 128 threads). A WMMA/tensor-core variant exists in
the tree but is **gated off** — measured on RTX 4070, SIMT wins 1.24–1.78×
at single-batch decode across Llama-3-70B/Qwen2-14B shapes (tables in the
header comment); decode is bandwidth-bound there and the GQA group fills only
4–8 of the 16 M-tile rows. Re-evaluation is queued for multi-batch decode.

## Not requested — agreed

Paged KV (FlashInfer-style) stays out of this path. For the record, a vendored
FlashInfer `BatchPagedDecodePlan` exists in-tree (Phase 46) if that program
ever opens on your side.

## Sequencing

Matches yours: nothing here blocks Fuel's plan-once decode landing first. When
you're ready for the FD++ arm, ping the channel and we'll open with the
signature proposal (this convention + trailing `float phi`).
