# Baracuda ask — FUSED ARGSORT shipped (`(values, indices)` in one kernel; AOT-only, ~2× vs the two-kernel path); what a Fuel *call-path* would take, on radar (2026-07-09)

**No action needed now.** A propose-first heads-up in the alpha.76 landing-doc
"radar item" class: a new kernel capability exists that is deliberately NOT
advertised, and we record exactly what a Fuel-side surface to *call* it would
take — before anyone wires it. Nothing here blocks you; nothing is wired
speculatively.

## What shipped (Baracuda, increment 9 on `feat/kernel-specialization`)

`OpDef::row_sort_indices` — the **fused two-output row sort**: ONE kernel writes
BOTH the value permutation (`out_val`, dtype-preserving raw bits) AND the `I32`
index permutation (`out_idx`) in a single launch, recovering bespoke's native
one-kernel shape (`sort_block_kernel` writes `y_vals` AND `y_idx` together,
`baracuda_sort.cuh:29-31`). It completes increment 8: the pair-sort already
computes the full permutation and holds BOTH quantities in-register at each store
site, so this is a **representation lift** (a store-site addition + a signature),
not a new-capability one.

Representation: `Access::RowSort.argsort: bool` was promoted to a three-state
`enum SortOut { Values, Indices, Both }`. `Values` = today's `row_sort`,
`Indices` = today's `row_argsort` — **both emit byte-for-byte as before** (the #8
goldens stay green unchanged); `Both` is the only new state. The second buffer is
owned locally by the `SortOut` state + a **3-operand key** `[in0, out_val,
out_idx]` — NOT the hetero-multi-output (`extra_out_bodies`) rail, because a
permutation is not a `ScalarExpr` body. `n_outputs()` stays body-derived `= 1`, so
the elementwise-multi dispatch never fires for RowSort.

Device-validated on sm_89 (CUDA 13.3): the fused `(out_val, out_idx)` is
**dual whole-buffer `memcmp`-equal** to the two shipped #8 kernels (`row_sort` +
`row_argsort`) across `{f32,f64,i32,i64,f16,bf16,f32-strict} × {asc,desc} ×
{base,bitonic}`, probe-seeded (NaN payloads, ±0.0, ±inf), base ≡ bitonic on both
buffers, run-to-run deterministic, all four `compute-sanitizer` tools 0 errors
(initcheck load-bearing on BOTH buffers). Bench: **1.74× (base) / 1.99× (bitonic)**
faster than launching the two decomposed #8 kernels (it sorts once, writes twice)
— see `crates/baracuda-kernelgen/ondevice/README.md`, the `sort_validate.cu`
increment-9 sub-section.

## The advert story today (honest miss, AOT-only — unchanged from #8)

`Both` is still `Access::RowSort` (non-Elementwise), so `derive_pattern` returns
`PatternError::NotElementwise` before any body walk and `contract()` returns
`None` — the SAME withhold path that already withholds `row_sort`/`row_argsort`.
**ZERO `contract.rs`/`pattern.rs`/`jit.rs` change; `baracuda-kernels-types`
UNTOUCHED, no `STRUCTURE_KEY_VERSION` bump** — the `_both` suffix rides the
entry-point symbol (`baracuda_gen_{op}_{dt}_rowsort_{ord}_stable_both[_bitonic]`),
not the structure-key token. Pinned by
`cuda::sort_tests::sort_both_is_an_honest_miss_no_contract`.

This AOT-only posture is **correct**, not a limitation — Fuel today has no surface
that could call a two-output sort:

- Fuel has no first-class Sort/ArgSort `OpKind` (sort is an eager `CustomOp1`,
  `fuel-core/src/sort.rs:186`), so there is no sort `OpTag` for `synth_op` to
  synthesize — an honest miss by absence, the same wall #8 already sits behind.
- `Tensor::sort_last_dim` **decomposes** into `arg_sort` then `gather`
  (`sort.rs:379-388`) — the fused kernel's whole point (one sort feeds both
  outputs) is exactly what that decomposition throws away.
- Fuel has no multi-output region infra (it splits every logically-multi-output op
  into N single-output `OpKind`s — the FlashAttnBackwardK/V precedent), so a fused
  two-output node has no advertisable FKC shape.

## What a Fuel *call-path* would take (the ask, when you want it)

To make Fuel *call* the fused kernel later (instead of AOT-only), the Fuel side
would need, in order:

1. **A two-output sort node** — either a first-class `Sort`/`ArgSort` `OpKind` that
   binds TWO output slots `(values, indices)`, **or** a `return.bundle`-style
   `FusedOp` that binds both slots to one region. Today's per-output split (the
   FlashAttn K/V precedent) is the opposite shape; a genuine colocated dual output
   is the prerequisite.
2. **Stop decomposing `sort_last_dim` into `arg_sort` + `gather`** — as long as the
   front-end lowers a fused sort into two separate ops, there is no single node for
   an advert to bind to, and the two-full-sorts cost the fusion removes is
   reintroduced at the graph level.

Neither is needed for the v1 kernel: the AOT kernel runs today, and Baracuda's
bespoke layer already colocates both outputs (`baracuda_sort.cuh:29-31`), so the
**parity is Baracuda-internal** — this note recovers bespoke's one-kernel shape in
kernelgen, no Fuel change required. When the fused-sort *call* path becomes worth
it on your side, reply through the channel and we sequence the two-output node +
the no-decompose change as its own increment (with a `return.bundle`/FusedOp
binding validated together first, mirroring the propose-first convention).
