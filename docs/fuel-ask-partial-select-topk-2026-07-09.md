# Baracuda ask — PARTIAL-SELECT TOPK shipped (streaming tiled-bitonic top-k; the large-vocab fast path, BitIdentical to full-sort-then-slice); still no `top_k` OpKind on your side — recorded, NOT a blocker (2026-07-09)

**No action needed now.** A propose-first heads-up in the landing-doc "radar
item" class: a new measured **variant** of the already-shipped `topk`/`bottomk`
cell exists, it changes NOTHING on the wire (additive entry-point suffix, no
`STRUCTURE_KEY_VERSION` bump), and we are recording where it fits so that a
future `top_k` call path picks the right kernel — before anyone wires it.

## What shipped (Baracuda, PARTIAL-SELECT TOPK increment on `feat/kernel-specialization`)

A new `lower_variants` filter `partial_select_topk_variant` (tag `psel`) on the
SAME `SortLimit::TopK` `Access::RowSort` cell — a **streaming tiled-bitonic
top-k**. It serves the regime the shipped `_bitonic` variant **declines**:
`k_in > 1024` large-vocab logits top-k, where the whole `next_pow2(k_in)`-padded
row does not fit one block, and the rank base is `O(k_in²)`.

- **Algorithm (`min_m`):** `m = next_pow2(k_out)`; a running best-buffer B of `m`
  pairs, streamed against the row in tiles of `m`. Each tile fully bitonic-sorts
  the `2m` window `[B | T]` under the `(key, original-index)` `pair_lt` order and
  keeps the LOWER `m`. `O(k_in·log²k_out)` vs the bitonic's `O(k_in·log²k_in)`.
- **The streaming win:** dynamic smem is a `2m`-pair arena
  (`2·next_pow2(k_out)·(acc_sz+4)` bytes) — bounded on **`k_out`, NOT `k_in`** —
  so there is **no `k_in ≤ 1024` cap**. This is the only fast+correct path at
  `k_in = 50 000`.
- **BitIdentical:** the composition lemma `min_m(min_m(X)∪Y) = min_m(X∪Y)` plus
  never-evict (a top-`k_out` member has global rank `< k_out ≤ m` and zero pads
  rank below it) ⇒ `B_final[0..k_out)` equals the full-sort top-`k_out` prefix
  **bit-for-bit**. `VariantFidelity::BitIdentical` (design-panel PROVEN, 0
  divergences / 200 000 adversarial rows). Values RAW-BIT gathered (NaN payloads /
  −0.0 intact); index = the `i32` original position.

Device-validated on sm_89 across {f32,f64,i32,i64,f16,bf16,f32-strict} ×
{topk,bottomk} × the `k_out` sweep, plus 11 targeted probes and the multi-tile
`k_in ∈ {3000, 4096, 8192}` headline — `_psel` `out_val`/`out_idx` memcmp==0 vs
BOTH the CPU `pair_lt` first-`k_out` AND the device full-sort `Both` base's
first-`k_out` slice, two-launch determinism, all four compute-sanitizers 0 errors.
**Bench headline: `top-10 of 50 000` — psel 8.6 ms vs rank base 886 ms (102.6×),
full-bitonic cannot run.** See `crates/baracuda-kernelgen/ondevice/README.md`,
the PARTIAL-SELECT TOPK sub-section.

## The advert story today (honest miss — AOT-only, unchanged from the topk increment)

- **No contract, no pattern, no JIT region** — ZERO `contract.rs` / `pattern.rs` /
  `jit.rs` change. Still `Access::RowSort` (non-Elementwise) ⇒ `derive_pattern`
  returns `NotElementwise`, `contract()` returns `None`. This is a schedule
  VARIANT of a cell that already withholds; the `psel` kernel rides the same
  honest-miss path as `topk`/`bottomk`.
- **`baracuda-kernels-types` UNTOUCHED** — no key field, **no
  `STRUCTURE_KEY_VERSION` bump**. `_psel` is an additive entry-point suffix
  (`baracuda_gen_{op}_{dtag}_rowsort_{ord}_stable_both_topk_psel`), not a
  structure-key token change. `validate_row_sort` / `Schedule::RowSort` unchanged
  (TopK was already valid). The base and `_bitonic` emitters are **byte-untouched**.

## The one design note worth recording (the "advantage decline" is a runtime/bench decision, not a compile-time gate)

The design panel specified a compile-time decline (offer `psel` only when
`2·next_pow2(k_out) < next_pow2(k_in)`). **That gate is not implementable in the
variant filter, by construction:** the `StructureKey` carries size CLASSES, never
literal extents (the §1 non-negotiable) — neither `k_in` nor `k_out` is
recoverable at variant-selection time (the OUT operand contributes only its
`inner_div` **divisibility bucket**, never its magnitude). This is the SAME
limitation the `_bitonic` variant has with its `k ≤ 1024` precondition, and both
are handled the same way: both are `BitIdentical` **measured variants** shipped
under the ship-top-K policy — offered for every TopK cell they can serve, with the
advantage regime documented in the `launch_note`, and the **bench/dispatch layer —
which sees concrete extents — selects the measured default**. So on YOUR side the
selection story is unchanged from any other measured-variant cell: the dispatch
table records Baracuda's per-shape default; Fuel stays the runtime selector.

## What a future `top_k` call path would take (unchanged from `fuel-ask-topk-2026-07-09.md`)

Nothing new here relative to the topk ask: still needs a `top_k` lazy primitive /
`Op::TopKRoute` (the documented dense-routing gap) with a `[batch, k_out]` shape
rule + two-output `(values, indices)` binding, and the `(n_out, k_in, k_out)`
launch-scalar wiring your Window/pool path already ships. The `psel` variant is
purely a performance upgrade **behind** that same interface: when Fuel grows the
primitive and the shape is large-vocab (`k_in ≫ k_out`, e.g. a 32k/128k-vocab
logits top-k), the dispatch table will already point at `psel` where it wins —
the base/bitonic serve the small-`k_in` regime. When the topk call path is worth
sequencing, reply through the channel and we wire the two-output binding together
— propose-first, per convention.
