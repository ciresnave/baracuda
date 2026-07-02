# Foundational roadmap — kernelgen ORDER-3 + the self-prioritizing loop

This directory holds **self-contained implementation briefs** for the first ~10 large, foundational
items on the `baracuda-kernelgen` / kernel-seam roadmap. Each brief is written so a **fresh Claude
Code session with no prior context** can execute it: objective, status/blockers, dependencies,
exact current-code touch points (file:line), the design delta, ordered implementation steps, an
on-device (sm_89 / RTX 4070) validation plan, an adversarial-verify checklist, and a definition of done.

**Origin:** these fell out of a "does Baracuda support FlashDecoding++?" question. Answer was no
(Baracuda has Flash-Decoding/Dao-2023 + FlashInfer paged decode, not FD++'s unified-max softmax /
flat-GEMM / heuristic dataflow), and FD++'s flat-GEMM would itself need the terminal MatMul/contraction
node — i.e. this same frontier. The selection is ordered by the dependency-critical path.

## How to use a brief

1. Pick the lowest-numbered brief whose prerequisites are met (see the graph below).
2. Read it end-to-end. For items flagged **design-open**, ratify the marked design decision first.
3. Implement on a git worktree; **validate on-device before anything lands** (house discipline:
   nvrtc-headerless compile + nvcc numeric on sm_89, compute-sanitizer where shared-mem/cross-thread,
   adversarial-verify pass after the change, lockstep release on ship).

## The plan — four phases

**Phase A — Foundational IR core** (all Baracuda-unblocked; start now)
```
01 layout/shape nodes ──┬──► 03 strided/multi-axis/keepdim reductions
   (keystone)           │
02 DAG w/ consumer cnts ─┘   04 integer accumulation (independent)
   (keystone)
```
**Phase B — Value delivery through the seam** — 05 (Fuel-blocked; design now), 06 (independent).
**Phase C — Self-prioritizing measurement loop** — 07 (harness now), 08 (Baracuda half now, feed Fuel-blocked).
**Phase D — Perf + terminal design** — 09 (anytime), 10 (design spike now; implementation is terminal).

## Two tracks

- **Build track (no external dependency):** 01 → 02 → 03/04 → 06 → 07 → 09, plus the 10 design spike.
- **Unblock track (relay to Fuel, then wire):** the four fused-reduce asks (unblocks 05), Fuel's
  telemetry emission layer (unblocks 08's feed), Fuel closing the §5 seam call-site.

## The briefs

| # | Brief | Blocked by | Effort | Prereqs | One-line |
|---|---|---|---|---|---|
| 01 | [Layout/shape IR nodes](01-layout-shape-ir-nodes.md) | design-open¹ | L | — | Per-operand `View` descriptor (BroadcastTo/Transpose/Permute/Reshape) so a fused op reads *through* a layout change in one pass; the keystone. |
| 02 | [DAG IR with consumer counts](02-dag-ir-consumer-counts.md) | none | L | — | Turn `ScalarExpr` tree → value-numbered DAG (reuse optimize.rs hash-cons) so shared interiors emit once + honest `consumers:>1`. |
| 03 | [Strided/multi-axis/keepdim reductions](03-strided-multiaxis-keepdim-reductions.md) | baracuda-internal | L | 01 | Extend `Access::Reduction` past contiguous-last-axis (adds `.axis`, strided inputs, keepdim broadcast-back, multi-input). |
| 04 | [Integer accumulation for reductions](04-integer-accumulation-reductions.md) | none | M | — | Int-typed accumulator path (i32/i64 Sum/Max/Min in `long long`); unblocks count/argmax-class later. |
| 05 | [RowReduce seam adoption + FKC contract](05-rowreduce-seam-adoption-fkc.md) | fuel | M | — (codegen done) | `region_to_op`→`Access::RowReduce` + FKC RowReduce contract so fused norms are seam-adoptable, not AOT-only. |
| 06 | [Fused residual-add LayerNorm + catalog](06-fused-residual-add-layernorm-catalog.md) | baracuda-internal | M | — | 2nd row-streamed input (residual) added before the norm; collapse Add+LayerNorm/RMSNorm to one launch; broaden AOT catalog. |
| 07 | [Per-arch dispatch table + bench-gate](07-perarch-dispatch-table-bench-gate.md) | none | L | — | §7 vendor-exclusion: per-`(op,structure-key,dtype,arch)` benchmark gate + committed dispatch-table artifact. |
| 08 | [Telemetry variant-selection consumer](08-telemetry-variant-selection.md) | fuel² | L | 07 synergy | Top-K equivalent-form emission + JSONL DispatchRecord/MissRecord ingest → ranked build matrix. |
| 09 | [f16/bf16 half2 packed-SIMD](09-half2-packed-simd.md) | none | M | coord. 01/03 | `half2`/`nv_bfloat162` packed lowering for f16/bf16 contiguous elementwise (Tier-A packed / Tier-B scalarize, determinism-safe). |
| 10 | [MatMul/contraction design spike](10-matmul-contraction-design-spike.md) | design-open | L | 01+02 design | *Design only:* `Access::Contraction` grammar, schedule axes, `ContractionKey`, §7 vendor gate + the exact "01 must / 02 must" requirement list. |

¹ Brief 01 recommends ratifying its representation choice first (a per-operand `View` descriptor over a
`ScalarExpr` node or a new `Access` variant — the blast radius on optimize.rs/pattern.rs/contract.rs differs).
² Brief 08's *feed* is Fuel-blocked; the schema, extractor, and reducer (the Baracuda half) are buildable now.

## Also

- `docs/design/kernel-specialization.md` is **stale** (still lists `ScalarExpr::Param` +
  `AddScalar`/`MulScalar` as not-emittable though they shipped; ORDER-3 reductions/RowReduce have landed).
  A 10-minute correction of the ORDER-3 status section is a foundational-hygiene quick win; several briefs
  note it in their "definition of done."
