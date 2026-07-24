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

Briefs **01–08** and **10** (the ORDER-3 IR ramp) have shipped and their standalone brief files have been
removed — see the **Status** section below. The live briefs are:

| # | Brief | Blocked by | Effort | Prereqs | One-line |
|---|---|---|---|---|---|
| 09 | [f16/bf16 half2 packed-SIMD](09-half2-packed-simd.md) | none | M | coord. 01/03 | `half2`/`nv_bfloat162` packed lowering for f16/bf16 contiguous elementwise (Tier-A packed / Tier-B scalarize, determinism-safe). |
| 11 | [Variant generators: measured-tradeoff backlog](11-variant-generators-backlog.md) | none | — | 07 synergy | Phase-2 charter: the tradeoff transforms (help some cells / hurt others / change bits) on top of the shipped pure-wins generator pass. |
| 12 | [IR expansion roadmap](12-ir-expansion-roadmap.md) | none | — | — | Expand the IR to express the full bespoke-kernel surface (cover, not call); the 13-agent inventory over all 23 kernel dirs. |

## Status

Briefs **01–08** and **10** have **shipped** (the ORDER-3 IR ramp — layout/shape, DAG, reductions,
integer accumulation, RowReduce seam, fused residual-add norm, per-arch dispatch, telemetry, and the
MatMul/contraction node via `Access::Contraction`) and their standalone brief files have been removed.
The remaining live briefs are **09** (f16/bf16 half2 packed-SIMD) and **11**/**12**. `docs/design/
kernel-specialization.md` was corrected (commit 351729b6) and now reflects the shipped ORDER-3 status.
