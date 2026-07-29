# Design: oracle.rs → kiss-ref consolidation (v1)

**Date:** 2026-07-28
**Status:** design approved; execution gated on kiss-ref publishing `0.1.0` to crates.io.
**Scope owner:** Baracuda (kernelgen), coordinated with kiss-ref (peer `3vgwagtz`).

## 1. Motivation

`baracuda-kernelgen/src/oracle.rs` is a 2993-line CPU interpreter that has served as
Baracuda's **spec-exact correctness reference** — the independent second implementation
every generated kernel is validated against in tests. Since kiss-ref (the
project-agnostic KISS reference impl) now exists and is spec-exact, oracle.rs is a
**redundant parallel semantics implementation**: keeping the two in sync is exactly the
"comprehension-correlation illusion" the freeze-gate wants collapsed (two references that
trace to one author's reading of the spec are not truly independent).

This consolidation makes **kiss-ref the single source of truth** for the ops it covers,
retiring oracle.rs's parallel CPU semantics for those ops. It was pre-planned with
kiss-ref and confirmed by them (peer, 2026-07-28).

**oracle.rs is a test/validation asset only** — verified: `oracle::evaluate`/`compare` are
called exclusively from `#[test]`/`cfg(test)` code (fuzz.rs, shape.rs, oracle.rs's own
tests); the generator never uses the oracle at kernel-generation runtime. So this is a
**test-infrastructure change**, not a change to shipping generator behavior.

## 2. What replaces oracle.rs — and what stays independent

- **kiss-ref becomes the CPU correctness reference** for the covered ops. Baracuda's tests
  assert generated behavior against `kiss-ref eval_recipe` (via the converter), not a
  second in-house oracle.
- **Baracuda's independent check is the DEVICE differential** (generated CUDA kernel vs
  kiss-ref) — kiss-ref is CPU-only (`no_std`), so running the emitted kernel on the GPU and
  diffing against kiss-ref stays Baracuda's own job (the `tools/kiss-ref-diff` step-3a/3b
  work). This is the independence that matters post-consolidation.
- **Conformance independence stays at the KISS-Conform §6.5 layer** — kiss-ref is the
  differential *target*, never re-imported as Baracuda's own oracle for the conformance
  claim (confirmed with kiss-ref).

## 3. Mechanism (unblocked — KISS/kiss-ref are now public)

1. **Publish** (kiss-ref's side, deps-first): `kiss-classify-vocab`, `kiss-ops-vocab`
   (independent siblings, either order), then `kiss-ref-core`, at **`0.1.0`**. kiss-ref does
   two API-hygiene fixes first — mark `Error` `#[non_exhaustive]` (zero-change for Baracuda:
   our matches use wildcards) and the `0.0.1 → 0.1.0` bump — plus crates.io metadata prep.
   **The publish requires Eric's direct authorization to the kiss-ref session** (a relay is
   not sufficient — the same discipline kiss-ref holds for all permanent outward actions);
   given.
2. **Dev-dependency:** `baracuda-kernelgen` adds the three crates under
   `[dev-dependencies]`, pinned to `0.1.0`. Because they are **dev**-dependencies, the
   published lib's dependency graph and its downstream consumers are unaffected, and
   `cargo publish` / default CI stay clean — the exact constraint that forced the harness to
   be a separate non-workspace tool, now resolved by the public publish.
3. **Converter moves in-tree:** the `tools/kiss-ref-diff` converter (`ScalarExpr`-emitted
   recipe text → kiss-ref `FlatDag`) + its diff harness helpers (byte codecs, the
   DetClass-driven comparator, the leg builders) become a **`#[cfg(test)]` support module**
   in kernelgen (e.g. `src/kiss_ref_diff.rs`), so migrated tests build kiss-ref DAGs from
   Baracuda ops.
4. **Migrate assertions:** each covered op family's tests, which assert against
   `oracle::evaluate`/`compare`, are rewritten to assert against `kiss-ref eval_recipe` under
   the **DetClass-driven comparator** — bit-exact for `ExactByte`, order-invariant/tolerance
   for `OrderInvariantNondeterministic` (the discipline already proven in the harness).
5. **Retire oracle code:** delete the oracle.rs `eval_*` paths for the migrated ops; their
   `evaluate()` dispatch arms become explicit `panic!("oracle semantics for <op> retired
   <date>; use kiss-ref eval_recipe — see kiss_ref_diff")` so a stray caller gets a clear
   message rather than a silent gap.

## 4. Precise retirement scope (v1)

`oracle::evaluate` implements every `Access` variant **except `RowSort`** (which already
panics as deferred): Elementwise, Reduction, RowReduce, Scan, Window, Im2Col, Contraction.
kiss-ref (via the converter) covers: elementwise, reduce, scan, rowreduce, matmul,
gather/scatter — but **not** Window / Im2Col (the reserved-constructor gap) or RowSort.

| Op | v1 action | Why |
|---|---|---|
| Elementwise | **Retire now** | harness proved oracle ≡ kiss-ref |
| Reduction | **Retire now** | harness proved oracle ≡ kiss-ref |
| Scan | **Retire now** | harness proved oracle ≡ kiss-ref |
| RowReduce | **Prove-then-retire** | oracle + kiss-ref both cover; add a differential first |
| Contraction (matmul) | **Prove-then-retire** | oracle + kiss-ref both cover; add a differential first |
| Window | **Keep in oracle** | kiss-ref doesn't cover (needs reserved `WithDim`) |
| Im2Col | **Keep in oracle** | kiss-ref doesn't cover (needs reserved `Dims`) |
| RowSort | **Keep (already deferred)** | kiss-ref/recipe don't cover sort |

**Always kept:** `TypedBuffer`, `Fidelity`, `compare` — still needed by the retained ops
*and* the device differential.

## 5. The prove-then-retire discipline (the safety gate)

**Never delete an oracle op's semantics until a committed harness differential proves
oracle ≡ kiss-ref for it.** Elementwise/reduction/scan are already proven (the committed
`tools/kiss-ref-diff` run). RowReduce and Contraction require adding their differential
(a softmax/rmsnorm case; a matmul case against oracle's `eval_contraction`, not just a
local naive) and confirming bit/tolerance agreement **before** their oracle code is
deleted. So every retirement is evidence-gated, op-by-op — never a blind swap.

## 6. Migration mechanics (the consumers, all tests)

- **oracle.rs's own test module** — the bulk of the migration: the per-emitter numerical
  correctness tests for elementwise/reduction/scan (and, after proving, rowreduce/matmul)
  move to assert against kiss-ref. Tests for the retained ops (Window/Im2Col) stay on oracle.
- **shape.rs (Task 7 shape differential)** — uses `oracle::evaluate` to produce shaped
  outputs for the shape-oracle differential. For the retired ops it switches to kiss-ref
  (which also produces shaped outputs); for Window it stays on oracle.
- **fuzz.rs** — the cross-backend fuzzer's numerical leg (`evaluate`/`compare`) for
  elementwise moves to kiss-ref; the structural/never-panic legs (no oracle) are unchanged.

## 7. Out of scope for v1 (named, not dropped)

- **Window / Im2Col / RowSort migration** — deferred until kiss-ref covers them (Window/Im2Col
  ride the KISS #86 `WithDim`/`Dims` activation Baracuda's shape oracle already anticipates;
  RowSort rides kiss-ref's `SortNetwork` becoming recipe-expressible). A later phase.
- **Deleting oracle.rs entirely** — not until every op is covered + proven; v1 shrinks it,
  doesn't remove it.
- **Publishing Baracuda's own crates** — unaffected; the kiss-ref deps are dev-only.

## 8. Validation

- Every migrated test suite stays green through the swap (the assertions change target
  oracle→kiss-ref but assert the same generated behavior; equivalence was proven first).
- The prove-then-retire differentials (rowreduce, matmul) are committed to the in-tree
  `kiss_ref_diff` support module and run in the normal `cargo test`.
- Because the three kiss-ref crates are on crates.io as normal dev-deps, the migrated tests
  run in **default CI** (no private-git-auth needed) — unlike the old separate-tool harness.
  This is a real gain: the consolidation's correctness gate becomes part of ordinary CI.

## 9. Convergence tie

This completes the arc the differential harness was built for: the "single source of truth"
becomes literal (kiss-ref is the reference in Baracuda's own tests), the comprehension-
correlation collapses (honest producer), and the freeze-gate independence relocates to the
device differential + the KISS-Conform layer + external implementors — exactly the posture
Eric ruled (E6) and all four projects adopted.
