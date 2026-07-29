# Design: oracle.rs → kiss-ref consolidation (v2)

**Date:** 2026-07-28 (v1); amended 2026-07-29 (v2, after execution).
**Status:** in execution. Tasks 0–2 landed. v2 reshapes the model after a Task-2 finding.
**Scope owner:** Baracuda (kernelgen), coordinated with kiss-ref (peer `3vgwagtz`).

## Amendment v2 (2026-07-29) — the reshaped model, after execution

Executing Task 2 surfaced that **oracle.rs is NOT fully redundant with kiss-ref**, and Eric
ruled the discipline: **retire-when-replaced — discard nothing from the oracle without an
equal-or-better equivalent already landed in kiss-ref (or another source); phase out the
oracle only as it is replaced.** The boundary was then locked with kiss-ref (peer
`3vgwagtz`, file-cited):

- **kiss-ref = the LOGICAL / value-semantics reference** (KISS-Ops math over dense logical
  tensors): elementwise math, folds, scans, matmul, gather/scatter, broadcast (logical
  stride-0), flip, raw-bit select (`resolve.rs:58`), compute-dtype cmp/select
  (`resolve.rs:67`, monomorphic `eval_op::<T>`), integer-lane WRAP values (`eval_int_op`,
  i128→width; int-lane drain in flight).
- **oracle.rs = the permanent Baracuda PLUMBING reference** for what a value-semantics
  reference structurally cannot model: **physical host layout** (negative strides,
  base_offset slices, axis-permuted consumer buffers), the **elementwise FRAME**
  (output-shape-from-operand; a const/partial-input body broadcasts across the frame —
  kiss-ref's value-DAG derives a const-only body as a *scalar*), and **float→int
  STORE-TRUNCATION** (no KISS-Ops cast op exists, so narrowing-store is Baracuda codegen).

**Consequence:** the consolidation collapses the duplicate **VALUE-semantics** reference
(the real comprehension-correlation risk), NOT the whole oracle. oracle.rs is **not heading
toward deletion** — it keeps a permanent plumbing role. Every retirement is gated: port the
value edge into `kiss_ref_diff` as the proven equal-or-better replacement FIRST, then delete
the redundant oracle self-test; keep the `evaluate` arm + all plumbing tests. This
**supersedes** the "redundant parallel implementation / shrink toward deletion" framing in
§1, §4, and §7 below.

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

## 4. Precise retirement scope (v2 — value-vs-plumbing, retire-when-replaced)

Retirement operates on oracle **SELF-TESTS**, not the `evaluate` arms: each arm stays alive
as long as ANY retained plumbing test needs it. For each op family, its VALUE-semantics
self-tests retire (edges ported into `kiss_ref_diff` first, as the proven equal-or-better
replacement); its PLUMBING self-tests (physical layout, int store-truncation, frame) stay.

| Op family | Value self-tests | Plumbing self-tests (permanent keep) | Gate |
|---|---|---|---|
| Elementwise (math) | **Retired (Task 2)** → kiss_ref_diff: add(+INF)/relu-signed-zero+NaN/affine/signed-zero-add/max_prop-min_prop-ties | raw-bit select, int8 store-trunc, strided/broadcast/flipped/permuted views, compute-dtype cmp/select, frame (fuzz leg) | **done** |
| Elementwise (select, compute-dtype) | **Next pass** (kiss-ref covers today: `resolve.rs:58/67`) | — | kiss-ref's strengthened select bit-identity test lands green |
| Reduction | **Prove-then-retire** value folds → kiss_ref_diff | int / strided reductions, frame | kiss-ref reduce covered; add differential first |
| Scan | **Prove-then-retire** value scans → kiss_ref_diff | int / strided scans, frame | kiss-ref scan covered; add differential first |
| RowReduce | **Prove-then-retire** (softmax/rmsnorm) → kiss_ref_diff | any layout/int edges | add differential first |
| Contraction (matmul) | **Prove-then-retire** → kiss_ref_diff | any layout/int edges | add differential first |
| Window / Im2Col | **Keep whole** | — | kiss-ref doesn't cover (reserved `WithDim`/`Dims`) |
| RowSort | **Keep (already deferred)** | — | kiss-ref/recipe don't cover sort |

**Permanent Baracuda territory (never retired):** physical host layout (negative strides,
base_offset, axis-permutation), the elementwise frame (const/partial-body broadcast),
float→int store-truncation. **Always kept:** `TypedBuffer`, `Fidelity`, `compare`, and every
`Access::*` arm.

**Retire-when-replaced gate (Eric's ruling):** an op's value self-test is deleted ONLY after
its edge is asserted against kiss-ref in `kiss_ref_diff` AND kiss-ref's coverage of that
value has landed + is green. kiss-ref's landing gates Baracuda's deletion.

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
