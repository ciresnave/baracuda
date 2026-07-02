# 02 (follow-up) — DAG dedup for fused reductions, cost, and the seam

> Continuation of `02-dag-ir-consumer-counts.md`. The **DAG representation** and
> the **elementwise emitter dedup** landed (off-device tested + on-device
> bit-exact). This brief is the remaining tail: extend the dedup to the
> reduction / RowReduce epilogue, make the contract cost DAG-aware, and update the
> seam. Self-contained for a fresh session.

---

## What landed (branch `feat/kernelgen-dag-ir`)

- **Representation** — `crates/baracuda-kernelgen/src/ir.rs`: `NodeId`, `DagNode`,
  `ExprDag`, `ExprDag::from_expr` (hash-cons interner; `Const` by `to_bits()`),
  per-node `consumers` (edge count, bumped only on new-node creation so shared
  leaves aren't over-counted), `to_expr` round-trip, `DagNode::is_leaf`. 6
  `dag_tests`: diamond (one `Mul`, `consumers == 2`), `x*x` (leaf shared, not
  hoisted), pure-chain unit-consumers + round-trip, `Const(NaN)` interns by bits,
  `Reduced` shared-but-never-merged, depth-8 diamond chain stays **linear** (11
  nodes, not `2^8`).
- **Emitter** — `crates/baracuda-kernelgen/src/backend.rs`: `lower_dag(dag, ctype,
  lo) -> (prelude, root_ref)` — post-order, memoized; hoists a shared **non-leaf**
  (`consumers > 1`) to `<ctype> tmpN = …;`, inlines everything else. `const_lit`
  factored out and shared with `lower_expr` (which stays the inlining primitive).
- **Wiring** — `crates/baracuda-kernelgen/src/cuda.rs`: `emit_scalar`,
  `emit_strided`, `emit_vectorized` route through `lower_dag`; empty prelude ⇒
  byte-identical to the old emitter (all 68 pre-existing goldens green). Two
  emitter tests: `shared_interior_is_hoisted_to_one_tmp`,
  `single_use_body_emits_no_tmp`.
- **On-device** — `ondevice/dag_validate.cu` + two `diamond` catalog cells: scalar
  and per-lane vectorized hoist, bit-exact on RTX 4070 (sm_89).

The internal DAG is transparent for every single-use body; only genuinely shared
interiors change the emitted text.

---

## Deferred

1. **Reduction / RowReduce epilogue dedup** (`cuda.rs` `emit_reduction`,
   `emit_row_reduce`). **Finding (2026-07-02): lower priority than the brief
   implied.** The headline case — Softmax's shared `exp(x - max)` feeding both the
   sum stage and the epilogue — is a **cross-pass** share (the reduction fold and
   the epilogue store loop are separate passes over different elements at different
   times), so it is *not* a `tmp`-hoist `lower_dag` can express; it is inherent to
   the two-pass structure. `lower_dag` only dedups *intra-expression* sharing, and
   **no current fused op** (Softmax / RmsNorm / LayerNorm) has an intra-epilogue
   shared interior — so routing these paths through `lower_dag` is **byte-identical
   today** and benefits zero current ops. Deferred until a fused op with a
   genuinely shared epilogue interior exists (then route the epilogue — a clean
   single-expr site — through `lower_dag`; the `Max`/`Min` fold `elem`, used at
   seed *and* loop, needs per-scope preludes and is the trickier site).
2. **Contract cost DAG-count** (`contract.rs` `count_flops`). **DONE (2026-07-02).**
   `count_flops` now counts distinct non-leaf `ExprDag` nodes, so a shared subtree
   is charged once — matching the emitter, which computes it once. `flops_per_elem`
   for a body with a duplicated subtree **drops** to the honest count, never rises;
   a body with no sharing is unchanged. Test `flops_count_dedups_shared_subtree`.
   (`ulp_bound` intentionally stays tree-based — over-stating the error bound is the
   safe direction; under-stating is not.)
3. **Seam `region_to_op`** (`jit.rs`). Hash-cons the region Fuel sends (dedup
   duplicated subtrees — pure codegen win, recipe still re-describes the sent
   region) and **rewrite the tree-only soundness docstring** (`jit.rs:401-405`).
   Honest-miss rule: an un-annotated cross-region shared interior declines, never
   emits a false `consumers: 1`.
4. **Pattern `consumers`** (`pattern.rs`). Keep `Some(1)` for an AOT non-root
   interior (still externally sole-consumer), but parameterize `walk` to read a
   DAG-derived count where the seam annotation says a value escapes — gated on
   **Fuel ask A** (does the frozen `PatternNode` grammar carry a per-node consumer
   count / escapes-region flag on import?). Until answered, the honest fallback
   in (3) holds; no wire change.

## Fuel ask (unchanged from the parent brief, §10 ask A)

Does `fuel_kernel_seam_types::PatternNode` carry / can it carry a per-node
consumer count or "escapes-region" flag on an **imported** region, and is
`consumers: >1` a matcher-honored annotation? If yes, Baracuda emits truthful
`consumers: N`; if no, the dedup-for-codegen + decline-on-unprovable-sharing
fallback stands. Propose-first via the Baracuda↔Fuel channel.
