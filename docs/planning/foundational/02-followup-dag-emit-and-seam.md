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
   `emit_row_reduce`). These still call `lower_expr` on sub-expressions inside
   fold / stage / epilogue contexts. Route each through `lower_dag` so Softmax's
   shared `exp(x - max)` (feeding both the sum stage `pre` and the epilogue) is
   computed once. Care: the `elem` in a `Max`/`Min` fold is used at the seed
   *and* in the loop — its prelude must sit at the right scope; the RowReduce
   `lower` closure lowers several exprs (stages + epilogue), each needing its own
   `tmp` scope. This is where shared interiors first appear in *real* ops, so it
   is the highest-value follow-up. Add the shared-`Reduced` Softmax case to
   `dag_validate.cu` + run compute-sanitizer `synccheck`/`racecheck`/`initcheck`
   on it (the warp-shuffle path must not be perturbed).
2. **Contract cost DAG-count** (`contract.rs` `count_ops` / `count_flops`). Today
   they walk the `PatternNode`/`ScalarExpr` tree and **double-count** a duplicated
   subtree. Count distinct DAG nodes instead so `flops_per_elem` drops (never
   rises) for a shared body. Golden update called out where sharing exists.
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
