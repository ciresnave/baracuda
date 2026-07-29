# oracle.rs → kiss-ref Consolidation (v1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make kiss-ref the single spec-exact CPU correctness reference for the ops it covers, retiring oracle.rs's parallel semantics for `elementwise`/`reduction`/`scan` (proven ≡) and `rowreduce`/`matmul` (prove-then-retire), keeping oracle.rs only for `Window`/`Im2Col`/`RowSort`.

**Architecture:** Publish the 3 kiss-ref crates → add as `baracuda-kernelgen` **dev-dependencies** → port the `tools/kiss-ref-diff` converter in-tree as a `#[cfg(test)]` module → retire each covered op's oracle semantics + self-tests, repointing the two oracle *consumers* (`shape.rs` Task 7, `fuzz.rs` numerical leg) to the kiss-ref path.

**Tech Stack:** Rust 2024, `kiss-ref-core`/`kiss-ops-vocab`/`kiss-classify-vocab` `0.1.0` (crates.io, dev-only), `cargo test -p baracuda-kernelgen`.

## Global Constraints

- **kiss-ref crates are `[dev-dependencies]` only** — the published `baracuda-kernelgen` lib's dependency graph and downstream consumers MUST stay unaffected; `cargo publish` / default CI stay clean.
- **Version pin: exactly `0.1.0`** for all three (`kiss-ref-core`, `kiss-ops-vocab`, `kiss-classify-vocab`) — confirmed by kiss-ref (published from their side; API additive-since-004e1a4; `Error` is `#[non_exhaustive]`). **Do not start any task until kiss-ref confirms `0.1.0` is live on crates.io** (they flag the exact versions).
- **Prove-then-retire:** NEVER delete an op's oracle semantics until a committed in-tree differential proves oracle ≡ kiss-ref for it. Elementwise/reduction/scan are already proven (the `tools/kiss-ref-diff` run); rowreduce/matmul need the differential added first (Tasks 5–6).
- **Keep** `TypedBuffer`, `Fidelity`, `compare`, and the `Window`/`Im2Col`/`RowSort` semantics — still needed.
- **Don't touch** `crates/baracuda-runtime/src/interop.rs` or the untracked `supertool` (pre-existing).
- **Formatting:** `rustfmt --style-edition 2024 <files>` before each commit. **Commit trailer** on every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01TUKx27gRHZfivD4N72euGK
  ```

## File Structure

| File | Change |
|---|---|
| `crates/baracuda-kernelgen/Cargo.toml` | add the 3 kiss-ref crates to `[dev-dependencies]` |
| `crates/baracuda-kernelgen/src/kiss_ref_diff.rs` (create) | `#[cfg(test)]` module: the converter + diff helpers, ported from `tools/kiss-ref-diff/main.rs` |
| `crates/baracuda-kernelgen/src/lib.rs` | `#[cfg(test)] mod kiss_ref_diff;` |
| `crates/baracuda-kernelgen/src/oracle.rs` | delete `eval_elementwise`/`eval_reduction`/`eval_scan` (+ later `eval_row_reduce`/`eval_contraction`), their `evaluate()` arms → `panic!`, and their self-tests |
| `crates/baracuda-kernelgen/src/shape.rs` | repoint the Task-7 differential's retired-op cases to kiss-ref |
| `crates/baracuda-kernelgen/src/fuzz.rs` | repoint the numerical leg to the recipe→converter→kiss-ref path |

---

### Task 0: Wire the kiss-ref dev-dependencies

**GATE:** kiss-ref must have published `0.1.0` (all three crates live on crates.io). Confirm via their peer flag before starting.

**Files:**
- Modify: `crates/baracuda-kernelgen/Cargo.toml`

**Interfaces:**
- Produces: the three crates available under `#[cfg(test)]` in kernelgen.

- [ ] **Step 1: Add the dev-dependencies**

In `crates/baracuda-kernelgen/Cargo.toml`, under `[dev-dependencies]` (create the section if absent), add:

```toml
# kiss-ref: the spec-exact reference impl, the single source of truth this crate's
# numerical tests assert against (oracle.rs consolidation). DEV-only — never in the
# published lib's dependency graph. Pinned; the API is additive-since-0.1.0.
kiss-ref-core = "0.1.0"
kiss-ops-vocab = "0.1.0"
kiss-classify-vocab = "0.1.0"
```

- [ ] **Step 2: Write a smoke test proving they resolve + eval_recipe runs**

Create `crates/baracuda-kernelgen/src/kiss_ref_smoke.rs` (temporary; folded into `kiss_ref_diff.rs` in Task 1):

```rust
//! Temporary: prove the kiss-ref dev-deps resolve and eval_recipe runs in-tree.
#[cfg(test)]
mod tests {
    use kiss_ops_vocab::Op;
    use kiss_ref_core::{eval_recipe, DetClass, FlatDag, Node, Tensor};

    #[test]
    fn kiss_ref_dev_dep_resolves_and_evaluates() {
        // relu(add(in0, in1)) over 3 values.
        let dag = FlatDag::new(
            vec![
                Node::Bind(0),
                Node::Bind(1),
                Node::Apply { op: Op::Add, children: vec![0, 1] },
                Node::Apply { op: Op::Relu, children: vec![2] },
            ],
            vec![3],
        );
        let a = Tensor::from_vec(vec![-1.0f32, 2.0, -3.0], &[3]).unwrap();
        let b = Tensor::from_vec(vec![0.5f32, -0.5, 1.0], &[3]).unwrap();
        let r = eval_recipe(&dag, &[a, b], &[], &[]).unwrap();
        let got: Vec<f32> = r.outputs[0].clone().into_data();
        assert_eq!(got, vec![0.0, 1.5, 0.0]);
        assert!(r.dets.iter().all(|d| matches!(d, DetClass::ExactByte)));
    }
}
```

Register it: add `#[cfg(test)] mod kiss_ref_smoke;` to `crates/baracuda-kernelgen/src/lib.rs`.

- [ ] **Step 3: Verify it resolves and passes**

Run: `cargo test -p baracuda-kernelgen kiss_ref_smoke`
Expected: PASS — the dev-deps download from crates.io, compile, and the test passes.

**If the API differs from this signature** (e.g. `FlatDag::new` arity, `eval_recipe` params), it means `0.1.0` diverged from the harness-tested rev — STOP and reconcile the exact `0.1.0` API with kiss-ref before proceeding (this is the compile-verification the gate exists for).

- [ ] **Step 4: Commit**

```bash
rustfmt --style-edition 2024 crates/baracuda-kernelgen/src/kiss_ref_smoke.rs
git add crates/baracuda-kernelgen/Cargo.toml crates/baracuda-kernelgen/src/kiss_ref_smoke.rs crates/baracuda-kernelgen/src/lib.rs
git commit -m "feat(kernelgen): add kiss-ref 0.1.0 dev-dependencies (oracle consolidation)"
```

---

### Task 1: Port the converter + diff harness in-tree

**Files:**
- Create: `crates/baracuda-kernelgen/src/kiss_ref_diff.rs`
- Delete: `crates/baracuda-kernelgen/src/kiss_ref_smoke.rs`
- Modify: `crates/baracuda-kernelgen/src/lib.rs`

**Interfaces:**
- Consumes: the kiss-ref dev-deps (Task 0); `crate::ir::OpDef`, `crate::recipe::semantics_dag`.
- Produces: `kiss_ref_diff::recipe_to_flatdag(op, rank) -> Result<FlatDag, String>`, `kiss_ref_diff::eval_recipe_for(op, shapes, inputs, params) -> RecipeEval<f32>`, `kiss_ref_diff::assert_conforming_eq(name, reference, candidate)` — the in-tree test API the migration tasks call.

- [ ] **Step 1: Port the converter + helpers**

Create `crates/baracuda-kernelgen/src/kiss_ref_diff.rs` as a `#[cfg(test)]` module. Port the converter (`IndexMap`, `DagBuilder`, `split_args`, `parse_const`, `parse_monoid`, `parse_oob`, `parse_bare_bind`, `parse_axes`, `parse_expr`, `recipe_to_flatdag`) and the diff helpers (`f32_bytes`, `bytes_f32`, `dense_strides`, `assert_bits_eq`, `assert_conforming_eq`, `kiss_ref_leg`) **verbatim from `tools/kiss-ref-diff/main.rs`** (that file is the tested reference), with two adaptations:
1. Wrap the whole file in `#![cfg(test)]`-style gating: the module is declared `#[cfg(test)] mod kiss_ref_diff;` so its contents only build under test.
2. Drop the `main()` and the `device_*` functions (device legs stay in the standalone `tools/kiss-ref-diff` tool — CI has no GPU). Keep only the CPU converter + comparators + `kiss_ref_leg`.

Expose the two entry points the migration tasks use:

```rust
/// Convert an OpDef's emitted recipe to a kiss-ref FlatDag (value lane).
pub(crate) fn recipe_to_flatdag(op: &OpDef, rank: usize) -> Result<FlatDag, String> { /* ported */ }

/// The kiss-ref leg: emitted recipe -> converter -> eval_recipe.
pub(crate) fn eval_recipe_for(
    op: &OpDef, shapes: &[Vec<usize>], inputs: &[Vec<f32>], params: &[f32],
) -> RecipeEval<f32> { /* the kiss_ref_leg body */ }
```

- [ ] **Step 2: Register the module, remove the smoke stub**

In `crates/baracuda-kernelgen/src/lib.rs`, replace `#[cfg(test)] mod kiss_ref_smoke;` with `#[cfg(test)] mod kiss_ref_diff;`. Delete `crates/baracuda-kernelgen/src/kiss_ref_smoke.rs`.

- [ ] **Step 3: Fold the smoke test in as the module's own test**

Inside `kiss_ref_diff.rs`, add a `#[cfg(test)] mod tests` with the relu_add differential (the smoke test from Task 0, plus one that exercises `recipe_to_flatdag` from a real `OpDef` — e.g. `OpDef::elementwise("relu_add", 2, &[F32], (input(0)+input(1)).relu())` → `recipe_to_flatdag` → `eval_recipe_for` → assert values).

- [ ] **Step 4: Verify**

Run: `cargo test -p baracuda-kernelgen kiss_ref_diff`
Expected: PASS — the in-tree converter round-trips a real OpDef through kiss-ref.

- [ ] **Step 5: Commit**

```bash
rustfmt --style-edition 2024 crates/baracuda-kernelgen/src/kiss_ref_diff.rs crates/baracuda-kernelgen/src/lib.rs
git add -A -- crates/baracuda-kernelgen/src/kiss_ref_diff.rs crates/baracuda-kernelgen/src/lib.rs crates/baracuda-kernelgen/src/kiss_ref_smoke.rs
git commit -m "test(kernelgen): port the kiss-ref differential converter in-tree (#[cfg(test)])"
```

---

### Task 2: Retire elementwise oracle semantics

**Files:**
- Modify: `crates/baracuda-kernelgen/src/oracle.rs`, `src/shape.rs`, `src/fuzz.rs`

**Interfaces:**
- Consumes: `kiss_ref_diff::eval_recipe_for`, `assert_conforming_eq` (Task 1).

- [ ] **Step 1: Repoint the `fuzz.rs` numerical leg to the converter path**

The `fuzz.rs` numerical property currently asserts `oracle::evaluate(elementwise) ≈ eval_ref`. Repoint it to fuzz the recipe→converter→kiss-ref path instead: for each random elementwise `op`, build `eval_recipe_for(&op, &shapes, &cols, &[])`, take `outputs[0].into_data()`, and compare against the existing `eval_ref` reference under the same tolerance. This turns an oracle self-consistency check into a **converter+kiss-ref fuzz** (broader coverage — the harness only covered fixed cases). Replace the `let actual = &evaluate(...)` block accordingly; remove the now-unused `oracle::{evaluate, compare}` import from fuzz.rs if elementwise was its only use.

- [ ] **Step 2: Repoint `shape.rs` Task-7 elementwise case to kiss-ref**

In `shape.rs`'s `oracle_differential_agrees_on_every_supported_variant`, the elementwise case uses `oracle::evaluate` to produce the shaped output. Replace it with `kiss_ref_diff::eval_recipe_for` for the elementwise op (the produced tensor's `.shape()` is the shaped output to compare against `output_shape`). Keep the `Window` case on oracle (retained).

- [ ] **Step 3: Delete the oracle elementwise semantics + self-tests**

In `oracle.rs`: delete `fn eval_elementwise` and its helpers used ONLY by elementwise; change the `Access::Elementwise =>` arm of `evaluate()` to `panic!("oracle: elementwise semantics retired 2026-07 — use kiss-ref via kiss_ref_diff")`. Delete the elementwise self-tests: `elementwise_add_contiguous`, `elementwise_relu_neg_zero_and_nan`, `elementwise_maxmin_prop_signed_zero_ties_keep_a`, `elementwise_affine_with_params`, `elementwise_broadcast_input`, `elementwise_flipped_input`, `elementwise_base_offset_slice`, `elementwise_permute_transpose`.

**Note on `elementwise_maxmin_prop_signed_zero_ties_keep_a`:** this pins the max_prop a-on-ties fix. Before deleting it, confirm the equivalent is covered — it is (the `tools/kiss-ref-diff` max_prop differential + KISS #79's tie-vector corpus). If you want belt-and-suspenders, port it as a `kiss_ref_diff` differential instead of deleting outright.

- [ ] **Step 4: Verify**

Run: `cargo test -p baracuda-kernelgen`
Expected: PASS — elementwise coverage now runs through kiss-ref; no oracle elementwise remains; no other test broke.

- [ ] **Step 5: Format + commit**

```bash
rustfmt --style-edition 2024 crates/baracuda-kernelgen/src/oracle.rs crates/baracuda-kernelgen/src/shape.rs crates/baracuda-kernelgen/src/fuzz.rs
git add crates/baracuda-kernelgen/src/oracle.rs crates/baracuda-kernelgen/src/shape.rs crates/baracuda-kernelgen/src/fuzz.rs
git commit -m "refactor(oracle): retire elementwise semantics — kiss-ref is the reference"
```

---

### Task 3: Retire reduction oracle semantics

**Files:** Modify: `crates/baracuda-kernelgen/src/oracle.rs`, `src/shape.rs`

- [ ] **Step 1: Repoint `shape.rs` Task-7 reduction case to kiss-ref** — same pattern as Task 2 Step 2, for the reduction op.
- [ ] **Step 2: Delete oracle reduction semantics + self-tests** — delete `fn eval_reduction` (+ reduction-only helpers); `Access::Reduction =>` arm → `panic!("oracle: reduction semantics retired — use kiss-ref")`. Delete self-tests `reduction_sum_last_axis`, `reduction_max_nan_sticks`, `reduction_mean_divisor`, `reduction_outer_axis` (and any other `reduction_*`).
- [ ] **Step 3: Verify** — `cargo test -p baracuda-kernelgen` PASS.
- [ ] **Step 4: Format + commit** — `refactor(oracle): retire reduction semantics — kiss-ref is the reference`.

---

### Task 4: Retire scan oracle semantics

**Files:** Modify: `crates/baracuda-kernelgen/src/oracle.rs`, `src/shape.rs`

- [ ] **Step 1: Repoint `shape.rs` Task-7 scan case to kiss-ref** — same pattern, for the scan op.
- [ ] **Step 2: Delete oracle scan semantics + self-tests** — delete `fn eval_scan` (+ `scan_identity` if scan-only); `Access::Scan =>` arm → `panic!`. Delete self-tests `scan_cumsum_forward`, `scan_cumsum_exclusive_first_pos_identity`, `scan_cummax_forward_and_reverse_and_exclusive`.
- [ ] **Step 3: Verify** — `cargo test -p baracuda-kernelgen` PASS.
- [ ] **Step 4: Format + commit** — `refactor(oracle): retire scan semantics — kiss-ref is the reference`.

---

### Task 5: Prove-then-retire rowreduce

**Files:** Modify: `crates/baracuda-kernelgen/src/kiss_ref_diff.rs`, `src/oracle.rs`, `src/shape.rs`

- [ ] **Step 1: Add the rowreduce differential (PROVE first)** — in `kiss_ref_diff.rs` tests, add a differential that builds a RowReduce op (softmax: 2 staged reduces + epilogue, per `recipe.rs`'s rowreduce recipe), evaluates it via BOTH `oracle::evaluate` AND `eval_recipe_for`, and asserts they agree bit/tolerance over an exactly-representable value set. This is the evidence gate — it must pass BEFORE Step 2.
- [ ] **Step 2: Verify the proof** — `cargo test -p baracuda-kernelgen kiss_ref_diff` PASS (oracle ≡ kiss-ref for rowreduce, proven in-tree). Commit this proof first: `test(kiss_ref_diff): prove oracle ≡ kiss-ref for rowreduce`.
- [ ] **Step 3: Retire** — repoint `shape.rs` rowreduce case; delete `fn eval_row_reduce` + `Access::RowReduce =>` arm → `panic!`; delete rowreduce self-tests. Verify `cargo test -p baracuda-kernelgen` PASS.
- [ ] **Step 4: Format + commit** — `refactor(oracle): retire rowreduce semantics — proven ≡ kiss-ref`.

---

### Task 6: Prove-then-retire matmul (contraction)

**Files:** Modify: `crates/baracuda-kernelgen/src/kiss_ref_diff.rs`, `src/oracle.rs`, `src/shape.rs`

- [ ] **Step 1: Add the contraction differential (PROVE first)** — in `kiss_ref_diff.rs` tests, build a matmul OpDef (`ContractionAxes::matmul()`), evaluate via BOTH `oracle::evaluate` (`eval_contraction`) AND `eval_recipe_for` (the `matmul[mk.kn]` recipe → kiss-ref `Matmul` node), assert agreement over an exactly-representable cell (e.g. `[2,3]·[3,2]`). Include a batched case (`ContractionAxes::batched_matmul()`). Evidence gate — must pass BEFORE Step 2.
- [ ] **Step 2: Verify the proof** — `cargo test -p baracuda-kernelgen kiss_ref_diff` PASS. Commit: `test(kiss_ref_diff): prove oracle ≡ kiss-ref for matmul`.
- [ ] **Step 3: Retire** — repoint `shape.rs` (note: `shape.rs` Task 7 does NOT cover Contraction, since `oracle::evaluate` was excluded there — verify; if so, no shape.rs change); delete `fn eval_contraction` + `Access::Contraction =>` arm → `panic!`; delete contraction self-tests (`contraction_matmul_identity_epilogue`, `contraction_matmul_relu_epilogue_over_reduced0`, `contraction_matmul_bias_relu_epilogue`, `contraction_batched_matmul_identity_epilogue`). Verify `cargo test -p baracuda-kernelgen` PASS.
- [ ] **Step 4: Format + commit** — `refactor(oracle): retire matmul semantics — proven ≡ kiss-ref`.

---

## Post-implementation

- **What remains in oracle.rs:** `Window`, `Im2Col`, `RowSort` semantics + their self-tests, `TypedBuffer`, `Fidelity`, `compare`. These stay until kiss-ref covers those ops (a later phase — Window/Im2Col ride the KISS #86 `WithDim`/`Dims` activation; RowSort rides kiss-ref's `SortNetwork` becoming recipe-expressible).
- **Report:** the retired ops (`evaluate()` panics on Elementwise/Reduction/Scan/RowReduce/Contraction now — an intentional honest signal, not a bug); the new CI reality (kiss-ref dev-deps mean the migrated numerical tests run in default CI — a gain over the old private-git harness); and that the standalone `tools/kiss-ref-diff` device harness stays for the on-hardware (step 3a/3b) diffs.

## Self-Review notes (addressed)

- **Spec coverage:** every design §4 scope row maps to a task (retire elementwise/reduction/scan = Tasks 2–4; prove-then-retire rowreduce/matmul = Tasks 5–6; keep Window/Im2Col/RowSort = untouched). The mechanism §3 = Tasks 0–1.
- **Un-verifiable-against-0.1.0 risk:** Task 0 Step 3 is the compile-verification gate; if `0.1.0`'s API differs from the harness-tested rev, it surfaces there before any migration.
- **Coverage preservation:** oracle self-tests → deleted (kiss-ref self-tests own semantics correctness); `fuzz.rs` numerical leg → repointed to fuzz the converter+kiss-ref path (a gain); `shape.rs` Task 7 → repointed to kiss-ref. No CI coverage lost.
