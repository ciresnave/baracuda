# int8/uint8 Reduction Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add S8 (int8) / U8 (uint8) support to the kernelgen reduction ops — `sum`/`max`/`min`/`prod` plus the predicate folds `any`/`all`/`count` — with wrapping two's-complement semantics byte-exact to KISS-Ops and differentially verified against kiss-ref.

**Architecture:** The reduction emitter already accumulates I32/I64 in `long long` and truncates at a `signed char*`/`unsigned char*` store; a single dtype assert declines S8/U8. Extend that assert + accumulator selection (Eric's decision: reuse `long long`), fix the empty-axis Max/Min identity for int, then surgically lift the int rejection of `Cmp*`/`Const` for the reduction-predicate position only (for any/all/count). Validate with an integer leg in `kiss_ref_diff.rs` (against kiss-ref's `eval_recipe_int`) plus an on-device `.cu` vs a CPU wrapping oracle on the RTX 4070.

**Tech Stack:** Rust (baracuda-kernelgen), CUDA (nvcc, sm_89), kiss-ref-core (crates.io, integer lane), the on-device `ondevice/*.cu` validators.

**Source spec:** `docs/superpowers/specs/2026-07-29-int8-reductions-design.md` (approved 2026-07-29).

## Global Constraints

- **Wrapping, not saturating** — KISS-OPS-6.2-0002. Never saturate, never UB on overflow.
- **Reduce identities** — KISS-OPS-6.11-0002: sum 0, prod 1, max = dtype-min (INT_MIN signed / 0 unsigned), min = dtype-max (INT_MAX signed / uint-max unsigned); empty axis → identity.
- **Exact-byte** — int sum/prod/max/min reductions are class exact-byte; the differential comparator is **bit-exact**, not ULP.
- **Determinism** — fixed-order fold, no FP atomics (moot for int; the reduction stays reproducible run-to-run).
- **Accumulator = `long long`** for S8/U8 (reuse the I32/I64 path); the store truncates. Byte-identical to accumulate-at-width by the wrapping homomorphism.
- **Do not touch** `crates/baracuda-runtime/src/interop.rs` (pre-existing working-tree change).
- **Out of scope:** integer `Mean` (float-contraction, keep rejected); `RowReduce` (float-only); int split-K perf variant (S8/U8 fall to the serial base — `log()` the decline, don't silently cap).
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` + `Claude-Session`.
- **Toolchain note:** all on-device steps require the repaired VS toolchain (14.51 update in flight). Do the Rust/CUDA source + the reading-only steps first; run the nvcc/on-device steps once the toolchain settles.

## File Structure

- `crates/baracuda-kernelgen/src/cuda.rs` — reduction emitter: admit S8/U8 in `int_acc` + the dtype assert (`emit_reduction`, ~2205); empty-axis Max/Min identity for int (~2500, ~2722).
- `crates/baracuda-kernelgen/src/plan.rs` — `assert_int_op_admissibility` (~1628): lift the `Cmp*`/`Const` rejection for the reduction-predicate position only.
- `crates/baracuda-kernelgen/src/kiss_ref_diff.rs` — add the integer differential leg (i128-backed, bit-exact comparator).
- `crates/baracuda-kernelgen/ondevice/reduction_int8_validate.cu` (new) — S8/U8 reduce cells + `exact_i8`/`exact_u8` + CPU wrapping oracle.
- `crates/baracuda-kernelgen/src/bin/kernelgen.rs` (catalog) — register the S8/U8 reduce cells the `.cu` validator consumes.

---

### Task 1: S8/U8 for sum/max/min/prod (emitter core)

**Files:**
- Modify: `crates/baracuda-kernelgen/src/cuda.rs:2225` (int_acc) and `:2226-2237` (assert message)
- Verify (no change expected): `promote_load_f32` (`cuda.rs:6411`), `out_ctype_of` (`cuda.rs:760`), the store closure (`cuda.rs:2346`)
- Test: `crates/baracuda-kernelgen/ondevice/reduction_int8_validate.cu` (new, Task 5 wires it)

**Interfaces:**
- Produces: S8/U8 reductions emit a kernel that accumulates in `long long` and truncates at the `signed char*`/`unsigned char*` store — byte-identical to kiss-ref's i128-wrap-to-width.

- [ ] **Step 1: Confirm the leaf load is native for S8/U8.** Read `promote_load_f32` (`cuda.rs:6411-6416`). Expected: S8/U8 hit the native `_` arm (`in{i}[idx]`), not the f16/bf16 widening. If S8/U8 are NOT handled, add them to the native arm. (No test yet — this is a read/confirm gate for Step 2.)

- [ ] **Step 2: Extend `int_acc` and the assert.** In `cuda.rs`:
```rust
let int_acc = matches!(
    plan.dtype,
    ElementKind::I32 | ElementKind::I64 | ElementKind::S8 | ElementKind::U8
);
assert!(
    matches!(
        plan.dtype,
        ElementKind::F16 | ElementKind::Bf16 | ElementKind::F32
            | ElementKind::F32Strict | ElementKind::F64
    ) || int_acc,
    "reduction: float or integer (i8/u8/i32/i64) dtypes only; got {:?}",
    plan.dtype
);
```
The `acc`/`zero`/`one` selection (`cuda.rs:2278-2300`) already keys off `int_acc` → S8/U8 get `long long`/`0`/`1` with no further change. The int-Mean guard (`:2240`) already covers the widened `int_acc`.

- [ ] **Step 3: Trace one S8 sum by hand.** Confirm the emitted kernel text for an S8 `Sum` reduction: `long long` accumulator, `out[oo] = (signed char)acc;`-equivalent via `octype`. No compile yet (toolchain); this is a source-level read of the generated string in a unit test if one exists, else a manual trace recorded in the commit message.

- [ ] **Step 4: Commit.**
```bash
git add crates/baracuda-kernelgen/src/cuda.rs
git commit -m "feat(reduce): admit S8/U8 in the reduction emitter (long-long accumulate, wrap store)"
```

### Task 2: Empty-axis Max/Min identity for int

**Files:**
- Modify: `crates/baracuda-kernelgen/src/cuda.rs` empty-extent Max/Min path (~2500-2501, ~2722-2724, the `has`-flag branch)
- Test: `reduction_int8_validate.cu` empty-axis case (Task 5)

**Interfaces:**
- Consumes: the `int_acc` gate from Task 1.
- Produces: an int Max over an empty reduced extent yields INT_MIN (signed) / 0 (unsigned); Min yields INT_MAX / uint-max — matching KISS-OPS-6.11-0002 and kiss-ref `int_identity`.

- [ ] **Step 1: Read the empty-extent Max/Min path.** Locate the `has`-flag logic (`cuda.rs:2500-2501`, `:2722-2724`) that returns 0 when the reduced extent is empty. Determine where to seed the monoid identity for int Max/Min.

- [ ] **Step 2: Seed the int identity.** For `int_acc && matches!(rop, ReduceOp::Max | ReduceOp::Min)`, initialize the accumulator to the per-dtype identity instead of relying on the `has` flag / 0:
  - Max: S8 → `-128`, U8 → `0`, I32 → `INT_MIN`, I64 → `LLONG_MIN`.
  - Min: S8 → `127`, U8 → `255`, I32 → `INT_MAX`, I64 → `LLONG_MAX`.
  Emit these as literals in the accumulator init. Leave the FP path unchanged (int-only branch).

- [ ] **Step 3: Verify Sum/Prod unaffected.** Confirm Sum still seeds 0, Prod seeds 1 (already correct); the change is Max/Min-only.

- [ ] **Step 4: Commit.**
```bash
git add crates/baracuda-kernelgen/src/cuda.rs
git commit -m "fix(reduce): seed int Max/Min empty-axis with the monoid identity (KISS 6.11-0002)"
```

### Task 3: int8 any/all/count admissibility lift

**Files:**
- Modify: `crates/baracuda-kernelgen/src/plan.rs:1628-1812` (`assert_int_op_admissibility`)
- Test: negative-control unit tests in `plan.rs` (or the kernelgen test module) + on-device any/all/count (Task 5)

**Interfaces:**
- Consumes: Task 1 (S8/U8 folds).
- Produces: int8 `any`/`all`/`count` (Sum/Max/Min fold + fused `Cmp*` predicate → U8 keep-mask / I64 count) is admitted; int8 **elementwise** `Cmp*`/`Const` still declines.

- [ ] **Step 1: Read the any/all/count lowering.** Determine exactly where the `Cmp*` and any `Const` appear for these folds — in the reduction body vs the `post` expr (`plan.rs:1774` walks the reduction `post`). Read the `Cmp` and `Select` arms of `walk` (below `plan.rs:1727`) to see their current int rejection.

- [ ] **Step 2: Write the negative-control test FIRST (must keep passing).**
```rust
#[test]
fn int_elementwise_cmp_still_declines() {
    // An elementwise int Cmp op must still be rejected (the lift must not loosen this).
    let op = /* build an Elementwise op with a Cmp* body at U8 */;
    let r = std::panic::catch_unwind(|| assert_int_op_admissibility(&op, ElementKind::U8));
    assert!(r.is_err(), "elementwise int Cmp must still decline after the reduction-post lift");
}
```

- [ ] **Step 3: Write the positive test (currently failing).**
```rust
#[test]
fn int_reduction_predicate_cmp_admitted() {
    // count = reduce-Sum over a Cmp* predicate on an S8 input, I64 out → must be admitted.
    let op = /* build the any/all/count reduction op at S8 input / hetero out */;
    assert_int_op_admissibility(&op, ElementKind::S8); // must not panic
}
```

- [ ] **Step 4: Lift the rejection, gated on the reduction-predicate position.** In `walk`, thread the position (reduction-post vs body vs elementwise). Allow `Cmp*` and the count/keep-mask `Const` **only** when the walk is on a reduction predicate/post at int dtype; keep the elementwise `Const`/`Cmp*` rejection. The `elementwise` bool already distinguishes elementwise from reduction; extend it (e.g. an `in_reduction_post: bool`) so the predicate `Cmp*`/`Const` are admitted without touching the elementwise arm. Keep the double-math hazard closed: the admitted int `Cmp*`/`Const` must lower to integer comparison / a 0/1 integer literal, never an f64 literal.

- [ ] **Step 5: Run both tests (host-only, no toolchain).**
```bash
cargo test -p baracuda-kernelgen int_elementwise_cmp_still_declines int_reduction_predicate_cmp_admitted
```
Expected: both pass.

- [ ] **Step 6: Commit.**
```bash
git add crates/baracuda-kernelgen/src/plan.rs
git commit -m "feat(reduce): admit int Cmp*/Const in the reduction predicate for i8/u8 any/all/count"
```

### Task 4: kiss_ref_diff integer differential leg

**Files:**
- Modify: `crates/baracuda-kernelgen/src/kiss_ref_diff.rs` (f32-only today, `:434-575`)
- Depends on: kiss-ref `eval_recipe_int` (peer `3vgwagtz`, additive 0.2.x minor; the "0.2" caret auto-pulls it). Interim fallback: call `kiss_ref_core::tensor_int::reduce` directly.

**Interfaces:**
- Consumes: emitted S8/U8 reduce `semantics_dag` text.
- Produces: `oracle_and_kiss_ref_int` returning `(Vec<i128>, Vec<i128>)` + `assert_int_bits_eq`.

- [ ] **Step 1: Add an integer eval entry point** mirroring `eval_recipe_for` but over `i128`, dtype-threaded (S8/U8), calling `eval_recipe_int` (or `tensor_int::reduce` until it lands).

- [ ] **Step 2: Add a bit-exact int comparator** `assert_int_bits_eq(a: &[i128], b: &[i128], dtype, label)` (int reductions are exact-byte).

- [ ] **Step 3: Write differential tests** for S8/U8 `sum`/`max`/`min`/`prod`, including a **signed-overflow-wrap** input and an **empty-axis** (identity) case. Run host-only where kiss-ref covers it; gate the emitted-kernel half behind the on-device feature.
```bash
cargo test -p baracuda-kernelgen kiss_ref_diff_int
```

- [ ] **Step 4: Commit.**
```bash
git add crates/baracuda-kernelgen/src/kiss_ref_diff.rs
git commit -m "test(reduce): differential S8/U8 reductions vs kiss-ref (bit-exact, wrap + empty-axis)"
```

### Task 5: on-device validators + catalog cells (RTX 4070)

**Files:**
- Create: `crates/baracuda-kernelgen/ondevice/reduction_int8_validate.cu` (model: `ondevice/reduce_validate.cu` — has `sum_i32`/`amax_i32`; `reduction_upgrades_validate.cu` — has `exact_i32`/`exact_i64`)
- Modify: `crates/baracuda-kernelgen/src/bin/kernelgen.rs` — register S8/U8 reduce cells (`sum_i8`, `sum_u8`, `amax_i8`, `amin_i8`, `prod_i8`, `any_i8`, `count_i8`, …)

**Interfaces:**
- Consumes: the S8/U8 reduce kernels emitted by Tasks 1-3.
- Produces: on-device pass/fail vs a CPU wrapping oracle (no bespoke int8 reduce sibling exists → CPU is the oracle, mirroring `int_validate.cu:14`).

- [ ] **Step 1: Add `exact_i8`/`exact_u8` comparators + a CPU wrapping-reduce oracle** (`wrapping_add`/`_mul`, `int_identity` init) in the new `.cu`.

- [ ] **Step 2: Register the S8/U8 reduce cells** in the catalog so `kernelgen` emits `baracuda_gen_{sum,amax,amin,prod}_{i8,u8}_reduce_*.cu`.

- [ ] **Step 3: Build + run on the 4070** (needs the repaired toolchain):
```bash
# once VS 14.51 is repaired:
cargo test -p baracuda-kernelgen --features <ondevice-feature> reduction_int8 -- --ignored --nocapture
```
Expected: S8/U8 sum/max/min/prod bit-exact vs the CPU oracle; any/all/count correct (U8 keep-mask / I64 count); empty-axis identity; div=0.

- [ ] **Step 4: Negative controls on-device:** int `Mean` still declines; int elementwise `Cmp*` still declines (belt-and-suspenders vs the host tests).

- [ ] **Step 5: Commit.**
```bash
git add crates/baracuda-kernelgen/ondevice/reduction_int8_validate.cu crates/baracuda-kernelgen/src/bin/kernelgen.rs
git commit -m "test(reduce): on-device S8/U8 reduction validators vs CPU wrapping oracle (RTX 4070)"
```

## Done criteria

- S8/U8 `sum`/`max`/`min`/`prod` emit and are **bit-exact** vs kiss-ref (differential) and vs the CPU wrapping oracle (on-device), including overflow-wrap + empty-axis identity.
- int8 `any`/`all`/`count` correct on-device (U8 keep-mask / I64 count).
- Negative controls hold: int `Mean` declines; int **elementwise** `Cmp*` declines (the admissibility lift didn't over-loosen).
- No regression in existing FP / I32 / I64 reduction tests.
- On-device run green on the RTX 4070 (sm_89).
