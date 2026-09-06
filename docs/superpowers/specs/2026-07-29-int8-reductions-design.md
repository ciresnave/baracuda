# int8/uint8 Reduction Support — Design

**Goal:** Add S8 (int8) / U8 (uint8) support to the Baracuda kernelgen reduction ops — `sum`, `max`, `min`, `prod`, and the predicate/count folds `any`/`all`/`count` — with wrapping two's-complement semantics that are byte-exact to KISS-Ops and differentially verified against kiss-ref.

**Status:** **IMPLEMENTED.** Decisions below were made by Eric on 2026-07-29 (four forks)
against a scoping report. The design shipped: `reduce_sum_int.cu` instantiates
`reduce_sum_u8` / `reduce_sum_i8` (`uint8_t` / `int8_t`) alongside the wider widths, with
`reduce_max_int.cu`, `reduce_min_int.cu`, `reduce_prod_int.cu`, `arg_reduce_int.cu` and the
`reduce_{all,any,count_nonzero}_fp_int_bool.cu` predicate folds covering the rest of the op
set named in the Goal — and `baracuda-cuda-emit/src/cuda.rs` recording that "U8 and S8 are
now COMPUTE dtypes". This document is the design record, not open work.

> ⚠ **DISCHARGED 2026-09-06.** This field read *"Design for review … **No code written
> yet**"* until today. That is the same sentence, in the same position, that cost `fuel`
> sixteen days on GAP-283 — three specs reading *"Design pass — no code yet"* against
> 30,180 lines of live implementation, which later misscoped a ratified cross-project gate.
> Replaced rather than annotated. Nothing below is edited: the four forks and their
> rationale are the record of why the implementation looks the way it does.

---

## 1. The correctness contract (settled by spec — no latitude)

- **Wrapping, not saturating.** KISS-OPS-6.2-0002 (`KISS/spec/ops.md:627`): integer `add`/`sub`/`mul` MUST be wrapping two's-complement modulo `2^bitwidth`, never UB, never saturating.
- **Reduce monoids + identities.** KISS-OPS-6.11-0002 (`ops.md:907`): `sum` (id 0), `prod` (id 1), `max` (id = dtype min: INT_MIN signed / 0 unsigned), `min` (id = dtype max: INT_MAX signed / uint-max); `max`/`min` NaN-propagating (moot for int); **empty axis → identity**.
- **Exact-byte.** Integer `sum`/`prod`/`max`/`min` reductions are class exact-byte (only *float* sum/prod left exact-byte per 6.0-0004). So the differential comparator is **bit-exact**, not ULP.
- **Width invariance (the key lever).** Wrapping `+`/`*` mod `2^b` is a ring homomorphism ⇒ order- and width-invariant. Accumulating an int8 fold in a wide `long long` and truncating once at the store is **byte-identical** to accumulating at 8-bit width. This is why the accumulator-width choice below is a code-reuse decision, not a correctness one.

kiss-ref 0.2.0 already implements this: `tensor_int::reduce` (`kiss-ref-core/src/tensor_int.rs:140`) folds in i128, wraps to width via `eval_int_op`/`wrap`, seeds `int_identity`. Our results agree with it by construction.

## 2. Decisions (the four forks)

| Fork | Decision | Rationale |
|------|----------|-----------|
| **Op scope** | `sum/max/min/prod` **+ `any/all/count`** | Full predicate coverage. Excludes `Mean` (a float-contraction, excluded both sides). |
| **Accumulator width** | **Reuse the existing `long long` accumulator** (as I32/I64) | Store truncates to `signed/unsigned char`. Keeps the coalesced warp-shuffle block-tree (64-bit `__shfl` exists). Byte-exact per §1 width-invariance. |
| **Empty-axis Max/Min** | **Fix to INT_MIN/INT_MAX for int** (seed accumulator with the monoid identity) | Closes a real spec gap — the current `has`-flag path returns 0 on an empty extent, disagreeing with §6.11-0002. Int-only change; FP path untouched. |
| **Validation** | **Add an integer `eval_recipe` path to kiss-ref** (peer) + int leg in `kiss_ref_diff.rs` + on-device `.cu` | Recipe-level int diffs benefit all future int ops, not just this. On-device proof independent of kiss-ref via a CPU wrapping oracle. |

## 3. Architecture / approach

Three units, testable independently:

### 3a. Emitter — admit S8/U8 in the reduction fold
`crates/baracuda-kernelgen/src/cuda.rs`, `emit_reduction` (~2205):
- Extend the `int_acc` predicate (`cuda.rs:2225`) and the dtype assert (`cuda.rs:2226-2237`) to admit `S8 | U8`. Keep the int-`Mean` rejection (`cuda.rs:2240`).
- Accumulator/`zero`/`one` selection (`cuda.rs:2278-2300`): S8/U8 route to the `long long` accumulator (no new accumulator ctype). The store closure (`cuda.rs:2346-2355`), `out_ctype_of`, and `demote_store_f32` already truncate to `signed/unsigned char` correctly — expected no change, verify.
- **Empty-axis identity:** for int Max/Min, seed the accumulator with the per-dtype monoid identity instead of the `has`-flag/return-0 path (`cuda.rs:2500-2501`, 2722-2724). Mirror kiss-ref `int_identity` (`tensor_int.rs:37`).

### 3b. Admissibility — lift the int post-expr gate for `any/all/count`
`crates/baracuda-kernelgen/src/plan.rs`, `assert_int_op_admissibility` (~1628):
- Today the reduction `post` walk (`plan.rs:1774`) rejects `Cmp*`/`Const` at int dtype (`plan.rs:1652-1702`), which blocks int8 `any/all/count` (they are a `Sum/Max/Min` fold + a fused `Cmp*` post + hetero out U8/I64).
- Lift the rejection **only for the reduction post-expression path** so an int8 `Cmp*`→U8 keep-mask / count is admissible, **without** loosening the elementwise int gate (which must keep rejecting `Cmp*`/`Const`/`Div`/float-fns for the compute-body). This is the subtlest part; the split must be surgical and tested both directions (int8 any/all/count admitted; int8 elementwise Cmp still declined).

### 3c. Validation
- **kiss-ref (peer, `3vgwagtz`):** integer `eval_recipe`/`FlatDag` path (i128-backed, wrapping-to-width, `int_identity`-seeded). Coordinated separately; a crates.io publish needs Eric's direct word to the kiss-ref session.
- **`crates/baracuda-kernelgen/src/kiss_ref_diff.rs`:** add an integer leg — dtype-threaded input buffers, a bit-exact int comparator, calling the new int recipe path (or `tensor_int::reduce` directly until it lands). Currently f32-only (`kiss_ref_diff.rs:434-575`).
- **On-device `.cu`:** new validator following `ondevice/reduce_validate.cu` (already has `sum_i32`/`amax_i32` cells) + `reduction_upgrades_validate.cu` (has `exact_i32`/`exact_i64` comparators). Add `exact_i8`/`exact_u8` + a CPU wrapping oracle (mirrors `int_validate.cu`), run on the RTX 4070.

## 4. Out of scope (explicit)

- **Integer `Mean`** — a float-contraction; rejected both in Baracuda and kiss-ref. Stays out.
- **RowReduce** (fused softmax/layernorm) — float-only by construction (`exp`/`div` reject at int). Not a target.
- **int split-K perf variant** — S8/U8 fall to the serial base like I32/I64 (correct, just not accelerated). A "long-long workspace" int split-K is deferred; log the decline, don't silently cap.

## 5. Validation / done criteria

- Bit-exact differential (Baracuda emitted int8 reduce vs kiss-ref int path) for `sum/max/min/prod` over S8 and U8, including a signed-overflow-wrap case and an empty-axis case (identity check).
- int8 `any/all/count` correctness (U8 keep-mask / I64 count) verified on-device.
- Negative controls: int8 elementwise `Cmp*` still declines (the admissibility split didn't over-loosen); int `Mean` still declines.
- On-device run green on the RTX 4070 (sm_89).
- No regression in existing FP / I32 / I64 reduction tests.

## 6. Global constraints

- Determinism is non-negotiable — fixed-order fold, no FP atomics (moot for int, but the reduction stays reproducible run-to-run).
- Do not touch `crates/baracuda-runtime/src/interop.rs` (pre-existing working-tree change, not ours).
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` + `Claude-Session`.
