# 04 — Integer accumulation for reductions — implementation brief

> Scope owner note: read the cited `file:line` anchors before touching anything — line numbers are from
> `feat/kernel-specialization` @ `f2c8d2c` (alpha.72) and will drift. Everything below is grounded in the
> actual code as of 2026-06-30.

## 1. Objective

Add an **integer-typed accumulator path** to the last-axis reduction emitter so `SumDim`/`CountNonzero`-class
reductions over `i32`/`i64` inputs fold in an integer accumulator (exact, no float round-off, no `2²⁴`
mantissa ceiling), instead of being rejected. Today `emit_reduction` hard-asserts float-only
(`cuda.rs:277-288`) and hard-codes the accumulator to `double`/`float` (`cuda.rs:302-303`); the elementwise
integer path already works (infix `+ - * /` over `i32`/`i64` — see the `integer_unary_binary_is_honest_miss_not_panic`
precedent at `jit.rs:1124-1135`), so integer reductions are the missing symmetric case. This is foundational
because it is the **prerequisite substrate for index/argmax-class and count-class reductions**: an exact
integer accumulator is the thing `ArgMax`/`ArgMin` (carry an index alongside the running extremum) and
`CountNonzero` (`i64` accumulator, per `OP-MATRIX.md:92`) are built on. It is a small, self-contained, AOT-only
change with no external blocker.

## 2. Status & blockers

- **Baracuda-unblocked. No Fuel dependency, no design-open blocker.** The entire change lives inside
  `baracuda-kernelgen` (`cuda.rs` emitter + IR docs + `bin/kernelgen.rs` catalog + tests).
- **Not exposed across the JIT trust boundary** — so no FKC/contract negotiation with Fuel is required. The
  §5 seam (`jit.rs` `region_to_op`) is elementwise-only: it hard-codes `Access::Elementwise` and `derive_pattern`
  rejects non-elementwise regions. Reductions are **AOT-authored only** (`jit.rs:10-16` + the `emit_reduction`
  header comment at `cuda.rs:260-264` — "reductions are not in the JIT vocabulary, so these never fire across
  the `synthesize` trust boundary"). Integer reductions inherit that: they are AOT catalog entries, never
  synthesized on demand. This is what makes the change low-risk.
- **One semantic decision is design-open but Baracuda-internal** (not a Fuel ask): whether `ReduceOp::Mean`
  is *legal* on an integer dtype, and if so whether it truncates (C integer division, matching
  `torch.mean` refusing int) or is simply **rejected** for int. Recommended default: **reject `Mean` for int
  at build time** (mirror PyTorch, which raises on `mean()` of an integer tensor) and support only `Sum`/`Max`/`Min`
  for the integer accumulator in this increment. Decided inside this brief (section 5); no cross-repo answer needed.

## 3. Dependencies & sequencing

**Must land before this: nothing.** Item 04 is independent per the initiative dependency graph — it depends
neither on **01 (layout/shape nodes)** nor **02 (DAG-with-consumer-counts)**. It touches only the existing
contiguous last-axis `Access::Reduction` path, whose shape facts (`n_out`, `k`) are already runtime kernel
args.

**What this enables downstream:**
- **Index/argmax-class reductions** (the natural follow-on the task memo names): an exact integer accumulator
  is the substrate for carrying an `i64`/`i32` index alongside a running extremum (`ArgMax`/`ArgMin`, cf. the
  bespoke `ArgReducePlan` at `OP-MATRIX.md:90`) and for `CountNonzero`'s `i64` accumulator (`OP-MATRIX.md:92`).
  Those are separate briefs; 04 just unblocks the accumulator typing they need.
- **03 (strided/multi-axis/keepdim reductions)**: independent of 04, but the two share the `emit_reduction`
  body. Sequence 04 **before or alongside** 03 so the accumulator-typing refactor (a new `AccKind`, section 5)
  lands once and 03 extends it rather than colliding. If 03 lands first, 04 rebases onto its axis machinery
  trivially (04 adds no new axis concept).
- **09 (f16/bf16 half2 packed-SIMD)**: orthogonal — 09 refines the *float* load/store path; 04 adds an
  *integer* path. No overlap, but both edit `emit_reduction`'s leaf-load/typing lines, so whichever lands
  second rebases the `load`/`acc` closures.

## 4. Current code — what exists today

### 4.1 The float-only assert and hard-coded accumulator (`cuda.rs`)

`emit_reduction` (`cuda.rs:265`) opens with two asserts and a hard-coded accumulator type:

```rust
// cuda.rs:277-288 — the float-only gate this brief removes/loosens
assert!(
    matches!(
        plan.dtype,
        ElementKind::F16 | ElementKind::Bf16 | ElementKind::F32
            | ElementKind::F32Strict | ElementKind::F64
    ),
    "reduction v1: float dtypes only (int needs integer-typed accumulation — follow-up); got {:?}",
    plan.dtype
);
// ...
// cuda.rs:302-303 — accumulator is double (f64/f32-strict) or float, never integer
let dbl = matches!(plan.dtype, ElementKind::F64 | ElementKind::F32Strict);
let acc = if dbl { "double" } else { "float" };
let zero = if dbl { "0.0" } else { "0.0f" };
```

The per-element `load` closure (`cuda.rs:305-310`) up-converts f16/bf16/f32-strict and loads f32/f64 natively;
for `I32`/`I64` there is no arm (the assert fires before this). The body is lowered (`cuda.rs:311-325`) with
`unary_f64`/`unary_f32` + `binary_f64`/`binary_f32` selected by `dbl` — **both are float-only** (they emit
`sqrtf`, `expf`, `powf`, float literals; see `cuda.rs:639-743`). The `Mean` finalize divides by `(acc)k`
(`cuda.rs:364-366`) and f16/bf16 store down-converts (`cuda.rs:369-373`).

### 4.2 The C-type + tag mapping already supports int (`cuda.rs`)

`scalar_ctype` **already** maps `I32 => "int"` and `I64 => "long long"` (`cuda.rs:60-61`), and `dtype_tag`
already returns `"i32"`/`"i64"` (`cuda.rs:96-97`). So `ctype` (the kernel-signature element type) and the
generated symbol name are already correct for integer inputs — the only gap is the *accumulator* type and the
*body lowering* inside `emit_reduction`.

### 4.3 The expression lowering seam (`backend.rs`)

`lower_expr` (`backend.rs:67-96`) is dtype-agnostic for the infix nodes (`Add`/`Sub`/`Mul`/`Div` emit
`(a + b)` etc.) but **two leaves are float-shaped**:

- `ScalarExpr::Const(v)` (`backend.rs:76-88`) formats the `f64` value via `{v:?}` — e.g. `1.0`, and maps
  non-finite to `NAN`/`INFINITY`. For an integer accumulator, `1.0` is a `double` literal (forces int→double
  promotion) and `NAN`/`INFINITY` are nonsensical.
- `ScalarExpr::Param(i)` (`backend.rs:71`) emits `p{i}`, and `param_args` declares every param as **`float`**
  (`cuda.rs:793-798`). An integer reduction body that references a `Param` would mix a `float` param into an
  integer fold.

For the **v1 integer-reduction scope** the recommended body vocabulary is `Input` + infix `Add`/`Mul` +
`ReduceOp::{Sum,Max,Min}` only (e.g. `sum(x)`, `sum(x*x)` for an integer sum-of-squares) — which avoids
`Const`/`Param`/`Unary`/`Binary` entirely and sidesteps the two float-shaped leaves. The brief still adds a
build-time reject for those leaves under an integer dtype (section 5) so an author can't silently emit
`double`-promoting source.

### 4.4 The IR follow-up note and the schedule wiring (`ir.rs`, `plan.rs`)

`Access::Reduction` docs (`ir.rs:284-292`) explicitly list "integer accumulation" as a follow-up; `ReduceOp`
(`ir.rs:117-127`) is `Sum`/`Mean`/`Max`/`Min`, dtype-agnostic. `build_plan` (`plan.rs:85-127`) maps
`Access::Reduction { op }` straight to `Schedule::Reduction { op }` (`plan.rs:87`) with **no dtype gate** —
dtype legality is deferred to the backend (`plan.rs:82-83`), i.e. to `emit_reduction`'s assert. So the schedule
path needs **no change**; only the emitter's dtype handling does.

### 4.5 The AOT catalog + the numeric oracle (`bin/kernelgen.rs`, `OP-MATRIX.md`)

The catalog emits a float `mean` reduction for `{F32, F16}` (`bin/kernelgen.rs:86-102`). The hand-written
bespoke reduction plans already cover the integer cases we diff against: `CountReducePlan` (`i64`
accumulator, `OP-MATRIX.md:92`) and `ArgReducePlan` (`OP-MATRIX.md:90`) — these are the **numeric oracle**
for the generated integer kernel.

## 5. Design / delta

### 5.1 Replace the boolean `dbl` with a three-way accumulator kind

The core change is turning the binary float/double choice into a three-way choice that admits an integer
accumulator. Introduce a small local enum inside `cuda.rs` (no IR change needed):

```rust
// cuda.rs, local to emit_reduction (or a small free fn `acc_kind(dtype) -> AccKind`)
enum AccKind { F32, F64, Int(&'static str) } // Int carries the C accumulator type

fn acc_kind(dt: ElementKind) -> Option<AccKind> {
    match dt {
        ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 => Some(AccKind::F32),
        ElementKind::F64 | ElementKind::F32Strict              => Some(AccKind::F64),
        // Accumulate i32 sums in `long long` to resist overflow on long axes
        // (k up to millions); i64 already widest. This mirrors the bespoke
        // CountReducePlan i64 accumulator (OP-MATRIX.md:92).
        ElementKind::I32 | ElementKind::I64                    => Some(AccKind::Int("long long")),
        _ => None,
    }
}
```

- `acc` string = `"float"` / `"double"` / `"long long"`; `zero` = `"0.0f"` / `"0.0"` / `"0"`.
- **`load` closure**: add integer arms that load natively (no up-convert): `I32`/`I64` → `in{i}[idx]`
  (the existing `_ =>` native arm at `cuda.rs:309` already covers this once the assert is loosened — verify it
  falls through, don't special-case).
- **body lowering**: add a third branch alongside `dbl` — for `AccKind::Int`, lower with an **integer**
  `unary`/`binary` seam. In the v1 scope (`Input` + infix only) no `unary`/`binary` closure is invoked, so the
  closures can be `|op, _| panic!("integer reduction body: no {op:?} in v1 int vocabulary")` — matching the
  existing "author-error backstop" panic style used for `reduced` at `cuda.rs:315`. (A real integer
  `unary`/`binary` seam — `abs`, integer `min`/`max`, no `powf` — is a follow-up when index reductions need it.)

### 5.2 Loosen the assert, keep it honest

Replace the float-only assert (`cuda.rs:277-288`) with an `acc_kind(plan.dtype)` lookup that **panics with a
clear author-error message** on an unlowerable dtype (Bool, FP8, S4/U4, etc.), preserving the current
"AOT-build-time backstop" contract (it never fires across the JIT boundary — section 2). Add a second
build-time guard:

```rust
// Mean on an integer dtype: reject (PyTorch raises on mean() of an int tensor).
assert!(
    !(matches!(acc, AccKind::Int(_)) && matches!(rop, ReduceOp::Mean)),
    "reduction: Mean is undefined for integer dtypes (no exact mean); use Sum or cast to float first"
);
```

### 5.3 Guard the float-shaped leaves for the int accumulator

Add a build-time check (in `emit_reduction`, or reuse the `plan.rs` validation style) that the reduction
**body contains no `Const`, `Param`, `Unary`, or `Binary` node when the accumulator is integer** — because
`lower_expr` would emit `double` literals (`backend.rs:76-88`) and `float p{i}` params (`cuda.rs:793-798`)
into an integer fold. This keeps v1 int reductions to `Input` + infix `Add`/`Sub`/`Mul` (exactly the
`sum(x)` / `sum(x*x)` shapes we want), and turns the two float-leaf hazards into an explicit rejection rather
than silently-wrong `double`-promoting source. A helper `expr_is_int_reduction_safe(body) -> bool` (walk the
tree, reject non-`Input`/non-infix nodes) belongs next to `params_used` (`cuda.rs:768`).

### 5.4 StructureKey / FKC / contract implications

- **StructureKey**: none new. `I32`/`I64` already have `size_of_kind` entries (4 / 8 bytes,
  `structure_key.rs:598-599`) and token round-trip (`i32`/`i64`, `structure_key.rs:820-847`). The reduction
  `OperandDesc`/`OpCategory::Reduction` cell shape is unchanged.
- **FKC / §5 contract**: unchanged. Integer reductions are AOT-only and never cross the JIT boundary, so the
  "honest-miss" contract that `jit.rs` upholds (e.g. `integer_unary_binary_is_honest_miss_not_panic`,
  `jit.rs:1124`) is untouched — but must be **re-verified green** (section 8) to confirm we didn't accidentally
  widen the JIT vocabulary.

## 6. Implementation steps

1. **IR docs (`ir.rs`)** — update the `Access::Reduction` doc (`ir.rs:284-292`) to note integer accumulation
   is now supported for `Sum`/`Max`/`Min` (`Mean` int-rejected). No enum/field change. Update the
   `emit_reduction` header comment's follow-up list (`cuda.rs:260-264`) to drop "integer accumulation".
2. **Emitter — accumulator kind (`cuda.rs`)** — add `acc_kind` (section 5.1), replace the `dbl`/`acc`/`zero`
   trio (`cuda.rs:302-304`) with the three-way derivation, and route the body-lowering branch
   (`cuda.rs:311-325`) to an integer seam for `AccKind::Int`.
3. **Emitter — assert loosening + Mean guard (`cuda.rs`)** — replace the float-only assert (`cuda.rs:277-288`)
   with the `acc_kind` panic and add the int-`Mean` reject (section 5.2). Confirm the `load` native arm
   (`cuda.rs:309`) and the store path (`cuda.rs:369-373`, which only down-converts f16/bf16 — int falls to the
   `_ => finalized` arm, correct) handle int with no extra code.
4. **Emitter — leaf guard (`cuda.rs`)** — add `expr_is_int_reduction_safe` and assert it for int accumulators
   (section 5.3), next to `params_used` (`cuda.rs:768`).
5. **Plan/schedule (`plan.rs`)** — **no change** (verify: `build_plan` already dtype-blind at `plan.rs:87`).
   Add a one-line comment noting int now flows through.
6. **Pattern/contract (`pattern.rs`, `jit.rs`, `contract.rs`)** — **no change** (reductions are AOT-only).
   Do not touch the JIT boundary; only re-run its tests.
7. **AOT catalog (`bin/kernelgen.rs`)** — add an integer `sum` reduction entry mirroring the float `mean`
   block (`bin/kernelgen.rs:86-102`): `OpDef::reduction("sum", 1, &[ElementKind::I32, ElementKind::I64],
   input(0), ReduceOp::Sum)` and an integer sum-of-products `input(0)*input(0)` variant, emitted over an
   `OpCategory::Reduction` key with an `i32`/`i64` `OperandDesc` (cf. `reduce_key` at `cuda.rs:952-957`).
8. **Docs (`OP-MATRIX.md`, `docs/design/kernel-specialization.md`)** — add a kernelgen integer-reduction row;
   note in the design doc that generated reductions now cover `{i32, i64}` `Sum`/`Max`/`Min`. (Flag: the
   design doc status text is known-stale per the memory index — correct only the reduction line, don't trust
   the rest.)

## 7. Test & on-device validation plan

**Unit tests (`cuda.rs` test module, next to `reduction_mean_of_squares_f32` at `cuda.rs:959`):**
- `reduction_sum_i32_uses_long_long_acc` — generate `OpDef::reduction("sum", 1, &[I32], input(0),
  ReduceOp::Sum)` over an `i32` `reduce_key`; assert name `baracuda_gen_sum_i32_reduce_sum`, signature contains
  `const int* __restrict__ in0`, body contains `long long acc = 0;` and `acc += in0[idx];`, store is `out[o] = acc;`
  (no `(float)`/`(double)` cast, no `__float2half`).
- `reduction_sum_of_squares_i64` — `input(0)*input(0)`, `i64`; assert `long long` acc and `acc += (in0[idx]*in0[idx]);`.
- `reduction_max_i32_peels_first` — `ReduceOp::Max` over `i32`; assert the `if (k > 0)` peel guard
  (`cuda.rs:353`) and the NaN-check line is **absent or harmless** for int (see adversarial section — the
  `e != e` NaN test is always-false for integers, so the compare-select degenerates to a plain `e > acc`;
  assert the emitted comparison is well-formed integer C).
- `reduction_mean_i32_rejected` — `#[should_panic]` on the int-`Mean` guard message.
- `reduction_int_body_with_param_rejected` — `#[should_panic]`: an int reduction whose body references
  `Param(0)`/`Const` trips the leaf guard.

**nvrtc headerless compile (house discipline — HEADERLESS):** feed each generated `i32`/`i64` `sum`/`max`/`min`
`.cu` to nvrtc with **no extra headers** (integer reductions need neither `cuda_fp16.h` nor `cuda_bf16.h` —
confirm `extra_include` returns `None` for int, `cuda.rs:67-72`) and assert a clean compile to PTX for
`sm_89`. This is the primary "does it even parse as valid CUDA" gate.

**nvcc numeric on sm_89 (RTX 4070):** compile + launch the generated `sum_i32` / `sum_i64` /
`sum_of_squares_i64` / `max_i32` / `min_i32` kernels on a `[256, 128]` and a `[4096, 1024]` input and
**diff bit-exactly** against:
- the **integer numeric oracle** = the bespoke `ReducePlan`/`CountReducePlan` integer path (`OP-MATRIX.md:89,92`)
  — or a trivial host `i64` fold, since integer sum is exact and deterministic. Bit-exact equality is the bar
  (no ULP tolerance — the whole point of the integer accumulator is exactness).
- **Overflow case**: an `i32` input whose axis sum exceeds `i32::MAX` but fits `i64` — assert the `long long`
  accumulator yields the correct wide result (this is the concrete win over a hypothetical `int` accumulator).
- **Empty axis (`k == 0`)**: `Sum` → `0`; `Max`/`Min` → the peel guard leaves the seeded `0` (documented,
  matches the float path's `k > 0` guard at `cuda.rs:353` — call out that int `amax` of an empty axis returns
  `0`, not an error, same as the float kernel).

**compute-sanitizer:** the last-axis reduction is **one thread per output cell with no shared memory and no
cross-thread communication** (`cuda.rs:336-362` — grid-stride over independent output cells). So `racecheck`/
`synccheck` have nothing to find, but run `initcheck` + `memcheck` on the `sum_i32` launch to confirm no OOB
on the `base = o*k` indexing for a ragged final block. (If/when a block-parallel tree fold with shared memory
is added — an explicit non-goal here — `racecheck` becomes mandatory.)

## 8. Adversarial-verify checklist (skeptic pass must probe THESE)

1. **`i32` accumulator overflow** — confirm we accumulate `i32` sums in `long long`, not `int`. A skeptic
   should construct an axis whose sum overflows `int` and verify the emitted `acc` type is `long long` and the
   nvcc numeric result is the true (non-wrapped) sum. (The float path never hit this; it's the signature risk
   of the int path.)
2. **`double` literal leaking into an integer fold** — grep the generated int source for `.0` / `0.0f` /
   `NAN` / `INFINITY` / `__float2` / `(float)` / `(double)`. Any of these in an `_reduce_sum` `i32`/`i64`
   kernel means a float leaf (`Const`/`Param`, `backend.rs:76-88`; `cuda.rs:793-798`) or the wrong `zero`
   slipped through — the leaf guard (section 5.3) and `zero = "0"` must prevent all of them.
3. **`Mean` int truncation** — verify int `Mean` is **rejected at build time**, not silently emitting C
   integer division `acc / k` (which truncates and diverges from `torch.mean`). The `#[should_panic]` test
   locks this.
4. **`Max`/`Min` NaN-check degeneracy on int** — the float peel-loop emits `acc = (e != e || e > acc) ? e : acc;`
   (`cuda.rs:359`). For integers `e != e` is always false (no int NaN) — confirm this is *harmless* (compiles,
   degenerates to `e > acc`), OR emit a clean integer compare-select without the `e != e` term. A skeptic
   should confirm the emitted int source is valid C and numerically matches a plain integer max/min (no
   accidental always-false short-circuit bug).
5. **Store-path cast** — confirm the int store hits the `_ => finalized` arm (`cuda.rs:372`) with **no**
   `__float2half`/`__float2bfloat16` and no truncating cast (`acc` is `long long`, `out` is `int` for `i32` —
   verify the implicit `long long → int` narrowing store is intended and correct for `Sum` results that fit,
   and documented for those that don't).
6. **JIT boundary didn't widen** — re-run `integer_unary_binary_is_honest_miss_not_panic` (`jit.rs:1124`) and
   confirm reductions are still unreachable from `synthesize` (a skeptic should try to construct a `JitRequest`
   that reaches `emit_reduction` and confirm `derive_pattern`/`region_to_op` still reject it — it must remain
   an honest miss, not a new capability).
7. **Empty-axis / ragged-block indexing** — `initcheck`/`memcheck` on `base = o*k` for `n_out` not a multiple
   of blockDim; confirm the grid-stride loop (`cuda.rs:338`) never reads past the input on the final block.

## 9. Definition of done

- [ ] `emit_reduction` accepts `I32`/`I64` for `Sum`/`Max`/`Min`; `Mean` int-rejected with a clear message.
- [ ] Accumulator is `long long` for both `i32` and `i64` (exact, overflow-resistant); `zero = "0"`; no float
      literal / cast / fp16 header in any generated integer-reduction source.
- [ ] Int reduction body restricted to `Input` + infix `Add`/`Sub`/`Mul` by a build-time leaf guard;
      `Const`/`Param`/`Unary`/`Binary` under an int dtype is a clean author-error panic.
- [ ] Unit tests green (int sum, int sum-of-squares, int max, `#[should_panic]` for Mean + Param-in-int-body).
- [ ] **On-device validated on sm_89 (RTX 4070)**: nvrtc headerless compile of every generated int reduction;
      nvcc numeric **bit-exact** vs the integer oracle, including the `i32`-overflow-into-`i64` case and the
      empty-axis case; `initcheck`/`memcheck` clean.
- [ ] Determinism preserved: one thread per output cell, no atomicAdd, no shared memory — explicitly noted as
      unchanged from the float path.
- [ ] **FKC honest-miss preserved**: JIT boundary tests green, reductions still unreachable from `synthesize`.
- [ ] AOT catalog emits int `sum` kernels (`bin/kernelgen.rs`); `OP-MATRIX.md` row added; the reduction line
      in `docs/design/kernel-specialization.md` corrected (rest of that doc left alone — known stale).
- [ ] Adversarial-verify pass run (find → dedup → skeptic-refute) with the section-8 checklist; findings
      resolved or documented.

## 10. Open questions / Fuel asks

- **No Fuel asks.** Integer reductions are AOT-only and never cross the §5 seam, so nothing here needs a
  cross-repo answer.
- **Internal decision, defaulted in this brief:** int `Mean` is **rejected** (matches PyTorch). If a consumer
  later needs a floor-division integer mean, that becomes an explicit `ReduceOp` semantic (e.g. a distinct
  `MeanFloor`) — out of scope here; do not silently truncate.
- **Accumulator width policy:** this brief accumulates *both* `i32` and `i64` in `long long`. Open refinement:
  should a `u32`/`u64` input family (not currently an `ElementKind` reduction input) later want an unsigned
  accumulator? Deferred — no unsigned integer reduction input exists in the kernelgen dtype set today
  (`ElementKind` int inputs are `I32`/`I64`; `U8`/`S8`/`S4`/`U4` are GEMM-operand-only per `element.rs:1012-1019`).
- **Follow-on (not this brief):** an integer `unary`/`binary` lowering seam (integer `abs`/`min`/`max`, no
  `powf`) is only needed once index/argmax-class reductions land; this brief deliberately restricts the int
  body vocabulary to infix ops and leaves that seam as a stub-panic.
