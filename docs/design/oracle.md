# CPU oracle + precision-first policy — design (2026-07-10)

**Status: shipped — and CARVED OUT of this repository.** The CPU oracle and the
precision-first `MorePrecise` variant policy both live in the external **`unpopped`**
crate now, not here. They were `crates/baracuda-kernelgen/src/{oracle,cuda,backend}.rs`
(commits `e57c8b3`, `85305948` / `541ecb6c`; device-proven at `3604b3db`) until the
extraction recorded in `2026-08-06-unpopped-extraction-plan.md`. The oracle is still the
numeric correctness reference; Baracuda consumes it as a dependency.

> ⚠ **DISCHARGED 2026-09-06.** The status word *shipped* was, and remains, correct — the
> four `crates/baracuda-kernelgen/…` paths under it were not. **That directory holds zero
> files in this repository** (control: `crates/baracuda-cuda-emit/` holds 22), so every
> citation beneath the authoritative Status line was dangling and a reader following one
> found nothing. **A true header over dead citations is in one way worse than a false one:
> it survives verification, and sends the reader somewhere that does not exist.** Now named
> by CRATE rather than by path — a path into another repository would recreate this defect
> the next time that repository moved a file. This is a `0.0.1-alpha` codebase — the scope caveats
below (v2 deferrals, the round-once residual) are real and load-bearing, not
polish.

The oracle is described obliquely in
[`backend-agnostic-emission-design.md`](backend-agnostic-emission-design.md) (as
the third leg of the correctness triangle); this doc is the primary spec for both
the oracle and the precision policy that rides on it.

---

## Part 1 — the CPU oracle (an independent plan-interpreter)

### What it is

`oracle::evaluate` (`oracle.rs:1053`) is an **independent** Rust interpreter that
computes what a generated CUDA (or CpuC) kernel computes — from the **same IR**
(`KernelPlan` + `OperandDesc`) the emitter reads, but via a **separate code
path**. It takes the plan, the operand descriptors, the input storage images
(`TypedBuffer` — a dtype-tagged raw byte blob + its logical layout, `oracle.rs:78`),
and the runtime scalar params, and returns the computed output storage images. It
is therefore three things at once: a differential test of the emitter
(`crate::cuda`), a GPU-free CI reference, and the shared oracle every backend
validates against.

### Independence — the whole point

The oracle earns its keys by sharing **zero lowering code** with the emitter
(`oracle.rs:8-25`). It does **not** call `lower_expr` / `lower_dag` / the
`Lowering` struct / `unary_f32` / `binary_f32` / `binary_int` / `cuda_select` /
`const_lit` / `offset_expr` / `gathered_offset_expr` / any `emit_*`. Every
`ScalarExpr` evaluation (`eval`, `oracle.rs:794`), every scalar op, all
index/fold/sort math, and the IEEE-754 half codec are **re-implemented here from
each op's DEFINITION**, not its CUDA spelling — so a shared spelling bug cannot
hide identically in both the emitter and its checker. The independent scalar
semantics live in `unary_op_f64` (`oracle.rs:522`), `binary_op_f64`
(`oracle.rs:596`), and `binary_op_int` (`oracle.rs:653`); the pure-Rust half
codec in `f16_to_f64` / `bf16_to_f64` / `f32_to_f16_bits` / `f32_to_bf16_bits`
(`oracle.rs:323`–`404`).

It **may** reuse the pieces *upstream* of both lowerings — the IR types
(`crate::ir`), `build_plan` / `KernelPlan`, and the operand role classifier
`rr_role` (a broadcast-mask predicate, not a spelling). That upstream reuse is
what bounds its honest scope:

> **The oracle catches EMISSION bugs, not IR-construction bugs.** It reads the
> same `build_plan` output the emitter does, so a wrong plan fools both. It
> *complements* — does not replace — the hand-written bespoke oracles and the
> `build_plan`-direct gate tests.

### Fidelity — the bit-exact-vs-tolerance dichotomy

`compare(expected, actual, fidelity)` (`oracle.rs:1885`) is the referee, with two
modes (`enum Fidelity`, `oracle.rs:1865`):

- **`BitExact`** — raw-storage `memcmp`. Used where the result is defined to the
  bit: identity bodies, `Im2Col`, permutation/movement, `Select` arm moves, and
  all integer arithmetic/bitwise/shift (exact-wrapping). The compare reports the
  first mismatching element index and the two hex bit patterns.
- **`Tolerant { rel, abs }`** — decode both sides to `f64` and accept within
  `abs + rel·max(|a|,|b|)`. Both-NaN compares equal (payload-agnostic), ±0 compare
  equal, and — deliberately — a non-equal infinity is **rejected** rather than
  swallowed by an infinite relative band (so an output overflowing to `inf`
  cannot silently pass; `oracle.rs:1926`).

### Precision posture (why the oracle is a genuine reference, not a peer)

Arithmetic accumulates in `f64`, never the emitter's "double-then-round-once f32"
convention — routing the oracle through the emitter's rounding would *couple* it
to the speller it is meant to check. Integer arithmetic is exact-wrapping in the
two's-complement width the emitter's C promotion uses (`op_width` / `wrap_bits`,
`oracle.rs:266`/`274`). Transcendentals are a genuinely **tighter** reference than
the kernel they validate: `Erf`/`Erfc`/`Gelu` use `libm` (~1 ULP `f64`, pure-Rust,
independent of the device `erf`/`erff`), `Lgamma` an in-house Lanczos series
(~1e-13); none is used on a bit-exact path.

The one subtlety is **discontinuous** ops. A continuous op (`exp`, `Max`, …) that
lands within ~1 ULP of a tie still returns a within-tolerance operand, so
accurate-`f64` + the tolerance band absorbs the gap. But a *discontinuous*
decision — the `Cmp*` family, a `Select` condition, `Sign`, `Step` — flips by a
full magnitude no tolerance catches (`x == 0.1` with `0.1` inexact in the compute
dtype). The emitter guards these with an explicit compute-dtype cast; the oracle
matches it with `round_to_compute` (`oracle.rs:507`) *before* the decision. This
rounds the operand **once** — exact for the leaf/`Const`/single-arith patterns
that feed real mask/threshold/sign decisions; a decision on a ≥2-deep sub-expression
landing within ~1 ULP of the threshold is the acknowledged measure-zero residual,
outside v1's tolerance-referee role.

### Access coverage (v1) and the v2-deferred gaps

**Covered (`oracle.rs:1059`):** `Elementwise` (+ multi-output / hetero),
`Reduction`, `RowReduce`, `Scan`, `Window`, `Im2Col` — the full `ScalarExpr`
vocabulary + layout math (contiguous / strided / broadcast / flipped / permuted /
runtime base-offset).

**Deferred to v2 (these `panic!` in `evaluate`, `oracle.rs:1066`):**

- `Access::Contraction` / MatMul — its own axis-role plumbing.
- `Access::RowSort` — the NaN-greatest `key_lt` / stable-index-tie / TopK
  comparator.
- gather / scatter — `ReadIndex`/`WriteIndex` OOB policies + FP-`atomicAdd`
  nondeterminism (only an order-independent invariant is checkable there).
- Differential fuzzing against the device side.

The emitter's own scope constraints are mirrored rather than silently
mis-computed: integer reductions are asserted `I32`/`I64`-only and integer `Mean`
is rejected, matching `emit_reduction`'s asserts (`cuda.rs:2207`–`2225`,
`oracle.rs:1173`).

### Role — the three-way triangulation

Because the oracle shares no lowering code, agreeing with it is a real third
leg, not a tautology. The correctness triangle is **CUDA emitter ↔ CpuC emitter ↔
Rust oracle**: the CpuC portable-C backend (`crates/baracuda-kernelgen/src/cpu_c.rs`)
was authored against the oracle's `eval_*` and its emitted C compiled + run
GPU-free on-box, so IR neutrality is proven *by execution*, and every generated
kernel (CUDA or CpuC) is numerically anchored to the same independent reference.

---

## Part 2 — the precision-first policy (`VariantFidelity::MorePrecise`)

### The variant

`MorePrecise` (`backend.rs:51`–`66`) is a schedule variant that is **strictly more
accurate than the default f32 lowering AND deterministic**. The default f32
reduction accumulates in a `float` running sum (error growing with the reduced
length); the variant forces a `double` accumulator and a **no-reassociation
serial fold**, yielding ~0.5 ULP(f32) of the correctly-rounded reduction. That
directedness — "closer to the true reduction", not merely "a different rounding" —
is the whole selection signal, and it is why `MorePrecise` is neither
`BitIdentical` (it differs from the default bits, so must never be chosen silently)
nor `ReassociatedDeterministic` (same-accuracy, different-rounding, undirected).

Because the serial double fold is a fixed order with per-op-deterministic doubles,
it is **bitwise-reproducible on any IEEE-754 hardware** — so its determinism
spelling is the *strongest*, `bitwise` (not `same_hardware_bitwise`;
`determinism_str`, `backend.rs:81`). It is never the silent default (it trades the
coalesced block-tree's throughput for accuracy); it is selectable only through an
honest FKC contract whose precision block advertises the tighter bound — the
caller's precision policy decides.

### When it is offered

`precision_first_variant` (`cuda.rs:2757`) dispatches on the schedule and offers
the variant for exactly two shapes:

- **`Schedule::Reduction`** — `precision_first_reduction` (`cuda.rs:2768`): routes
  through `emit_reduction(.., precision=true)`'s serial per-output general nest
  with a forced `double` accumulator (`cuda.rs:2200`–`2266`).
- **`Schedule::RowReduce`** — `precision_first_rowreduce` (`cuda.rs:2826`): the
  `_prec` serial-per-row double fold from `emit_row_reduce_impl(.., precision=true)`,
  for the fused softmax / layernorm / rmsnorm cells.

Both arms **decline** every cell they cannot strictly improve:

- **f32 only.** `F64` / `F32Strict` already accumulate in `double` (the variant
  would be a byte-identical duplicate); `F16`/`Bf16`'s low-mantissa *store* rounds
  the extra accumulation away; integer reductions are exact.
- **Sum/Mean only.** `Max`/`Min` are a pick and `Prod` a product — order-exact, so
  the double fold is bit-identical to the base (no distinct offer). The RowReduce
  arm requires at least one Sum/Mean stage.
- **Injective dense output.** A flipped (negative-stride) or broadcast (stride-0)
  output would alias / write OOB in the serial nest, so it is declined (the block
  base serves it via its own path).

### The device-proven result

On-device two-sided validation (RTX 4070 / sm_89 / CUDA 13.3, nvrtc; commit
`3604b3db`, `tests/precision_first_numerics.rs`): an outer-axis f32 `Sum` of
20,000,000 ones per column, folded in a **serial** thread so base and variant
share the exact same fold order and only the accumulator width differs.

- **base (serial `float`)** — the running sum **saturates at 2²⁴ = 16,777,216**
  (each `+1.0` stalls once the sum exceeds the f32 mantissa's reach): relerr 0.161,
  i.e. ~16 % wrong.
- **prec (serial `double`, the `MorePrecise` variant)** — **20,000,000, exactly
  equal to the f64 oracle** (relerr 0.0).

This isolates the precision win cleanly and retroactively validates the reduction
variant (`85305948`), whose numeric check had been deferred to a CUDA box.

### The honest nuance — accuracy vs. reproducibility

The reduction is the *clean* accuracy demonstrator because its base is a serial
`float` fold that genuinely saturates. The **RowReduce base is different**: it
folds in a **block-tree**, which stays accurate for realistic per-row lengths (the
partial sums don't individually reach 2²⁴). So for RowReduce the variant's real
value is **bitwise reproducibility** (a fixed, hardware-independent fold order
matching the CPU oracle), not a dramatic accuracy delta. The policy offers both
under the same `MorePrecise` fidelity — but this doc records that the "strictly
more accurate" claim is *dramatic* for the serial-base reduction and *primarily
about reproducibility* for the block-tree-base RowReduce. Do not oversell the
RowReduce accuracy win.

---

## Anchors

- `oracle.rs`: `evaluate` `1053`, independence doc `8-25`, `eval` `794`,
  `unary_op_f64` `522`, `binary_op_f64` `596`, `binary_op_int` `653`,
  `round_to_compute` `507`, half codec `323`/`339`/`344`/`397`,
  `compare` + `enum Fidelity` `1862`–`1947`, access dispatch + v2 panics `1059`.
- `backend.rs`: `enum VariantFidelity` `27`, `MorePrecise` doc `51`,
  `determinism_str` `81`, `struct Variant` `98`, `Backend` trait `113`.
- `cuda.rs`: `precision_first_variant` `2757`, `precision_first_reduction` `2768`,
  `precision_first_rowreduce` `2826`, `emit_reduction` precision path `2200`–`2266`.
- Commits: oracle `e57c8b3`; precision-first reduction `85305948` + RowReduce
  `541ecb6c`; device numerical proof `3604b3db`.
- Related: [`backend-agnostic-emission-design.md`](backend-agnostic-emission-design.md)
  (the CUDA ↔ CpuC ↔ oracle triangle), [`axis-role-vocabulary.md`](axis-role-vocabulary.md).
