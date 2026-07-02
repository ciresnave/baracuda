# 03 — Strided/multi-axis/keepdim reductions — implementation brief

## 1. Objective

Extend `Access::Reduction` past its v1 corner — **contiguous, last-axis, float-dtype, single-input** — so the AOT generator can specialize the real reduction shapes Fuel's `ReducePlan` / `SoftmaxPlan` / `NormalizationKind` already serve: reducing a **non-last (outer/middle) axis**, reducing **multiple axes at once**, reducing a **strided or broadcast input**, keeping the reduced axis with **keepdim** broadcast-back layout, and **multi-input (weighted) reductions**. This is foundational because reductions are the second op family (after elementwise) the generator must cover to be useful to Fuel at all: every norm, softmax, mean/var, and loss reduction routes through a reduce over some axis set, and today the generator can only emit the one contiguous-trailing-axis case. Concretely it turns the design doc's **predicate #9 (reduce-axis position `{Inner, Outer, Middle, Multi}`)** and **#10 (reduce-extent vs block)** from named-but-empty axes into live specialization, and it lights up the `StructureKey::reduce_axes` field (already carried, always empty today) as a real join token so the runtime dispatch, the FKC `accept` predicate, and telemetry all agree on *which axes a reduction kernel reduces*.

## 2. Status & blockers

**Baracuda-unblocked.** The entire change is AOT-side codegen + keying. Nothing in it crosses the JIT `synthesize` trust boundary: reductions are not in the JIT vocabulary (`region_to_op` in `jit.rs:406-425` hardcodes `Access::Elementwise`, and `derive_pattern`/`pattern.rs:80-82` rejects non-elementwise), so the build-time asserts in `emit_reduction` are author-error backstops, not runtime guards. All the machinery this needs already exists in-repo:

- `StructureKey.reduce_axes: AxisMask` field + token codec are **already present** (`structure_key.rs:215`, token form `x{:02x}` at `to_token`/`from_token`) — but `structure_key()` hardcodes `AxisMask::EMPTY` (`structure_key.rs:423`). We add the derivation, we do **not** add the field or the wire format.
- The strided elementwise emitter (`emit_strided`, `cuda.rs:187-249`) already does per-operand rank-unrolled coord-unravel, broadcast-axis offset dropping, and fully-broadcast hoisting — the exact address machinery a strided/keepdim reduction needs. We reuse its shape/stride approach rather than inventing new offset code.

**Design-open (small, decided in §5, not cross-repo):** the boundary between "extend `Access::Reduction`" and "reductions that need `Access::RowReduce`'s block-tree" — i.e. whether outer-axis reductions get their own thread-per-column schedule or ride the existing sequential fold. Resolved in §5 as: keep the sequential-fold schedule for correctness-first v2, split the schedule token so a block-parallel outer-axis kernel is a later drop-in.

**Not blocked by Fuel.** Item 05 (the RowReduce seam) is the Fuel-blocked one (see `docs/fuel-ask-fused-reduce-seam-2026-06-25.md`); **this item is the plain-reduction path and needs no Fuel answer** to land AOT. The only cross-repo touch is confirming the `reduce_axes` token addition is acceptable on the telemetry/FKC wire — but since the field already exists and already serializes, that is a confirmation, not a blocker (see §10).

## 3. Dependencies & sequencing

**Should land before this:**
- **01 (layout/shape nodes)** — this item deliberately **reuses 01's shape-fact machinery** for the keepdim broadcast-back output layout and for arbitrary-axis offset math. If 01 lands first, the keepdim output layout and the strided-input offset expressions share one shape/axis representation instead of this item growing a private one. If 01 is not ready, this item can still ship the **axis-set + strided-input + multi-input** slice using the existing `StructureKey` operand strides directly (the `emit_strided` approach), and adopt 01's shape nodes for keepdim in a follow-up. **Recommended: sequence 01 first for keepdim; the rest is independent.**

**Independent of / synergistic with:**
- **04 (integer accumulation for reductions)** — orthogonal: 04 changes the accumulator *type*, this changes the *axis geometry*. They touch the same `emit_reduction` asserts (`cuda.rs:286-297`), so whichever lands second rebases the assert block; no logical conflict. Coordinate the assert edit.

**What this ENABLES downstream:**
- **03 → non-last-axis norms + keepdim** directly (the reason it exists).
- **03's axis/shape handling feeds 10 (MatMul design spike)** — a contraction is a reduction over a shared axis; the `.axis`/`reduce_axes` representation this item lands is the same axis-fact vocabulary 10's design must reference.
- **Sharpens the honest-miss story for the whole reduction family:** once `reduce_axes` is a real key component, a reduction kernel built for `{last axis}` correctly *misses* a request to reduce `{axis 0}`, instead of both hashing to the same empty-mask cell.

## 4. Current code — what exists today

### IR (`crates/baracuda-kernelgen/src/ir.rs`)
`Access::Reduction` carries **only** the combine op — no axis, no keepdim (`ir.rs:289-292`):
```rust
Reduction {
    /// The associative combine (+ implied identity).
    op: ReduceOp,
},
```
The doc comment (`ir.rs:285-288`) names the exact v1 scope and the follow-ups this brief implements: *"v1 covers the contiguous last-axis float-dtype case … Strided inputs, arbitrary/multiple axes, keepdim layout, and integer accumulation are follow-ups."* `ReduceOp` = `Sum | Mean | Max | Min` (`ir.rs:117-127`). The builder `OpDef::reduction` (`ir.rs:344-359`) sets `access: Access::Reduction { op }`.

### Emitter (`crates/baracuda-kernelgen/src/cuda.rs`)
`emit_reduction` (`cuda.rs:265-377`) is one thread per output element, a **sequential** fold over the trailing axis. Its scope is enforced by three build-time asserts that are exactly the walls to move:
```rust
// cuda.rs:277  — float dtypes only
assert!(matches!(plan.dtype, F16|Bf16|F32|F32Strict|F64), "reduction v1: float dtypes only …");
// cuda.rs:289  — single input
assert!(plan.n_inputs == 1, "reduction v1: single-input only …");
// cuda.rs:294  — all operands contiguous; base = o*k assumes a dense last axis
assert!((0..plan.key.n_operands as usize).all(|i| plan.key.operands[i].contig == Contiguity::Contig),
    "reduction v1: contiguous operands only (base = o*k assumes a dense last axis); …");
```
The addressing that assumes last-axis-contiguous is `cuda.rs:339`: `long long base = o * k;` then `idx = base + j` (`cuda.rs:344`). That `base = o*k` linear form is *only* valid when the reduced axis is the trailing contiguous run and the output is dense — it is the single line that must generalize for outer/middle/multi-axis and strided inputs. The Mean divisor `k` (`cuda.rs:365`) is the reduced *extent*; for multi-axis it becomes the product of reduced extents.

### Plan / schedule (`crates/baracuda-kernelgen/src/plan.rs`)
`Schedule::Reduction { op }` (`plan.rs:36-39`) carries only the combine. `build_plan` routes `Access::Reduction { op: rop } => Schedule::Reduction { op: rop }` (`plan.rs:87`) — a straight map, no axis inspection. The `KernelPlan` already carries `key: &StructureKey` (`plan.rs:67`) and `access: &Access` (`plan.rs:74`), so the emitter can read axis facts off either without a signature change.

### Keying (`crates/baracuda-kernels-types/src/structure_key.rs`)
`StructureKey.reduce_axes: AxisMask` **exists** (`structure_key.rs:215`) with the doc *"Reduced-axis set for reduction-class ops; `AxisMask::EMPTY` otherwise (always empty in v1)."* `structure_key()` sets it unconditionally empty (`structure_key.rs:423`: `reduce_axes: AxisMask::EMPTY, // reduction keying: follow-up`). The token codec **already** round-trips it (`to_token` emits `x{:02x}` or `-`; `from_token` parses it; the round-trip test `token_round_trips` at `structure_key.rs:991` covers the empty case). `AxisMask` has `set`/`is_set`/`is_empty`/`EMPTY` (`structure_key.rs:133-158`). `OperandKey` carries `bcast: AxisMask` + `flipped: bool` + `contig` + strides-derived facts; per-operand strides live on `OperandDesc.strides` (`structure_key.rs:301`).

### Seam / pattern (unchanged by this item, but must stay honest)
`jit.rs:406-425` `region_to_op` hardcodes `Access::Elementwise`; `pattern.rs:80-82` `derive_pattern` returns `PatternError::NotElementwise` for anything else; `contract.rs:68` calls `derive_pattern(op).ok()` and returns `None` (skips the cell — honest miss) when it fails. A reduction op therefore already produces **no FKC contract** today, which is correct: the generator does not yet advertise reductions on the seam. This item must **preserve** that — a strided/multi-axis reduction still must not leak a bindable elementwise contract.

### Design doc rows (stale — flag, don't trust status text)
`docs/design/kernel-specialization.md:67-68` names predicate #9 `{Inner, Outer, Middle, Multi}` and #10; `:187` shows `reduce: op_class.reduction_axes(&canon)` as the intended derivation hook; `:432-435` files reductions-with-`.axis` under the pending "ORDER 3" workstream. The doc's ORDER-3 text is the design intent for exactly this item.

## 5. Design / delta

### 5a. IR: axis set + keepdim on `Access::Reduction`
Add two fields. `axes` is a canonical-axis bitmask (reuse `AxisMask` from `baracuda-kernels-types`, already a dependency); `keepdim` selects whether reduced axes collapse (rank drops) or stay size-1 (broadcast-back output). Default construction (via the existing `OpDef::reduction`) must keep behaving as last-axis, so add a back-compat builder and make the old one delegate.

```rust
// ir.rs — Access::Reduction
Reduction {
    op: ReduceOp,
    /// Canonical axes reduced (bit i ⇒ axis i). Empty ⇒ the legacy
    /// last-axis default (OpDef::reduction preserves this).
    axes: AxisMask,
    /// Keep reduced axes as size-1 (broadcast-back) vs collapse them.
    keepdim: bool,
},
```
`OpDef::reduction(...)` keeps its signature and sets `axes: AxisMask::EMPTY, keepdim: false` — **byte-identical output to today** (empty ⇒ "trailing axis", the existing meaning). Add `OpDef::reduction_axes(name, n_inputs, dtypes, body, op, axes, keepdim)` for the new cases. Rationale: never break the ~7 existing reduction tests / the RmsNorm mean-of-squares core; the empty mask is the sentinel for "the one axis we always did".

### 5b. Schedule: name the axis geometry so the emitter branches on it
`Schedule::Reduction` is `Copy`, so it cannot carry a `Vec`; but it can carry the *classification* (predicate #9) as a small enum, with the full axis mask read off `plan.key.reduce_axes` / `plan.access` in the emitter (same pattern `RowReduce` already uses — schedule carries `n_stages`, the `Vec` rides on `access`).

```rust
// plan.rs
pub enum ReduceAxisClass { InnerContig, Outer, Middle, Multi } // predicate #9

Schedule::Reduction {
    op: ReduceOp,
    class: ReduceAxisClass,
    keepdim: bool,
}
```
`build_plan` computes `class` from `key.reduce_axes` + the input operand's contiguity: empty-or-trailing-mask + contiguous ⇒ `InnerContig` (today's fast path, unchanged); a single non-trailing axis ⇒ `Outer`/`Middle`; ≥2 axes ⇒ `Multi`. **v2 correctness-first decision:** all four classes lower to the **same sequential fold**, just with generalized addressing (below). Splitting `Outer` off to a thread-per-column / block-parallel kernel (design doc #9/#10) is a *later* perf drop-in behind the `class` token — the token exists now so that kernel is additive, not a re-key.

### 5c. Emitter: generalize the address from `base = o*k` to a strided reduce nest
Replace the single-axis `base = o*k` + `idx = base+j` with the same **kept-axes unravel / reduced-axes fold** shape `emit_strided` already uses for elementwise. For each output element:
1. Unravel the output linear index over the **kept** axes only (using the output shape) → per-kept-axis coords `ck`.
2. Compute the input base offset from those kept coords via the input's strides (dropping broadcast axes, exactly as `offset_expr`, `cuda.rs:619`).
3. Loop the **reduced** axes (nested, one loop per set bit in `axes`, or a single flattened loop over the reduced-extent product) accumulating `in0[base + Σ reduced_coord·stride]`.
4. `keepdim` changes only the *output* offset computation: with keepdim the output has the reduced axes as size-1 (stride 0), so the store offset unravels over the full rank with the reduced-axis terms dropped; without keepdim the output rank is the kept-axes count.

Concretely the emitter reads `plan.key.operands[0].strides`-equivalent facts and `plan.key.reduce_axes` and emits, per reduced axis `d`, a `for (cd = 0; cd < shape[d]; ++cd)` wrapping the accumulate, with the input offset accumulating `cd * s0[d]`. The **Mean divisor** becomes `Π shape[d] over d in axes` (a runtime product of the reduced extents, passed as one `long long k` kernel arg = the reduced-element count, exactly as today — the host already knows it). The existing accumulate/finalize/store body (`cuda.rs:340-374`), the NaN-propagating Max/Min peel (`cuda.rs:348-362`), the f16/bf16 up/down-convert (`cuda.rs:305-310, 369-373`), and the float/double accumulator choice (`cuda.rs:302-303`) are **reused verbatim** — only the index generation changes.

The three asserts (`cuda.rs:277/289/294`) relax:
- **contiguity** (`:294`): no longer require all-contiguous; instead require the *reduced* axes' addressing be expressible from strides (always true given `OperandDesc.strides`) and keep a guard that the **output** is dense/keepdim-broadcast (the store must not alias). A truly pathological non-representable layout stays an assert (author backstop).
- **single-input** (`:289`): relax to allow the RowReduce-style row-streamed `x` + per-column weight roles (§5d).
- **float-only** (`:277`): unchanged by *this* item (integer accumulation is item 04); leave the assert, note the seam with 04.

### 5d. Multi-input reductions
Adopt the role machinery `RowReduce` already ships (`plan.rs:129-149` `RrRole`/`rr_role`, and `validate_row_reduce` at `plan.rs:174-293`): input 0 is the row-streamed reduced tensor, inputs 1.. are per-column `[k]` weight/bias (broadcast over the reduced/outer axes). The load index is role-aware (`in_i[idx]` vs `in_i[j]`) exactly as `emit_row_reduce`'s `load` closure (`cuda.rs:416-427`). For a plain (non-fused-epilogue) multi-input reduction the weight multiplies inside the fold; the same validate guard ("column input forbidden inside a stage" logic) applies — reuse `validate_row_reduce`'s expression walker rather than writing a second one. **Do not reimplement the OOB guards** — the bare-rank-1-`[k]` misclassification guard (`plan.rs:236-246`) is load-bearing and already tested (`rowreduce_bare_rank1_weight_rejected`).

### 5e. StructureKey: derive `reduce_axes` instead of hardcoding empty
`structure_key()` gains a reduction-aware branch: when `op == OpCategory::Reduction` (or a reduction-family category), compute `reduce_axes` from the operand shapes — the reduced axes are those where the **output** extent is 1 (keepdim) or absent (collapsed) while the **input** extent is > 1. This is the same "which axes collapsed" inference the doc's `op_class.reduction_axes(&canon)` hook (`:187`) intends. Guard it so **non-reduction ops stay `AxisMask::EMPTY`** (no behavior change for the elementwise pilot). The token already serializes this; the only new test is that a real reduction key now emits `…|x01` (or similar) instead of `…|-`.

### 5f. FKC / contract implications (honesty)
`emit_reduction` output must **not** become bindable-as-elementwise. Two honest options; **choose (i) for this item:**
- **(i) Preserve honest miss:** reductions continue to produce **no FKC contract** (`derive_pattern` still `NotElementwise` → `contract()` returns `None`). The `reduce_axes` key still tags the cell for **telemetry and runtime dispatch** (both consume `to_token`), so the miss is correctly *attributed to the right axis-set* even though no contract is advertised. This is the minimal honest step and keeps item 05 (the seam) as the place where reduction advertisement is designed.
- (ii) Advertise a reduction contract (a `cost.class: reduction`, an `accept.structure_key` carrying the real `reduce_axes`, no `pattern:` block) — **deferred to item 05**, noted here so the `reduce_axes`-in-token work done now is the prerequisite.

## 6. Implementation steps

1. **IR** (`ir.rs`): add `axes: AxisMask`, `keepdim: bool` to `Access::Reduction`; import `AxisMask`; keep `OpDef::reduction` delegating (empty mask / no keepdim); add `OpDef::reduction_axes(...)`. Update the `#[non_exhaustive]` doc comment to reflect the now-shipped scope.
2. **Plan** (`plan.rs`): add `ReduceAxisClass`; extend `Schedule::Reduction` with `class` + `keepdim`; classify in `build_plan` from `key.reduce_axes` + operand-0 contiguity. Reuse/extract the `RrRole` role classifier for the multi-input reduction path (share `rr_role`, do not fork it).
3. **Emitter** (`cuda.rs`): rewrite `emit_reduction`'s index generation to the kept-unravel / reduced-fold nest (reuse `offset_expr`, `is_fully_broadcast` from the strided path); role-aware load for multi-input (reuse the `emit_row_reduce` load-closure shape); Mean divisor = reduced-extent product; keepdim store-offset branch. Relax the `:289`/`:294` asserts (keep `:277` for item 04); keep the Max/Min NaN peel + f16/bf16 convert + accumulator-type logic untouched.
4. **Validation** (`plan.rs`): route multi-input reductions through the existing `validate_row_reduce`-style guards (share the walker + the bare-`[k]` OOB guard); add a `validate_reduction` for the plain path if the RowReduce validator's assumptions (epilogue, stages) don't fit cleanly — but reuse its guard *bodies*.
5. **Keying** (`structure_key.rs`): replace the hardcoded `reduce_axes: AxisMask::EMPTY` with a reduction-aware derivation gated on `OpCategory::Reduction` (+ any reduction-family category in scope); non-reduction ops unchanged. No codec change (already present).
6. **Contract honesty** (`contract.rs`/`pattern.rs`): **no code change** — verify (test) that a reduction op still yields `None` from `contract()` and `NotElementwise` from `derive_pattern`. Add a regression test pinning that.
7. **FFI/build wiring**: the generator binary (`bin/kernelgen.rs`) — add the new reduction cells to whatever the AOT catalog enumerates; ensure the emitted symbol name disambiguates axis-class (extend the `_reduce_{tag}` name, e.g. `_reduce_{tag}_ax{hex}` / `_kd`), so two kernels reducing different axis-sets never collide on one `__global__` symbol.
8. **Catalog / docs** (`OP-MATRIX.md`, `docs/design/kernel-specialization.md`): mark reductions' axis-position (#9) and keepdim as shipped in the generator; **correct the stale ORDER-3 status line** (`:432-435`) to reflect that `Access::Reduction` now carries axis/keepdim; do not touch the (separately stale) Param/AddScalar status text beyond flagging.

## 7. Test & on-device validation plan

**Unit (emitter source-shape, `cuda.rs` tests — mirror the existing `reduction_*` tests):**
- Last-axis reduction still emits **byte-identical** source to today (golden: the existing `reduction_mean_of_squares_f32` assertions must pass unchanged — proves the empty-mask default is a true no-op).
- Outer-axis reduction (reduce axis 0 of `[R, C]`): emits a fold over the outer axis with a `c*s0[0]`-style input offset and a `C`-wide output; assert no `base = o*k`.
- Multi-axis reduction (reduce `{0,1}` of `[A,B,C]` → `[C]`): nested reduced loops (or flattened), divisor = `A*B` for Mean.
- Keepdim: output offset unravels full rank with reduced-axis terms dropped (size-1 output axes); collapsed vs keepdim differ only in the store offset.
- Multi-input weighted reduction: role-aware `in1[j]` (per-column) not `in1[idx]`; reuse the `wrmsnorm` column-index assertions' shape.
- f16/bf16 outer-axis reduction still accumulates in float (convert on load/store) — reuse `reduction_f16_accumulates_in_float`'s pattern at a non-trivial axis.

**Keying (`structure_key.rs` tests):** a reduction key over `[R,C]`-reduce-last emits `…|xNN` with the right bit; reduce-axis-0 emits a *different* token than reduce-last (proves the honest miss); round-trip through `from_token` (extend `token_round_trips` with a non-empty `reduce_axes`); non-reduction op still emits `-`.

**Honesty (`contract.rs`/`pattern.rs` tests):** `derive_pattern(reduction_op) == Err(NotElementwise)`; `contract(reduction_op, …) == None`. Regression-pin so a future edit can't silently advertise a reduction as elementwise.

**nvrtc HEADERLESS compile (house discipline):** compile every new cell's emitted source with nvrtc, **no extra includes** beyond the fp16/bf16 operator headers the emitter already conditionally adds — outer-axis, middle-axis, multi-axis, keepdim, weighted, and each of f32/f16/bf16/f64. A compile failure here catches a malformed generated offset expression before device run.

**nvcc numeric on sm_89 (RTX 4070):** compile + launch each cell; diff against a **numeric oracle**:
- Oracle = the generic strided reduction sibling / a host reference (`torch.sum`/`mean`/`amax`/`amin` with the same `dim=` and `keepdim=`). The design doc's "differential-test every generated cell against the generic strided oracle" (`:338`) is the standing rule.
- Cases: outer / middle / multi axis; keepdim true/false; a strided (transposed) input; a broadcast weight input; empty-axis (`k==0`) guard; NaN-in-input for Max/Min (must propagate, matching `torch.amax`); Mean divisor exactness for multi-axis.

**compute-sanitizer:** the **plain reduction path is one-thread-per-output with no shared memory**, so `initcheck` (uninitialized reads on the strided offset math) + `memcheck` (the OOB the relaxed contiguity assert used to backstop — especially the too-short-weight case and keepdim store aliasing) are the relevant tools. `synccheck`/`racecheck` only if item 05's block-parallel outer-axis kernel is pulled forward (it is not, per §5b) — note that when it is, they become mandatory.

## 8. Adversarial-verify checklist

Run the multi-agent find → dedup → skeptic-refute pass after the change; probe specifically for:

- **`base = o*k` residue:** any surviving assumption that the reduced axis is the trailing contiguous run — outer/middle-axis offset math must not silently reduce the wrong axis (produces a plausible-but-wrong number the oracle catches, but a skeptic should find it in the emitted source first).
- **Mean divisor drift:** multi-axis Mean must divide by the **product** of reduced extents, not just the last; a single-axis `k` left in place is a silent 1/N error.
- **keepdim store aliasing / OOB:** with keepdim the output has size-1 reduced axes (stride 0) — verify two output elements never map to the same offset, and the store never runs past the (smaller) output buffer. This is a memcheck-visible UAF-class bug on a defensive path (the house has been bitten by a reintroduced UAF before).
- **NaN misroute (regression class):** the Max/Min peel's `e != e` NaN-propagation must survive the index rewrite — the house previously caught a `fmaxf` NaN misroute; confirm the compare-select (not `fmaxf`) is still emitted at the new axis geometry.
- **Weight-role misclassification:** the bare-rank-1-`[k]` weight OOB guard (`plan.rs:236-246`) must still fire — a strided/multi-axis reduction must not open a new path around it. Add the adversarial too-short-weight case.
- **Honest-miss leak:** confirm no code path lets a reduction op reach `derive_pattern`'s elementwise arms or produce a bindable contract; a strided reduction must not be advertisable as an elementwise cell (the `region_to_op` hardcode is the wall — verify it's untouched).
- **Empty / degenerate axes:** `k==0` (empty reduced axis) guard preserved (Sum→identity, Max/Min→no OOB seed read, `cuda.rs:353`/`476` semantics); a size-1 reduced axis (trivial reduce) must still be correct.
- **Determinism:** the sequential fold is order-fixed (one thread per output, no atomicAdd) — the house determinism guarantee. If any class is later moved to block-parallel, the tree-reduce order must be documented as the new deterministic order (per house discipline); for **this** item, confirm no atomicAdd was introduced.
- **`reduce_axes` derivation false-positive:** an elementwise op with a size-1 axis must **not** be mis-derived as a reduction (the gate must be `OpCategory`, not shape-shape inference alone).

## 9. Definition of done

- `Access::Reduction` carries `axes` + `keepdim`; `OpDef::reduction` output is byte-identical to pre-change (all existing `reduction_*` tests green, unmodified).
- Outer, middle, multi-axis, keepdim, strided-input, and multi-input (weighted) reductions emit correct CUDA; new unit tests green.
- `structure_key()` derives `reduce_axes` for reduction-category ops (non-empty token where axes are reduced), leaves non-reduction ops empty; token round-trips; new key tests green.
- **On-device validated on sm_89 (RTX 4070):** every new cell nvrtc-headerless-compiles and nvcc-numeric-matches the strided/host oracle across the §7 case matrix; compute-sanitizer `memcheck`/`initcheck` clean (including the keepdim-store and too-short-weight adversarial cases).
- **FKC honest-miss preserved:** reduction ops still yield no contract and `NotElementwise`; a regression test pins it; no elementwise-advertisement leak.
- **Determinism preserved:** one-thread-per-output sequential fold, no atomicAdd; documented if any class later goes block-parallel.
- Adversarial-verify pass run and clean (find → dedup → skeptic-refute); the §8 failure modes each probed.
- `OP-MATRIX.md` + `docs/design/kernel-specialization.md` updated: axis-position (#9)/keepdim marked shipped in the generator; the stale ORDER-3 reductions status line corrected.
- Lockstep release discipline honored (all crates bump + full republish per `publish_alpha*.ps1`) when this ships.

## 10. Open questions / Fuel asks

1. **`reduce_axes` on the wire (confirm, not blocker):** the `StructureKey` token already serializes `reduce_axes` (`x{:02x}`/`-`), and this item starts *populating* it for reduction cells. Confirm with Fuel that a now-non-empty `reduce_axes` field in a telemetry/`accept` token is expected and correctly consumed on their side (they call `structure_key`/`to_token`, so it should be transparent — this is a heads-up that reduction tokens change shape from `…|-` to `…|xNN`).
2. **Axis-set canonicalization agreement (for when 05 advertises reductions):** the reduced-axis *bit numbering* must match Fuel's canonical axis order (post-squeeze/collapse) or a Baracuda kernel keyed for `{axis 0}` won't join a Fuel request that numbers the same logical axis differently. For **this** item (no reduction contract emitted) it only affects telemetry attribution; it becomes load-bearing at item 05. Ask Fuel to confirm the canonical axis numbering the `reduce_axes` bits index (design doc §canonicalization step 3, `:145`, says reductions are permutation-invariant only *within* kept/reduced groups — pin the exact ordering).
3. **Keepdim representation vs item 01:** whether keepdim's broadcast-back output is best expressed via item 01's shape/layout nodes (preferred) or via the operand-stride-0 convention this item uses standalone. Decide when 01's shape-node shape is final; until then this item uses the stride-0 keepdim convention (self-contained, no 01 dependency for the non-keepdim cases).
4. **Outer-axis block-parallel kernel (internal, deferred):** design doc predicate #9 `Outer` wants a thread-per-column / block-parallel kernel and #10 wants a shared-memory-tree vs shuffle threshold from the CC. This item ships the sequential fold for all classes and reserves the `class` token; the block-parallel `Outer` kernel is a follow-up perf item — confirm that ordering (correctness-first) is acceptable, or pull it forward if a Fuel workload is bandwidth-bound on outer-axis reductions today.
