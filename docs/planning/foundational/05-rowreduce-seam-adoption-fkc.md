# 05 — RowReduce seam adoption + FKC contract — implementation brief

> Baracuda item 05 of the foundational kernel-specialization set. Scope: make the already-shipped fused row-reduce norms (RmsNorm / LayerNorm / weighted-RmsNorm / Softmax) **adoptable through the live §5 JIT seam** instead of AOT-only, by teaching the seam adapter to lower a `MeanDim/SumDim → broadcast → elementwise` region to `Access::RowReduce` and by emitting the FKC RowReduce contract + `pattern:` block. **Codegen is done; this is the seam *encoding*.**
>
> `blocked_by: fuel` — the exact wire encoding is pinned to Fuel's answers to the 4 asks in `docs/fuel-ask-fused-reduce-seam-2026-06-25.md`. A defined subset (RmsNorm/LayerNorm structural scaffolding) can be built and unit-tested behind the seam now; final wire-up and Softmax wait on Fuel.

---

## 1. Objective

Fuse RmsNorm / LayerNorm / weighted-RmsNorm / Softmax are **already synthesized and numerically validated on sm_89** — one block per row, warp-shuffle + shared-mem tree reduce, no per-op hand-written CUDA (`Access::RowReduce { stages, epilogue }`, codegen in `crates/baracuda-kernelgen/src/cuda.rs`, validated by the `nvrtc_compiles_rowreduce_kernels` test at `jit.rs:1280`). But they are **AOT-only**: the live §5 seam is elementwise-only, so a Fuel-chosen region containing a reduction honest-misses at two points — `region_to_op` hardcodes `Access::Elementwise` (`jit.rs:419`), and `derive_pattern` rejects any non-elementwise access with `NotElementwise` (`pattern.rs:80-82`), which makes `contract()` return `None` (`contract.rs:79`). This item wires the region→`RowReduce` lowering into the seam adapter and emits the FKC `RowReduce` contract + `pattern:`, so a fused-reduce region flows *region in → kernel + recipe + link out → cost-gated adoption*. It is foundational because it is the **only** path by which the shipped fused norms reach the runtime through Fuel's strategist; it directly unlocks FKC §8's `RmsNorm` / `FusedLinear`-adjacent norm targets and is the template every later fused-reduce family (residual-add LayerNorm, multi-axis) reuses.

---

## 2. Status & blockers

**Baracuda-unblocked (codegen + AOT validation): DONE.** `Access::RowReduce`, `ReduceStage`, `Reduced(i)` (`ir.rs:264-305`), the `RrRole`/`validate_row_reduce` legality gate (`plan.rs:129-293`), and the block-parallel `Schedule::RowReduce` codegen (`plan.rs:45`, `cuda.rs`) all ship. RmsNorm (1 stage), Softmax (2 stages), weighted-RmsNorm + LayerNorm (multi-input, per-column weight/bias) all compile headerless under nvrtc on sm_89 (`jit.rs:1280-1373`).

**Fuel-blocked (the seam *encoding*): the 4 asks in `docs/fuel-ask-fused-reduce-seam-2026-06-25.md`.** The seam consumes Fuel's *frozen* grammar (`fuel-kernel-seam-types` v0.10.2, `Cargo.toml:29`). We cannot finalize the region→`RowReduce` recognizer or the emitted `pattern:` until Fuel confirms:
- **(a)** the `MeanDim`/`SumDim` + `OpAttrs.axis = Some(-1)` + shared-`Bind` + `AddScalar{eps}` encoding, broadcast-back implicit at the consuming binary op. *Unblocks RmsNorm + LayerNorm recognition.*
- **(b)** Softmax's last-axis-max spelling — the frozen `OpTag` has `MeanDim`/`SumDim` but **no** `MaxDim` (only `MaxAll`/`ReduceMaxTo`). *Unblocks Softmax only.*
- **(c)** whether `match_region` matches `reduce → (implicit broadcast) → elementwise` as-is, or needs an explicit `BroadcastTo` node between the reduction and its consumer. *Governs whether our `pattern:` emits `BroadcastTo` and whether `region_to_op` must consume one.* *Unblocks RmsNorm + LayerNorm.*
- **(d)** the cost-expr variables for a row-reduce op — is a per-row `k` binding available in Fuel's `cost_expr` core, or must cost stay in `n`? *Governs the emitted `cost` string; a correct-but-coarse `n`-only fallback exists, so this is refinement, not a hard block.*

**What proceeds now (design-open → build-ready, no Fuel answer required):**
1. The IR-facing recognizer `region_to_row_reduce` (region `PatternNode` → `Access::RowReduce { stages, epilogue }`) — a pure function over *our* internal `PatternNode`, testable with hand-built nodes independent of the frozen grammar's final tag names. Structurally, RmsNorm and LayerNorm need only ask (a) + (c) to be *pinned*; the lowering logic (reduction node → stage, rest → epilogue, reduction result → `Reduced(i)`) is grammar-name-agnostic and can be written against a small internal alias set today.
2. `derive_pattern` / `contract` RowReduce branch: teach them to emit a `pattern:` for a `RowReduce` op instead of rejecting it (`pattern.rs:80`, `contract.rs`). The *serialization* is Baracuda-internal; only the reduction node's **spelling** (`MeanDim` vs `ReduceMaxTo` vs a new `MaxDim`, ask b) and the presence of an explicit `BroadcastTo` (ask c) are Fuel-pinned. Emit behind those two as named constants so pinning is a one-line change.
3. Full unit-test scaffolding + the AOT numeric oracle diff (already have the on-device kernels).

**Cannot ship to the live `BaracudaSynthesizer` seam call until (a)+(c) [RmsNorm/LayerNorm] and additionally (b) [Softmax] are answered** — because the `optag_name` map (`jit.rs:660-697`) and the emitted `pattern:` tag names must match Fuel's frozen grammar byte-for-byte or `match_region` will never fire.

---

## 3. Dependencies & sequencing

**Must land before / consumed by this item:**
- The `Access::RowReduce` IR + codegen + `validate_row_reduce` (DONE — `ir.rs`, `plan.rs`, `cuda.rs`). This item consumes them unchanged.
- The elementwise §5 seam (`synthesize` / `synthesize_op` / `region_to_op` / `BaracudaSynthesizer`, `jit.rs`) — DONE; this item extends `region_to_op` and the `Synthesizer::synthesize` category/cost logic.

**Coexists with (no ordering constraint, disjoint touch points):**
- **Item 01** (layout / shape nodes) — if Fuel answers ask (c) with an explicit `BroadcastTo`, item 01's shape-node handling and this item's `region_to_op` must agree on the `BroadcastTo` spelling; otherwise disjoint.
- **Item 03** (strided reductions) — extends `Access::Reduction`/`RowReduce` to strided inputs; this item stays on the v1 contiguous-last-axis case and inherits item 03's strided support later for free.

**Unlocks downstream:**
- Fused norms adoption through the §5 JIT seam → FKC §8 `RmsNorm` and the norm half of `FusedLinear` targets.
- The recognizer + contract shape is the reusable template for residual-add LayerNorm (a second row-streamed input) and multi-axis reductions (both explicitly scoped as follow-ups in the Fuel-ask §Scope).

---

## 4. Current code — what exists today

### 4.1 `region_to_op` hardcodes `Access::Elementwise` — `jit.rs:406-425`

```rust
fn region_to_op(
    region: &PatternNode,
    n_inputs: u8,
    name: &str,
    dtype: ElementKind,
) -> Result<(OpDef, PatternNode), JitError> {
    let mut next_param = 0u8;
    let body = node_to_expr(region, &mut next_param)?;
    let op = OpDef {
        name: name.to_string(),
        n_inputs,
        body,
        dtypes: vec![dtype],
        access: Access::Elementwise,          // <-- hardcoded; no reduction path
    };
    let pattern = derive_pattern(&op)?;       // <-- rejects non-elementwise (see 4.2)
    Ok((op, pattern))
}
```

`node_to_expr` / `synth_op` (`jit.rs:427-466`) only knows infix arithmetic, unary math, and non-infix binary fns — there is **no** case for a reduction node, so a `MeanDim`/`SumDim` op-name would fall through to `JitError::UnsupportedOp` (`jit.rs:462`). The seam front-end's `optag_name` (`jit.rs:660-697`) likewise returns `None` for every reduction tag (the `_ => return None` arm at `jit.rs:695`, whose comment explicitly lists "reductions … — not synthesized").

### 4.2 `derive_pattern` rejects non-elementwise — `pattern.rs:79-82`

```rust
pub fn derive_pattern(op: &OpDef) -> Result<PatternNode, PatternError> {
    if !matches!(op.access, Access::Elementwise) {
        return Err(PatternError::NotElementwise);   // <-- RowReduce honest-miss here
    }
    ...
}
```

`walk` has a defensive `Reduced(_) => Err(ScalarParamUnsupported)` arm (`pattern.rs:194`) that "never fires on the elementwise path" — that comment is the marker for exactly what changes. Consequently `contract()` (`contract.rs:58-80`) calls `derive_pattern(op).ok()` → `None` → the `op_line` match hits `None => return None` (`contract.rs:79`): **a RowReduce op advertises no contract at all today.**

### 4.3 The RowReduce codegen that already exists (consumed unchanged)

- **IR:** `Access::RowReduce { stages: Vec<ReduceStage>, epilogue: ScalarExpr }` (`ir.rs:299-304`); `ReduceStage { pre, op: ReduceOp }` (`ir.rs:264-271`); `ScalarExpr::Reduced(u8)` — the per-row reduced scalar leaf, legal only inside a RowReduce (`ir.rs:27-34`); `OpDef::row_reduce(...)` sets `body = epilogue` so existing body-walkers work unchanged (`ir.rs:367-385`).
- **Schedule:** `Schedule::RowReduce { n_stages, block }` (`plan.rs:45-50`); `build_plan` routes `Access::RowReduce` → validate + block-parallel tree reduce (`plan.rs:90-99`).
- **Legality gate:** `validate_row_reduce` (`plan.rs:174-293`) — float dtype, `x` (input 0) row-streamed + contiguous, inputs 1.. per-column `[k]` weight/bias broadcasting every outer axis (not reversed, not a bare rank-1 `[k]`), full-width contiguous output, expression legality (`Reduced(s)` only references produced stages, no `Param`, no column input inside a `stage.pre`). `RrRole::{RowStreamed, ColBroadcast}` + `rr_role` classify by broadcast mask (`plan.rs:129-149`).
- **The confirmed caller pre-condition** (`plan.rs:166-173`, Fuel-ask §4): the structure key carries broadcast masks but **no numeric extents**, so a too-short weight/bias keys identically to a correct one and reads OOB. `validate_row_reduce` cannot check `weight.extent[-1] == x.extent[-1]`; the seam caller (holding live `OperandDesc` extents) must assert it before the request crosses.

### 4.4 The live seam call — `BaracudaSynthesizer::synthesize` — `jit.rs:765-827`

Today derives `op_category` from operand arity only (`UnaryElementwise`/`BinaryElementwise`/`TernaryElementwise`, `jit.rs:779-783`) and emits an elementwise `cost = "n * (n_inputs+1)"` (`jit.rs:802`). Neither is reduction-aware. `region_op_id` (`jit.rs:835-844`) already hashes region + operands into a stable `entry_point`, reused as-is.

---

## 5. Design / delta

### 5.1 The region → `RowReduce` lowering (`region_to_op`, `jit.rs`)

Split `region_to_op` into an elementwise path (unchanged) and a new `region_to_row_reduce` recognizer, dispatched on whether the region contains any last-axis reduction node:

```rust
fn region_to_op(region: &PatternNode, n_inputs: u8, name: &str, dtype: ElementKind)
    -> Result<(OpDef, PatternNode), JitError>
{
    if contains_reduce(region) {
        return region_to_row_reduce(region, n_inputs, name, dtype);
    }
    // ... existing elementwise path (Access::Elementwise) ...
}
```

`region_to_row_reduce` walks the region and:
1. Collects each last-axis reduction node (`MeanDim`/`SumDim`, and — pending ask b — the Softmax max node) in dataflow order → one `ReduceStage { pre, op }` per node, where `pre` is the reduction node's operand lowered to `ScalarExpr` and `op` is `ReduceOp::{Mean, Sum, Max}` (`ir.rs:117-127`, matching `MeanDim → Mean`, `SumDim → Sum`).
2. Replaces each reduction node **in the surrounding expression** with a `ScalarExpr::Reduced(i)` leaf (`i` = its stage index) — this is exactly the "reduced scalar broadcast across the row" that `Reduced` models (`ir.rs:27-34`), so the implicit broadcast-back at the consuming binary op (Fuel-ask §3 / ask c) requires **no** node in our IR: it *is* the `Reduced(i)` leaf.
3. The remaining top-level expression (reductions substituted) → the `epilogue`.
4. Builds `OpDef::row_reduce(name, n_inputs, &[dtype], stages, epilogue)` (`ir.rs:367`) → `Access::RowReduce`.

Worked mapping (RmsNorm, Fuel-ask §1):

```
region:   Mul( Bind0, Rsqrt( AddScalar{eps}( MeanDim{axis:-1}( Sqr( Bind0 ) ) ) ) )
              │                                  └─ reduction node → stage 0
              │
lowers to: stage[0] = ReduceStage { pre: Sqr(Input(0)), op: Mean }
           epilogue = Mul( Input(0), Rsqrt( Add( Reduced(0), Const(eps) ) ) )
```

Note `AddScalar{eps}` lowers to `Add(_, Const(eps))` — the fused eps is a **compile-time constant** in a RowReduce (`validate_row_reduce` forbids `Param`, `plan.rs:274-276`; eps rides as `Const`). This is a delta from the elementwise path, where `AddScalar` → runtime `Param`. The recognizer must fold a scalar-op eps attribute into `Const`, not `Param`, when it sits between a reduction and the epilogue.

Two-stage Softmax (pending ask b) maps the same way: stage 0 = `Max` over `Input(0)`, stage 1 = `Sum` over `Exp(Sub(Input(0), Reduced(0)))`, epilogue = `Div(Exp(Sub(Input(0), Reduced(0))), Reduced(1))` — matching the `sm(dt)` fixture at `jit.rs:1303-1317`.

**StructureKey / op_category:** the seam call (`jit.rs:779`) must key a row-reduce region as `OpCategory::Normalization` (RmsNorm/LayerNorm/weighted-RmsNorm) or `OpCategory::Softmax` (`sku.rs:54-59`; tokens `nrm`/`sft`, `structure_key.rs:869-870`), **not** an elementwise category — the schedule legality (`Schedule::RowReduce`) is category-gated. Derive it from the reduction op kind: any `Max` stage → `Softmax`; else `Normalization`.

### 5.2 The FKC RowReduce contract + `pattern:` (`pattern.rs`, `contract.rs`)

Replace the blanket `NotElementwise` reject (`pattern.rs:80-82`) with a RowReduce-aware branch. The `pattern:` for a RowReduce op must describe the **primitive subgraph it replaces** (Fuel's `match_region` recognizes it and `decompose:` expands back). Structurally that subgraph is: reduction node(s) over the shared input, feeding — via the implicit-broadcast consuming op (ask c) — the epilogue arithmetic, with the reduced input a repeated `bind: i` (the node-identity guard, `pattern.rs:12-14`).

Reconstruct the pattern tree from `Access::RowReduce { stages, epilogue }` by re-inflating each `Reduced(i)` leaf in the epilogue back into the reduction node (`op: MeanDim`/`SumDim`/`<max-spelling>` with `OpAttrs.axis = -1`) wrapping `stage[i].pre`, then walking the whole thing with the existing `walk`/`node_lines` machinery (`pattern.rs:181-397`). Two Fuel-pinned knobs, isolated as named constants so pinning is a one-line change:
- **`REDUCE_MAX_OP`** — the last-axis-max op name (ask b): `"MaxDim"` if Fuel adds it, else `"ReduceMaxTo"` with the `[…,1]` target in attrs.
- **`BROADCAST_NODE`** — `None` if `match_region` handles the implicit broadcast (ask c, our lean), else emit an explicit `"BroadcastTo"` node between each reduction and its consumer.

`contract.rs` changes: `derive_pattern(op)` now returns `Ok(_)` for a RowReduce, so `contract()` (`contract.rs:58-154`) emits `fused_op: <name>` + the `pattern:` block (a RowReduce is always ≥2 graph ops → `is_fusion`). The `cost.class` should read `reduction`/`normalization` rather than the hardcoded `elementwise` (`contract.rs:137`), and `determinism: bitwise` stays load-bearing and **correct**: the shipped kernel is one-block-per-row with a warp-shuffle/shared-mem tree reduce and **no `atomicAdd`**, so it is deterministic (do not weaken this line). `precision`: the `rsqrt`/`exp` transcendentals relax `mode` to `approximate` with the existing `ulp_bound` (`contract.rs:223-263`), which already walks `Reduced` as a 0-ulp leaf.

### 5.3 Cost (ask d) — `jit.rs:802`

A row-reduce kernel is one launch, ~`(n_stages + 1)` passes over each row of extent `k`; the primitive path is several full-pass reduction + broadcast + elementwise launches. Preferred: `cost = format!("n * {}", n_stages + 1)` with `n` = out-elem count (each pass touches the row once), refined to a `k`-aware form (`n_stages * n + n` reads + `n` writes) **if** Fuel's `cost_expr` core binds a per-row `k` (ask d). The `n`-only form is a correct, conservative fallback that unblocks adoption regardless of Fuel's answer.

### 5.4 Honest-miss discipline (unchanged invariant)

Everything outside the confirmed v1 envelope must stay an **honest miss**, never a fake contract or a panic across the trait boundary: non-last-axis reductions, mixed dtype (`MixedDtype`, `jit.rs:310-314`), a reduction over a per-column operand, a bare rank-1 `[k]` weight, or an unspellable dtype (`fkc_dtype → None`, `contract.rs:329`). `region_to_row_reduce` must route each to a typed `JitError` (→ `SeamResponse::Declined`), preserving the property that the planner's miss signal is honest by construction. The extent pre-condition (§4.3) is a *caller* assert, not a synthesizer check — document it at the new recognizer exactly as `validate_row_reduce` documents it.

---

## 6. Implementation steps

1. **`jit.rs` — `contains_reduce` + dispatch.** Add a region predicate (any node whose op-name is a last-axis reduction) and branch `region_to_op` to `region_to_row_reduce` when true. *(edits `crates/baracuda-kernelgen/src/jit.rs`)*
2. **`jit.rs` — `region_to_row_reduce`.** Walk region → collect reduction nodes to `ReduceStage`s (dataflow order), substitute each with `Reduced(i)`, build the epilogue, fold eps `AddScalar` → `Const`, construct `OpDef::row_reduce`. Route all out-of-envelope shapes to typed `JitError`. *(jit.rs)*
3. **`jit.rs` — reduction op-name aliases.** Add the internal names the recognizer parses (`"MeanDim"`, `"SumDim"`, and `REDUCE_MAX_OP`) to `node_to_expr`/`synth_op`'s vocabulary *as reduction markers only* (they must not fall through to `UnsupportedOp` inside a row-reduce region). *(jit.rs)*
4. **`jit.rs` — `optag_name` (seam front-end).** Add `OpTag::MeanDim`/`SumDim` (+ the ask-b max tag) → the internal reduction names, replacing the `_ => return None` miss for those tags (`jit.rs:695`). Guard behind the frozen-grammar tag names. *(jit.rs)*
5. **`pattern.rs` — RowReduce branch.** Replace the `NotElementwise` reject (`pattern.rs:80-82`) with reconstruction of the pattern tree from `Access::RowReduce` (re-inflate `Reduced(i)` → reduction node with `axis:-1`, optional `BROADCAST_NODE`). Emit the two Fuel-pinned knobs as named constants. *(crates/baracuda-kernelgen/src/pattern.rs)*
6. **`contract.rs` — cost class + fused_op path.** Let a RowReduce op reach the `fused_op:` + `pattern:` branch; set `cost.class` to `reduction`/`normalization`; keep `determinism: bitwise`. *(crates/baracuda-kernelgen/src/contract.rs)*
7. **`jit.rs` — `BaracudaSynthesizer` category + cost.** Derive `OpCategory::Normalization`/`Softmax` for row-reduce regions (from the stage ops) and emit the `n_stages`-aware `cost` (§5.3). Add the extent pre-condition doc + (where the caller holds extents) the `weight.extent[-1] == x.extent[-1]` assert. *(jit.rs)*
8. **Tests.** Unit tests (§7) for `region_to_row_reduce`, the emitted `pattern:`/contract, honest misses, and the seam round-trip; extend the ignored on-device tests with the numeric oracle diff. *(jit.rs, pattern.rs, contract.rs test mods)*
9. **Release.** Lockstep bump + full republish of all crates (house discipline) once the Fuel-pinned knobs are answered and wired.

---

## 7. Test & on-device validation plan

**Unit (host, no CUDA):**
- `region_to_row_reduce` maps the RmsNorm region (§5.1) to `stages == [ReduceStage { pre: Sqr(Input(0)), op: Mean }]`, `epilogue == Mul(Input(0), Rsqrt(Add(Reduced(0), Const(eps))))`. Assert eps is `Const`, not `Param`.
- Two-stage Softmax region → `[Max, Sum]` stages + the `Div(.., Reduced(1))` epilogue (matches the `sm` fixture at `jit.rs:1303`).
- LayerNorm / weighted-RmsNorm regions → multi-input `RowReduce` with the per-column weight/bias as inputs 1.. (matches the `wrms`/`ln` fixtures at `jit.rs:1341-1364`).
- `derive_pattern` now returns `Ok` for a `RowReduce` op; `to_fkc` emits the reduction node with `axis:-1`, the repeated shared `bind`, and (per ask c) the correct broadcast shape. Golden-string assertions on the `pattern:` YAML.
- `contract()` for a RowReduce op emits `fused_op:` + `pattern:` + `cost.class` reduction/normalization + `determinism: bitwise`; returns `None` for an unspellable dtype (honest miss preserved).
- Honest-miss coverage: non-last-axis reduction, mixed dtype, reduction over a column operand, bare rank-1 `[k]` weight — each a typed `JitError` / `Declined`, never a panic (mirror `synthesizer_declines_never_panics`, `jit.rs:939`).
- Seam round-trip: a `SeamRequest` carrying an RmsNorm region → `SeamResponse::Synthesized` with the row-reduce `entry_point`, `Normalization` category, and the `n_stages`-aware cost; `take_kernel` retrieves the artifact.

**On-device (mandatory for kernel changes — house discipline):**
- **nvrtc headerless compile on sm_89 (RTX 4070):** the fused RmsNorm/Softmax/LayerNorm/weighted-RmsNorm kernels the seam now emits compile headerless (already covered by `nvrtc_compiles_rowreduce_kernels`, `jit.rs:1280` — extend it to compile via the *seam* path, proving the region→kernel→PTX round-trip, not just direct `OpDef::row_reduce`). f32 + f16 (`__half2float` + `cuda_fp16.h`) + f64/f32-strict (double `__shfl_down_sync`) all in the matrix.
- **nvcc numeric on sm_89:** the numeric oracle is the reference math evaluated in f64 — RmsNorm `x·rsqrt(mean(x²)+eps)`, LayerNorm `(x−μ)·rsqrt(var+eps)·w+b`, Softmax `exp(x−max)/Σexp(x−max)`. Build the seam-emitted kernel via the nvcc host harness, run it against random `[n_out, k]` inputs, diff against the f64 oracle within the declared `max_ulp`. This is the existing AOT oracle; the delta to prove is only that the **seam-lowered** `OpDef` is byte-identical to the AOT one.
- **compute-sanitizer:** RowReduce uses shared memory + cross-thread warp shuffles, so run `--tool memcheck` and `--tool racecheck` on the seam-emitted RmsNorm and (crucially) the multi-input LayerNorm — the per-column `in_i[j]` index is the confirmed OOB surface (§4.3). Prove clean with correctly-sized operands; the mismatch case is a caller pre-condition, not a kernel bug.

---

## 8. Adversarial-verify checklist

A skeptic pass (run after every substantive change — house discipline) must probe:

- **Wrong stage order / dependency.** A Softmax whose `Sum` stage reads `Reduced(0)` (the max) before it is produced must be caught by `validate_row_reduce` (`plan.rs:270`), not silently mis-lowered. Confirm the recognizer emits stages in dataflow order.
- **eps folded as `Param`, not `Const`.** If `region_to_row_reduce` leaves eps as a runtime `Param`, `validate_row_reduce` panics (`plan.rs:274-276`) — an AOT backstop, but on the JIT boundary a panic *crosses the trait* and crashes the host. Assert eps → `Const` and that a stray `Param` in a row-reduce region is a typed `Declined`, never a panic (the exact class the `synthesizer_declines_unlowerable_dtype_never_panics` regression guards, `jit.rs:951`).
- **Column-operand OOB.** A LayerNorm weight with `extent[-1] != x.extent[-1]` keys identically to a correct one and reads OOB (`plan.rs:166`). Prove: (i) the seam caller asserts the extent equality before the request crosses; (ii) compute-sanitizer is clean with matched extents. Do not claim the synthesizer can detect the mismatch — it structurally cannot.
- **Bare rank-1 `[k]` weight misclassifies as row-streamed.** `rr_role` treats an empty bcast as `RowStreamed` (`plan.rs:143`); a rank-1 `[k]` weight would read `in_i[row*k+j]` past its buffer. Confirm inputs 1.. are rejected unless per-column broadcast (`plan.rs:240-246`).
- **`pattern:` tag / broadcast mismatch with Fuel.** If `REDUCE_MAX_OP` or `BROADCAST_NODE` don't match Fuel's frozen grammar exactly, `match_region` silently never fires — a *quiet* adoption failure, not a crash. Golden-test the emitted YAML against Fuel's confirmed encoding once (a)–(c) land.
- **Determinism regression.** Confirm the emitted kernel is one-block-per-row with **no `atomicAdd`** (grep the generated source) — the `determinism: bitwise` contract line is load-bearing and must not be weakened to `atomic`/`nondeterministic`.
- **Elementwise regression.** A plain elementwise region (no reduction) must still take the unchanged `Access::Elementwise` path and produce a byte-identical contract to today (the `contains_reduce` dispatch must not perturb the elementwise case). Re-run the full existing `jit.rs`/`pattern.rs`/`contract.rs` test suites.
- **Category mis-key.** A row-reduce region keyed as an elementwise `OpCategory` would select the wrong (elementwise) schedule and mis-lower. Assert `Normalization`/`Softmax` keying.

---

## 9. Definition of done

- [ ] `region_to_op` dispatches a reduce-bearing region to `region_to_row_reduce`, which lowers RmsNorm / weighted-RmsNorm / LayerNorm (and, once ask b lands, Softmax) regions to `Access::RowReduce` with correct stages + `Reduced(i)` epilogue + `Const` eps.
- [ ] `derive_pattern` + `contract` emit a valid FKC `fused_op:` + `pattern:` block for a `RowReduce` op (reduction node with `axis:-1`, shared `bind`, ask-c broadcast shape), replacing the `NotElementwise` / `None` honest-miss; `cost.class` reflects the reduction, `determinism: bitwise` retained.
- [ ] `BaracudaSynthesizer::synthesize` keys row-reduce regions as `Normalization`/`Softmax` and emits an `n_stages`-aware cost; the extent pre-condition is documented and (caller-side) asserted.
- [ ] Every out-of-envelope shape is a typed `JitError` → `SeamResponse::Declined` — **no panic crosses the trait boundary**, no fake contract.
- [ ] Unit tests green; the seam-path kernels compile headerless under nvrtc on sm_89; nvcc numeric diff vs the f64 oracle within `max_ulp`; compute-sanitizer clean on the shared-mem/cross-thread LayerNorm path.
- [ ] The two Fuel-pinned knobs (`REDUCE_MAX_OP`, `BROADCAST_NODE`) are set to Fuel's confirmed answers and the emitted `pattern:` is golden-tested against Fuel's encoding.
- [ ] Lockstep release: all crates bump + full republish.

---

## 10. Open questions / Fuel asks

All four live in `docs/fuel-ask-fused-reduce-seam-2026-06-25.md`; this item is blocked on them.

- **Ask (a) — region encoding.** Confirm `MeanDim`/`SumDim` with `OpAttrs.axis = Some(-1)`, shared `Bind` (repeated index = node-identity guard), `AddScalar{eps}`, broadcast-back implicit at the consuming binary op. **Unblocks:** RmsNorm + LayerNorm recognition in `region_to_row_reduce` and the emitted `pattern:`.
- **Ask (b) — Softmax last-axis max.** The frozen `OpTag` has no `MaxDim` (only `MaxAll`/`ReduceMaxTo`). Pin the spelling: (i) `ReduceMaxTo` with `[…,1]` target in attrs, or (ii) a new `MaxDim` (our lean — cleanest mirror of `MeanDim`/`SumDim`, and `ReduceOp::Max`/`Min` already exist, `ir.rs:118-127`). Sets `REDUCE_MAX_OP`. **Unblocks:** Softmax only (RmsNorm + LayerNorm ship without it).
- **Ask (c) — reduce→broadcast→elementwise matching.** Does `match_region` match a `MeanDim` result feeding a broadcasting `Mul`/`Sub`/`Div` implicitly, or does the region need an explicit `BroadcastTo` node? Sets `BROADCAST_NODE` (and whether `region_to_row_reduce` must *consume* a `BroadcastTo`). **Unblocks:** RmsNorm + LayerNorm (the epilogue shape and `Reduced(i)` re-inflation depend on it).
- **Ask (d) — cost variables.** Is a per-row `k` binding available in Fuel's `cost_expr` core, or must cost stay in `n`? Sets the emitted `cost` string. **Unblocks:** cost-gating refinement — a correct `n`-only fallback exists, so this is the only ask that does *not* hard-block adoption.
