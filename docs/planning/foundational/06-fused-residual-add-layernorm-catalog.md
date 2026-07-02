# 06 — Fused residual-add LayerNorm + AOT catalog breadth — implementation brief

## 1. Objective

Add a **fused residual-add** to the RowReduce norm family: a *second row-streamed
input* (`residual`, shape `[n_out, k]`, same layout as `x`) that is summed with `x`
**before** the reduction/normalization, so a transformer block's
`y = LayerNorm(x + residual) * w + b` collapses into one kernel launch instead of an
elementwise `Add` kernel feeding a norm kernel. This is distinct from the
already-shipped per-column `[k]` weight/bias (`ColBroadcast` role) — it is a *new
operand role*, `Residual`, whose feature axis varies (it is read `in_i[idx]`, not
`in_i[j]`) and which is legal to reference inside a reduction stage `pre` (because it
enters the statistics). It is foundational because it (a) proves the RowReduce role
machinery (`RrRole`) generalizes past the single-row-streamed-input assumption baked
into `validate_row_reduce`, unblocking every residual-fused normalizer (RmsNorm,
LayerNorm) that dominates real LLM blocks, and (b) motivates broadening the AOT
catalog (`bin/kernelgen.rs`) so the enumerated cells actually cover the shapes Fuel
will request. The residual-add is the last elementwise op that survives *outside* the
current fused norms; folding it in removes the final pre-norm launch.

## 2. Status & blockers

**Baracuda-unblocked (build the whole codegen half now).** The residual-add is a pure
AOT emitter + validator change; it never crosses the JIT trust boundary because
RowReduce is not in the elementwise `region_to_op` seam yet (jit.rs:406-425 hardcodes
`Access::Elementwise`; the RowReduce seam is item 05, itself Fuel-blocked). So the IR
variant/role, `validate_row_reduce`, `emit_row_reduce`, the AOT catalog entry, and the
on-device validation can all land and republish independently — exactly like the
already-shipped multi-input weight/bias RowReduce did.

**Fuel-blocked (the seam wiring only).** Making a residual-add norm *adoptable through
the §5 seam* depends on item 05, which is blocked on the four asks in
`docs/fuel-ask-fused-reduce-seam-2026-06-25.md` (a: `MeanDim`+axis+shared-`Bind`
encoding; b: Softmax last-axis-max spelling; c: `match_region` reduce→broadcast→
elementwise; d: cost-expr variables). The residual-add adds a *fifth* implicit ask —
the region encoding of a **two-tensor add feeding the reduction** (an `Add(Bind0,
Bind1)` node whose result is shared by the stat reduction and the epilogue) — but that
ask is only relevant to item 05's seam path, and that doc's Scope (line 98) already
names "a second row-streamed input (fused residual-add LayerNorm)" as the follow-up.
**Do not** wire the seam here. Build AOT + contract emission; leave a documented TODO
in `region_to_op`.

**Design-open:** whether the fused output should optionally *also* write back the
pre-norm sum `x + residual` (some blocks reuse it as the next residual). See §10.

## 3. Dependencies & sequencing

**Must land before this:** nothing new. It builds directly on the **shipped multi-input
RowReduce** (`RrRole::ColBroadcast` + `validate_row_reduce`, plan.rs:131-293; the
multi-input `emit_row_reduce` load path, cuda.rs:416-427). It does **not** depend on
item 01 (layout/shape nodes) or item 02 (DAG): the residual is another full-width
contiguous operand, indexed exactly like `x`, so it needs no new shape facts and no
shared-interior DAG (the residual sum `x+residual` is recomputed in the stage and the
epilogue, the same tree-recompute the current epilogue already does for `x`).

**What this enables / relates to downstream:**
- **05 (RowReduce seam + FKC contract):** this adds the residual case to the set of
  RowReduce ops that 05 must teach `region_to_op` + FKC `pattern:` to encode; sequence
  05 to cover it (the `Add(Bind0, Bind1)`-before-reduce shape) once Fuel answers.
- **06's own catalog half** feeds **07 (per-arch dispatch table + bench-gate)** and
  **08 (telemetry variant-selection)**: a broader enumerated catalog is the corpus
  those harnesses sweep and rank. Coordinate the catalog-shape enumeration with 07 so
  the bench harness and the generator iterate the *same* cell list.
- **09 (f16/bf16 half2)** touches the same `emit_row_reduce` load/store path; the
  residual load is one more site the packed path must cover — sequence 09 after this
  so it sees the final role set.
- Independent of **03/04/10**.

## 4. Current code — what exists today

**IR (`crates/baracuda-kernelgen/src/ir.rs`).** `Access::RowReduce { stages: Vec<ReduceStage>, epilogue: ScalarExpr }` (ir.rs:299-304). A `ReduceStage { pre: ScalarExpr, op: ReduceOp }` (ir.rs:264-271). `ScalarExpr` leaves relevant here: `Input(u8)`, `Const(f64)`, `Reduced(u8)` (ir.rs:17-47). `OpDef::row_reduce(name, n_inputs, dtypes, stages, epilogue)` sets `body = epilogue` (ir.rs:367-385). There is **no operand-role field** on the IR — role is inferred *entirely* from the structure key's broadcast mask at emit/validate time.

**Role classification (`plan.rs`).** The one place roles are assigned:
```rust
// plan.rs:143-149
pub(crate) fn rr_role(o: OperandKey) -> RrRole {
    if o.bcast.is_empty() { RrRole::RowStreamed } else { RrRole::ColBroadcast }
}
```
`RrRole` has exactly two variants (plan.rs:131-136): `RowStreamed` (empty bcast → `in_i[base+j]`) and `ColBroadcast` (any bcast → `in_i[j]`). **A residual operand has an empty bcast** (it is a full `[n_out,k]` contiguous tensor), so today it classifies as `RowStreamed` — which is what we want for the *index*, but `validate_row_reduce` then **rejects it**:
```rust
// plan.rs:240-246  — inputs 1.. MUST be column-broadcast
for i in 1..n {
    assert!(is_col[i],
        "RowReduce input {i} must be a per-column [k] weight/bias ... not a bare row-streamed tensor");
}
```
and the comment at plan.rs:237-239 states the intent explicitly: *"A second row-streamed input — residual fusion — is a deliberate follow-up."* Also relevant: the stage-`pre` check forbids a **column** input inside a stage (plan.rs:262-268, `in_stage` guard) — but a residual input **must** be allowed inside a stage, since it enters the statistics. And Input0 is asserted non-column (plan.rs:232-235).

**Emitter (`cuda.rs`).** `emit_row_reduce` (cuda.rs:397-531). The load index is role-aware:
```rust
// cuda.rs:416-427
let load = |i: u8| {
    let pos = match rr_role(plan.key.operands[i as usize]) {
        RrRole::RowStreamed => "idx",       // base + j
        RrRole::ColBroadcast => "j",        // per-column, same every row
    };
    /* f16/bf16/f32-strict up-convert, else in{i}[{pos}] */
};
```
Stage folds emit `for (j = threadIdx.x; j < k; j += blockDim.x) { idx = base + j; acc_i += pre; }` then a `block_sum/max/min` broadcast to `r{i}` (cuda.rs:480-515). The epilogue re-streams `out[idx] = stored` (cuda.rs:517-528). A residual input is *already indexed correctly by this loader* (empty bcast → `idx`) — the only blocker is `validate_row_reduce`.

**Pattern / contract.** `derive_pattern` rejects non-elementwise (`Access::RowReduce` → `PatternError::NotElementwise`, pattern.rs:80-82), so `contract()` (contract.rs:58-154) returns `None` for **every** RowReduce op today (contract.rs:68-80: `derive_pattern(op).ok()` → `None` → `return None`). RowReduce norms currently emit **no FKC contract at all** — an honest miss on the seam (this is correct and must be preserved until item 05).

**AOT catalog (`bin/kernelgen.rs`).** The enumeration (kernelgen.rs:104-167) builds `rmsnorm` (1 input), `softmax` (1 input, 2 stage), `wrmsnorm` (2 input: x + col weight), `layernorm` (3 input: x + col weight + col bias). Operand descs: `x = [4096,1024] stride [1024,1]`, `col = [4096,1024] stride [0,1]` (rank-aligned broadcast weight/bias), `full = [4096,1024] stride [1024,1]`. All at `OpCategory::Normalization`/`Softmax`, `ArchSku::Sm89`, f32 only.

**Types.** `NormalizationKind::{RMSNorm, LayerNorm, ...}` (`crates/baracuda-kernels-types/src/ops.rs:674-693`) is the bespoke-kernels enum; kernelgen keys use `OpCategory::Normalization` and do not require a new kind for a residual variant (the op is named at the `OpDef` level).

## 5. Design / delta

### 5.1 A new operand role: `Residual`

Add `RrRole::Residual` and make `rr_role` a **three-way** classifier. The key insight:
a residual and the row-streamed `x` are indistinguishable by broadcast mask alone (both
empty bcast). So `rr_role` **cannot** stay a pure function of `OperandKey` for the new
case — the residual/`x` distinction is *positional* (which input index), not
*structural*. Two clean options; pick **(A)**:

**(A) Positional residual — role is `(mask, index-role)`.** Keep `rr_role(OperandKey) ->
RrRole` returning `RowStreamed | ColBroadcast` (unchanged, still total, still the
loader's index picker — a residual *loads* exactly like a row-streamed input, so it
maps to `RowStreamed` there). Introduce the residual **only** as a validation +
epilogue-legality concept keyed on input index, carried by a small per-op annotation.
Since the IR has no role field, the least-invasive carrier is: **residuals are inputs
`1..=n_res` declared row-streamed (empty bcast) that validate accepts, and everything
after them is column-broadcast.** Concretely, relax `validate_row_reduce` so an input
`i >= 1` may be **either** a legal residual (empty bcast, contiguous, rank ≥ 1) **or** a
column operand — and require residuals to precede columns (a fixed operand order:
`[x, residual_0..residual_{r-1}, col_0..col_{c-1}, out]`). The loader already does the
right thing for both (empty bcast → `idx`, bcast → `j`). This needs **no IR change** and
**no new field** — the role is fully recoverable from `(input index, bcast mask)` given
the ordering convention.

Because a residual is now a legal in-stage input, the stage-`pre` guard must change from
"reject any non-`x` input" to "reject only *column* inputs" — which is already exactly
what the code checks (`is_col[*i]`, plan.rs:262-268). So **the stage guard needs no
change**; it already permits a row-streamed residual and forbids only column operands.
This is the elegant part: the existing `is_col` gate is precisely the residual-safe gate.

> Rejected alternative (B): add an explicit `roles: Vec<RrRole>` to `Access::RowReduce`.
> Cleaner conceptually but a wider IR/serialization change and it duplicates information
> the structure key already carries. Defer unless item 05's seam encoding forces an
> explicit role list (it may — see §10).

### 5.2 What `validate_row_reduce` must now enforce

Replace the blanket "inputs 1.. must be column" (plan.rs:240-246) with an ordered
classification:

```rust
// pseudo-delta, plan.rs validate_row_reduce
// operands: [x, residual*, col*, out]; residuals precede cols.
let mut seen_col = false;
for i in 1..n {
    let o = key.operands[i];
    match rr_role(o) {                       // (mask-based)
        RrRole::RowStreamed => {             // a residual
            assert!(!seen_col, "residual input {i} must precede all column weight/bias inputs");
            assert!(o.contig == Contiguity::Contig,
                "residual input {i} must be contiguous (indexed base+j like x)");
            assert!(!o.flipped, "residual input {i} must not be reversed");
            // is_col[i] stays false → legal inside a stage pre, indexed in_i[idx]
        }
        RrRole::ColBroadcast => { /* existing checks */ seen_col = true; is_col[i] = true; }
    }
}
```
Keep every existing column check (feature axis not broadcast, broadcasts every outer
axis, not flipped — plan.rs:212-229). Keep Input0 non-column (plan.rs:232-235). Keep the
finite-`Const`, `Reduced`-ordering, no-`Param` expression checks (plan.rs:259-292)
verbatim. The **bare rank-1 `[k]` weight OOB guard** (the must-fix from the shipped work,
tested at cuda.rs:1223-1235) is *preserved* by the ordering rule: a bare `[k]` weight has
empty bcast, so it would now be *accepted as a residual* — which reads `in_i[base+j]`,
still OOB if it is only `k` elements. **This is the critical adversarial point (§8):**
the residual relaxation *reopens* the exact OOB the shipped guard closed, unless we keep
the caller extent pre-condition (a residual's extent must be the full `[n_out,k]`, same
as `x`). Document it identically to the weight extent pre-condition (plan.rs:160-173) and
the fuel doc §4 (lines 74-83): the structure key has no numeric extents, so
`residual.extent == x.extent` is a **caller pre-condition** the boundary holding
`OperandDesc` must assert. The *shape-legality* difference we can check (residual is
row-streamed contiguous, not a `[k]` broadcast view) still rejects the bare-`[k]`-passed-
*as-weight* case, because a residual must be full-rank contiguous with a non-broadcast
outer axis — a bare `[k]` has rank 1 and no outer axis, so it can only be input 0's peer,
never a column. Add an assert that a residual's rank matches `x`'s rank and its outer
axes are **not** broadcast (distinguishing it from a mis-passed column).

### 5.3 Emitter (`emit_row_reduce`) — essentially free

The loader (cuda.rs:416-427) already emits `in_i[idx]` for an empty-bcast operand, so a
residual input is loaded correctly with **zero emitter change**. The op author writes the
`x + residual` sum directly in the IR: the stage `pre` becomes e.g.
`(input(0) + input(1)).unary(Sqr)` and the epilogue references `(input(0) + input(1))`
wherever it used `input(0)`. The emitter lowers that tree unchanged. The only thing to
verify: `n_inputs` counts residuals, and the kernel signature loop (cuda.rs:466-468)
emits `const T* in1` for the residual — which it already does for any input. **No emitter
delta is expected; confirm by golden test.** (If we later want to avoid recomputing
`x+residual` in both stage and epilogue, that is the deferred temp-binding pass, not this
brief.)

### 5.4 New AOT catalog ops (`bin/kernelgen.rs`)

Add residual-fused variants to the enumeration:
- `res_rmsnorm` — 2 inputs `[x, residual]`: `pre = Sqr(x+residual)` (Mean), epilogue
  `(x+residual) * rsqrt(reduced0 + eps)`.
- `res_layernorm` — 3 inputs `[x, residual, weight, bias]` → 4 inputs actually
  `[x, residual, col_w, col_b]`: stage0 `Mean(x+residual)`, stage1 `Mean(Sqr((x+residual)
  - r0))`, epilogue `((x+residual) - r0)*rsqrt(r1+eps)*w[j] + b[j]`. This exercises
  **residual (row-streamed) + column (broadcast) inputs in the same op** — the full role
  matrix.
- Optionally `res_wrmsnorm` (x + residual, then weighted RmsNorm) for symmetry.

Operand order per §5.1: `[x, residual, col_w?, col_b?, out]`, residual full-width
contiguous (`stride [1024,1]`, empty bcast), cols `stride [0,1]`. Also broaden the
existing catalog while here (independent, low-risk): additional dtype cells (f16/bf16 for
the norms — the emitter already up-converts) and a second row extent, to give 07/08 a
richer sweep corpus. Keep each new cell's `OpCategory` at `Normalization`.

### 5.5 FKC / contract implications

`contract()` still returns `None` for these (RowReduce → `derive_pattern` NotElementwise
→ None). **This is intentional and must be preserved** — the honest miss is what keeps
the planner's miss signal truthful until item 05 wires the RowReduce contract path. Add a
test asserting `contract(res_layernorm) == None`. The residual region encoding
(`Add(Bind0, Bind1)` feeding the reduction) is a note for item 05 / a new Fuel ask, not
code here.

## 6. Implementation steps (ordered checklist)

1. **IR (`ir.rs`)** — no variant change needed. Add a `row_reduce` doc note that inputs
   are ordered `[x, residual*, col*]` and that residual inputs are row-streamed and
   legal inside stages. (If §10 resolves toward explicit roles, add `roles` here instead
   — but default to no IR change.)
2. **Role/validate (`plan.rs`)** — extend `RrRole` doc; relax `validate_row_reduce`
   (plan.rs:240-249) to accept an ordered residual run before the column run per §5.2;
   add residual rank/outer-axis/contiguity/flip asserts; keep every existing column,
   Input0, expression, and finite-`Const` check. Extend the extent-precondition rustdoc
   (plan.rs:160-173) to name the residual extent equality.
3. **Emitter (`cuda.rs`)** — confirm no change; add a code comment at the `load` closure
   (cuda.rs:416-427) noting a residual is an empty-bcast row-streamed input loaded at
   `idx`. (Only touch code if a golden test surprises.)
4. **Pattern/contract (`pattern.rs`, `contract.rs`)** — no code change; add a test that
   RowReduce (incl. residual) still returns `NotElementwise` / `None` (honest-miss
   preservation). Add a `// residual region encoding: item 05` TODO note near
   contract.rs:68.
5. **Seam (`jit.rs`)** — no change; add/confirm a TODO at `region_to_op` (jit.rs:406)
   that RowReduce (incl. residual) is not yet a seam-adoptable region (points at item 05
   + the fuel-ask doc).
6. **Catalog (`bin/kernelgen.rs`)** — add `res_rmsnorm`, `res_layernorm` (+ optional
   `res_wrmsnorm`) to the enumeration (kernelgen.rs:104-167) with the `[x, residual,
   col*, out]` operand order; broaden dtype/shape cells for 07/08.
7. **Docs (`OP-MATRIX.md`, `docs/design/kernel-specialization.md`)** — record the
   residual-add RowReduce capability. Note: `OP-MATRIX.md` documents the *bespoke*
   kernels crate, not kernelgen; add the kernelgen residual entry where the kernelgen
   catalog is tracked (§12 of `kernel-specialization.md`) and flag that doc's stale
   status text (per house note) rather than trusting it.

## 7. Test & on-device validation plan

**Unit / golden (`cuda.rs` tests, alongside cuda.rs:1197-1254):**
- `res_rmsnorm` golden: stage fold emits `acc0 += ((in0[idx] + in1[idx])*(in0[idx] +
  in1[idx]));`, epilogue `out[idx] = ((in0[idx] + in1[idx]) * rsqrtf((r0 + 1e-5)));`,
  and asserts **no** `in1[j]` (residual is `[idx]`, not per-column).
- `res_layernorm` golden with residual **and** column weight/bias: assert stage reads
  `(in0[idx] + in1[idx])`, columns read `in2[j]`/`in3[j]`, epilogue combines both.
- Negative: a residual placed *after* a column input → panic (ordering rule).
- Negative: preserve the bare-rank-1-`[k]` rejection (residual must be full-rank
  contiguous with non-broadcast outer axis).
- `contract(res_layernorm)` is `None` (honest-miss preserved).

**nvrtc headerless compile:** every new cell's `.cu` must compile under **nvrtc with no
headers** (the house header-light invariant; f16/bf16 cells include only
`cuda_fp16.h`/`cuda_bf16.h`). Verify `res_rmsnorm`/`res_layernorm` f32/f16/bf16.

**nvcc numeric on sm_89 (RTX 4070):** compile each residual kernel with nvcc, launch on
`[4096,1024]` (and a small `[7, 13]` odd shape to stress the grid-stride tail + partial
warps), diff against a **CPU/PyTorch numeric oracle**:
- `res_rmsnorm(x, r) == RMSNorm(x + r)` (torch: `F.rms_norm(x+r, ...)`),
- `res_layernorm(x, r, w, b) == F.layer_norm(x+r, ..., w, b)`.
  f32 correctly-rounded-ish tolerance; f16/bf16 fold-in-f32 tolerance matching the
  existing norm tests.

**compute-sanitizer** (mandatory — this kernel has `__shared__` block-reduce + cross-
thread shuffle): run `synccheck`, `racecheck`, `initcheck` on `res_layernorm` (two
stages, residual + columns) at `[4096,1024]` and at a shape with `k` not a multiple of
32 and `n_out` not a multiple of `gridDim`. Must be **clean** — especially initcheck,
because the residual doubles the global loads per element and any mis-indexed residual
read is an OOB initcheck will catch.

## 8. Adversarial-verify checklist

Run the multi-agent find → dedup → skeptic-refute pass after the change. Probe
specifically for:
1. **Reopened OOB (highest risk).** The residual relaxation *undoes* the input-1..-must-
   be-column guard that closed the bare-`[k]` OOB. Confirm a too-short residual (extent <
   `x`) is (a) rejected where structurally distinguishable (rank/outer-axis mismatch) and
   (b) documented as a caller extent pre-condition where structurally identical. Verify
   compute-sanitizer initcheck is clean *only* when extents match, and that the
   ordering/rank asserts fire on a mis-passed `[k]`.
2. **Role misclassification.** Ensure a residual (empty bcast) is never treated as a
   column (`in_i[j]`, dropping the row dependence → silent wrong result reading only the
   first row's values). Golden-assert `in1[idx]` not `in1[j]`.
3. **Stage-guard regression.** Confirm the `is_col` in-stage guard still forbids *column*
   inputs inside a stage (reducing a weight is nonsense) while *allowing* the residual —
   i.e. the guard didn't get loosened to "allow all inputs."
4. **NaN/determinism.** `x + residual` must not perturb the NaN-propagating max path
   (Softmax residual variant, if added) or the one-block-per-row determinism (no
   atomicAdd introduced; still one block per output row). Diff against oracle with NaN
   injected in `residual`.
5. **f16/bf16 up-convert.** The residual add must happen in the **accumulate type**
   (float/double), not in the narrow input type — confirm the emitted sum is
   `(__half2float(in0[idx]) + __half2float(in1[idx]))`, matching the existing up-convert
   contract; a narrow-type add would lose precision and break the oracle diff.
6. **Ordering-convention drift.** If a caller ever emits `[x, col, residual]` (wrong
   order), validate must reject, not silently mis-index.

## 9. Definition of done

- `RrRole` / `validate_row_reduce` accept an ordered `[x, residual*, col*, out]` operand
  layout; all existing RowReduce tests (single-input, weighted, layernorm, bare-`[k]`
  rejection, in-stage column rejection) **still green**.
- New golden tests for `res_rmsnorm` + `res_layernorm` (residual indexed `[idx]`, columns
  `[j]`) pass; ordering + rank negatives panic as specified.
- `bin/kernelgen.rs` emits the residual variants (+ broadened dtype/shape cells);
  `kernelgen <dir>` runs clean.
- **On-device validated on sm_89:** nvrtc headerless compile green for every new cell;
  nvcc numeric matches the PyTorch oracle within tolerance for f32/f16/bf16;
  compute-sanitizer synccheck/racecheck/initcheck **clean** (incl. odd `k`, partial
  warps, and a shape where extents match).
- **FKC honest-miss preserved:** `contract()` returns `None` for every RowReduce op
  (test-asserted); no seam adoption path added; `region_to_op` TODO points at item 05.
- Determinism preserved: one block per output row, no atomicAdd, block-reduce unchanged.
- Docs updated: kernelgen catalog capability recorded; `kernel-specialization.md` §12
  residual entry added and its stale status text flagged; extent pre-condition rustdoc
  extended to name the residual.
- Lockstep release: all crates bumped + full republish (`publish_alpha*.ps1` shape).

## 10. Open questions / Fuel asks

- **Ask (e) for item 05's seam (not this brief):** how is a *two-tensor residual add
  feeding a reduction* encoded in the frozen region grammar? Proposed: an `Add(Bind0,
  Bind1)` node whose result is the shared input to the stat reduction *and* the epilogue
  (a shared interior — which the current pure-tree IR recomputes rather than shares). This
  extends the `docs/fuel-ask-fused-reduce-seam-2026-06-25.md` asks; raise it *with* 05,
  not before Fuel answers a-d.
- **Explicit roles vs positional (design-open):** we chose positional ordering (`[x,
  residual*, col*]`) with no IR change. If item 05's seam encoding needs an explicit role
  list to reconstruct the operand order from a region, revisit adding `roles:
  Vec<RrRole>` to `Access::RowReduce` (rejected alternative B, §5.1).
- **Write-back of `x + residual` (design-open):** some transformer blocks reuse the
  pre-norm sum as the next block's residual. Should the fused op optionally emit a second
  output (the sum) to save a recompute downstream? This is a *dual-output* kernel shape
  the current single-`out` emitter does not model — defer unless a Fuel/consumer ask
  requires it; note it so it is not silently dropped.
- **Multiple residuals:** the design admits `residual*` (a run), but real blocks use
  exactly one. Ship single-residual; the `n_res > 1` path is validated-but-untested
  unless a consumer needs it.
