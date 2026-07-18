# Recipe-convergence follow-ups (2026-07-18) — sub-agent work briefs

Each section below is a **self-contained brief** for one follow-up. A fresh agent
should be able to execute a brief with only this doc + the repo. Do them
independently (no shared state); each ends green on its own.

---

## Shared context (read first — every brief depends on this)

**Where the recipe convergence stands.** Baracuda emits a neutral **KISS-Ops
Semantics recipe** for a kernel's op DAG; a recipe-carrying contract advertises to
Fuel's *recipe-import* path (Fuel resolves the recipe to its primitive floor and
verifies numerically). The grammar co-design with Fuel is **closed**; the
authoritative replies are saved in the repo:

- `docs/fuel-reply-recipe-grammar-2026-07-15.md` — the 6-question co-design (AGREED).
- `docs/fuel-reply-recipe-schema-2026-07-15.md` — §6.4-0009 node schema + 4 open items pinned.
- `docs/fuel-reply-matmul-attr-2026-07-16.md` — matmul role-vector attr CONFIRMED.
- The FKC `OutputDesc` grammar (Fuel, 2026-07-17): `dtype_rule: passthrough(<role>) | fixed(<DType>)`; `shape_rule: same_as(<role>) | from_params(…)`; **no `from_recipe` form**; all fields omittable; `shape_rule` is a claim verified vs the recipe (not evaluated yet), `dtype_rule` IS interpreted (builds the binding-key output slot).

**The recipe emitter** is `crates/baracuda-kernelgen/src/recipe.rs`:
`semantics_dag(op: &OpDef) -> Option<String>` matches `&op.access` and returns a
compact **functional** recipe string (a co-pin *surface*; Fuel flattens it to the
§6.4-0009 flat table + canonicalizes on ingest — so the exact literal grammar is a
strawman, but keep it consistent with the existing arms). Helpers:
`expr_to_recipe(e, reduced: Option<&str>)` (threads the fold-node string a
`Reduced(0)` leaf resolves to), `contraction_roles`, `reduce_monoid`,
`reduce_axes_code`, `const_repr`, `unary_kiss_name`, `binary_kiss_name`. Established
recipe forms (mirror them):
- elementwise: `add(relu(in0), in1)`; leaves `in<i>` (=Bind), `const(<v>)`, `iota(<axis>)` (Coord), `runtime_scalar(<slot>)` (Param).
- contraction: `matmul[<lhs>.<rhs>](in0, in1)` role chars `b`=Batch/`m`=FreeM/`n`=FreeN/`k`=ContractedK; `Reduced(0)`→the matmul node; fused bias/act composes as elementwise over it (`relu(add(matmul[mk.kn](in0,in1), in2))`).
- reduction: `reduce[<monoid>,<axes>,<keepdim>](<pre-map=op.body>)` + `post` over `Reduced(0)`; monoid∈{sum,prod,max,min}, Mean→honest miss; axes `last`|`0x<hex>`, keepdim `kd`|`nokd`.
- scan: `prefix_scan[<monoid>,<axis>,<excl>](<pre>)`; reverse = `flip[<axis>](prefix_scan[…](flip[<axis>](<pre>)))`.

**Auto-wiring into contracts** is `crates/baracuda-kernelgen/src/contract.rs`
`contract()`: it computes `let recipe = crate::recipe::semantics_dag(op);` and
`let recipe_carrying = recipe.is_some() && !matches!(op.access, crate::ir::Access::Elementwise);`.
A `recipe_carrying` op advertises `fused_op: <op.name>` (the `None` op_line arm) +
a `semantics: <recipe>` line, with the return block OMITTING `shape_rule` (recipe
is the shape authority) and emitting a real `dtype_rule` (`fixed(<dtype>)` hetero /
`passthrough(in0)` uniform). **Withhold discipline preserved:** `bundle()`
(`recipe_import=false`) still withholds these; `bundle_kisc` admits ONLY to a
recipe-import peer via `contract_carries_recipe` (the semantics line). So **a new
NON-elementwise recipe arm auto-advertises with no contract.rs change.** (An
elementwise op with an *indexed* read/write — gather/scatter — is `Access::Elementwise`
so it is NOT caught by `recipe_carrying`; see Brief 3.)

**Honest-miss discipline (non-negotiable):** never fabricate a token. If a node
has no confirmed KISS-Ops name, `semantics_dag`/`expr_to_recipe` returns `None`
(→ the op stays a withheld honest miss). `unary_kiss_name`/`binary_kiss_name`
are exhaustive (no catch-all) so a new IR op forces a decision.

**Workflow for every brief:** strict TDD (RED test first, watch it fail, then
GREEN). Run `cargo test -p baracuda-kernelgen --lib recipe::` and `… contract::`
and the full `… --lib`; `cargo clippy -p baracuda-kernelgen --lib`; `cargo fmt -p
baracuda-kernelgen`. Baseline at time of writing: **kernelgen lib 703 green.**
Commit with the trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
Mirror the existing recipe/contract tests (e.g. `recipe::tests::contraction_recipe_is_a_matmul_node_with_the_epilogue_over_it`, `contract::tests::contraction_advertises_a_recipe_carrying_contract`).

---

## Brief 1 — RowReduce recipe arm (softmax / rmsnorm / layernorm)

**Goal.** `semantics_dag` covers `Access::RowReduce { stages: Vec<ReduceStage>, epilogue }`
so softmax/rmsnorm cells advertise recipe-carrying contracts (auto-wires via
`recipe_carrying`). Fuel framed this as pre-map→fold→post with NO new node kind —
each stage is a `reduce[…]` fold; the epilogue is elementwise over the stage
results AND the row-streamed inputs.

**Key structural facts.** `ReduceStage { pre: ScalarExpr, op: ReduceOp }`
(ir.rs:1061). Stage `i` produces `Reduced(i)` (the RowReduce doc, ir.rs:1123). The
`epilogue` references `Input`s (the full-width row-streamed input — RowReduce is
reduce→broadcast→elementwise, so `Input` IS admissible here, unlike Reduction) and
`Reduced(0..n)`. RowReduce reduces the **last axis** (contiguous).

**The one real refactor.** `expr_to_recipe`'s `reduced: Option<&str>` handles only
`Reduced(0)`. Generalize it to resolve `Reduced(i)` for `i` in `0..n_stages` —
e.g. change the param to `reduced: &[String]` (index by `i`), with `Reduced(i)` →
`reduced.get(i)` (None → honest miss). Update the existing single-`Reduced(0)`
call sites (contraction/reduction/scan pass a 1-element slice; elementwise passes
`&[]`). Recipe shape (strawman, consistent with the others): each stage `i` =
`reduce[<monoid>,last,nokd](<stage.pre with Reduced(0..i)>)`; then
`expr_to_recipe(epilogue, &[stage0_str, stage1_str, …])`. A softmax (2 stages:
max, then sum of exp(x−max)) becomes e.g.
`div(exp(sub(in0, <stage0>)), <stage1>)` where stage0=`reduce[max,last,nokd](in0)`
and stage1=`reduce[sum,last,nokd](exp(sub(in0, reduce[max,last,nokd](in0))))`.
(Confirm the exact softmax IR by reading `OpDef::row_reduce` + the rowreduce tests.)

**Gotcha.** A stage's `pre` may reference earlier stages' `Reduced(j<i)` — thread
the already-built stage strings when building stage `i`. `Mean` monoid → honest
miss (as in reduction). Watch `n_stages` from `stages.len()`.

**Acceptance.** A `recipe::tests` case for softmax + rmsnorm (assert the exact
recipe string). A `contract::tests` case: the cell advertises `fused_op` +
`semantics` + no `shape_rule` + `dtype_rule: passthrough(in0)`, withheld from a
non-recipe-import bundle, admitted to a recipe-import peer. Existing rowreduce
honest-miss tests (if any assert no contract) flip to advertise. Full lib green.

---

## Brief 2 — Mean reduction recipe (sum-fold + div-by-extent)

**Goal.** Retire the `Mean` reduction honest miss. Fuel's position: `Mean` is not a
monoid — it's a `sum` fold + a `div`-by-extent epilogue (their `MeanDim` decomposes
exactly that way). So a Mean reduction's recipe is
`div(reduce[sum,<axes>,<keepdim>](<pre>), <extent>)`.

**The design question (resolve first, likely a short Fuel co-pin).** The divisor is
the **reduced-axis extent** — a shape-derived value, not a `const`/`param`/`input`.
The recipe surface has no token for it yet. Options: (a) a new source op
`reduce_extent{axes}` (a leaf like `iota`/`runtime_scalar`, resolved by Fuel from
the interface rank/shape); (b) express it via the FKC `shape_rule: from_params(…)`
channel; (c) emit `const(<extent>)` if the extent is statically known (it is NOT —
`StructureKey` carries size *classes*, not literal extents, so a literal const
would be wrong). Recommend (a) `reduce_extent{axes}` and **draft a short
propose-first to Fuel** (`docs/fuel-ask-reduce-extent-2026-07-…md`) confirming the
op name + that its attr is the reduced-axis set, mirroring how `runtime_scalar`
was pinned. Do NOT emit a literal-extent const.

**Implementation (after the extent token is pinned).** In the `Access::Reduction`
arm, when `rop == Mean`: build `reduce[sum,<axes>,<keepdim>](<pre>)`, wrap as
`div(<that>, reduce_extent{<axes>})`, then apply the `post` over `Reduced(0)` —
where `Reduced(0)` resolves to the `div(...)` node (the finalized mean), matching
the "post sees the post-Mean value" ordering pinned at
`plan::assert_valid_reduction_post` / the `Access::Reduction` doc (ir.rs:1107).

**Gotcha.** Integer Mean is out of scope (the emitter rejects `int_acc && Mean`) —
keep it an honest miss. Update the reduction test that asserts `Mean → None`.

**Acceptance.** `recipe::tests`: an f32 Mean reduction → the sum+div-extent recipe.
`contract::tests`: it advertises recipe-carrying. Full lib green. (Blocked on the
extent-token co-pin — if Fuel hasn't answered, stop after drafting the propose-first
+ the RED test, and leave Mean an honest miss.)

---

## Brief 3 — Gather + Scatter recipe arms (indexed read/write)

**Goal.** Emit recipes for data-dependent gather/scatter so those cells advertise
recipe-carrying contracts. Fuel's pinned schemas:
`gather{axis, oob_policy, index_operand, index_dtype∈{u32,i32,i64}}` (child_edges
`[data, index]`, positional roles) and
`scatter{axis, scatter_combine∈{assign,atomic-add,atomic-max,atomic-min}, oob_policy, index_operand, index_dtype}`.
Fuel has `Gather` + `ScatterAdd` (=`scatter{atomic-add}`); other scatter combines
are Fuel-side gaps (honest-miss those per tier-3).

**Investigate first (the structural subtlety).** Gather/scatter are NOT `Access`
variants — they ride `op.read_index: Vec<ReadIndex>` (ir.rs:1838, `ReadIndex::Indexed`)
and `op.write_index: WriteIndex` (`WriteIndex::ScatterIndexed`). A gather op is
likely `Access::Elementwise` **with** an indexed `read_index`, so:
1. `semantics_dag`'s `Access::Elementwise` arm currently emits an elementwise recipe
   that IGNORES the index — that is WRONG for a gather. Add a check: if any
   `op.read_index` is `Indexed` (or `op.write_index` is `ScatterIndexed`), emit the
   `gather{…}`/`scatter{…}` node instead of / wrapping the plain body.
2. `recipe_carrying = … && !Elementwise` in `contract()` EXCLUDES a gather (it's
   Elementwise-access). Broaden the gate to also fire for an indexed read/write
   (e.g. `!op.read_index.iter().all(ReadIndex::is_direct) || !op.write_index.is_direct()`),
   OR compute `recipe_carrying` from "the recipe differs from a plain elementwise
   body." Confirm the exact condition by reading `contract()`.
3. **Reconcile with the existing `gather_advert`** (contract.rs ~650): a Model-A
   u32 gather already advertises `op_kind: Gather`. Decide: does the recipe
   SUPERSEDE the gather_advert op_kind (emit the `gather{…}` recipe + let it ride
   the recipe-import path), or COMPLEMENT it (keep op_kind for the primitive path,
   add semantics for recipe-import)? Likely complement — keep the op_kind advert
   (a known Fuel primitive imports directly, no recipe needed) and ONLY emit a
   recipe for the gathers currently honest-missed (i32/i64 index, or a fused gather
   body). Read the gather honest-miss guards (contract.rs ~369) to see which
   gathers withhold today — those are the target.

**Recommendation.** Scope v1 to the gathers/scatters that are honest misses today
(no `gather_advert`), giving them a recipe advert. Leave the already-advertising
u32 gather on its op_kind path. Draft a short propose-first only if a surface
question arises; the schemas are already pinned.

**Acceptance.** `recipe::tests` for a gather (data+index) and a scatter-add.
`contract::tests`: a previously-honest-missed indexed op now advertises
recipe-carrying (withheld / admitted split). Full lib green. **Higher-risk brief**
— touches the gather_advert + the recipe_carrying gate; keep the change surgical and
re-run the whole contract suite.

---

## Brief 4 — Elementwise pattern-miss recipe-import retirement (band / erfc / triu / …)

**Goal.** Retire the withhold for ELEMENTWISE ops that `derive_pattern` rejects but
that carry a valid recipe — e.g. `BitAnd`→`bit_and(in0,in1)` (`NoFkcName`),
`Erfc`→`erfc(in0)` (`NoFkcName`), `triu` (`CoordUnsupported`, recipe uses
`iota`/`cmp_ge`/`mul`). These already have recipes from `semantics_dag` (elementwise
arm) but `contract()` withholds them because `recipe_carrying` is scoped to
NON-elementwise (85f1bbec/cf573f34 deliberately deferred this).

**Why deferred / the care required.** (1) Not every such op is safely
recipe-importable — the recipe primitives must be in Fuel's floor. `erfc`, `bit_and`
etc. are confirmed KISS-Ops tokens (that's why `unary/binary_kiss_name` maps them),
so their recipe IS resolvable — but VERIFY against the saved Fuel replies /
KISS-Ops op set which primitives Fuel's floor actually has, and honest-miss the
rest. (2) The return block for an ELEMENTWISE op is the normal
`same_as(in0)`/`passthrough(in0)` (out shape+dtype = input) — those are TRUE, valid
FKC forms, so KEEP them (do NOT omit shape_rule for elementwise; that's only for
non-elementwise where the shape differs).

**Implementation.** Broaden the `contract()` op_line: an elementwise op whose
pattern derivation FAILED (`pattern == None`) but `recipe.is_some()` advertises
`fused_op: <op.name>` + `semantics`, keeping the elementwise return block
(`same_as(in0)` + `passthrough`/`fixed(U8)`). This is a NEW branch distinct from the
non-elementwise `recipe_carrying` one. Preserve the same withhold discipline
(bundle withholds; bundle_kisc admits with recipe-import). Watch the existing
guards that return None BEFORE the op_line (multi-output, addressing-view, offset,
gather, select) — those stay honest misses regardless (they have no valid recipe /
a real ABI gap). Only the `NoFkcName`/`CoordUnsupported` elementwise misses flip.

**Acceptance.** `contract::tests`: `int_ops_rate_zero_ulp_and_emit_no_contract` and
`vocab_ops_have_no_contract_until_fuel_names_them` (and the `triu`/`Coord` test)
FLIP to assert recipe-carrying adverts (fused_op + semantics + `same_as(in0)` +
recipe-import withhold split), `derive_pattern` still the same Err. Verify NO
elementwise op with a KNOWN Fuel op_kind changed (they still emit `op_kind:` via the
`Some(p)` arm). Full lib green. **Medium-risk** — re-run the whole contract suite;
several honest-miss tests flip.

---

## Brief 5 — Device-validate the B13/B14 contraction growth on the RTX 4070

**Goal.** On-device numeric proof for this session's contraction growth: the
**fused bias/activation epilogue** (B13, commit e16fac7d) and the **batched
`[B,M,K]·[B,K,N]`** kernel (B14, 4cd604cd). The fused-relu epilogue path is already
RTX-4070-proven (`ondevice/contract_validate.cu`, item 10); the NEW code (the
`in2[col]` bias load; the `b = blockIdx.z` batch offset + per-operand batch
strides) is only CPU-oracle-validated so far.

**Context.** This is a CUDA box (RTX 4070 Laptop / sm_89 / CUDA 13.3 — see the
memory note "cuda-box-local-validation"). The manual on-device harnesses live in
`crates/baracuda-kernelgen/ondevice/` (see `ondevice/README.md`); they are NOT
wired into `cargo test` — run via `nvcc` from a VS dev shell (`Enter-VsDevShell`
so `nvcc` finds `cl.exe`). The CPU numeric reference is `oracle::eval_contraction`
(oracle.rs) which now covers bias (reads `Input(2)` at column) + batched (rank-3
loop) — use it as the ground truth, or the in-`.cu` f64 oracle like
`contract_validate.cu` does.

**Do.** Extend `ondevice/contract_validate.cu` (or add a sibling
`contract_bias_batched_validate.cu`) to: (a) generate the fused `matmul_bias_relu`
cell + the batched `bmm` cell via `bin/kernelgen` (or the library), (b) launch them
(batched: grid `.z = B`; bias: pass the `[N]` bias buffer), (c) diff vs a host f64
reference (matmul+bias+relu; per-batch matmul) — bit-exact for exactly-representable
inputs, tolerance otherwise, and vs cuBLAS batched (`cublasSgemmStridedBatched`) if
convenient. Run `compute-sanitizer` (memcheck + initcheck) on the small shapes.
Record the result table in `ondevice/README.md` (matching the existing entries'
format + the "Last run (RTX 4070…)" line).

**Acceptance.** A README section documenting a PASS (bias load correct; batch
strides correct; sanitizers 0). This is validation-only — no src change unless a
device bug is found (then TDD-fix it). **Self-contained**, no Fuel dependency.

---

## NOT actionable yet (do NOT pick these up)

- **Pooling (`Access::Window`), Sort (`Access::RowSort`), Im2Col (`Access::Im2Col`)
  recipes** — Fuel confirmed these are **Fuel-side basis gaps** (no clean
  Pool/Sort primitive; conv/im2col is Fuel's own missing `Im2Col`/`Col2Im`). They
  stay tier-3 honest misses (`semantics_dag → None`) until Fuel fills the gap.
  Emitting a recipe would reference primitives Fuel can't resolve.
- **Shared-header role-enum codes** — Fuel offered to host the
  `{Batch,FreeM,FreeN,ContractedK}` codes next to the Scan `{role,index}` codes.
  Baracuda's `AxisRole` discriminants already match (0-3); adopt the shared header
  only once Fuel publishes it (Fuel-gated).
- **General body-carrying `Op::Scan` recipe** (SSM/Mamba `scan_placeholder`, the
  affine-pair semiring) — Fuel shipped `Op::Scan` but Baracuda has no such op to
  emit yet; N/A until Baracuda grows one.
