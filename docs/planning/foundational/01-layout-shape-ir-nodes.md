# 01 — Layout/shape IR nodes — implementation brief

> Scope: `crates/baracuda-kernelgen` (the AOT kernel-specialization generator +
> the §5 Baracuda↔Fuel JIT seam). This is the **keystone** foundational item:
> items **03** (strided/multi-axis/keepdim reductions) and **10** (MatMul design
> spike) build directly on the axis/shape-fact representation decided here, and
> the §1 headline win — "fusions across a layout change, skipping the contiguize
> round-trip" — is unlocked by it. Get the representation right before anything
> downstream leans on it.

---

## ⚠️ Update 2026-07-01 — course-corrected to convention (c); steps 2–4 reverted

Verified (3 adversarial agents incl. a steelman): layout adaptation for an **input read is already handled**
by the generic strided emitter + structure-key specialization — a transpose is just a strided read
(`strided_2d_unravels`), which the design doc always intended ("specialize on structure, not literal shapes").
So the per-operand **`View` EMIT** work (§5.2 `permute_offset_expr`/`view_tag`; steps 3–4) and the
**`StructureKey` perm-field + `STRUCTURE_KEY_VERSION → 2`** bump (§5.3; step 2) are **REDUNDANT and have been
reverted** (`plan.rs`/`cuda.rs` back to generic strided; only `ir.rs`'s `View` IR is kept, on branch
`feat/kernelgen-layout-views`).

**Stride convention: (c), not Fuel's owner-decided (a).** Fuel keeps its existing pre-permute
(`Layout::permute`); the perm rides in `OpAttrs.perm` for **recognition only**; Baracuda uses the existing
generic strided cell. Decisive point: (a)'s entire cost only repairs a mis-key **(a) itself creates**
(producer-order strides misclassify as `Contig` → untransposed read). Counter-proposal relayed to Fuel:
`docs/fuel-reply-layout-stride-convention-2026-07-01.md` (asks Fuel to **stand down peel-the-permute**).

**Item 01 now collapses to:** the `View` IR (recognition carrier — kept) + emitting the FKC `Permute`
recognition pattern (steps 5–6, `pattern.rs`/`contract.rs`) so Fuel's matcher routes a transpose-fused
subgraph to the **existing generic strided cell**. No new emit, no key change. Those steps are **Fuel-gated**
on the (c) confirmation + F1 (`OpAttrs.perm`) + the `match_node` attr routing.

**Caveat (why we keep `View`):** (c) dominates for **elementwise** only; a future perm-aware *specialized*
schedule (shared-mem-tiled transpose, or perm × reduction-axis in items 03/10) could revive (a)'s compile-time
perm. The §5.2 / §5.3 / step-2–4 / §10 Q-D2 material below is retained for history but is **superseded** by
this note. See memory `kernelgen-ir-frontier.md`.

---

## 1. Objective

Add a first-class notion of **layout/shape transforms** (`BroadcastTo`,
`Transpose`, `Permute`, `Reshape`, and the squeeze/unsqueeze rank-adjusters) to
the kernelgen IR so a fused op can read an input **through** a layout change and
compute the epilogue in one kernel pass, instead of forcing a materialized
`contiguize`/transpose copy first. Today the IR describes only the *value* math
(`ScalarExpr`) and a coarse *iteration pattern* (`Access`); the per-operand
layout facts live entirely in the `StructureKey`/`OperandKey` (contiguity,
broadcast mask, flip) and the emitter consumes them **only** on the `Strided`
schedule (`cuda.rs::emit_strided`). There is no IR node that says "input 0 is the
transpose of a `[K, M]` producer" — so the generator cannot express, recognize
(derive an FKC `pattern:`), or emit a *fusion that crosses* that transform. This
brief adds that expressiveness. It is foundational because the axis/shape-fact
carrier it defines is the same machinery **03** needs to name a non-last reduction
axis and a keepdim output, and the same shape-fact vocabulary **10** needs to
state MatMul's contraction/free axes and pick a schedule.

---

## 2. Status & blockers

- **Baracuda-unblocked (build the whole AOT half now).** The AOT path —
  `ir.rs` node(s), `plan.rs` schedule, `cuda.rs` emit, `pattern.rs`/`contract.rs`
  representation, `StructureKey` facts — is entirely within Baracuda's ownership
  and can be designed and shipped without waiting on Fuel. The strided emitter
  already lowers arbitrary per-operand strides/broadcast (`emit_strided`,
  `offset_expr`), so the *codegen* substrate for reading-through-a-layout largely
  exists; this work is about **representing** the transform in the IR and
  threading it through derive/contract, not inventing new CUDA.

- **Fuel-blocked (only the live §5 seam adoption).** The JIT seam
  (`jit.rs::region_to_op`, `jit.rs::seam::optag_name`) is elementwise-only and
  cannot ingest a layout op **because Fuel's frozen region grammar carries no
  shape facts for them.** `fuel_kernel_seam_types::OpTag` *does* already list
  `Transpose, Permute, Reshape, BroadcastTo, Unsqueeze, Squeeze`
  (`fuel-kernel-seam-types/src/lib.rs:52`), but `OpAttrs`
  (`.../lib.rs:71-80`) carries only `scalars: Vec<f64>` and `axis: Option<i64>`
  — **no permutation vector, no target shape, no broadcast target**. A
  `Transpose`/`Reshape` region node is therefore unusable across the seam until
  Fuel extends `OpAttrs` (see §10, ask **F1**). Until then `optag_name` keeps
  these tags in its `_ => None` honest-miss arm — which is correct and must stay
  correct.

- **Design-open (one decision, called out in §5 and §10 Q-D1).** *Where* a
  layout transform lives in the algorithm/schedule split — a new `Access`
  variant, a shape-fact-carrying `ScalarExpr` leaf, or a separate "view
  descriptor" on the operand — is the one genuinely open design choice, and the
  recommendation below (a per-operand **view descriptor** consumed by the
  emitter, *not* a `ScalarExpr` node) should be ratified before coding.

**What can proceed right now:** the IR view-descriptor design, the `cuda.rs`
lowering of a fused-op-through-a-view, the `StructureKey` fact it keys on, the
AOT catalog entries + on-device validation, and the honest-miss wiring that keeps
the seam declining layout regions until F1 lands.

---

## 3. Dependencies & sequencing

**Must land before this:** nothing. This item depends on no other. (That is why
it is the keystone — the memory/plan graph lists it as depending on NOTHING.)

**What this enables downstream:**

- **03 — strided/multi-axis/keepdim reductions.** 03 needs to *name* a reduction
  axis that is not the contiguous last axis, and to describe a keepdim output
  shape. Both are shape facts; 03 shares the axis-index + shape-fact carrier this
  item defines rather than inventing a parallel one. Sequence 03 immediately
  after 01.
- **10 — MatMul/contraction design spike.** 10's grammar needs to state
  contraction vs. free axes and reconcile A/B operand layouts (row/col-major) —
  it consumes 01's shape facts (and 02's DAG). 10 is explicitly a *design*, so it
  only needs 01's *design* ratified, not its full implementation.
- **§1 fusion-across-layout win.** The headline "op + transform + epilogue in one
  pass" (`kernel-specialization.md` §1) is directly this item.
- **06 — fused residual-add LayerNorm** benefits (a residual `x + shortcut` where
  `shortcut` arrives transposed/broadcast can fuse without a pre-copy), but does
  not strictly depend on 01.

Independent of / does not block: 04 (integer accumulation), 05 (RowReduce seam
adoption, Fuel-blocked on a different ask), 07/08 (dispatch table + telemetry),
09 (half2 SIMD — but 09 touches the same emit paths, so *coordinate* the emit
changes; sequence 09 after 01/03).

---

## 4. Current code — what exists today

Everything below is quoted from the live tree so a fresh session can locate the
exact touch points.

### 4.1 The IR: value math vs. iteration pattern (`crates/baracuda-kernelgen/src/ir.rs`)

`ScalarExpr` is a pure **tree** of per-coordinate math — no layout node, no DAG
(`ir.rs:16-47`):

```rust
pub enum ScalarExpr {
    Input(u8), Const(f64), Param(u8), Reduced(u8),
    Add(Box<ScalarExpr>, Box<ScalarExpr>), Sub(..), Mul(..), Div(..),
    Unary(UnaryOp, Box<ScalarExpr>),
    Binary(BinaryOp, Box<ScalarExpr>, Box<ScalarExpr>),
}
```

`Access` is `#[non_exhaustive]`, and its own doc-comment names the growth path
(`ir.rs:273-305`):

```rust
/// `#[non_exhaustive]`: windowed/stencil and gather patterns are still the growth
/// path; arbitrary/multiple reduction axes, strided-input reductions, and keepdim
/// layout extend [`Access::Reduction`] later.
pub enum Access {
    Elementwise,
    Reduction { op: ReduceOp },
    RowReduce { stages: Vec<ReduceStage>, epilogue: ScalarExpr },
}
```

`OpDef` (`ir.rs:312-324`) has `name`, `n_inputs`, `body: ScalarExpr`,
`dtypes`, `access`. **No shape/rank/perm field anywhere on the op** — the op is
layout-agnostic and the *cell* (`StructureKey`) carries all layout facts.

### 4.2 Where layout facts live today: `StructureKey`/`OperandKey`

`crates/baracuda-kernels-types/src/structure_key.rs:167-216`. Each operand
carries (`OperandKey`, `:167`):

```rust
pub struct OperandKey {
    pub contig: Contiguity,   // Contig | InnerContig | Strided | Broadcast
    pub bcast: AxisMask,      // which axes have stride 0
    pub vec_width: VecWidth,
    pub inner_div: DivBucket,
    pub flipped: bool,        // any negative stride (reversed view)
}
```

`StructureKey` (`:188`) additionally carries `rank: u8` (the raw iteration rank =
widest operand rank), `reduce_axes: AxisMask` (always `EMPTY` in v1), and
`operands: [OperandKey; MAX_OPERANDS]`. `AxisMask` (`:130-158`) is a `u8`
bitmask, `MAX_RANK == 8`. **Crucially, the key carries a broadcast *mask* and a
flip *bool* and per-axis strides via the source `OperandDesc`, but it does NOT
carry a permutation order or a shape-change (reshape) fact** — a transpose shows
up only as `Contiguity::Strided` + non-unit inner stride (see the
`transposed_view_is_strided` test, `:974-980`), and a reshape is invisible
(both sides are just contiguous). `to_token`/`from_token` (`:616-703`) round-trip
every current field; a new keyed fact must extend both and bump
`STRUCTURE_KEY_VERSION` (`:48`).

### 4.3 The emit paths (`crates/baracuda-kernelgen/src/cuda.rs`)

`Cuda::lower` dispatches on `Schedule` (`cuda.rs:37-50`):
`Vectorized`/`Scalar`/`Strided`/`Reduction`/`RowReduce`. The relevant one is
`emit_strided` (`:187-249`), which already:

- takes a runtime `shape[]` + per-operand `s{i}[]` stride arrays + output `so[]`;
- unravels the linear index row-major over `plan.key.rank` (`:218-222`);
- **drops broadcast axes' offset terms** via `offset_expr` (`:619-632`, keyed on
  `o.bcast.is_set(d)`);
- **hoists a fully-broadcast operand** to a loop-invariant register load
  (`is_fully_broadcast`, `:613-615`; test `fully_broadcast_operand_is_hoisted`,
  `:874-885`).

The `cuda.rs:864` reference in the memory note is the `strided_2d_unravels`
test's comment — "Transposed (column-major) views: all operands strided, none
broadcast" — i.e. a transpose is *already* handled today, but **only as an
opaque strided operand**, never as a *named* transform that a fusion can be
recognized across. `offset_expr` computes `c{d}*s{d}[d]` from the *runtime*
stride array; it does not know the *structural* fact "this is a transpose of a
`[K,M]` producer," so it cannot participate in `derive_pattern`.

### 4.4 The schedule decision (`crates/baracuda-kernelgen/src/plan.rs`)

`build_plan` (`:85-127`) maps `Access::Elementwise` to
`Vectorized`/`Scalar`/`Strided` purely from the operands' `Contiguity`/`VecWidth`
(`:100-116`): any non-contig operand ⇒ `Schedule::Strided`. `Schedule`
(`:18-51`) is `#[non_exhaustive]` and `Copy`. So the *plan* already routes a
strided/broadcast cell to the strided emitter; what's missing is the IR's
*statement* of the transform.

### 4.5 The FKC pattern derivation (`crates/baracuda-kernelgen/src/pattern.rs`)

`derive_pattern` (`:79-107`) **rejects any non-elementwise op up front**
(`:80-82`):

```rust
if !matches!(op.access, Access::Elementwise) {
    return Err(PatternError::NotElementwise);
}
```

`PatternNode` (`:37-55`) is the derived FKC subgraph — `Op { op, operands,
consumers, extract }` + `Bind(u8)`. `unary_name`/`binary_name` (`:215-252`) map
IR ops to FKC §4.1 graph-`Op` names. There is **no layout-op case** — a layout
transform has no `ScalarExpr` node to walk, so it cannot currently be emitted
into a `pattern:` tree at all.

### 4.6 The live §5 seam (`crates/baracuda-kernelgen/src/jit.rs`)

`region_to_op` (`:406-425`) **hardcodes** `access: Access::Elementwise` and
builds the body via `node_to_expr`/`synth_op` (`:427-466`). `synth_op` covers
the arithmetic, unary, binary-fn, and scalar-param ops only; anything else is
`JitError::UnsupportedOp`. On the Fuel-facing side, `seam::optag_name`
(`:660-697`) maps `OpTag`→name and puts every unlisted tag (including
`Transpose`/`Reshape`/`BroadcastTo`/`Permute`/`Squeeze`/`Unsqueeze`) in the
`_ => None` honest-miss arm (`:694-696`), which becomes a `JitError::UnsupportedOp`
→ `SeamResponse::Declined` (`:820-824`). **The seam never panics on a layout
region today; it declines it.** That property is load-bearing and must be
preserved.

### 4.7 The inward optimizer (`optimize.rs`) & link (`link.rs`)

`optimize` (`optimize.rs:446-451`) is an e-graph over `ScalarExpr` only. If a
layout transform becomes a `ScalarExpr` node, the e-graph's `ENode` and every
match arm must grow to handle it (a strong argument **against** making it a
`ScalarExpr` node — see §5). `link_entry` (`link.rs:36-42`) keys purely on
`key.to_token()` + `revision_hash`, so a new keyed layout fact flows through the
link registry for free once `to_token` carries it.

---

## 5. Design / delta

### 5.1 The core decision — a per-operand **view descriptor**, not a `ScalarExpr` node

**Recommendation: model a layout transform as a per-input `View` descriptor
attached to the `OpDef`, consumed by the emitter's address computation, NOT as a
`ScalarExpr` node and NOT (initially) as a new `Access` variant.**

Rationale, grounded in the current code:

- A layout transform changes *where* an operand's element is read, not *what
  math* is applied. `ScalarExpr` is explicitly per-coordinate value math
  (`ir.rs:2-8`), and `lower_expr` (`backend.rs:67-96`) has no addressing concept
  — its `leaf` closure is handed the already-computed access string. Putting a
  transform in `ScalarExpr` would (a) force `optimize.rs`'s `ENode` +
  every rule/cost/build arm to grow (`optimize.rs:31-45`, `342-436`), (b) force
  `pattern.rs`/`contract.rs`'s `count_flops`/`ulp_bound`/`params_used` walkers to
  grow, and (c) blur the algorithm/schedule split the crate is built around. High
  blast radius, wrong layer.
- A new `Access` variant is the right home for a *global* iteration-shape change
  (that is exactly what `Reduction`/`RowReduce` are). A per-operand *read-through*
  is more naturally a per-operand fact. We keep `Access` free for 03/10's genuine
  loop-nest changes and add `View` orthogonally.

Concrete IR shape (`ir.rs`):

```rust
/// How input operand `i` is read relative to the op's iteration space — a
/// structural (compile-time) layout fact the emitter folds into address math.
/// `Identity` = read at the iteration coordinate (today's behavior).
#[derive(Clone, Debug, PartialEq, Default)]
pub enum View {
    #[default]
    Identity,
    /// Read the transpose/permutation of the producer: iteration axis `d`
    /// indexes producer axis `perm[d]`. Rank = op iteration rank; `perm` is a
    /// permutation of `0..rank`. (Transpose is the rank-2 special case.)
    Permute { perm: Vec<u8> },
    /// Broadcast a lower-rank / size-1 producer up to the iteration shape:
    /// `bcast` marks the iteration axes the producer does NOT vary along
    /// (stride 0). Rank-aligned to the iteration space.
    Broadcast { bcast: AxisMask },
    /// Reshape: the producer is contiguous with a different logical rank/shape
    /// but the SAME linear element order, so reading is a pure linear-index
    /// pass-through (no per-axis stride math). Carries the producer rank for
    /// the contract/keying only.
    Reshape { producer_rank: u8 },
}

pub struct OpDef {
    // ...existing fields...
    /// Per-input view (index `i` ↔ `Input(i)`). Empty ⇒ all `Identity`
    /// (back-compat: every existing OpDef is view-free). Length, when present,
    /// MUST equal `n_inputs`.
    pub views: Vec<View>,
}
```

Add an `OpDef::with_views(...)` builder and default `views: Vec::new()` in the
existing `elementwise`/`reduction`/`row_reduce` constructors so **every current
call site and test is unchanged**.

> **Why `Reshape` carries no stride math:** a reshape of a contiguous tensor is
> the identity linear-index map — the emitter already walks a linear `i`
> (`emit_scalar`/`emit_vectorized`). `Reshape` matters for *recognition* (the
> FKC pattern must state it) and for *keying* (a reshape-fused op is a distinct
> cell), not for address arithmetic. A reshape of a *non-contiguous* producer is
> a genuine gather and is **out of scope** — decline it (see §8).

### 5.2 Lowering: fuse-through-a-view in one pass (`cuda.rs`)

The `Strided` emitter already computes a per-operand offset from a coordinate
unravel (`emit_strided`, `offset_expr`). Extend the address computation so the
per-operand offset is taken **through** its `View`:

- `View::Identity` → today's `offset_expr` (coordinate `c{d}` × stride).
- `View::Permute { perm }` → the operand's coordinate on iteration axis `d` maps
  to producer axis `perm[d]`; emit `c{perm_inverse[d]}` bindings, i.e. index the
  producer with the permuted coordinate. For the contiguous-producer common case
  this is `Σ c[perm[d]] * producer_stride[d]` — one multiply-add chain, no copy.
- `View::Broadcast { bcast }` → reuse the existing broadcast-axis drop + the
  fully-broadcast hoist (`is_fully_broadcast`) verbatim; `Broadcast` is the named
  IR form of what the mask already does.
- `View::Reshape` → linear pass-through: the operand reads `in{i}[i]` on the
  contiguous/vectorized schedule with no change.

**Fusion-across-layout is then automatic:** the epilogue `ScalarExpr` is lowered
exactly as today (`lower_expr` with the `leaf` closure), and the `leaf` closure
returns the view-aware access string. One kernel reads the transposed/broadcast
input, applies the epilogue, writes the output — the "skip the contiguize
round-trip" win, with no new schedule and no new math node.

For the AOT pilot, a view-bearing op routes to `Schedule::Strided` (a permuted or
broadcast operand is non-contig by construction, and `build_plan` already routes
non-contig ⇒ `Strided`). The only `plan.rs` change is to pass the views through
`KernelPlan` so the emitter can read them (add `views: &'a [View]` to
`KernelPlan`, mirroring how `access` is threaded, `plan.rs:74`).

### 5.3 `StructureKey` facts it needs

The current key already distinguishes a strided/broadcast/flipped operand, so a
**transpose** and a **broadcast** are *already keyed* (they change `Contiguity`
/`bcast`/`flipped`). Two gaps:

1. **Permutation identity.** Two different transposes of the same-rank tensor can
   land in the same `Contiguity::Strided` bucket but need different address code.
   The runtime `s{i}[]` stride array disambiguates the *emitted* code, but the
   *key* must distinguish them so the right prebuilt cell is chosen and the FKC
   `accept` predicate is honest. **Add a per-operand `perm: PermCode`** (a compact
   canonical permutation code, e.g. the Lehmer/factorial-number-system index of
   the permutation, `u16` — `MAX_RANK=8` ⇒ `8! = 40320 < 2^16`). `Identity`
   permutation ⇒ code 0, so every existing cell is unchanged.
2. **Reshape fact.** A reshape changes the logical rank; key it via the existing
   `rank` plus a small **`view_kind` per operand** (`{Identity, Permute,
   Broadcast, Reshape}`, 2 bits) so a reshape-fused cell is distinguishable from a
   plain one. `Broadcast`/`Permute` here mirror the `View` enum for round-trip.

Both extend `OperandKey`, `to_token`/`from_token` (add fields to the
`<contig>/<bcasthex>/<vec>/<div>/<flip>` operand encoding, `structure_key.rs:622-629`
and `parse_operand:705-745`), and **bump `STRUCTURE_KEY_VERSION`** to 2
(`:48`). `Default` on the new fields must be the identity so a defaulted tail
operand still hashes equal (the `[OperandKey; MAX_OPERANDS]` tail invariant,
`:211-212`).

> Keep the added key surface **minimal**: `perm` + a 2-bit `view_kind` per
> operand. Do NOT add the full producer shape to the key — that is a numeric
> extent, which the whole design forbids keying on (`kernel-specialization.md`
> "Specialize on structure, not on literal shapes"). The producer shape rides the
> runtime `shape[]`/`s{i}[]` args, exactly as extents do today.

### 5.4 `pattern.rs`/`contract.rs`/FKC implications

- **`derive_pattern` must learn layout ops.** When an `OpDef` carries a
  non-`Identity` view on `Input(i)`, the derived `PatternNode` for that leaf is no
  longer a bare `Bind(i)` — it is an `Op { op: "Transpose"|"Permute"|"Reshape"
  |"BroadcastTo", operands: vec![Bind(i)], .. }` wrapping the bind, with the
  transform's shape facts carried in a new `attrs`-like field on `PatternNode`
  (perm vector / target shape). This is the emit side of the same `OpAttrs` gap
  Fuel has on the ingest side (§10 F1) — the two must agree on the encoding.
- The `NotElementwise` guard (`pattern.rs:80-82`) stays: a view-fused op is still
  `Access::Elementwise` (views are per-operand, not a loop-nest change), so it
  passes the guard. Only `Reduction`/`RowReduce` remain non-derivable.
- **`contract.rs` layout tokens.** `layout_token` (`contract.rs:269-276`) already
  emits `contiguous|inner_contiguous|strided|broadcast` per operand from
  `Contiguity`; a permuted/reshaped operand should advertise its structural view
  there too so the `accept.inputs[i].layout` gloss is honest. `awkward_layout`
  (`:280-285`) already returns `handles_strided` for a strided operand — a
  view-fused cell is `handles_strided` by construction.
- **`bytes_per_elem`/`flops_per_elem`** are unaffected (a view changes addressing,
  not element count or arithmetic).

---

## 6. Implementation steps

Each step names the file it edits. Do them in order; each is independently
compilable + testable.

1. **IR — add `View` + `OpDef.views`.** `ir.rs`: add the `View` enum (§5.1),
   the `views: Vec<View>` field, `views: Vec::new()` in every existing
   constructor, and an `OpDef::with_views` / builder. Import `AxisMask` from
   `baracuda-kernels-types`. Unit-test the enum + default. *No behavior change
   yet.*
2. **StructureKey facts.** `baracuda-kernels-types/src/structure_key.rs`: add
   `perm: PermCode` (Lehmer `u16`) and a `view_kind` (2-bit enum) to `OperandKey`;
   derive them in `derive_operand_key` from the source `OperandDesc` strides
   (a permuted operand is detectable as a strided non-flip whose sorted-|stride|
   order ≠ axis order); extend `to_token`/`from_token`/`parse_operand`; bump
   `STRUCTURE_KEY_VERSION` to 2. Round-trip test.
3. **Plan threading.** `plan.rs`: add `views: &'a [View]` to `KernelPlan`, set it
   from `op.views` in `build_plan`; a view-bearing op keeps routing to
   `Schedule::Strided` (add an assert that a `Permute`/`Broadcast` operand is
   non-contig, matching the existing routing).
4. **Emitter — view-aware addressing.** `cuda.rs`: in `emit_strided`, replace the
   per-operand `offset_expr` call with a `view_offset_expr(view, operand, ...)`
   that handles `Identity`/`Permute`/`Broadcast`/`Reshape` (§5.2). Reuse
   `is_fully_broadcast`/the hoist for `Broadcast`. Name the generated symbol with
   a view tag (e.g. `..._strided_r2_pT` for a transpose) so distinct views can't
   collide on a symbol. Golden-string tests per view.
5. **Pattern derivation.** `pattern.rs`: when `op.views[i]` is non-`Identity`,
   wrap the `Bind(i)` leaf in the corresponding layout `Op` node with its shape
   facts; extend `PatternNode` with a shape-facts field (perm/target-shape) and
   `to_fkc` serialization for it. Keep the `NotElementwise` guard. Tests:
   `transpose_fused_pattern`, `broadcast_fused_pattern`.
6. **Contract.** `contract.rs`: advertise the per-operand structural view in
   `layout_token`/the `accept.inputs` gloss; confirm `awkward_layout` returns
   `handles_strided`. Test the emitted `accept`/`return` blocks name the view.
7. **Seam — keep the honest miss, then wire when F1 lands.** `jit.rs`: leave
   `optag_name`'s layout tags in the `_ => None` arm (a `system-reminder`-style
   comment: "layout ops decline until `OpAttrs` carries shape facts — ask F1").
   Add a `region_to_op` branch (behind the `seam` feature) that, *once F1 exists*,
   reads the perm/target-shape from `OpAttrs` and sets `OpDef.views`. Add a test
   that a `Transpose` region **Declines** (never panics) under today's grammar.
8. **AOT catalog + OP-MATRIX.** `bin/kernelgen.rs`: emit a transpose-fused and a
   broadcast-fused pilot cell (e.g. `relu(transpose(a) + b)`), alongside the
   existing pilots. Update `OP-MATRIX.md` (Elementwise section) with the
   layout-fused row. Update `docs/design/kernel-specialization.md` §12 status
   (and flag the stale Param/AddScalar/MulScalar "not-emittable" note per the
   memory index — that shipped).

---

## 7. Test & on-device validation plan

House discipline (memory): nvrtc **headerless** compile + nvcc numeric on sm_89
(RTX 4070), compute-sanitizer where shared mem / cross-thread, adversarial-verify
after every substantive change.

- **Unit (Rust, `cargo test -p baracuda-kernelgen`):**
  - `View` default = `Identity`; `OpDef` back-compat (existing constructors emit
    empty `views`).
  - `StructureKey` round-trips with a permuted + reshaped operand
    (`token_round_trips` extended); `PermCode(identity) == 0`; a version-2 token
    parses and a version-1 token is still distinguishable by the `version` field.
  - `emit_strided` golden strings for `Transpose`(rank-2), a rank-3 `Permute`, a
    `Broadcast` (matches the existing hoist), and a `Reshape` (linear
    pass-through) — assert the address expression, not just the symbol name.
  - `derive_pattern` emits `op: Transpose`/`op: BroadcastTo` wrapping the bind,
    with the perm/shape facts; a reshape-of-non-contiguous op is rejected.
- **nvrtc headerless compile (`--features nvrtc -- --ignored`, extend the
  existing `nvrtc_compiles_*` tests in `jit.rs`):** each new emitted view kernel
  (f32 + f16) compiles to PTX with a `.entry`, proving header-light portability
  (the `cstdint` regression guard).
- **nvcc numeric on sm_89 — the oracle diff.** The design's stated oracle is the
  **generic strided kernel** (`kernel-specialization.md` §1, §10): build a small
  host harness that, for a `[M,K]` input, computes `relu(Aᵀ + b)` (and a
  broadcast variant) two ways — (1) the generated view-fused kernel, (2) an
  explicit `contiguize/transpose` copy + the plain elementwise kernel — and
  asserts bitwise-equal (both are exact/correctly-rounded arithmetic). This
  *directly* proves the fusion is a faithful skip of the round-trip, which is the
  whole point.
- **compute-sanitizer:** the view kernels are one-thread-per-output with no shared
  memory, so `initcheck` (no uninitialized reads at the permuted offsets) and
  `memcheck` (the permuted/broadcast index never reads OOB — the adversarial
  concern) are the relevant tools; `synccheck`/`racecheck` are N/A here (no
  `__syncthreads`). Run `memcheck` on the rank-3 `Permute` and the broadcast-hoist
  cases specifically.

---

## 8. Adversarial-verify checklist

Specific failure modes a skeptic pass must probe for **this** change:

- **Permutation inversion bug.** `perm` vs. `perm⁻¹` confusion: does iteration
  axis `d` index producer axis `perm[d]` or does producer axis `d` come from
  iteration axis `perm[d]`? Off-by-inversion silently transposes the *wrong* way
  and passes shape checks. The oracle diff (§7) catches it; assert the *direction*
  explicitly in a golden test with a non-symmetric shape (`[3,5]`, not `[4,4]`).
- **OOB read on a mis-declared view.** The `rowreduce_bare_rank1_weight_rejected`
  precedent (`cuda.rs:1223-1235`) shows how a mis-classified operand reads past
  its buffer. A `Reshape`/`Permute` whose declared producer rank/perm disagrees
  with the runtime `shape[]`/`s{i}[]` must be an OOB. Because the key abstracts
  extents away (the `validate_row_reduce` caller-precondition note,
  `plan.rs:161-173`), add a build-time `debug_assert`/validate that `perm` is a
  true permutation of `0..rank` and that a `Reshape` producer is contiguous;
  document that extent-agreement is the caller's precondition (same trust level as
  `n_out`/`k`).
- **`Reshape` of a non-contiguous producer** silently emitting a linear
  pass-through (a wrong gather). Must be **rejected**, not emitted. Test it.
- **Key collision.** Two distinct transposes hashing to one `StructureKey` ⇒ the
  wrong prebuilt cell dispatched, or a dishonest FKC `accept`. Verify `PermCode`
  distinguishes `[1,0,2]` from `[2,1,0]` and that `to_token` is injective over
  them (round-trip test with both).
- **Version-skew honesty.** A v1 token must not silently parse as a v2 key with a
  defaulted (identity) view — confirm `from_token` keys the version and a
  consumer can reject a mismatched schema.
- **Seam still declines, never panics.** A `Transpose`/`Reshape` region under the
  *current* (shape-fact-less) `OpAttrs` must return `SeamResponse::Declined`, not
  unwind across the trait boundary (the `synthesizer_declines_unlowerable_dtype`
  precedent, `jit.rs:951-975`). Test every layout `OpTag` declines.
- **Vectorization legality.** A permuted/flipped operand must NOT be handed a
  `float4` path (the `negative_stride_is_flipped` test already asserts flip ⇒
  `Scalar`, `structure_key.rs:964-971`); confirm a `Permute` operand keys as
  `Scalar` vec width so the emitter never `ld.128`s a strided address.

---

## 9. Definition of done

- [ ] `View` enum + `OpDef.views` land; all existing constructors/tests compile
      unchanged (empty `views` back-compat).
- [ ] `StructureKey` carries `perm` + `view_kind`, round-trips, and
      `STRUCTURE_KEY_VERSION` is bumped to 2; identity view ⇒ byte-identical to a
      v1-shaped cell modulo the version field.
- [ ] `cuda.rs` emits a correct one-pass fused-through-a-view kernel for
      `Transpose`/`Permute`/`Broadcast`/`Reshape`; golden-string tests green.
- [ ] **On-device validated on sm_89:** nvrtc headerless compile green for the new
      view kernels (f32 + f16); nvcc numeric **bitwise-equal to the
      generic-strided oracle** for a transpose-fused and a broadcast-fused op;
      `compute-sanitizer memcheck`/`initcheck` clean on the rank-3 permute and the
      broadcast-hoist cases.
- [ ] `derive_pattern`/`contract` represent the view (layout `Op` node in the
      `pattern:` tree, honest `accept.inputs[i].layout`); a reshape-of-non-contig
      op is rejected.
- [ ] **FKC honest-miss preserved:** every layout `OpTag` still `Declines`
      (never panics) across the seam under today's `OpAttrs`; a test proves it.
- [ ] Adversarial-verify pass (multi-agent find → dedup → skeptic refute) run and
      clean, with the §8 failure modes each explicitly probed.
- [ ] `bin/kernelgen.rs` emits a layout-fused pilot; `OP-MATRIX.md` +
      `kernel-specialization.md` §12 updated; the stale
      Param/AddScalar/MulScalar "not-emittable" note in the design doc corrected.
- [ ] Lockstep release discipline honored (all crates bump + republish via the
      `publish_alpha*.ps1` shape) when this ships.

---

## 10. Open questions / Fuel asks

**Design decisions to ratify before coding (Baracuda-internal):**

- **Q-D1 (the keystone decision) — ✅ RATIFIED 2026-06-30: per-operand `View`
  descriptor (§5.1).** Chosen over a `ScalarExpr` node (largest blast radius —
  ~8 exhaustive `ScalarExpr` matches + the `ENode` mirror in optimize.rs + the
  e-graph rules/cost, and it puts addressing in the value-math layer) and a new
  `Access` variant (wrong granularity — per-op not per-operand, and it crowds the
  `Access` space 03/10 need). `View` keeps the algorithm/schedule split intact,
  leaves the value-math walkers untouched, and gives 03/10 a clean shape-fact carrier.
- **Q-D2.** `PermCode` encoding for the key — Lehmer/factorial-index `u16` (§5.3).
  Confirm `u16` (fits `8!`) and that identity ⇒ 0.
- **Q-D3.** Whether `Reshape` is worth AOT-emitting at all in this item, or
  whether it should be *design-only* here (recognized/keyed) and left to emit
  under 03/10 where a genuine rank change matters. Reshape is the lowest-value,
  highest-ambiguity view (it's a no-op on address math for contiguous producers);
  consider shipping only `Transpose`/`Permute`/`Broadcast` emit in v1 and deferring
  `Reshape` emit.

**Fuel asks (cross-repo — the seam adoption is blocked on these):**

- **F1 — `OpAttrs` shape facts for layout ops (the blocker).** Fuel's
  `fuel_kernel_seam_types::OpTag` already lists
  `Transpose/Permute/Reshape/BroadcastTo/Unsqueeze/Squeeze`
  (`fuel-kernel-seam-types/src/lib.rs:52`), but `OpAttrs` (`:71-80`) carries only
  `scalars` + `axis: Option<i64>`. A layout region is **unusable** across the seam
  until `OpAttrs` (or the region node) carries the transform's shape facts — a
  **permutation vector** for `Transpose`/`Permute`, a **target shape / broadcast
  target** for `Reshape`/`BroadcastTo`, and the **dim list** for `Squeeze`/`Unsqueeze`.
  Propose the exact field shape (and its `extract:`/re-read semantics, mirroring
  how scalar params are re-read live) in a `docs/fuel-ask-layout-shape-facts-*.md`
  channel doc. Baracuda designs the emit-side encoding (§5.4) to match whatever
  Fuel lands.
- **F2 — Canonical permutation encoding agreement.** For the FKC `pattern:` a
  layout `Op` node must encode its perm/shape in a form Fuel's matcher canonicalizes
  identically (the §3a.2a "both sides canonicalize before matching" principle the
  crate already relies on, `pattern.rs:26-31`). Agree whether the perm is expressed
  absolute (`perm=[1,0]`) or relative-to-input-rank, and how a `BroadcastTo`
  target interacts with the operand's existing broadcast mask, so a Baracuda-emitted
  transpose-fused pattern matches a Fuel-discovered transpose subgraph.
- **F3 (confirm-only).** Fuel's roadmap reply already lists "layout nodes
  (`Reshape`/`BroadcastTo`/`Transpose`) with shape facts" as part of the item-3
  norm/linear workstream (`docs/fuel-reply-fkc-patterns-2026-06-19.md:175-176`).
  Confirm this brief's `View` vocabulary + F1 field shape is the agreed
  realization of that line, so the two repos converge rather than fork.
