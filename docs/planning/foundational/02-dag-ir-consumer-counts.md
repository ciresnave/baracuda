# 02 — DAG IR with consumer counts — implementation brief

> Foundational item **02** of the kernel-specialization sequencing plan. Turns the
> op-IR `ScalarExpr` from a pure **tree** into a **DAG** with shared interiors +
> consumer counts, so an interior value feeding 2+ consumers is representable,
> emitted once on-device, and described honestly across the FKC seam.

---

## 1. Objective

Today `ScalarExpr` (`crates/baracuda-kernelgen/src/ir.rs:16-47`) is a pure tree —
every child is a `Box<ScalarExpr>`, so the only way to express "value `t` feeds two
consumers" is to **duplicate the whole subtree that computes `t`**. That has two
consequences, both wrong-in-the-limit: (a) the CUDA emitter `lower_expr`
(`backend.rs:67`) re-renders each duplicated subtree literally, so a diamond
`f(g(x), g(x))` emits `g(x)` twice and a chain of `k` diamonds is `O(2^k)` source
text and redundant device recomputation; (b) the FKC pattern emitter
(`pattern.rs:187`) stamps `consumers: Some(1)` on **every** interior node
unconditionally — an honest claim only while the IR literally cannot represent a
multi-consumer interior. Build a value-numbered DAG (hash-cons + explicit
consumer counts) so a shared interior is (i) **stored once**, (ii) **emitted once**
(as a named `tmp`), and (iii) **described truthfully** to Fuel (`consumers: >1`
where a value is genuinely re-used, so Fuel's §3a.4 sole-consumer fusion-safety
guard is fed real data instead of a hardcoded `1`). This is **foundational**: it
is the representation `10 MatMul` reconstruction needs, the thing `optimize.rs`'s
e-graph already assumes internally but cannot surface, and the enabler of CSE and
accurate FKC `consumers` for every fused op the seam ever synthesizes.

---

## 2. Status & blockers

**Baracuda-unblocked (the whole codegen + representation half).** Nothing here
needs a Fuel answer to build: the DAG representation, the hash-cons interner, the
`let tmp = …;` emission in `cuda.rs`, the consumer-count computation, and the
CSE win in `optimize.rs` are all internal to `baracuda-kernelgen`. The existing
e-graph in `optimize.rs` (`memo: HashMap<ENode, Id>`, union-find, `find`/`canon`,
`optimize.rs:47-141`) is the value-numbering substrate — reuse it, do not build a
second one.

**Fuel-blocked (only the seam-honesty tail).** One narrow question must be
answered before Baracuda **emits** `consumers: >1` across the wire: does Fuel's
frozen `PatternNode` grammar carry a per-node consumer count on **import**, and is
`consumers: >1` a legal, matcher-honored annotation (see §10 ask A). Fuel's region
grammar is a **tree** (`fuel_kernel_seam_types::PatternNode { Op | Bind | SeeThrough
| Any }`, per `docs/fuel-reply-jit-frozen-types-2026-06-21.md:66-67`), so a
*Fuel-supplied* region with a shared interior arrives as **duplicated subtrees**
today. The part that **can proceed now**: the internal DAG, the emitter dedup, the
consumer-count analysis, and *conservatively-correct* FKC emission (see §5, "honest
fallback") — a shared interior with any consumer outside the fused region is a
**miss**, not a false `consumers: 1`. Nothing here regresses the live §5 seam,
which stays elementwise-only.

**Design-open:** whether the DAG becomes the *primary* IR (all producers build
DAGs) or a *derived* form (tree authored, DAG computed by hash-consing before
emit). This brief picks **derived** (§5) to keep the eDSL and every existing
`OpDef` unchanged — flag if the sequencing owner wants the stronger form.

---

## 3. Dependencies & sequencing

**Must land before this: nothing.** Item 02 is a keystone with no upstream
dependency (confirmed in the dependency graph). It shares machinery with the
existing `optimize.rs` e-graph but does not depend on any other foundational item.

**What this enables downstream:**

- **10 (MatMul/contraction design spike)** — a contraction graph is inherently a
  DAG (a shared activation feeds multiple accumulations); `region_to_op` cannot
  reconstruct that sharing while the IR is a tree. 02 is a hard prerequisite for
  10's reconstruction path. (10 also depends on **01 layout/shape**; 02 and 01 are
  the two keystones 10 sits on.)
- **Fusion discovery / FKC `consumers`** — accurate `consumers: >1` is exactly the
  §3a.4 fusion-safety signal Fuel's `match_region` engine consumes
  (`docs/fuel-reply-jit-frozen-types-2026-06-21.md:99-100`); today every fused op
  Baracuda emits claims `consumers: 1` on every interior.
- **CSE in `optimize.rs`** — the e-graph already *finds* common subexpressions
  (shared e-classes); 02 gives extraction a DAG target so a re-used value is
  extracted once instead of rebuilt into the output tree
  (`optimize.rs:build`, lines 420-436, currently always rebuilds a tree).
- Indirectly de-risks **05 (RowReduce seam adoption)** and **06 (fused residual
  LayerNorm)** epilogues, whose multi-`Reduced` / multi-`Input` epilogues are the
  first real shared-interior bodies (Softmax's `exp(x − max)` feeds both the sum
  stage and the epilogue — see `ir.rs:361` doc + the softmax test at
  `jit.rs:1303-1317`).

02 does **not** depend on or block 03/04/07/08/09; it is orthogonal to the
reduction-axis and dispatch-table work.

---

## 4. Current code — what exists today

### 4.1 The IR is a pure tree (`ir.rs:16-47`)

```rust
pub enum ScalarExpr {
    Input(u8), Const(f64), Param(u8), Reduced(u8),
    Add(Box<ScalarExpr>, Box<ScalarExpr>),   // ← Box children: a tree, never shared
    Sub(Box<ScalarExpr>, Box<ScalarExpr>),
    Mul(Box<ScalarExpr>, Box<ScalarExpr>),
    Div(Box<ScalarExpr>, Box<ScalarExpr>),
    Unary(UnaryOp, Box<ScalarExpr>),
    Binary(BinaryOp, Box<ScalarExpr>, Box<ScalarExpr>),
}
```

There is **no** node id, no arena, no `Rc`/index — a value that appears twice is
two distinct `Box` allocations. `#[derive(PartialEq)]` compares by structure, so
"same value" is only ever detected by deep structural equality, never identity.

### 4.2 The emitter re-renders every subtree (`backend.rs:65-96`)

`lower_expr` is the single shared body-walker (CUDA / future Slang all go through
it). It recurses **structurally**:

```rust
ScalarExpr::Add(a, b) => format!("({} + {})", lower_expr(a, lo), lower_expr(b, lo)),
ScalarExpr::Unary(op, x) => (lo.unary)(*op, lower_expr(x, lo)),
```

A subtree reachable by two paths is walked — and its text emitted — **once per
path**. There is no `tmp`/`let` mechanism and no memo. This is the concrete
`O(2^depth)` blowup + redundant device recompute. `cuda.rs` `emit_scalar` /
`emit_vectorized` / `emit_strided` (the three schedule emitters, e.g.
`cuda.rs:159`) each build a `Lowering` with a `leaf` closure and call `lower_expr`
inline into `out[i] = <expr>;` — none declares intermediates.

### 4.3 Pattern emission hardcodes `consumers: 1` (`pattern.rs:181-212`)

```rust
fn walk(e, is_root, path, extracts) -> Result<PatternNode, PatternError> {
    let consumers = if is_root { None } else { Some(1) };   // ← every interior gets 1
    ...
}
```

`PatternNode::Op.consumers` is `Option<u32>` (`pattern.rs:47-48`): `Some(1)` on
interiors (the FKC §3a.4 sole-consumer guard), `None` (= `any`) on the root. The
module doc (`pattern.rs:12-16`) states the intent: "a reused **input** → a repeated
`bind: i` … for free, because a shared operand is literally the same `Input(i)`" —
node-identity is free **for inputs**, but a shared **interior** cannot be
expressed, so `walk` never sees one and `Some(1)` is currently always true. The
serializer `node_lines` (`pattern.rs:371-372`) emits `consumers: {c}` verbatim.

### 4.4 `region_to_op` is tree-only *by construction* (`jit.rs:395-425`)

The docstring is explicit (`jit.rs:401-405`):

> The region's per-node `consumers`/`extract` fields are **ignored** and
> regenerated by `derive_pattern` under the sole-consumer rule — **sound because
> the IR is a pure tree (no shared interiors)**, so the only fusable shape
> (sole-consumer interiors) is the only representable one.

`node_to_expr` (`jit.rs:427`) rebuilds a fresh `ScalarExpr` per `PatternNode`, so
a Fuel region whose two operands point at the *same* producer node becomes **two
independent subtrees**. That soundness argument is exactly what 02 changes — so
02 must **update this docstring and the regeneration logic** (§5).

### 4.5 The e-graph already hash-conses (`optimize.rs:47-141`)

```rust
struct EGraph {
    parent: Vec<Id>,                     // union-find
    class_nodes: HashMap<Id, Vec<ENode>>,
    memo: HashMap<ENode, Id>,            // ← hashcons: identical e-nodes share a class id
}
fn add(&mut self, n: ENode) -> Id { let c = self.canon(&n); if let Some(&id) = self.memo.get(&c) { return self.find(id); } … }
```

`ENode` (`optimize.rs:31-45`) is the id-keyed op shape (`Add(Id, Id)`, `Const(u64)`
NaN-safe-by-bits, etc.) — **this is the DAG node the whole IR wants**. `add_expr`
(`optimize.rs:143`) is already the tree→DAG interner. But `extract` (`optimize.rs:398`)
and `build` (`optimize.rs:420`) reconstruct a **`ScalarExpr` tree**, discarding the
sharing the e-graph found. 02 wires that sharing through to emit.

### 4.6 Contract op-counting walks the pattern tree (`contract.rs:160-165`)

`count_ops` recurses `PatternNode` structurally to decide primitive
(`op_kind`) vs fusion (`fused_op` + `pattern:`). If a DAG's shared interior is
serialized once (not duplicated), `count_ops` must count it once — relevant to
`is_fusion` and `flops_per_elem` (`contract.rs:138`, `count_flops` at
`contract.rs:204`, which today double-counts a duplicated subtree's flops).

---

## 5. Design / delta

### 5.1 Representation — value-numbered DAG (derived, not authored)

Keep `ScalarExpr` (the tree) as the **authoring** surface — the `Expr` eDSL,
every `OpDef`, and the seam boundary all stay identical. Introduce a **derived**
DAG produced by hash-consing the authored tree just before emission:

```rust
// ir.rs (new)
type NodeId = u32;

/// A value-numbered op DAG: nodes stored once, children referenced by id.
/// Built from a `ScalarExpr` by hash-consing (structural equality → shared id).
pub struct ExprDag {
    nodes: Vec<DagNode>,            // dense arena; index == NodeId
    root: NodeId,
    consumers: Vec<u32>,           // consumers[id] = #distinct parents that reference id (root gets +? see 5.3)
}

/// A DAG node — the op shape with id children (mirrors optimize.rs::ENode,
/// intentionally: they are the same abstraction). Leaves carry their payload.
pub enum DagNode {
    Input(u8), Const(f64), Param(u8), Reduced(u8),
    Add(NodeId, NodeId), Sub(NodeId, NodeId), Mul(NodeId, NodeId), Div(NodeId, NodeId),
    Unary(UnaryOp, NodeId), Binary(BinaryOp, NodeId, NodeId),
}
```

`Const(f64)` is not `Hash`; intern on `f64::to_bits()` exactly as `optimize.rs`'s
`ENode::Const(u64)` does (NaN-safe by bits, `optimize.rs:33`, `ir.rs`-side conversion
in `add_expr` at `optimize.rs:146`). **Do not** hash-cons `Reduced` and `Param`
away across positions in a way that violates the RowReduce contract — a `Reduced(i)`
is legitimately a shared leaf (that's the point of Softmax), and interning it once
is *correct* and *desirable*; the only invariant is that it stays a leaf and is
never folded across rows (`optimize.rs:33-35` documents this — the DAG inherits
it for free because a leaf has no children to fold).

### 5.2 Building the DAG — reuse the interner, don't fork it

Factor the hash-cons out of `optimize.rs` so both the optimizer **and** the emitter
build DAGs the same way. Two clean options; pick one and document it:

- **(A, preferred) `ExprDag::from_expr(&ScalarExpr) -> ExprDag`** in `ir.rs`, a
  standalone hash-cons interner (~40 lines, a `HashMap<DagNode, NodeId>` memo). Then
  make `optimize.rs` build its e-graph from an `ExprDag` (or keep its `EGraph` and
  add an `ExprDag`-producing `extract`). This keeps `ir.rs` dependency-free of the
  e-graph.
- **(B) `optimize::optimize_to_dag(&ScalarExpr) -> ExprDag`** — extend the e-graph's
  `extract`/`build` (`optimize.rs:398-436`) to emit an `ExprDag` (share by e-class
  id) instead of a tree. Folds optimization + sharing into one pass. More powerful
  (the optimizer's CSE lands automatically) but couples `ir.rs` to `optimize`.

Both are correct. **(A) is the lower-risk first cut** — sharing without semantic
change — with (B) as the follow-up once (A)'s emit path is trusted. The JIT synth
path (`jit.rs:363-367`) already calls `optimize(&op.body)` before `generate`; after
02, that becomes `optimize`-then-`from_expr`, or the single `optimize_to_dag`.

### 5.3 Consumer counts — the two distinct meanings (do not conflate)

There are **two** consumer notions and 02 must keep them separate:

1. **Intra-body sharing (Baracuda-internal, always safe).** `consumers[id]` =
   number of distinct **parent nodes inside this op body** that reference `id`.
   Drives the emitter: `> 1` ⇒ hoist to a named `tmp`. This is unconditionally
   Baracuda-unblocked and needs no Fuel input.
2. **FKC `consumers:` (cross-region fusion-safety, §3a.4).** The FKC field answers
   "outside the fused region, does this interior value have other consumers?" —
   the guard that says fusing is safe only if the interior is *dead* outside. For an
   **AOT-authored** op body, the body **is** the whole region, so a value used only
   inside the body has `consumers` (external) `= 1` **only if it is the region
   root or a value not otherwise escaping**. The current `Some(1)` was correct
   *because a tree interior is by definition sole-consumer within the region*. With
   a DAG, an interior used by `m` parents **inside the region** is still legally
   `consumers: 1` **externally** (it does not escape) — so the FKC value for an
   AOT body remains `1` for a non-root interior. The change is the **emitter**, not
   the FKC number, for the AOT case.

**Where FKC `consumers: >1` actually becomes real:** the *seam* path, when Fuel
hands a region in which a node inside the region also feeds a consumer **outside**
it. That information must come from Fuel's region annotation (§10 ask A). Until it
does, 02's rule at the seam is the **honest fallback**: if a candidate shared
interior cannot be proven sole-consumer-outside-region, **decline** (typed miss),
never emit a false `consumers: 1`. `region_to_op` (§5.5) enforces this.

### 5.4 Emitter — hoist shared interiors to `tmp` (`backend.rs` + `cuda.rs`)

Replace the structural `lower_expr` recursion with a **memoized DAG walk** that
emits each node with `consumers > 1` once, into a declared temporary, and
references the temporary thereafter:

```rust
// backend.rs — new signature; lower_expr(&ScalarExpr, …) kept as a thin
// `from_expr(e); lower_dag(&dag, …)` wrapper for callers not yet ported.
pub struct DagLowering<'a> { pub leaf, pub reduced, pub unary, pub binary }  // unchanged seams

/// Emit `dag` as `(prelude, root_ref)`: `prelude` is the block of
/// `<ctype> tmpN = <expr>;` declarations for every multi-consumer node in
/// topological order; `root_ref` is the string that names the DAG's value.
pub fn lower_dag(dag: &ExprDag, ctype: &str, lo: &DagLowering) -> (String, String);
```

A node with `consumers[id] == 1` is inlined at its use site (identical text to
today — zero diff for every existing single-use body); a node with
`consumers[id] > 1` is emitted once as `float tmp3 = (a + b);` in the prelude and
referenced as `tmp3`. The three `cuda.rs` schedule emitters change from
`out[i] = <lower_expr(...)>;` to `<prelude> out[i] = <root_ref>;`. Ctype comes from
the existing `scalar_ctype` path (`plan.dtype`). Topological order is a post-order
DFS over the DAG (dense arena → cheap `Vec<bool>` visited). **Vectorized** and
**strided** schedules reuse the same prelude, keyed on the per-lane/per-coord leaf
spelling their `Lowering.leaf` already provides.

### 5.5 Seam / `region_to_op` — reconstruct sharing, or miss honestly

Update `region_to_op` (`jit.rs:406-425`) and its docstring (`jit.rs:401-405`). The
tree-only soundness argument no longer holds. Two sub-cases:

- **Fuel region is a tree with duplicated subtrees** (today's wire reality): hash-cons
  it (`from_expr` over the built `ScalarExpr`) — structurally-identical subtrees
  collapse to one DAG node, so `f(g(x), g(x))` emits `g(x)` once. This is a pure win
  and needs no Fuel change. The re-fused **pattern** still serializes the region
  Fuel sent (recipe fidelity, `jit.rs:376-377`), so matching is unaffected.
- **Fuel region carries an explicit shared-interior / external-consumer annotation**
  (future, §10 ask A): honor it — emit `consumers: N` on that node. Absent the
  annotation, treat any interior as sole-consumer-within-region (safe for the
  re-fuse pattern) and rely on Fuel's import-time canonicalization (§3a.2a,
  `pattern.rs:29-31`).

The `consumers`-regeneration note at `jit.rs:401-405` must be rewritten to:
"interior sharing within the region is deduplicated by hash-consing; cross-region
consumer counts, when Fuel supplies them, are honored; absent them, an interior is
treated as region-local and the emitted `pattern:` re-describes the region Fuel
sent."

### 5.6 StructureKey facts it needs — **none new**

02 is a pure IR/emit change. It does **not** add a `StructureKey` field or a
schedule variant — a DAG body lowers under the *same* `Schedule::{Scalar,
Vectorized, Strided, Reduction, RowReduce}` cell as its tree form; only the emitted
text between `out[i] = ` and `;` changes (now possibly preceded by a `tmp` prelude).
This is deliberate: keeping the key stable means every existing golden contract and
`structure_key` token is byte-identical **except** where dedup removes duplicated
flops (see §7 golden-diff).

### 5.7 Contract implications (`contract.rs`)

- `count_ops` (`contract.rs:160`) and `count_flops` (`contract.rs:204`) walk the
  `ScalarExpr`/`PatternNode` **tree** and today **double-count a duplicated
  subtree**. After 02, cost is computed over the **DAG** (each node once) — a more
  honest `flops_per_elem`. Add a DAG-aware `count_flops` (count distinct nodes) and
  route the contract through it. This is a **contract value change** for any body
  with sharing — call it out in the golden update.
- `is_fusion` (`n_ops > 1`) and `op_kind` vs `fused_op` selection are unaffected in
  shape (a single-op DAG is still a primitive).

---

## 6. Implementation steps (ordered checklist)

1. **IR — DAG type + interner** (`ir.rs`): add `NodeId`, `DagNode`, `ExprDag`, and
   `ExprDag::from_expr(&ScalarExpr) -> ExprDag` (hash-cons on `DagNode` with
   `Const` interned by `f64::to_bits()`). Compute `consumers: Vec<u32>` during the
   interning walk (increment on each *distinct-parent* reference). Unit-test the
   interner in isolation (§7.1).
2. **IR — consumer analysis** (`ir.rs`): finalize the intra-body `consumers[]`
   semantics (distinct parents, not raw edge count — a node used twice by the *same*
   parent, e.g. `x*x` as `Mul(a,a)`, still counts as one shared value → still
   hoisted; that is correct and desirable). Document the two-meaning split from §5.3.
3. **Emitter — DAG lowering** (`backend.rs`): add `lower_dag(dag, ctype, lo) ->
   (prelude, root_ref)` (post-order DFS, hoist `consumers > 1` to `tmpN`). Keep
   `lower_expr` as a `from_expr`→`lower_dag` shim so no caller breaks in one commit.
4. **Emitter — wire the schedules** (`cuda.rs`): change `emit_scalar` /
   `emit_vectorized` / `emit_strided` (and the `RowReduce`/`Reduction` epilogue
   emit) to `let (prelude, r) = lower_dag(...); write("{prelude}out[i] = {r};")`.
   Ctype via the existing `scalar_ctype`. Ensure the `Lowering.reduced` /
   `Lowering.leaf` closures are unchanged (the DAG references leaves the same way).
5. **Optimizer — DAG-aware extraction** (`optimize.rs`, option B follow-up): make
   `extract`/`build` (`optimize.rs:398-436`) emit an `ExprDag` sharing by e-class id,
   or add `optimize_to_dag`. First cut may skip this (option A) and hash-cons the
   optimized *tree*.
6. **Pattern / consumers** (`pattern.rs`): keep `walk`'s `consumers: Some(1)` for
   the AOT-authored non-root interior (still correct externally, §5.3), but make it
   **DAG-derived** where a shared interior is region-root-reachable by ≥2 paths and
   the annotation says it escapes — parameterize `walk` to read a consumer count
   rather than a literal `1`. Preserve byte-identical output for all existing
   single-use bodies (golden lock).
7. **Contract cost** (`contract.rs`): add DAG-based `count_ops`/`count_flops`
   (distinct-node count), route `contract` through them. Update the golden for any
   body with sharing.
8. **Seam / `region_to_op`** (`jit.rs`): hash-cons the built `ScalarExpr` (dedup
   duplicated subtrees) and **rewrite the tree-only soundness docstring**
   (`jit.rs:401-405`). Add the honest-miss rule for un-annotated cross-region
   sharing (§5.5). No FFI/wire type change in this commit (the seam types stay
   frozen — §10 ask A gates any `consumers` wire addition).
9. **FFI/build wiring**: none required for the internal DAG (no new crate, no new
   `#[repr(C)]`). If §10 ask A is answered "yes, carry `consumers`", a *separate*
   follow-up threads it through `fuel_kernel_seam_types::PatternNode` — out of scope
   for this brief's core.
10. **Catalog / docs** (`OP-MATRIX.md`, `docs/design/kernel-specialization.md`): mark
    "DAG IR with consumer counts" **done** in the IR roadmap
    (`kernel-specialization.md:428-435`, the ORDER-3 list), and correct the stale
    `ir.rs` description at `kernel-specialization.md:390` ("a value DAG" — it lies
    today; make it true). Add a DAG row to OP-MATRIX if it tracks IR capabilities.

---

## 7. Test & on-device validation plan

### 7.1 Unit — interner + consumer counts (`ir.rs` tests)
- `from_expr` on a **diamond** `Add(Mul(Input0,Input1), Mul(Input0,Input1))` yields
  **one** `Mul` node, `consumers[mul] == 2`, `nodes.len()` counts it once.
- `x*x` (`Mul(Input0, Input0)`) → one `Input0` node, `consumers[input0] == 2`, and
  the `Mul`'s two children are the **same** id (same-parent-twice still one shared
  value).
- A pure chain (no repeats) round-trips to a DAG with **all** `consumers == 1` and
  `from_expr` then a `to_expr` reconstruction equals the original (semantic identity).
- `Const` NaN/inf intern by bits (two `Const(f64::NAN)` share one node) — mirrors
  the `optimize.rs` NaN-by-bits invariant.
- `Reduced`/`Param` leaves intern once but are never merged with a structurally
  different node; a shared `Reduced(1)` in a Softmax-shaped epilogue counts correctly.

### 7.2 Unit — emitter dedup (`cuda.rs` / `backend.rs` tests)
- The diamond body emits its shared `Mul` as **one** `tmpN = (in0[i] * in1[i]);`
  and references `tmpN` twice — assert the source contains exactly one `(in0[i] *
  in1[i])` and two `tmp` uses.
- Every existing single-use body emits **byte-identical** source to pre-02 (lock via
  the existing golden tests in `cuda.rs`/`pattern.rs`/`contract.rs` — this is the
  "no regression" gate; the DAG must be transparent for non-shared bodies).
- Deep synthetic diamond chain (`k = 8`): source length is **linear**, not `2^8`
  (regression guard against the tree blowup this whole item exists to kill).

### 7.3 nvrtc headerless compile (sm_89, RTX 4070; `--features nvrtc`, `--ignored`)
Extend the existing headerless-compile suite (`jit.rs:1210-1373` pattern):
- A shared-interior f32 body (diamond) compiles headerless → `.entry` present.
- The same body at f16 (compute-in-float round-trip) compiles headerless (guards the
  `tmp` ctype spelling for the half path — `__half tmpN` vs `float`).
- A RowReduce epilogue with a shared `Reduced` (Softmax `exp(x−max)` feeding both
  the sum stage `pre` and the epilogue, `jit.rs:1303-1317`) still compiles after the
  DAG rewrite — the epilogue is where shared interiors first appear in real ops.

### 7.4 nvcc numeric on sm_89 (the correctness oracle)
Host harness (the established `nvcc` numeric path — the shape used for reductions):
diff device output against a **host `f64` reference** evaluated over the **same DAG**
(walk `ExprDag`, memoizing values — the reference must also share, so it is a genuine
oracle for the sharing, not just the tree). Cases:
- Diamond `g(x) = (a+b); out = g*g` — verify device `out == host g*g` bit-close
  (elementwise, `correctly_rounded`, so **bitwise** where the ops are exact).
- A transcendental shared interior `h = exp(a−b); out = h/(h+1)` — verify within the
  declared `max_ulp` (the shared `expf` must be computed once; correctness is
  independent of sharing, but this pins that dedup didn't perturb the math).
- Softmax (shared `exp`) f32 + f32-strict — numeric parity with the pre-02 kernel
  (dedup must be a **no-op on values**, only on text).

### 7.5 compute-sanitizer
The elementwise/strided DAG kernels have **no shared memory / no cross-thread**
(one thread per output element), so `racecheck`/`synccheck` are N/A there. Run
`initcheck` on the diamond kernel to confirm the `tmp` prelude reads no
uninitialized value. **Do** run the full `synccheck`/`racecheck`/`initcheck` on the
**RowReduce** case (7.3 Softmax) — it has the warp-shuffle + shared-mem tree reduce,
so the DAG rewrite of its epilogue must not perturb the barrier structure.

---

## 8. Adversarial-verify checklist (skeptic pass targets for THIS change)

1. **False `consumers: 1` across the seam** — the headline risk. Probe: a Fuel
   region where an interior escapes the region; assert Baracuda **misses honestly**
   (typed decline) rather than emitting `consumers: 1` and letting Fuel fuse an
   unsafe region. This is the §3a.4 correctness contract — a wrong `1` is a *silent
   miscompile of someone else's graph*.
2. **`Reduced`/`Param` over-merging** — a skeptic must check the interner does not
   fold a `Reduced(i)` across rows or hoist it out of its stage in a way that
   violates the RowReduce contract (`optimize.rs:33-35`, `plan.rs:270-276`). Assert
   a shared `Reduced` stays a per-row leaf and the `validate_row_reduce` guards
   (`plan.rs:174-293`) still fire.
3. **Emit-order / `tmp` scoping bug** — a `tmp` referenced before its declaration
   (topo-order error), or a `tmp` name colliding with `p{i}` params or `in{i}` leaves.
   Probe: a body mixing shared interiors, params, and multiple inputs; assert every
   `tmpN` is declared before use and names are collision-free.
4. **Const/NaN interning divergence** — two `Const(NaN)` must share (by bits), but a
   folded const must not change device bits (the `Rsqrt`/transcendental
   non-fold rules in `optimize.rs:184-213` must survive if option B is taken).
5. **Vectorized/strided lane aliasing** — in the vectorized schedule a `tmp` holds a
   **vector** (`float4`), and a shared interior used in two lanes must not be
   mis-hoisted to a scalar. Probe the diamond under `Vectorized{width:4}`.
6. **Golden drift for non-shared bodies** — assert the emitter is a **pure identity**
   for every existing single-use body (no stray `tmp`, no reordered text). A skeptic
   should diff a corpus of current goldens.
7. **Cost double-count regression direction** — after switching `count_flops` to the
   DAG, confirm `flops_per_elem` **drops** (or is equal) for shared bodies and never
   *rises* (a rising cost would be a new bug, not the fix).
8. **UAF/aliasing on the `tmp` prelude** — this codebase has a history of
   defensive-path UAF reintroduction (per house discipline); confirm the prelude
   holds no dangling reference into a dropped arena and that `lower_dag`'s borrow of
   the DAG outlives the returned strings.

---

## 9. Definition of done

- [ ] `ExprDag` + `from_expr` interner in `ir.rs`, with the two-meaning consumer
      semantics (§5.3) documented in the type's rustdoc.
- [ ] `lower_dag` in `backend.rs`; the three `cuda.rs` schedule emitters (+ RowReduce
      epilogue) route through it; `lower_expr` kept as a transparent shim.
- [ ] A shared interior is emitted **once** (named `tmp`) and computed once on-device;
      a diamond chain of depth `k` produces **linear** source (regression test green).
- [ ] **Byte-identical output** for every existing single-use body (all current
      `cuda.rs`/`pattern.rs`/`contract.rs` goldens green with no edits, except the
      intentional shared-body flops-count golden update, which is called out).
- [ ] nvrtc headerless compile green on sm_89 for f32 + f16 diamond and the shared-
      `Reduced` Softmax epilogue; `.entry` present.
- [ ] nvcc numeric on sm_89: shared-interior kernels match the host **DAG** oracle
      (bitwise where exact, within `max_ulp` for transcendentals); Softmax parity
      with pre-02.
- [ ] compute-sanitizer `initcheck` clean on the diamond; full
      `synccheck`/`racecheck`/`initcheck` clean on the RowReduce case.
- [ ] **FKC honest-miss preserved**: no path emits `consumers: 1` for an interior
      that could escape a Fuel region; un-annotated cross-region sharing declines.
- [ ] `region_to_op` hash-conses the region (dedup) and its **tree-only soundness
      docstring is rewritten** (`jit.rs:401-405`).
- [ ] Adversarial-verify pass (find → dedup → skeptic refute) run and clean on §8's
      failure modes.
- [ ] Docs updated: `kernel-specialization.md` IR roadmap marks DAG-with-consumer-
      counts **done** and the `ir.rs`-description line (`:390`) corrected;
      `OP-MATRIX.md` reflects the new IR capability.
- [ ] Lockstep release respected (all crates bump + full republish, `publish_alpha*.ps1`
      shape) — `baracuda-kernelgen` is `publish = false`, but any change to
      `baracuda-kernels-types` (only if §10 ask A lands a wire field) triggers the
      lockstep bump; a pure `baracuda-kernelgen`-internal 02 does **not**.

---

## 10. Open questions / Fuel asks

**Ask A (Fuel, gates only the seam-honesty tail, not the internal DAG).** Does the
frozen `fuel_kernel_seam_types::PatternNode` grammar carry, or can it carry, a
**per-node consumer count / "escapes-region" flag** on the region Fuel hands us —
and is `consumers: >1` a legal, matcher-honored annotation on an **imported**
`pattern:`? Two consequences: (1) if yes, Baracuda can emit truthful
`consumers: N` for a genuinely re-used interior and Fuel's §3a.4 guard fuses safely;
(2) if no / not yet, Baracuda's rule is the honest fallback — dedup for codegen, but
**decline** any region with an un-provable cross-region shared interior rather than
emit a false `consumers: 1`. Reference: `docs/fuel-reply-jit-frozen-types-2026-06-21.md`
(§4 `PatternNode` as-converged, §6 `match_region` sole-consumer guard). Propose-first
per the Baracuda↔Fuel channel discipline.

**Ask B (Fuel, informational).** When a Fuel-supplied region already contains a
shared interior expressed as **duplicated subtrees** (today's tree wire), is it
acceptable for Baracuda to collapse them (identical codegen, the recipe still
re-describes the region Fuel sent)? We believe yes — it is a pure codegen win with
no matching impact (§3a.2a canonicalization) — but confirm there is no Fuel
assumption that a duplicated subtree implies *distinct* runtime values.

**Design-open (Baracuda-internal, sequencing owner to rule).** Primary-DAG vs
derived-DAG (§2/§5.1): this brief builds the **derived** form (author trees,
hash-cons before emit) to keep the eDSL and every `OpDef` untouched. If 10 (MatMul)
or a future contraction grammar wants producers to build DAGs directly, that is a
larger IR migration and should be its own item — flag if it should be pulled forward.

**Design-open (Baracuda-internal).** Option A (standalone `from_expr` interner) vs
Option B (fold sharing into `optimize.rs`'s e-graph extraction, §5.2). A lands
sharing with zero optimizer coupling; B lands sharing **plus** the optimizer's CSE
in one pass but couples `ir.rs`↔`optimize`. Recommend A first, B as a fast-follow.
