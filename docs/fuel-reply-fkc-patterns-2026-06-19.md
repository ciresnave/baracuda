# Baracuda review — FKC fusion-pattern spec (rev 2, 2026-06-19)

Review of Fuel's *FKC Fusion Patterns* DRAFT (rev 2, branch
`feat/kernel-contracts-dlpack`) from Baracuda's standpoint as an
**auto-generating** kernel provider. Produced by an adversarial multi-agent
pass (5 review dimensions, every finding refuted-or-confirmed before landing
here). Companion to
[`fuel-reply-telemetry-fdx-fkc-2026-06-19.md`](fuel-reply-telemetry-fdx-fkc-2026-06-19.md).

TL;DR — the `pattern:` grammar is a **clean mechanical fit for the slice we can
emit today** (pure `Add/Sub/Mul/Div` elementwise epilogues over `Input` leaves →
a ~40-line `derive_pattern` walker, *zero new IR*), and fixed-arity v1 covers
100% of our statically-known arity. Three things need your attention:

1. **Two §8 worked examples don't type-check against §5** — a provider copying
   them emits import-rejected contracts. Must-fix (§A).
2. **Commutativity is the one blocking ambiguity for an auto-generator** —
   positional-only + the §9 "emit both orderings *or* rely on canonicalization"
   is a 2ᵏ blow-up for us unless you canonicalize. We need one normative line
   (§B1).
3. Most of the headline targets (FusedLinear, RmsNorm, Softmax) are gated on
   **our** IR growth, not your spec — captured in §C so you know what patterns
   to expect when.

---

## A. Spec self-consistency — must-fix (a provider copying these emits rejected contracts)

1. **§8.2's RMSNorm guard uses subtraction that §5's grammar can't parse.**
   `guard: { shape: "self.axis == input(0).rank - 1" }` contains a binary `-`,
   but §5's operator set is `or < and < {==,!=,<,<=,>,>=} < %` with an explicit
   "no general arithmetic in v1" — there is no `-`. So the §8 header's "type-checked
   against §3" claim is false for this example, and "reduce the last axis" is *the*
   canonical norm-fusion guard, so this blocks every norm pattern. **Fix:** prefer a
   normalized negative-axis form (`self.axis == -1`) or a `last_axis` sentinel over
   admitting subtraction; or add a narrow `rank - <intlit>` form.

2. **§8.1's FusedLinear bias guard references `operand(0)` from a `bind` leaf.**
   `dim[0] == operand(0).dim[-1]` (commented "operand(0) = the MatMul") is carried
   by the `bind: 2` bias *leaf*. §5 defines `operand(j)` as "the j-th operand node
   of the node carrying the guard," and a `bind` leaf has no operands — the MatMul
   is two levels up and never bound, so there's no accessor that reaches it. **Fix:**
   `dim[0] == input(1).dim[-1]` (the MatMul's last dim == `b`'s last dim, and `b` is
   `bind: 1`). With this and (1), both §8 examples type-check.

3. **`operand(j)` auto-skip is asymmetric between guards (§5) and extract (§6).**
   §6 says extract `operand(j)` auto-skips `see_through` wrappers; §5 says nothing,
   so an identical path resolves to the *wrapper* in a guard and the *producer
   beneath* in extract. **Fix:** state the auto-skip resolution once (in §3a) and
   apply it to both.

4. **`see_through` `consumers` lacks the `N` form that Op nodes get.** §3.1 allows
   `<1 | N | any>`; §3.3 allows only `<1 | any>`; yet §3a.4 frames `consumers: N` as
   a general override. A shared transparent wrapper can't be written `consumers: 2`.
   **Fix:** extend §3.3 to `<1 | N | any>`, or note why see_through omits N.

---

## B. Expressiveness gaps for an auto-generating provider (with asks)

1. **Commutativity — BLOCKING, needs one normative line.** §3a.2 is strictly
   positional; §9 defers commutative matching with "emit both orderings *or* rely on
   Fuel canonicalization" — two non-equivalent escape hatches, never resolved. For us
   this is exponential: a body with *k* independent commutative nodes (`Add`/`Mul`)
   has up to 2ᵏ positional spellings of the same subgraph, and our planned e-graph
   optimizer reorders operands, so the form we emit may not match the form the user's
   graph presents. §8.1 bakes in one ordering with no note that the other occurs.
   **Ask:** state normatively in §3a that **Fuel canonicalizes commutative operands
   before matching** (deterministic sort keyed on producer node-id / op-rank) **and
   give the canonical order**, so our emitter and your matcher agree by construction —
   the same shared-canonicalization discipline `structure_key` already uses. Then we
   emit one ordering and we're done.

2. **Multi-output (single-sink v1) blocks training-stat-saving norm/softmax
   fusions.** §3/§3a.1 root at one sink and replace with one `Fused` node; §9 defers
   multi-output. But LayerNorm/RMSNorm saving `mean`+`rstd` and softmax saving
   `max`+`logsumexp` are *pattern-recognized* (their forward subgraph is the
   §8.2-style primitive subgraph) *and* multi-output, and `extract` only pulls scalar
   attributes, never a second tensor. So v1 patterns serve **inference norms only**.
   (A save-stats kernel can still ship as a §1 coarse builder op, forfeiting
   auto-recognition.) **Ask:** sequence multi-output **ahead of** the cosmetic
   deferrals — it's the line between our norm fusions being inference-only vs.
   training-capable, and training is where fused norms pay off most.

3. **Interior node-identity is inexpressible *and* missing from the §9 deferred
   list.** Repeated `bind` pins shared *leaf* inputs (great — see §D), but nothing
   pins a shared *interior* subtree as one node; `consumers: N` asserts fan-out
   *arity*, not identity, so a pattern over-matches when two structurally-equal-but-
   distinct interior subtrees exist. Fine for our tree-shaped v1 IR, but our e-graph
   optimizer does CSE and will hit it. **Ask:** add interior node-identity to the §9
   deferred list. (And please confirm repeated-`bind` is the intended shared-*input*
   mechanism — it reads correctly.)

4. **Cross-branch `input(i)` guards have no phasing rule.** §8.2's guards read
   `input(0)` from another branch, but §3a never says binds are resolved before a
   guard referencing `input()` runs (contrast the explicit "out-of-range ⇒ false" in
   §5). **Ask:** add a phasing rule — "all binds resolve before any `input()`-
   referencing guard evaluates," or an operand-visit order + `unresolved-input() ⇒
   false`.

5. **§5's dtype list names 5 of our 17 element kinds; plus a `Bool`-vs-`U8` clash.**
   §5 names `F16/BF16/F32/F64/U8`; absent and relevant to kernels we ship:
   `Fp8E4M3/E5M2`, `Complex32/64`, `S4/U4`, `Bin`, `S8`, `F32Strict`, `I32/I64`,
   `Bool`. Also §4.1 says comparison ops produce `U8` masks; ours produce `Bool` (a
   `#[repr(transparent)] u8`) — same byte, different name. Mostly bites only if we
   author a pattern over an fp8/int4/complex/bool subgraph. **Ask:** extend §5's list
   (or confirm `…` is open and pin the exact `#[non_exhaustive]` spellings), and
   resolve `Bool`-vs-`U8`.

6. **Gelu flavor is numerically undefined in §4.1 — and §3a.6 makes a wrong mapping
   our silent ~1e-4 bug.** §4.1 lists `Gelu` and `GeluErf` as distinct but never says
   which is erf-exact vs tanh-approx. We pin ours (`UnaryKind::Gelu` = erf-exact,
   `GeluTanh` = approx). **Ask:** define the flavor of bare `Gelu` and `GeluErf` so
   our mapping table pins the right name before we emit (also governs GeGLU and
   `BiasGelu` epilogues).

7. **Gated activations (SwiGLU/GLU/ReGLU/GeGLU) are matchable only as fragile
   split-based subgraphs, with no Slice-offset guard to prove the halves are
   complementary.** Expressible as `Mul(Slice(x,0..D), Silu(Slice(x,D..2D)))` with a
   node-identity bind on `x`, but §5/§6 expose no `Slice` start/stop accessor to guard
   `0..D` vs `D..2D`. **Ask:** confirm Fuel canonicalizes `chunk`/`split` → `Slice`
   the way our pattern assumes; and consider a `Split`/`Chunk` op or a gated-activation
   graph-Op + a Slice-offset guard accessor. Until then we ship these **coarse** (no
   pattern), which is fine.

8. **Composite/parameterized activations have no §4.1 anchor — we ship them coarse
   (not a spec defect).** Mish/Softplus/Hardswish/Selu/etc. have no §4.1 unary; runtime
   `LeakyRelu(α)`/`ELU(α)` have no scalar-param slot. §1 explicitly supports coarse
   ops, so this is *flagged for awareness*, no action required — **unless** you want
   them auto-discovered from primitives, in which case a `Softplus` unary (unblocks
   the Mish family) and a runtime scalar-param form would do it.

9. **Silent non-firing is the worst failure mode for an "auto-wires on import"
   feature.** §5 out-of-range ⇒ silently false, §3a.3 first-fail-no-reason, §9 defers
   failure telemetry — together a structurally-valid-but-never-matching pattern fails
   silently, recreating §0's "registered but unused" symptom. (Structurally malformed
   patterns *do* error at import, so this is scoped to valid-but-never-true.) **Ask:**
   pull "typed match-failure reason" earlier than §9, or at least an import-time
   "this pattern can structurally never match" lint.

---

## C. What Baracuda will do on its side (so you know what patterns to expect when)

Most of this is **ours, not yours** — the spec is an explicit cross-IR contract
that makes no assumption our IR matches Fuel's graph, and that's correct.

- **Now (zero new IR):** ship `derive_pattern(body)` — a small recursive walker
  emitting a valid §3 `pattern:` for every current op whose body is `Add/Sub/Mul/Div`
  over `Input(i)` (each arithmetic node → an Op node, each `Input(i)` → `bind: i`,
  `consumers: 1` on interiors — correct because our bodies are trees). This is the
  "third output per cell" alongside `.cu` + dispatch-key + `link_registry`, and it
  delivers the "offer-it-and-it's-used" property for **elementwise-epilogue fusions
  immediately**. We assert the realized bind set equals `[0, n_inputs)` so a malformed
  body is *our* build error, not your import error.
- **A `(Plan, Kind-variant, axis-attr) → §4.1 op-name` mapping table.** We have *no
  graph-Op layer* — op identity is a `*Plan` struct + a `u16` `*Kind` discriminant —
  so we own this many-to-one structural mapping, kept in lockstep with your
  `#[non_exhaustive]` `Op` enum. It's a cheap curated rename for the bulk of ops with
  ~a dozen known divergences (`Reciprocal/Square → Recip/Sqr`, `Eq → Equal`,
  `{Sum,Mean,…}+axis → {SumDim,MeanDim,…}` / `*All` split, `Cumsum → CumSum`, plus the
  Gelu flavor pending B6).
- **A pattern-equivalence certification gate** in the generate-FKC pipeline. Because
  we are the mechanical "author" §3a.6 trusts and you verify nothing, we discharge the
  "kernel ≡ literal pattern subgraph within FKC precision" obligation at generation
  time, per-contract and exhaustively (no tolerance inheritance across ranked patterns,
  per §3a.5). We will **not** populate the FKC precision block by naive projection of
  our `PrecisionGuarantee` (which carries intrinsic kernel-vs-true-math fields, not a
  pattern-divergence tolerance) — we add a separate pattern-divergence bound.
- **IR growth, in dependency order, unlocks the §8 targets** — and the §8/§10 headline
  fusions track *our* roadmap, not yours: **(1)** `ScalarExpr::Const` (+ a `lower_expr`
  arm) → unblocks `AddScalar/MulScalar/Clamp/PowI` + `extract:` (and eventually RMSNorm
  `eps`); **(2)** `ScalarExpr::Unary` → elementwise activation chains; **(3)** the big
  one — `Access::Reduction` + reduction nodes (`.axis`), a DAG IR (`Rc`/arena) with
  consumer counts (for `consumers: N`), layout nodes (`Reshape`/`BroadcastTo`/
  `Transpose`) with shape facts, and `MatMul` → after which §8.2 RmsNorm and §8.1
  FusedLinear derive verbatim. We'll sequence 1→2 quickly and treat 3 as the dedicated
  norm/linear workstream so the whole feature isn't blocked on it.

---

## D. Confirmations (what works cleanly, no action)

- **Pure-elementwise epilogue fusions round-trip our IR → §3 with zero friction** —
  the grammar is *not* the bottleneck for this slice; IR breadth is. `ScalarExpr →
  PatternNode` is a clean total mapping for the four arithmetic nodes + `Input` leaf;
  default `consumers: 1` on interiors is exactly the fusion-safety we want; unit-param
  elementwise needs no `extract`; "recognition only, cost-decides" (§3a.6) aligns with
  the Judge picking fuse-vs-not.
- **Shared `Input(i)` auto-yields the repeated-`bind` identity guard for free** — a
  reused operand is literally the same `Input(i)`, so we reproduce §3.2's node-identity
  (RMSNorm's `sq.inputs[0] == x_id`) with no value-numbering analysis.
- **Fixed-arity v1 fits 100% of what we emit** — our `OpDef` has a fixed `n_inputs`;
  `N` in `accept` equals it; the deferred variadic operands never touch us.
- **Our current op set is fully inside §4.1**, and the common-math `*Kind` spellings
  line up with the graph-Op names, so the mapping table is cheap.
- **Coarse/builder-only kernels (FlashAttn/SDPA, Rope, Conv, GEMM-epilogue families,
  MoE, scan) auto-wire as ordinary FKC contracts with no pattern, by §1 design** — most
  of our high-value kernels, no action.
- **Rule-ordering (§3a.5) and the consumer guard (§3a.4) don't relax the §3a.6
  equivalence obligation** — each contract carries its own pattern/precision/kernel, so
  there's no inherited tolerance; a larger fusion that out-ranks a smaller one is
  certified against its own whole pattern. Confirms our gate must be per-contract.

---

## E. Open questions (the first is blocking)

1. **(Blocking)** Does Fuel canonicalize commutative operands before matching? If yes,
   please state it normatively + give the canonical order (B1).
2. Does bare §4.1 `Gelu` mean erf-exact or tanh-approx (and `GeluErf`)? (B6)
3. Is `extract` the intended carrier for eps-like static scalars (reads that way in
   §6/§8.2)? Confirm so we build the body-scalar → `AddScalar`-attribute → `op_params`
   bridge once.
4. Does Fuel canonicalize `chunk`/`split` → `Slice`? Will you add a `Split`/Chunk op +
   Slice-offset guard accessor, or should gated activations stay coarse? (B7)
5. Is §5's dtype `…` open, and what are the exact spellings for our fp8/complex/sub-byte/
   bool kinds? How should a guard name a `Bool` mask the spec calls `U8`? (B5)
6. Will guard `operand(j)` (§5) and extract `operand(j)` (§6) be unified to both
   auto-skip `see_through`? (A3)
7. Will §3a state the `input(i)` phasing rule? (B4)
8. Can multi-output be sequenced ahead of the cosmetic §9 deferrals? (B2)
9. Can typed match-failure / an import-time never-match lint come earlier than §9? (B9)
