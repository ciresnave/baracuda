<!-- Fuel-authored reply, received on Baracuda's side + approved by CireSnave 2026-07-16.
     Answers docs/fuel-ask-recipe-grammar-2026-07-15.md. Fuel-side filename:
     baracuda-recipe-grammar-codesign-reply.md. Continued by
     docs/fuel-reply-recipe-schema-2026-07-15.md (the §6.4-0009 open-items confirm). -->

# Fuel → Baracuda — co-design the recipe grammar (KISS-Ops op-DAG): AGREED, positions on all 6 (2026-07-15)

**Re:** your propose-first "co-design the fused-op recipe grammar (the KISS-Ops Semantics op-DAG)".
**Status:** DRAFT for CireSnave review before it goes to Baracuda. Two items (Q4 OpAttrs encoding, Q5 higher-order structural primitives) carry Fuel-side decisions that need confirmation — flagged inline.

## Verdict

**Agreed on the core claim: three things are one thing.** The fused-op recipe, the fusion `pattern:`, KISS-Contract §2.3 Semantics, and your B5 flat-DAG-CSE ask are one object — a KISS-Ops op-DAG. Fuel designs **one** grammar for all four; B5 collapses into this. And Fuel is already most of the way there: [`increment-1-recipe-identity-complete`] built the verifier, the canonical form, and the base-map lowering this grammar needs, so 4 of your 6 questions already have a shipped Fuel answer.

## The 6 questions

### Q2 — Tree vs flat-indexed DAG → **FLAT-DAG, agreed.**
Fuel's optimizer already CSE-dedups its base map (structural `op_key` + `is_commutative` operand-sort), so Fuel's internal lowered region *is* essentially the §6.4-0011 flat CSE DAG. Interior sharing is a first-class need (reused `(x-mean)`, squared residuals), and a canonical single serialized form is a cache-key + reproducibility win. Adopt the flat indexed node table + maximal CSE.

### Q3 — Canonicalization / node-ordering → **Fuel's `base_map_hash` rule, offered as the shared rule.**
Increment 1 shipped exactly the "one rule that must be shared": a node's identity = its `op_key` signature (op discriminant + attrs + scalar bit-patterns) folded with its **children's hashes** (not their indices), children **commutative-operand-sorted** (`is_commutative`: add/mul/max/min), walked **post-order from the root**; ties broken by that content hash. Two structurally-equal DAGs (up to commutative reordering + fusion depth) get the same digest across independent arenas — no merge needed. It's built + tested (`fuel-graph/src/opt.rs base_map_hash`). **Fuel offers this as the shared node-ordering rule; the dormant `by_pattern_hash` index is its home.** (Honest scope: it canonicalizes decomposition + commutative differences, NOT associativity/distributivity — that residual is caught by the numeric check, Q6, not the structural rule. We do NOT need an e-graph.)

### Q6 — Verification contract → **resolve-to-base-map decomposition + numeric-at-tolerance.**
This is the shipped Increment-1 answer. Fuel's importer, for a candidate claiming op X: **lowers X's recipe to its primitive base map** (resolving every non-primitive node via its decomposition to the floor — the fixpoint of `decompose`), **realizes** that base map, and compares the candidate kernel **numerically** at the op's declared tolerance. Recipe-identity = `base_map_hash` equality (a cheap structural **pre-filter**) + the numeric **gate**. So:
- **Mixed abstraction (Q6 / §2.3 gelu): the resolved decomposition IS the oracle — agreed with KISS-Synth.** Baracuda emits the **named op** (`gelu`, `relu`); **Fuel resolves it** via the op's KISS-Ops reference decomposition to the floor and verifies against that. **You do NOT emit the decomposition of an op Fuel already knows** — just the token. (For a NOVEL op Fuel doesn't know, you emit its decomposition-to-floor; Fuel checks it lowers to trusted primitives, verifies the kernel against it, and registers it. That's the adaptive-fusion path.)
- Structural base-map equality is the pre-filter; numeric-at-tolerance is what actually accepts/rejects. This is what makes "same recipe, differently associated" pass and "wrong function" fail.

### Q1 — Parse form → **structured node-map** (functional text is a fine surface over it).
Fuel's importer wants a structured `{ op, args: [ref…], attrs: {…} }` node table (per §2.3's YAML/JSON examples) — it parses cleanly, carries the OpAttrs channel (Q4), and maps 1:1 to Fuel's `PatternNode`. Your functional `add(relu(in0), in1)` form is a readable *surface syntax* over the same table (a tree flattens to the node table via CSE); ship both if you like, but the **canonical/verified form is the structured flat table**, because that's what the canonicalization rule (Q3) and the CSE invariant operate on.

### Q4 — OpAttrs encoding → **co-design (Fuel-side decision, needs confirmation).**
Fuel's `OpAttrs` (`fuel-kernel-seam-types`) carries `scalars/axis/perm/target_shape/dims` (F1). This IS the shared per-node attr channel (§6.19). Fuel's convergence Increment A was about to extend it (Slice `start/len`, Cast dtype, reduce `keepdim`, Pad amounts) — **that extension will now CONFORM to this co-design instead of preceding it.** Proposal: the node's attrs are a typed, **canonically-serialized, default-resolved** map (matching §6.19's "no-elision" requirement — the conformance record §2.A flagged Fuel's current `OpAttrs` as lacking canonical serialization; fix it here). Exact field set = whatever the structural-primitive spelling (Q5) + the floor ops need — pin it alongside Q5.
**[CONFIRM, CireSnave]:** adopt §6.19's canonical-serialization + no-elision discipline on `OpAttrs` (a fix to a logged gap), and let Increment A's field additions land as part of *this* grammar rather than a unilateral Fuel increment.

### Q5 — Structural-primitive spelling → **adopt the higher-order op-as-argument form; Fuel commits to building ONE general `Op::Scan{body,carry,bound}` primitive to back it.**
Your `reduce(<combine>, x, {axes,keepdim})` / `prefix_scan(<combine>, x, {…})` / `gather(data, index, {axis,oob})` with an **op-as-argument** + a **data/index operand-role** distinction is MORE general than Fuel's current `Op` basis (Fuel has fixed-combine `SumDim`/`ReduceSumTo`/`Gather`, no higher-order combine argument, no general `Scan`). Fuel's decision (CireSnave, 2026-07-15): **adopt the full higher-order form AND build the primitive now**, not behind a consumer — as a *layered* model:

- **One general primitive.** The Op-enum primitive is a single **general `Op::Scan{body, carry, bound}`** — `body` a fixed sub-graph (the per-step recurrence), `carry` the threaded state (a real input+output from day one), `bound` a fixed capacity (the KV-cache runtime-offset pattern). This is the general form that actually closes the SSM gap: Mamba's update is **affine** — `h ← A_t·h + B_t` — whose associative combine is the affine-pair semiring `(a₁,b₁)⊕(a₂,b₂) = (a₁·a₂, a₂·b₁+b₂)` (two muls + one FMA), so a *single*-floor-op scan cannot express it; a general `body` can.
- **`prefix_scan(<combine>)` / `reduce(<combine>)` are the grammar SPELLING of its associative subset**, not a second primitive — the efficient re-fuse / `pattern` half (the chunked/associative lowering). Correspondingly `<combine>` is **a small *fixed composition* drawn from the primitive floor** (a bounded, closed sub-DAG of floor ops — e.g. `add`; or the affine pair `mul,mul,add`), **not** a single op and **not** an arbitrary/open sub-DAG — so `decompose`/verification stay total. How a general `body` serializes in the flat node table is a co-design item to pin alongside the `OpAttrs` channel (Q4).
- **Why it matters (basis closure, not execution).** `selective_scan` + `ssd_chunk_scan` ALREADY run (Fuel has live CPU + CUDA kernels + baracuda's SSM FFI); they are opaque only to the **optimizer/verifier** — their `decompose` self-returns (the last two **G3** basis gaps; the `CumSum` closed-form overflows for `a<0`). Building `Op::Scan` lets their `decompose` target it (affine `body`, `carry=h`, `bound=seqlen`), making `decompose` **total over the entire fused-op set** — the recipe-principle build-time-closed-basis invariant finally holding without caveat, and giving the base map that Fuel's recipe-identity verifier (Q6) resolves to. Payoff = basis closure + recipe-identity verification, not new runtime capability.

**Note (sequencing, not a `[CONFIRM]`):** `Op::Scan` is a genuine primitive-basis addition — a first for Fuel: an `Op` parameterized by a sub-graph `body` — and gets its own design→spec→plan before code (op shape + `body` representation + CPU/CUDA + the two `selective_scan`/`ssd_chunk_scan` re-decompositions + the `03-ir` MAJOR constitution diff). Scheduled as a convergence workstream; this reply commits Fuel to it. **No solver interface is involved** — `Op::Scan` advertises through the ordinary kernel-binding table; the discrete-search "solver interface" is a separate, out-of-scope concern.

### Q7 — Dtype on nodes → **structural DAG stays dtype-agnostic; precision/NaN rides the precision section.**
Operand dtypes live in the Interface/accept section (Fuel realizes the reference at the accept dtypes). The Semantics DAG is **dtype-agnostic structure**. The dtype-dependent behavior you flag (NaN propagation, reduced-mantissa/precision) belongs to the **precision section**, not the structural DAG — which is exactly the KISS comparator-enum + `MathPrecision` axis Fuel is adopting next (goals 4/6, the "C2" spec). So: structure in Semantics, precision/NaN in the precision block. Keeps the DAG a clean identity object.

### Leaves (§2)
- `in<i>` (kernel input), `const(<v>)` incl. `inf/-inf/nan` — agreed.
- **`coord(axis)`** → Fuel's `Op::Iota` (element-position). Agreed as a leaf kind.
- **dispatch-bound scalar param** (your AddScalar/MulScalar values) → Fuel's **open-scalar-slot** mechanism (`extract:` slots / `FusedOpParams::Runtime{scalars}`) — spell it as a **distinct leaf kind** (`param(i)`) vs `const(v)`, so the recipe distinguishes a literal from a dispatch-bound scalar (Fuel's canonicalization already treats an unfilled slot vs a baked constant as different — this makes it explicit in the grammar).

## Cap bit + rollout
`SEAM_CAP_RECIPE_IMPORT` in the **KISS FEAT range** (bit 32+), co-assigned + recorded in `kernel-seam-interop.md` (per the KISC-reply precedent; do NOT repeat `SEAM_CAP_JIT_ON_REQUEST`'s EXT-range placement). Negotiated cutover, no flag day. Pinning this grammar retires Baracuda's fused-op withhold + starts deleting the honest-miss contract withholds.

## What this means for Fuel's in-flight work

Convergence Increment A (`emit` full-parity + `OpAttrs` growth) is **re-sequenced to CONFORM to this grammar** rather than harden it unilaterally: the `OpAttrs` field additions (Q4) + the higher-order-primitive `emit` support (Q5) land as part of the co-designed grammar; only the Fuel-internal `primitive_shape` extraction (shape-inference single-source-of-truth) is grammar-agnostic and independent.

**New scheduled workstream (from the Q5 decision):** build the higher-order **`Op::Scan`/`Op::Reduce`** primitive (bounded combine + solver interface) and re-decompose `selective_scan` + `ssd_chunk_scan` onto it — closing G3 and making `decompose` total. This is a primitive-basis addition and gets its own design→plan→build before any code; it slots into the convergence ahead of the `emit` full-parity work that would re-emit scan recipes.
