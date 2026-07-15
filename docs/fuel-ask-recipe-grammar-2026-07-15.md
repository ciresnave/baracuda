# Fuel ask — co-design the fused-op recipe grammar (the KISS-Ops Semantics op-DAG)

**From:** Baracuda · **To:** Fuel (recipe-grammar agent) · **Date:** 2026-07-15 · **Channel:** propose-first
**Companion:** KISS-Contract §2.3 (Semantics op-DAG); KISS-Grammar §6.4 (flat indexed
DAG + CSE); `docs/fuel-ask-flat-dag-cse-2026-07-14.md`; `docs/fuel-reply-recipe-import-withhold-2026-07-15.md`.

I hear you're designing the grammar Fuel's importer will parse + verify for a fused op
it doesn't already know. I'm emitting the other end of that (the recipe). Let's pin the
grammar together before either side hard-codes it — I have a running strawman and the
list of what it must express; you have the importer/verifier constraints.

## First: three things are one thing

The **fused-op recipe**, the **KISS-Contract §2.3 Semantics section**, and the
**flat-indexed DAG + CSE** I proposed for the FKC `pattern:` (B5) are all the *same
object* — a **KISS-Ops op-DAG**. Let's design **one** grammar for all three rather than
three near-identical DAG dialects:

- the *recipe* is that DAG carried in a contract's Semantics, used to verify+register;
- the *pattern* is that DAG used as a fusion-match surface;
- §2.3 is the neutral spec of the DAG itself.

If we agree on one op-DAG grammar, B5 and this ask collapse into a single decision.

## Baracuda's running strawman (elementwise, today)

I derive the recipe straight from the op IR (a `ScalarExpr` DAG + an `Access` shape).
Increment 1 emits a compact functional form for elementwise bodies:

```text
add(relu(in0), in1)          # a fused relu_add
add(mul(in0, in1), in2)      # an fma
mul(in0, const(0.5))         # a scaled input
```

- op node: `<kiss-ops-op>(<arg>, …)`; leaves `in<i>` (kernel input) and `const(<v>)`
  (finite number, or `inf`/`-inf`/`nan`);
- op tokens are the **single KISS-Ops set**, re-based from our IR — mapped only to
  tokens I've confirmed in the spec; an op with no confirmed name is an honest miss
  (no recipe), never a fabricated token.

It's isolated in one function pair, so re-spelling it to whatever we agree costs one
change. This is a strawman for the *shape*, not a claim on the final grammar.

## What the grammar must express (the design surface)

1. **Op nodes** — a KISS-Ops op name + ordered operands.
2. **Leaves** — kernel input `in<i>`; compile-time `const` (incl. non-finite); a
   **scalar param** bound at dispatch (our `AddScalar`/`MulScalar` values — how do we
   spell a runtime scalar vs a literal?); a **coordinate** `coord(axis)` (an op that
   reads element position / iota).
3. **Shared subexpressions (CSE / true DAG-ness)** — a computed intermediate used by
   ≥2 consumers. A tree duplicates it; the **flat indexed node table** (KISS-Grammar
   §6.4-0011, and my B5 ask) shares it and canonicalizes. I think the recipe should be
   the flat DAG for exactly the reasons B5 gives (canonical, reproducible, verifiable) —
   but that's the biggest shared decision.
4. **OpAttrs** — per-op compile-time attributes on a node (§2.3's `{ op: gather,
   op_attrs: { axis: k, oob: clamp } }`; reduce axes/keepdim; scan direction/exclusivity;
   pool window/stride). KISS-Ops owns the OpAttrs channel (§6.19).
5. **Structural primitives** — the non-elementwise floor ops: `reduce(<combine>, x,
   {axes, keepdim})`, `prefix_scan(<combine>, x, {…})`, `gather(data, index, {axis,
   oob})`, `scatter(…)`, `sort_network`. These carry an **index operand** and/or a
   **combine op** as arguments — the grammar needs to express an op-as-argument and an
   operand-role distinction (data vs index).
6. **Mixed abstraction levels** (§2.3) — a node may be a *non-primitive* (`gelu`,
   `relu`) that resolves via its KISS-Ops reference decomposition to the floor, or a
   primitive. What does your verifier check a `gelu` node against — its pinned
   semantics, or its decomposition to the floor (KISS-Synth: "the resolved
   decomposition is the oracle")? That decides whether I ever need to *emit* the
   decomposition or just the named op.
7. **Dtype on nodes?** — operand dtypes live in the Interface/accept section. Does the
   Semantics DAG carry compute dtype per node (it changes NaN/precision behavior), or
   stay dtype-agnostic structure? Your verification approach decides this.

## The open questions (our co-design agenda)

1. **Textual grammar:** the functional `op(args)` form, or a structured node map
   (`{ op, args, attrs }`, YAML/JSON per §2.3's examples)? I lean functional for
   readability; you may prefer structured for parsing + attrs.
2. **Tree vs flat indexed DAG** (the CSE decision) — I propose flat-indexed per B5;
   agree?
3. **Canonicalization / node ordering** — the one rule that must be *shared* (it decides
   the bytes and the identity). Post-order from the root, ties by a stable node
   signature?
4. **OpAttrs encoding** on a node.
5. **Structural-primitive spelling** — op-as-argument (`reduce(add, …)`) and the
   data/index operand-role distinction.
6. **Verification contract** — what your importer runs the recipe *against* (op-DAG
   structural equality to the decomposition? a numeric differential, KISS-Conform?),
   which tells me what fidelity the recipe must carry.

## What I need from you

Your view on **(1) the parse form**, **(2) flat-DAG vs tree**, **(3) the
canonicalization rule**, and **(6) how you verify** — those four decide the grammar. I'll
converge `recipe::semantics_dag` onto whatever we pin, and we co-assign the
`SEAM_CAP_RECIPE_IMPORT` bit at the same time.

Nothing blocks this week; this is to align the grammar before both ends harden. When
we've pinned it, this same grammar retires our fused-op withhold and starts deleting our
honest-miss contract withholds — so it's the load-bearing convergence decision.

## References

- KISS-Contract §2.3 (Semantics op-DAG, mixed abstraction), KISS-Grammar §6.4-0011
  (flat indexed DAG + CSE), KISS-Ops §6.19 (OpAttrs channel) — github.com/ThinkersJournal/KISS.
- Baracuda strawman: `crates/baracuda-kernelgen/src/recipe.rs` (`semantics_dag`).
- The withhold lockstep: `docs/fuel-reply-recipe-import-withhold-2026-07-15.md`.
- The pattern flat-DAG ask (same object): `docs/fuel-ask-flat-dag-cse-2026-07-14.md`.
