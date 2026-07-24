# Fuel ask — extend the FKC pattern to a flat indexed DAG with shared computed nodes (CSE)

**From:** Baracuda · **To:** Fuel · **Date:** 2026-07-14 · **Channel:** propose-first
**Companion:** KISS-Grammar §6.4-0011 (flat indexed node table + maximal common-subexpression dedup).

This is a shared-seam change: the FKC `pattern:` grammar. It is **not** urgent and
needs no code from you this week — but it is the one convergence item where Baracuda
can't move unilaterally, because the pattern is what *you* parse. Flagging it now so
we design it together.

## TL;DR

1. The FKC v1 `pattern:` is a **nested tree** (`PatternNode = Op(operands…) | Bind(i)`,
   [`pattern.rs:36-55`](../crates/baracuda-kernelgen/src/pattern.rs)). It can share an
   *input* (`Bind(i)` reused) but **cannot share a computed intermediate** — a fused
   region like `(a+b) * (a+b)` serializes as **two independent `Add` subtrees**.
2. That has three costs: (a) the serialized pattern is **non-canonical** — the same
   fused region can emit differently depending on how the tree was built, undermining
   reproducible emission; (b) interior sharing is **inexpressible**, so a genuinely
   DAG-shaped region is either duplicated or declined; (c) it **diverges from
   KISS-Grammar §6.4-0011**, which mandates a *flat indexed node table with maximal
   CSE* as the neutral region form.
3. Proposal: FKC adopts a **flat indexed node table** — nodes addressed by `u32`
   index, operands referenced by index — so a shared computed subexpression is **one
   node referenced by several consumers** (true CSE). `Bind(input)` is unchanged; the
   new capability is sharing *computed* nodes. This is exactly the KISS-Grammar flat
   DAG.

## Why it matters

- **Canonical, reproducible emission.** With a flat CSE'd table, one fused region has
  exactly one serialized form. Today two structurally-identical regions can produce
  different `pattern:` bytes (duplicated subtrees in different orders), which is a
  reproducibility hole and a cache-key hazard.
- **Interior sharing becomes expressible.** Any region whose DAG isn't a pure tree
  (a shared normalization, a reused `(x - mean)`, a squared residual) is representable
  without duplication — and without a decline.
- **FKC == the KISS-Grammar neutral form.** §6.4-0011 already pins the flat indexed
  DAG + CSE as the advertisable-op region grammar. Adopting it on our seam means the
  FKC `pattern:` *is* the neutral form, not a tree dialect that has to be converted.

## The shape (for discussion, not final)

- A pattern is a **node table**: `nodes: [Node]`, where `Node = Op { op, operands: [NodeRef] } | Bind(input_index)`, and `NodeRef` is a `u32` index into the table (or an input bind). The **root** is a designated index.
- **CSE invariant:** no two nodes in the table are structurally identical (same op + same operand refs + same attrs). A shared subexpression appears once and is referenced by index from every consumer.
- **Determinism:** a pinned node-ordering rule (e.g. post-order from the root, ties broken by a stable node signature) so the table serializes canonically. (Our internal `canonicalize` sig is Baracuda-private today — §6.4-0011's ordering would replace it as the *shared* rule.)
- `consumers:` / `extract:` and the op-attribute channel ride as node fields, as they do now.

## What changes on each side

- **Baracuda (us):** migrate `PatternNode` from the nested tree to the flat indexed
  table + CSE during `derive_pattern`, and emit the indexed table from `to_fkc`.
- **Fuel (you):** the FKC importer reads a node table (operands are indices, not
  inline subtrees) and reconstructs the DAG. A pure-tree pattern is just the special
  case where every node has exactly one consumer.

## Compatibility & rollout

- **Negotiated, not a flag day.** Gate on a seam capability bit (`FLAT_DAG_PATTERN`).
  While a peer doesn't advertise it, we keep emitting the current tree form (which is
  the CSE table with sharing disabled); once both sides advertise it, sharing turns on.
- **No op-vocabulary change.** Same `op` names, same `Bind`, same attrs — only the
  *shape* of the container and the addition of computed-node sharing.

## What we need from you

1. A read of KISS-Grammar §6.4-0011 and a thumbs-up (or pushback) on moving the FKC
   `pattern:` to the flat indexed DAG + CSE.
2. Agreement on the **node-ordering / canonicalization rule** (this is the one part
   that must be *shared*, since it decides the byte form) and the `NodeRef` encoding.
3. The **capability bit** for negotiated cutover.

## References

- KISS-Grammar §6.4-0011 (flat indexed node table + maximal CSE) — github.com/ThinkersJournal/KISS.
- Current tree form: [`pattern.rs:36-55`](../crates/baracuda-kernelgen/src/pattern.rs)
  (`PatternNode`), the `to_fkc` emitter, and the Baracuda-internal `canonicalize` sig
  ([`pattern.rs:30-31`](../crates/baracuda-kernelgen/src/pattern.rs) — "need not equal
  Fuel's", which the shared ordering rule would change).
