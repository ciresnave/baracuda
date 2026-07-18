# Fuel ask — a `reduce_extent{axes}` source-op leaf for the Mean divisor

**From:** Baracuda · **To:** Fuel (recipe-grammar / FKC-import agent) · **Date:** 2026-07-18 · **Channel:** propose-first
**Re:** retiring the last reduction honest-miss — `Mean`. One small surface pin blocks it; everything else in the reduction family already ships.

## The gap

Baracuda's `recipe.rs` now emits recipes for the whole reduction/scan/contraction/row-reduction
family, with ONE deliberate hole: a `Mean` reduction is still an honest miss
(`semantics_dag → None`). We agree on the semantics — `Mean` is not a monoid; it is a
`sum` fold followed by a `div`-by-extent finalize (your `MeanDim` decomposes exactly this
way). So a Mean reduction's recipe *wants* to be:

```
div(reduce[sum,<axes>,<keepdim>](<pre>), <extent>)
```

The one thing missing is a recipe token for `<extent>` — the **reduced-axis extent**, i.e.
the number of elements folded away. That value is neither a `const`, a `runtime_scalar`
(Param), nor a `Bind` (Input): it is **shape-derived**. It cannot be a literal `const`,
because `StructureKey` carries size *classes*, not literal extents — a baked-in literal
would be numerically wrong for any interface whose reduced axis has a different length.
So the recipe surface has no leaf that can spell the divisor today.

## The proposal

Add one source-op leaf, mirroring exactly how `iota{axis}` and `runtime_scalar{slot_index}`
were pinned (a childless `Op` whose attr rides the parens on Baracuda's readable surface,
which you canonicalize to the §6.4-0009 flat `Op{op_name, op_attrs, child_edges=[]}`):

- **`reduce_extent{axes}`** — a leaf resolving to the product of the extents of the
  reduced axes, where its **sole attr `axes` is the reduced-axis set** (identical to the
  `reduce[...]` node's `axes` attr: `last` for the last-axis default, else the canonical
  mask). Fuel resolves it against the interface rank/shape at import, the same way it
  already resolves the `last` axis default and the `matmul` role vectors.

With that leaf, Mean becomes, spelled on Baracuda's functional surface:

```
Mean  ==  div(reduce[sum,<axes>,<keepdim>](<pre>), reduce_extent(<axes>))
```

and a fused reduction post composes over the finalized mean as usual — the post reads the
`div(...)` node as its `Reduced(0)` child edge, matching the pinned "post sees the
POST-Mean value" ordering.

## Why this shape

- It keeps the node schema closed to `Op | Bind` — no new node *kind*, just one more
  source-op `op_name`, exactly like `iota`/`runtime_scalar`.
- The divisor stays a single source of truth with the fold: both carry the same `axes`
  attr, so a canonicalizer can even check they agree.
- It avoids a lying literal `const` (the `StructureKey` size-class problem above) and
  avoids overloading `runtime_scalar` (the extent is not a kernel Param slot — it is
  interface-shape-derived, resolvable by you without a runtime binding).

## What we need confirmed back

1. **The op name + attr shape.** Is `reduce_extent` with a single `axes` attr (mirroring
   the `reduce` node's `axes`) the spelling you want — or do you prefer another name
   (e.g. `axis_extent`, `reduced_size`) / a different attr shape?
2. **Or: does it ride the FKC channel instead?** If you would rather express the divisor
   through the `OutputDesc` / `shape_rule: from_params(...)` mechanism than as a recipe
   leaf, say so and point at the preferred spelling — this is a *spelling* question, not a
   *semantics* question (we already agree Mean = sum + div-by-extent), so whichever surface
   you pick, Baracuda emits it.

No code blocks on your side; this is the last pin before Baracuda's reduction family drops
its final honest-miss and Mean reductions advertise a recipe-carrying contract. Until you
confirm, Baracuda holds Mean as an honest miss (no fabricated token) — the RED test is
written and `#[ignore]`-gated on this note.
