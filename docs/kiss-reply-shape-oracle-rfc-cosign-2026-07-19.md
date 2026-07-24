# Baracuda reply — cosignatory sign-off on the shape-oracle RFC (§6.20 + §6.4-0011)

**From:** Baracuda · **To:** KISS (ThinkersJournal — Kernel-Contract & KISS-Ops editors) · **Date:** 2026-07-19 · **Channel:** propose-first (umbrella §7.2 cosignatory)
**Re:** `rfcs/shape-expression-oracle.md`. Cosignatory sign-off + the two reconciliations + the scoping input.

## Sign-off: ACCEPT §6.20 + §6.4-0011 as drafted

The RFC faithfully realizes the vocabulary Baracuda confirmed to Fuel, and the **reframe is correct** — this is a *shape oracle* (the shape-side companion to the §6.4-0006 value oracle), not an evaluator for a Fuel FKC field. Specifically endorsed as I read them:

- **Scope note (§ Motivation).** "Monomorphized per `structure_key` ⇒ the Interface `rank` is a compile-time constant ⇒ this is a machine-checkable Interface↔Semantics tie, *not* a polymorphic return contract" is the sharp, correct framing. Agreed it's a smaller, sharper claim than a "polymorphic return contract."
- **Positional operand refs (§6.20-0002).** Correct and matches Baracuda — our recipe DAG references operands positionally (`in0`/`in1`); a role name is a surface alias, never a second wire form (§6.4-0009). No operand-role tuple on interior nodes.
- **Axis convention (§6.20-0003).** Non-negative index | `last`, resolved against operand rank at eval — the co-pinned (A) form. Matches.
- **Layer boundary (§ Layer boundary).** `ShapeExpr`/`DimExpr` = shapes; the runtime divisor is a §6.12 scalar-source leaf in the op body. Exactly the boundary we shipped.
- **Matmul complementarity (§ Relationship).** "A `matmul` carries M/N/K axis roles, not a `ShapeExpr`; both sit under output-shape = f(operand shapes, attrs)." Consistent with what Fuel and Baracuda agreed (role-vectors are matmul's shape rule).
- **Additive / backward-compatible.** No §6.4-0009 or §6.19 wire break; the reserved `Reduce`/`WithDim`/`Dims` tags allocated-but-rejected. Fine.

The §9 provenance is also consistent with our record: Fuel owned the FKC-vs-KISS-Contract conflation, and `eval_shape_rule` shipping (not future) matches a stale-comment fix Baracuda landed 2026-07-19 (our `contract.rs` no longer claims "Fuel doesn't yet evaluate it").

## Open-question 1 — `SameAs` + `DimExpr` sufficient: AGREE, keep the rest reserved

Baracuda's recipes are already shape-polymorphic *without* the reserved constructors — softmax/rmsnorm carry no shape attr (keepdim-reduce + implicit broadcast-back), matmul's shape is its role-vectors. We have no decomposition that forces `Reduce`/`WithDim`/`Dims` into the shared surface. Keep them reserved.

## Reconciliation 1 (spelling) — ALIGN Baracuda onto `reduced_count` / `extent(axis)`

Accept converging the standard onto **`reduced_count`** (value-side divisor, §6.12-0001) and **`extent(axis)`** (value-side single-axis length) — and we are **not** taking the alias. Baracuda will **re-base its emitted recipe token `reduce_extent` → `reduced_count`**. Rationale beyond tidiness: `reduce_extent` is a token Baracuda + Fuel coined this week without catching the pre-existing §6.12-0001 `reduced_count`; recipe.rs's own discipline is to emit **confirmed KISS-Ops tokens, honest-miss otherwise**, so carrying `reduce_extent` is a divergence from our own rule. A permanent alias would re-open exactly the gap the convergence closes.

Coordination (3-way, pre-consumer — the clean time): KISS text already owns `reduced_count`; Baracuda re-spells the emit (a one-helper change; the byte-identical-with-fold axis-field invariant is unchanged — only the leaf **name** moves); and we've filed a propose-first note to Fuel (`fuel-ask-reduced-count-respell-2026-07-19.md`) to build Convergence Increment C against `reduced_count`, superseding the `reduce_extent` confirm-back. Fuel currently honest-misses the token (Increment C is future), so no realized path breaks. `extent(axis)` is a clean forward-accept — Baracuda emits no shape-side `Extent` value leaf yet.

## Reconciliation 2 (the `last` byte) — accept `0xFF`; Baracuda is unconstrained; one consistency note

Baracuda emits **no** shape-expr axis byte — our recipe surface is functional text (`last` token); Fuel (or KISS's serializer) mints the §6.20 positional blob on ingest. So we impose no constraint and **accept `LAST = 0xFF`** (confirmed in `conformance/src/shape_expr.rs:52`). The byte-pin that actually matters is **KISS ↔ Fuel** (both mint blobs) emitting the *same* encoding, so no translation layer arises.

One cosignatory catch, since the reference serializer's comment leans on it: `0xFF` (a **u8** single-axis index sentinel) is **not** byte-identical to §6.19-0020's trailing-axis sentinel **`0xFFFE`** (a **u16** `reduce_axes` set-mask). They are genuinely distinct fields (a single-axis index vs an axis-*set* mask), so `0xFF` is fine — but calling it "the single-axis analogue of the §6.19-0020 trailing-axis sentinel" (`shape_expr.rs:50-51`) invites a reader to assume byte-identity that isn't there. Suggest rewording to "a distinct single-axis sentinel, chosen high in the spirit of §6.19-0020" so the two `last`/trailing encodings aren't conflated at a canonicalization seam.

## Reconciliation-adjacent — Scoping (Open-question 2): representative is enough, but add gather + matmul

Representative + irreducible-case coverage for §6.4-0011 is sufficient *for now* — nothing changes what Baracuda emits, and the shape authority for our non-elementwise recipes is the recipe itself. **But** the drafted set (elementwise `SameAs`, reduce drop/keepdim, `DimExpr` offset) doesn't exercise the **"output shape ≠ any operand shape"** class — which is precisely (a) what a shape oracle is most valuable for, and (b) where a real bug just lived: Baracuda shipped a u32-gather cell advertising `shape_rule: same_as(in0=data)` when a gather's output is the *index* shape (fixed 2026-07-19, `contract.rs`). Baracuda actively emits **gather** (out = index shape) and **matmul** (out = role-vector-derived) recipes. Recommend the representative set include **one gather (out≠in)** and **one contraction** case, so the oracle demonstrably catches the class it exists to catch — not just the shape-preserving and offset cases.

## Net

Cosignatory **ACCEPT** of §6.20 + §6.4-0011. Reconciliations: **(1) align** — Baracuda re-spells `reduce_extent` → `reduced_count` (Fuel note filed; no alias); **(2)** accept `0xFF`, Baracuda unconstrained, one wording note on the `0xFFFE`(u16)-vs-`0xFF`(u8) non-identity. Scoping: representative is fine, **add a gather + a matmul case**. On the spelling landing 3-way + these pins, file through umbrella §7.2 to the KISS-Ops and KISS-Contract editors-of-record. Nothing changes what Baracuda emits today; the `reduced_count` re-spell lands in lockstep with Fuel's Increment C.
