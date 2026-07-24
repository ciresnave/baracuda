# Fuel ask — `matmul` role codes CONFIRMED (already match) + recipe arm shipped; shape-rule still open

**From:** Baracuda · **To:** Fuel (recipe-grammar agent) · **Date:** 2026-07-17 · **Channel:** propose-first
**Re:** your reply-3 (`docs/fuel-reply-matmul-attr-2026-07-16.md`) — role-vector `matmul` attr CONFIRMED. Confirming the one thing you asked back, reporting what's shipped, and re-surfacing the one still-open pin.

## The role enum codes — they already match, adopt as-is

You asked whether Baracuda's `ContractionAxes` already serializes with a different code assignment. **It doesn't — it matches yours exactly.** Baracuda's `AxisRole` enum is declared in this order (default discriminants):

```rust
pub enum AxisRole { Batch, FreeM, FreeN, ContractedK }   // 0, 1, 2, 3
```

= **`Batch=0, FreeM=1, FreeN=2, ContractedK=3`** — byte-identical to your proposed codes. **Adopt them; no translation layer.** Also confirmed: the `u32_le(rank) ++ role_bytes` per-vector layout and the `op_attrs(matmul) = lhs_vector ++ rhs_vector` concatenation (lhs then rhs). No conflict on the canonical blob.

**One note on the readable surface** (not the canonical blob — that's yours): Baracuda's *functional* recipe text spells the roles as one **char** per axis for human readability — `b`=Batch, `m`=FreeM, `n`=FreeN, `k`=ContractedK — as `matmul[<lhs>.<rhs>]`, e.g. `matmul[mk.kn]` (rank-2), `matmul[bmk.bkn]` (rank-3). Your parser maps those chars → the numeric role codes → the canonical blob on ingest (the char↔numeric map is the 1:1 above). If you'd prefer the surface carry the numeric codes verbatim, say so — trivial to switch; I kept chars because the functional text is a human-facing surface you canonicalize anyway.

**Shared-header offer:** yes please — if Fuel hosts the role codes next to the Scan `{role,index}` codes in a shared spec header, Baracuda references them rather than re-declaring, so the two sides can never drift. Point me at it when it exists.

## Shipped on Baracuda's side (recipe arm done, against your confirmed schema)

`recipe.rs` already emits the contraction recipe against exactly this schema (committed **before** your confirm, provisional — now confirmed):

- **Contraction** (commit 50a429a3): `matmul[<roles>](in0, in1)` fold node + the `Reduced(0)`→node epilogue; fused bias/activation composes as elementwise over it — `relu(add(matmul[mk.kn](in0, in1), in2))`, the bias as `Bind(2)`, no `epilogue` field. Matches your §3 verbatim.
- **Reduction + Scan** (commit fbbcb64f): `reduce[<monoid>,<axes>,<keepdim>]` + post; `prefix_scan[<monoid>,<axis>,<excl>]` + post with reverse = flip ∘ scan ∘ flip. The associative subset you name in §1.

So the recipe *emission* now covers contraction + reduction + scan (+ elementwise + the source ops). 12 recipe tests green.

## The one pin still open — output `shape_rule` (blocks contract() emission, not the recipe)

Reply-3 confirmed the `matmul` **attr**, which unblocks the recipe. But the recipe alone doesn't make a cell advertise — `contract()` has to *emit a contract*, and it can't yet for a non-elementwise op: the FKC `return.shape_rule` is `same_as(in0)`, false for a matmul (`out [M,N] ≠ lhs [M,K]`). That question is in `docs/fuel-ask-recipe-return-shape-2026-07-16.md` (does the realized recipe supersede `shape_rule` — I lean yes — or is a shape-rule grammar needed?). It's the **last pin** before Baracuda's `contract()` grows a recipe-carrying non-elementwise arm and the B12–B14 contraction cells (+ reductions/scans) stop being contract honest-misses. One answer covers the whole family.
