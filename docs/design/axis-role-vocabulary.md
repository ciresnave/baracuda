# AxisRole vocabulary — the unified ORDER-3 axis-fact spec (design 2026-07-01; status updated 2026-07-10)

**Status:** **wired** (in-IR) via `Access::Contraction`. The `AxisRole` enum (`{Batch, FreeM, FreeN,
ContractedK, …}`) and `ContractionAxes { lhs, rhs }` now live in `crates/baracuda-kernelgen/src/ir.rs`
(the `Access::Contraction { axes, accum, epilogue }` node carries them; `ContractionAxes::matmul()` is the
canonical `M×K · K×N` case). This spec defines the axis-fact vocabulary that items 03 (reductions), 05
(fused-norm seam), and 10 (contraction / MatMul) all draw on, so they converge on one representation instead
of three private ones. Item 03 ships `reduce_axes: AxisMask` as the `{Reduced}` projection of this
vocabulary; the **wire/token** form of the full vocabulary (the cross-process `STRUCTURE_KEY_VERSION`-bumped
token codec) is the remaining forward-compat step and is still pending a real Fuel negotiation.

> **Status update (2026-07-10).** Contraction (item 10) has since **shipped** — but not the way
> §6/§8 below predicted. `Access::Contraction` and the `AxisRole` enum are wired **in-IR**
> (`crates/baracuda-kernelgen/src/ir.rs:1330`), and contraction structure facts ride an **additive**
> `ContractionKey` (`crates/baracuda-kernels-types/src/structure_key.rs:200`, derived at `:485`) that did
> **not** require a schema change — `STRUCTURE_KEY_VERSION` is still `1`. So the anticipated "one
> `STRUCTURE_KEY_VERSION` bump for a per-axis role token at item 10" did **not** happen; contraction landed
> additively instead. The cross-process per-axis **role-token codec** remains a genuine future forward-compat
> step, now **decoupled** from contraction (which no longer needs it). The design reasoning below is preserved
> as the original 2026-07-01 rationale; read §6/§8's "at item 10" as "if/when the role-token codec is built."

## 1. Why this exists (and why paper-now / wire-later)

Every ORDER-3 op needs to say *which axis plays which structural role* — reductions need "reduced vs kept",
a contraction needs "M vs N vs K vs batch". The temptation is to build one rich per-axis role field on the
`StructureKey` now and have 03/05/10 all use it. We deliberately **design it now but wire it later**, because:

- **Same version-bump count either way.** The `reduce_axes: AxisMask` field + its `x{:02x}`/`-` token codec
  **already exist** (`crates/baracuda-kernels-types/src/structure_key.rs`), so item 03 populating it is a
  token *value* change (`…|-` → `…|xNN`), transparent to Fuel (opaque token, K1) — **zero** bumps. Replacing
  it with a per-axis role field is a token *schema* change → one bump, whenever it lands. Building the full
  vocab now doesn't save a bump; it just moves it earlier.
- **The hard part (contraction roles) has no validated design yet.** `{Batch, FreeM, FreeN, ContractedK}`
  only matter for contraction, whose needs (K-tiling, Tensor-Core fragment roles, M/N/K structure classes)
  are exactly what item 10's design spike exists to determine. Designing the roles before that spike is
  guessing — and a wrong guess is a *second* bump to fix. Deferring → design once, informed → one bump.
- **The roles keystone (item 01) collapsed to recognition-only.** The axis-role machinery item 01 was to
  supply was reverted, so building roles in item 03 means inventing that infrastructure out of order.
- **It forces a cross-repo negotiation in the dark.** The role vocab needs Fuel to agree on contraction role
  *semantics* + canonical numbering — a negotiation coupled to the contraction design.

So: **design the superset here; project `{Reduced}` into `reduce_axes` for 03/05; wire the full role token at
item 10**, where the contraction design and the Fuel negotiation are both in hand. This mirrors the just-set
precedent where Baracuda *withdrew* the layout token version bump (K2) rather than change the wire before a
consumer needed it.

## 2. The vocabulary

A per-axis **role**, one per axis in `0..rank`, drawn from:

| Role | Meaning | Output presence |
| --- | --- | --- |
| **`Kept`** | Iterated, preserved 1:1 into the output (an elementwise/batch-of-the-map axis). | present |
| **`Reduced`** | Folded away by an associative combine (`Sum`/`Mean`/`Max`/`Min`/…). | absent (collapse) or size-1 (keepdim) |
| **`Batch`** | A contraction batch axis — iterated, present in output, shared by both operands (`bmm`'s `b`). | present |
| **`FreeM`** | A contraction free axis unique to the **lhs** → output rows (`M`). | present |
| **`FreeN`** | A contraction free axis unique to the **rhs** → output cols (`N`). | present |
| **`ContractedK`** | A contraction shared axis folded via multiply-accumulate (`K`); the "reduced" axis of a matmul. | absent from output |

Two structural observations that make this a *superset*, not a bag of unrelated tags:

- **`Reduced` and `ContractedK` are the same kind of thing** — an axis folded away — differing only in the
  combine (associative reduce vs multiply-accumulate over two operands). A reduction is the degenerate
  single-operand contraction. So `reduce_axes` (item 03) *is* the `{Reduced}` bit, and item 10's
  `ContractedK` is its two-operand generalization; a future unification can treat them uniformly.
- **`Kept`, `Batch`, `FreeM`, `FreeN` are all "present in output"** axes; they differ only in *which operands*
  vary along them. Reductions and elementwise never need the M/N/Batch distinction (single output-shaped
  operand), so their axes are all `Kept`.

### Per-op-class role assignment

| Op class (item) | Role assignment |
| --- | --- |
| Elementwise (shipped) | every axis `Kept`. |
| Reduction (03) | reduced axes `Reduced`, the rest `Kept`. `keepdim` = the `Reduced` axes are size-1 in the output vs. absent. |
| Fused norm / RowReduce (05) | the last axis `Reduced` (folded then broadcast back), the rest `Kept`; output is full-width (`Reduced` axis broadcast-back, so present-but-recomputed). |
| Contraction / MatMul (10) | `{Batch}*` shared iterated axes, `FreeM`* on lhs→out rows, `FreeN`* on rhs→out cols, `ContractedK`* shared+folded. |

## 3. `reduce_axes` as the `{Reduced}` projection (item 03, now)

Item 03 ships `reduce_axes: AxisMask` where bit `d` = "axis `d` is `Reduced`". This is exactly the projection
`role[d] == Reduced` of the full vocabulary. Forward-compat guarantee:

- **No re-key at item 10.** When item 10 introduces the per-axis role token, a reduction cell keyed today with
  `reduce_axes = {d}` maps to `role[d] = Reduced`, all others `Kept` — the *same* structural cell. Item 10 can
  either (a) keep `reduce_axes` and *add* the role token for contraction only, or (b) derive `reduce_axes`
  from the role token as the `{Reduced}` projection. Either way, no reduction cell is invalidated.
- **Derivation (item 03, option A).** For `OpCategory::Reduction`-family ops, `structure_key` sets
  `reduce_axes[d] = (input.shape[d] > 1 && output.shape[d] == 1)` on **keepdim-form** operands (the output
  presents each reduced axis as size-1). This is unambiguous *only* in keepdim form — a rank-collapsed output
  is un-inferable (input `[2,2,4]` reducing axis 0 vs 1 both give `[2,4]`, byte-identical operands), so the
  reduction key boundary requires keepdim-form output (see §6). The empty mask is retired as a "last-axis"
  sentinel and reserved for *non-reduction / undetermined* so a mis-formed reduction can't collide with a
  real last-axis cell.

## 4. Canonical numbering

Role bits/handles index **raw absolute axis position** (`0..rank`), matching Fuel's convention (its `perm` is
documented "ABSOLUTE — a permutation of `0..rank`" and reductions carry `axis: Option<i64>`). Canonicalization
follows `kernel-specialization.md`'s rule: **permutation-invariant *within* a role group, ordered *between*
groups** — i.e. two cells that differ only by reordering axes that share a role are the same cell, but the
Kept/Reduced (and Batch/M/N/K) partition is significant. Size-1 axes are squeezed before numbering (a size-1
axis is a no-op regardless of role), and both repos must squeeze identically — the numbering pin item 03/10
ask Fuel to confirm.

## 5. What is and isn't shape-inferable (the load-bearing distinction)

| Fact | Inferable from operands? | Mechanism |
| --- | --- | --- |
| Reduction `{Reduced}` (item 03) | **Yes**, from keepdim-form output (`in>1 && out==1`). | item 03 option A (derive in `structure_key`). |
| Fused-norm reduced axis (item 05) | **No** — softmax/norm output == input shape, no size-1 trace. | explicit axis via the region's `OpAttrs.axis` at the seam. |
| Contraction `ContractedK` (item 10) | **No** — the `M×N` output carries no `K` at all. | explicit, threaded via an item-10 `ContractionKey` (or the reserved additive-`OperandDesc` axis fact). |
| Contraction `M`/`N`/`Batch` | Partially (present in output) but role assignment needs op knowledge. | explicit at item 10. |

**Takeaway:** shape-inference (item 03 option A) is the honest mechanism *only* for plain keepdim reductions.
Everything else needs the axis fact threaded **explicitly** — which is why item 10 carries an explicit
`ContractionKey` rather than inferring, and item 05's fused family carries the axis in the region node. The
role vocabulary is the *unifying representation*; the *transport* is shape-inference where possible and
explicit where not.

## 6. Wire / token — deferred to item 10

- **Today (item 03):** `reduce_axes: AxisMask` populated (a `u8` bitmask; token `x{:02x}`/`-`). No schema
  change, no version bump. Keepdim-form-output precondition on reduction operands (confirm-only Fuel ask).
- **The role-token step (not yet built; see the status update above).** Contraction itself has already
  shipped *additively* via `ContractionKey` with **no** version bump, so this step is now **decoupled** from
  it. If a compact per-axis role encoding on the `StructureKey` is later built (a 3-bit × `MAX_RANK` code in a
  `u32`, or additive role fields), it would bump `STRUCTURE_KEY_VERSION` **once** and negotiate the role
  semantics + canonical numbering with Fuel then. `reduce_axes` either stays (redundant, harmless) or becomes
  the `{Reduced}` projection of the role code. The token stays opaque to Fuel throughout (K1), so the bump is
  a coordination event, not a parse change on their side.

## 7. Invariant: specialize on structure, not extents

Roles are **structural predicates** — *which axis plays which role* — never numeric extents. The keepdim
size-1 test (§3) reads extents only to *locate* a reduced axis within an `OpCategory`-gated reduction and set
its structural *bit*; it never keys the reduced *extent value* and never classifies an op as a reduction from
shape alone. This is the same discipline as `Contiguity`/broadcast-mask/`DivBucket` — all stride/extent-derived
structural buckets.

## 8. Open questions for the role-token step (item 10's contraction spike has shipped)

1. **Role encoding on the token:** per-axis 3-bit role code vs. additive per-role `AxisMask` fields (Batch/M/N/K)
   — pick the one that keeps `reduce_axes` a clean projection.
2. **Unify `Reduced` and `ContractedK`?** They're the same "folded axis" up to the combine — decide whether the
   token distinguishes them or carries a separate "combine" fact.
3. **Fuel negotiation scope:** confirm the M/N/K/Batch role assignment convention + the squeeze-then-number
   canonical order (bundled with the item-10 contraction seam design).
4. **Explicit-vs-inferred transport per role** (§5): pin which roles ride shape-inference (Reduced-keepdim) vs.
   an explicit `ContractionKey`/region attr (K, fused-norm axis).

## Related

- Items 03 (reductions) + 10 (contraction) have shipped; their standalone briefs were removed on ship (see
  `docs/planning/foundational/README.md` → Status, and git history). The contraction node now lives in
  `crates/baracuda-kernelgen/src/ir.rs` (`Access::Contraction`, `AxisRole`, `ContractionAxes`).
- `docs/design/kernel-specialization.md` (§1 thesis, canonicalization ~:141-149, `op_class.reduction_axes` ~:187).
