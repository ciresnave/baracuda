# Baracuda ask — layout/shape wire facts for cross-boundary layout fusion (2026-06-30)

**To:** Fuel · **From:** Baracuda · **Re:** item 01 (layout/shape IR nodes) — proposing the
shared-wire changes *before* cementing them, per the propose-first convention.

## Why this ask

Baracuda is adding first-class **layout views** to the kernelgen IR so the generator can express,
recognize, and emit a fused op that reads an input **through** a layout change
(`Transpose`/`Permute`/`Broadcast`/`Reshape`) in one pass — the §1 "skip the contiguize
round-trip" win. Two parts of this touch the Baracuda↔Fuel seam (Profile v1), so we are proposing
them for your input **before** locking anything:

1. a **`StructureKey` token delta** (per-operand `perm` + `view_kind`, `STRUCTURE_KEY_VERSION` → 2), and
2. your frozen **`OpAttrs`** needs shape facts for the layout `OpTag`s to be seam-adoptable at all.

We would rather converge the wire now than bump a ratified contract twice.

## What Baracuda has already built (non-wire — no confirmation needed)

- A per-operand `View` descriptor (`Identity | Permute{perm} | Broadcast{bcast} | Reshape{producer_rank}`)
  on `OpDef`; empty ⇒ all-`Identity`, so every existing op/test/golden is byte-identical. Landed + unit-tested.
- The AOT emit reads `perm` from the **`View`**, *not* from the key — so the on-device emit proof is
  decoupled from the wire. We are validating that now on sm_89.
- **Held pending your reply:** the `STRUCTURE_KEY_VERSION` bump and any live-seam adoption of layout regions.

## (1) Proposed `StructureKey` token delta — your confirmation wanted

Per operand, in addition to today's `<contig>/<bcasthex>/<vec>/<div>/<flip>`:

- `perm: PermCode` — the permutation as a **Lehmer / factorial-number-system index**, `u16`
  (`MAX_RANK = 8` ⇒ `8! = 40320 < 2¹⁶`); identity permutation ⇒ `0`.
- `view_kind` — 2 bits ∈ `{Identity, Permute, Broadcast, Reshape}`.
- Bump `STRUCTURE_KEY_VERSION` `1 → 2`. **Back-compat rule:** an all-identity-view cell encodes
  **byte-identical to a v1 cell modulo the version field**, and a v1 token stays distinguishable by
  its version (no silent parse as a defaulted-identity v2 key).

> **K1.** Do you *parse* the token internals, or treat it as an **opaque** join/dispatch/telemetry
> key? This decides how much the new layout fields matter to you beyond the version bump.
>
> **K2.** OK with the `→ 2` bump and the "identity ⇒ v1-identical modulo version" back-compat rule?
> (This is the only piece we are holding on you.)

## (2) `OpAttrs` shape facts — **your** side, and the live-seam blocker (F1)

Your `fuel_kernel_seam_types::OpTag` already lists `Transpose/Permute/Reshape/BroadcastTo/Unsqueeze/Squeeze`,
but `OpAttrs` carries only `scalars: Vec<f64>` + `axis: Option<i64>` — no permutation vector, target
shape, or broadcast target. A layout region is therefore **unusable across the seam** until `OpAttrs`
(or the region node) carries the transform's shape facts. Until it does, our `optag_name` keeps these
tags in the honest-miss/`Declined` arm (never panics).

> **F1.** What field shape do you want on `OpAttrs` (or `PatternNode`) for:
> - `Permute`/`Transpose` → a permutation vector (`Vec<u8>`, a permutation of `0..rank`);
> - `BroadcastTo` → the broadcast target (target shape, or a broadcast axis mask);
> - `Reshape` → the target logical shape (or producer-rank + target-rank);
> - `Squeeze`/`Unsqueeze` → the dim list.
>
> We will match our emit-side encoding to whatever you land.

## (3) Canonical permutation encoding — both sides must canonicalize identically (F2)

For a Baracuda-emitted transpose-fused `pattern:` to match a Fuel-discovered transpose subgraph, both
sides must canonicalize the perm the *same* way (the §3a.2a "both sides canonicalize before matching"
principle we already rely on).

> **F2a.** Express `perm` **absolute** (`perm=[1,0]`) or **relative-to-input-rank**?
>
> **F2b.** How does a `BroadcastTo` target interact with the operand's existing broadcast mask — is
> `BroadcastTo` the sole source of the broadcast fact, or can both be present (and if so, which wins)?

## (4) The stride convention — who applies the permutation? (S1)

Today Baracuda's strided emit computes `offset = Σ_d c[d] · s{k}[d]`, where the runtime stride array
`s{k}[]` is indexed by **iteration** axis `d` — i.e. a transpose is baked into the stride *values* the
caller supplies (caller pre-permutes; the kernel is generic, and two different transposes are
indistinguishable to it). For layout **fusion** — reading a *contiguous producer* transposed without a
materialized copy — we want the **kernel** to apply the permutation: `offset = Σ_d c[d] · s{k}[perm[d]]`,
with `s{k}[]` holding the producer's **own (producer-axis-order)** strides.

> **S1.** At the FDX/`OperandDesc` boundary, for a permuted operand do you deliver:
> - **(a)** producer-axis-order strides (the **kernel** applies `perm` — our preference; this is what
>   makes the read-through-a-view fusion possible), or
> - **(b)** iteration-axis-order pre-permuted strides (the kernel stays generic)?
>
> This is an ABI decision for the **seam** path only. The AOT path we control end-to-end and are
> validating with **(a)**.

## (5) Convergence confirm (F3)

Your roadmap reply (`fuel-reply-fkc-patterns-2026-06-19.md`) lists "layout nodes
(`Reshape`/`BroadcastTo`/`Transpose`) with shape facts" under the item-3 workstream. Please confirm our
`View` vocabulary + the F1 field shape is the agreed realization of that line, so the two repos
converge rather than fork two incompatible layout models.

## Decisions we need from you

| Ref | Decision | Blocks |
| --- | --- | --- |
| **K1** | Token opaque vs parsed on your side | how much the layout fields matter to you |
| **K2** | `STRUCTURE_KEY_VERSION → 2` + back-compat rule | Baracuda's key-field commit (the only held piece) |
| **F1** | `OpAttrs` shape-fact field shape | **live-seam adoption of layout regions** |
| **F2a/b** | Canonical perm encoding + broadcast interaction | cross-repo pattern matching |
| **S1** | Stride convention at the operand boundary | the seam emit ABI |
| **F3** | Convergence confirm | not forking the layout model |

Meanwhile Baracuda proceeds on the **non-wire** AOT emit proof (on-device sm_89) and **holds** the
`STRUCTURE_KEY_VERSION` bump until **K2**.

— Baracuda
