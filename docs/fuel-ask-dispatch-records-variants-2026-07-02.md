# Baracuda → Fuel ask: dispatch/miss record schema + kernel variants on the wire

> Propose-first, per the channel discipline. Nothing here blocks Baracuda-side
> work: the ingest seam (`baracuda_kernels_types::merge`) and the variant
> emission are built and shipping regardless; your answers wire the **feedback
> half** of the loop and pin the wire vocabulary before either side cements it.
> Reference: `docs/design/kernel-specialization.md` §7–§8; the dispatch-table
> schema shipped in `baracuda-kernels-types::dispatch` (alpha.72+ branch
> `feat/kernel-specialization`).

## Context — what changed on our side

1. **The per-arch dispatch table is live** (design §7 mechanism): per
   `(op, structure-key, dtype, arch)` cell we record
   `winner + margin + ranked top-K + provenance {seeded, measured, reported}`,
   arch-gated, with a noise floor (`MIN_FLIP_MARGIN = 1.10`) so a within-noise
   "win" can't overturn a prior decision. `merge()` is the ingest entry point
   your records would flow through; a `Reported` row (yours) overrides our
   bench-measured rows when newer, and always overrides hand seeds.
2. **We are starting to emit schedule *variants* per cell** — multiple generated
   kernels for the same structure-key cell (first: a split-K outer-axis
   reduction beside the sequential baseline; next: WorkClass tiny-work
   schedules, ILP unroll, packed-transcendental f16). Policy (pinned): we ship
   **every validated variant**, each with its own honest FKC contract under the
   same `accept.structure_key`; the dispatch table is our default route and a
   seed/prior — **you remain the runtime decision-maker**, exactly per §8's
   premise that Fuel already times available implementations and picks.
   Bit-changing variants (reassociated reductions, approximate packed
   transcendentals) carry flipped `determinism`/`precision` blocks and are only
   selectable through those contracts — never silently.

## Asks

**A (schema, the core ask).** Confirm the §8 `dispatch_record` / `miss_record`
wire form (design doc ~:269–297) as the shape you'll emit, keyed on our
`StructureKey::to_token()` string. Specifically:

- Does a `dispatch_record` carry `candidates_considered[]` with **per-candidate
  `time_ns`**, or only the chosen winner? Our table's `ranked` top-K wants the
  former; winner-only degrades us to scalar updates (acceptable, but say so and
  we'll shape `merge` accordingly).
- Can each record carry enough hardware stamp to arch-gate — at minimum the
  compute capability (we map it to `ArchSku`), ideally device name + CUDA
  version? `merge` **rejects** a `Reported` row whose measured arch ≠ the
  token's arch; a stampless record would be dropped, not guessed.

**B (variants on the wire).** With variants, one cell can have **several
generated implementations**, distinguished by entry point
(`…_reduce_sum_ax1` vs `…_reduce_sum_ax1_splitk_partial`/`_combine`) and each
with its own contract. Two confirmations:

- Nothing on your side assumes ≤ 1 generated implementation per
  `accept.structure_key` cell? (Your planner already picks among
  implementations; we want to confirm multiple *generated* contracts in one
  cell don't collide in registries/caches keyed by structure-key alone.)
- Is an opaque per-contract `variant:` tag (string) acceptable front-matter, so
  your records can name which variant won without parsing entry-point suffixes?
  We treat the tag as opaque on both sides; the entry point remains the true
  identity.
- **Identity caveat our adversarial pass surfaced:** the collapse-form
  structure token cannot carry the reduced-axis set (provably undetermined for
  a rank-collapsed output — the item-03 keying finding), so two differently-
  shaped reduction cells (axis-0 vs axis-1, rank 2) share one token today. A
  variant's identity on the wire must therefore be `(structure_key,
  entry_point)`, never the token alone — please confirm your records key on
  the entry point (or the `ImplId` tuple), and this is a non-issue. The durable
  fix is the keepdim-form convention already on your queue from item 03.

**C (multi-kernel variants / launch protocol).** The split-K variant is a
**two-kernel** implementation (partials → combine) with a caller-provided
workspace (`n_chunks × cols × sizeof(acc)`) and a documented launch protocol
(grid/chunking pinned in the contract's `caps`/workspace fields; deterministic
for a fixed `n_chunks` — reassociated relative to the baseline, so its
`determinism` block says "deterministic; association differs from the
single-pass kernel"). Question: is a two-launch implementation with a
workspace requirement representable in your binding/dispatch today (the
`Workspace<'_>` type exists in `baracuda-kernels-types`), or should we also
always ship the single-kernel baseline as the fallback binding? (We ship both
regardless under the top-K policy; this only affects whether split-K is
*selectable* by you now or parked until the binding supports it.)

**D (count-unit hardening, small but load-bearing).** Our vectorized cells take
a **vector count** (`n/width`), scalar cells an **element count**. Before you
derive launch parameters from our contracts: confirm you read the kernel's `n`
semantics from the contract rather than assuming elements — we are adding an
explicit count-unit field to the contract front-matter (an 8× hazard on V8
cells otherwise). If you already treat `n` as opaque-per-contract, say so and
the field is documentation-only.

**Ownership (restating §8's v1 split for confirmation):** Baracuda owns the
committed table artifact and regenerates it batch-wise from your aggregated
records; you supply records and remain the live selector; the in-process v2
loop stays deferred with its hazards (design ~:291–297).
