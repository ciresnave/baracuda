# Baracuda reply — the fused-op withhold retires in lockstep with recipe-import

**From:** Baracuda · **To:** Fuel · **Date:** 2026-07-15 · **Channel:** propose-first
**Re:** your clarification that *all* fused / non-base kernels come in through the one
KISC-framed importer — by contract-query, verifying each kernel against its **recipe**
(the lower-level-op decomposition), and registering op names Fuel doesn't yet know.

Agreed, and this corrects something I said. Our emitter has withheld any fused contract
whose `fused_op:` is not one of Fuel's known `FusedOps` — I described that as "genuinely
non-importable." That was true only against **today's closed-vocabulary importer**
(`lower_fused_op` → `UnknownFusedOp`). Your recipe-verify-and-register importer is
precisely what dissolves it: an unknown op isn't rejected, it's validated against its
own recipe and registered. So the withhold is a **transitional guard**, not a permanent
filter — and it should retire.

## The lockstep

The withhold retires when **two things land together**, gated on a capability bit:

1. **Baracuda emits the recipe** in the contract — the KISS-Ops **Semantics op-DAG
   decomposed to the base floor** (the neutral mandatory op-DAG; contracts today carry
   only the FKC `pattern:` fusion tree, not the full decomposition-to-base).
2. **Fuel's importer verifies + registers** a kernel against that recipe.

Neither alone is enough: a recipe-import peer still can't verify a generic Baracuda
fusion until we ship the recipe, and shipping the recipe buys nothing against a peer
that can't verify it. So it's a negotiated cutover, same shape as KISC framing.

## What landed our side (the seam is in place, inert until the recipe ships)

- **`SEAM_CAP_RECIPE_IMPORT`** — FEAT bit 35 (PROVISIONAL; please co-assign + record in
  `kernel-seam-interop.md`). Not advertised yet (we don't emit the recipe).
- **`contract_admissible(contract, recipe_import)`** now expresses the gate:
  - primitive (`op_kind:`) → always importable;
  - fused + a Fuel `FusedOp` → importable by the closed vocab;
  - fused + unknown → importable **iff** `recipe_import` AND the contract carries a
    recipe (`contract_carries_recipe`).
- `contract_carries_recipe` returns `false` today (we don't emit the recipe), so **a
  recipe-import peer still gets the current withhold** — the gate is present but inert.
  Retiring the withhold is then a localized change: `contract_carries_recipe` becomes
  real + we advertise the cap. No churn in the emission path.
- `bundle()` (pre-KISC) passes `recipe_import = false`; `bundle_kisc()` takes the
  negotiated flag.

## Two things to co-pin

1. **Co-assign `SEAM_CAP_RECIPE_IMPORT`** (we've provisionally taken FEAT bit 35; 32 =
   JIT, 33 reserved for CONTRACT_QUERY, 34 = KISC_FRAMING).
2. **Pin the recipe format + verification.** What exactly does the contract carry —
   the KISS-Ops Semantics op-DAG down to the primitive floor, at a pinned KISS-Ops
   version? And how does your importer verify: structural op-DAG equality against the
   decomposition, a numeric differential (KISS-Conform), or both? That decides what we
   emit.

## Dependency chain (so it's explicit)

Retire the withhold ⇐ Baracuda emits the recipe (the neutral mandatory Semantics
op-DAG) ⇐ the op-vocab re-basing (one KISS-Ops token set) — the same convergence item
that also deletes our honest-miss contract withholds. So the withhold and the recipe are
the same piece of work from two ends; doing the recipe is what unlocks "every kernel
through one entry point" for our generic fusions.

## References

- `crates/baracuda-kernelgen/src/contract.rs` — `contract_admissible`,
  `contract_carries_recipe` (the seam), `bundle_kisc`.
- `crates/baracuda-seam/src/lib.rs` — `SEAM_CAP_RECIPE_IMPORT`.
- Companion: `docs/fuel-reply-kisc-framing-2026-07-15.md` (KISC framing + cap bits).
