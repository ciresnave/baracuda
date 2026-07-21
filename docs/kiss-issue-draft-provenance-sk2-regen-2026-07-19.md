# ~~DRAFT~~ FILED + RESOLVED — KISS GitHub issue (ThinkersJournal/KISS)

> **STATUS (updated 2026-07-21): filed as KISS #60, resolved, and merged to `origin/main`.**
> The provenance regen landed (fixture golden `relu_add_generated_r1_cell` + PROVENANCE.md carry the sk2 token); `fix/60` merged to `origin/main` at `b666df5` under Eric's direction. The recorded three-way condition-1 byte-match (KISS fixture ↔ Fuel deriver ↔ Baracuda `aca0aa85` live sm_89 generator run) now lives on main. Scope: condition-1 same-namespace reproduction only — NOT §6.4-0004, NOT a KISS-Classify freeze (E6). Kept for the historical record.
>
> _Original draft (Baracuda-authored, provider corner, per [[kiss-change-via-issue-not-edit]]) follows._

---

**Title:** `conformance/cuda/generated`: regenerate the `relu_add` provenance to the `sk2` token (Baracuda generator now emits KISS-Classify-conformant keys)

**Labels:** conformance, provenance, good-first-issue

---

## Summary

The committed conformance artifacts under `conformance/cuda/generated/` record the Baracuda reference generator's `structure_key` for the `relu_add` cell in the **pre-spec `sk1|…|sm89`** form. Baracuda's generator has now aligned its token codec to KISS-Classify — schema `sk2`, the namespaced `cuda:sm89` target (§6.8), and the `ix32` index-width code (§6.7-0003) — so a fresh regeneration emits the **spec-conformant** token. This asks to regenerate the two artifacts so the recorded provenance is conformant and the token-level freeze-gate head-to-head compares the right bytes.

## What changed upstream (Baracuda)

Baracuda landed the D8 codec alignment (`baracuda` commit `aca0aa85`, branch `feat/kiss-convergence`):

- `STRUCTURE_KEY_VERSION` 1 → 2
- arch token `sm{80,89,90a}` → `cuda:sm{80,89,90a}` (**KISS-CLASSIFY-6.8-0002**)
- index-width field `i32`/`i64` → `ix32`/`ix64` (**KISS-CLASSIFY-6.7-0003** — deliberately distinct from the `i32`/`i64` **dtype** tokens)

The regenerated cell key (verified by re-running the generator, not hand-derived) is:

```
sk2|bin|f32|cuda:sm89|ix32|grid|r1|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-
```

Only the `sk1→sk2`, `sm89→cuda:sm89`, and `i32→ix32` fields change; the rank (`r1`) and the operand sub-keys are byte-identical.

## The stale artifacts

Two committed files carry the old token:

1. **`conformance/cuda/generated/baracuda_gen_relu_add_f32_co_v4.cu`** — the header comment (line 2):
   ```
   // op: relu_add | cell: sk1|bin|f32|sm89|i32|grid|r1|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-
   ```
   The kernel **body** is byte-identical after regeneration — the token lives only in this comment, not in the emitted CUDA.

2. **`conformance/cuda/generated/PROVENANCE.md`**:
   - the "Cell (generator's key)" table row (records the same `sk1|…` token);
   - the note that says *"the generator's key is schema `sk1|…|sm89` … it predates the KISS-Classify `structure_key` spec, which is `sk2|…|cuda:sm89`"* — that divergence is now **closed** on the generator side.

## Requested change

Regenerate from the current Baracuda generator and refresh the provenance:

```sh
# in the baracuda repo, on feat/kiss-convergence (>= aca0aa85)
cargo run -p baracuda-kernelgen --bin kernelgen -- /tmp/gen
cp /tmp/gen/baracuda_gen_relu_add_f32_co_v4.cu \
   <KISS>/conformance/cuda/generated/
```

Then in `PROVENANCE.md`:

- update the "Cell (generator's key)" row to the `sk2|bin|f32|cuda:sm89|ix32|grid|r1|…` token above;
- reword the note — the generator now emits the KISS-Classify `sk2|…|cuda:sm89|ix32` form, so the divergence is **closed** rather than "out of scope";
- bump the recorded generator `Commit` hash to `aca0aa85` (or the eventual merge commit).

## Why this is low-risk

- The kernel **body** is unchanged, so the on-device numeric differential (`tests/device.rs::generated_relu_add_matches_kiss_on_device`) stays green — it certifies the `relu` **semantics** (KISS-Ops §6.15), which the codec change does not touch.
- This is a provenance/metadata refresh that brings the recorded token into conformance with the very spec this suite tests against — and it makes the recorded token usable for a token-level provider↔consumer byte-match (the KISS-CLASSIFY-8-0004 freeze-gate condition), which the old `sk1|…|sm89` form could not satisfy.

## Context

Companion to Baracuda's D1–D8 reconciliation reply (provider corner) and the `fuel-ask-target-capability-namespace` ask that Fuel ACKed. This closes the last schema-lag item (reconciliation **D8**) on the artifact side.
