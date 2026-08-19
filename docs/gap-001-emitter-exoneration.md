# GAP-001 — `baracuda-cuda-emit` emitter exoneration (version bisect)

**Date:** 2026-08-19
**Owner:** Baracuda (emitter of record for the CUDA kernels under test)
**Why this file exists:** Fuel's `Cargo.toml` pinned `baracuda-kernelgen = "=0.0.1-alpha.77"` for eleven days with a comment naming `alpha.78` as an "all-zero-output regression," bisected `.76 PASS / .77 PASS / .78 FAIL`. That comment is being rewritten. A retraction that survives only in the artifact being retracted is not a record — so the emitter-owner attestation is preserved here, independently, where it outlives the pin comment.

## Scope

**This exonerates the VERSION BISECT. It does NOT clear the emitted code of the intermittent** (see §2).

## 1. No version boundary — the alpha.78-specific accusation is invalid

- The emitted `relu(add)` f32 kernel — scalar **and** vectorized — is **byte-identical across the 77→78 bump** (Baracuda on-device gate 5, commit `7bd90baf`, RTX 4070). No golden `.cu` changed; the `emit_scalar` signature is unchanged; the delta was a hoist refactor, behaviorally identical.
- The `count_unit` launch contract (`count_unit: elements`, class `elementwise`, `n = elem_count() = 7`) is **emitted correctly and read in the same unit** by the Fuel consumer. Emit and launch agree.
- Fuel's GAP-001 discriminator (Fuel lane `fuel-gap029-qwen2`, 2026-08-19) measured a **~25% NONDETERMINISTIC** failure at kernelgen `=0.0.1-alpha.78`: **20 fresh-process repeats, NaN-prefilled output → 15/20 clean, 5/20 all-7-elements-never-written**, with the **mock-PTX control passing 20/20** through the identical loader / launcher / prefill / readback.
- A ~25% intermittent across three single bisect runs yields `PASS/PASS/FAIL` by chance (p ≈ 0.25 per trial). **There is no version boundary.**

## 2. The ~25% intermittent is a separate, currently UNATTRIBUTED defect

- Leading candidate by elimination (three of four candidate mechanisms — wrong count, marshaling mismatch, body defect — are deterministic by construction): **a state-dependent write not landing in the buffer that is read.**
- **Its origin — emitted code vs. consumer buffer-lifetime / grid handling — is OPEN.** Gate 6 proved on-device output-correctness in its runs but did **not** run enough repeats to rule out a ~25% intermittent in the emitted kernel.
- So the emitted kernel is cleared of the **bisect**, **not** of the **intermittent** — and neither is the consumer.

## 3. What would close it

- **Version:** the same 20× repeat at `=0.0.1-alpha.77`. If it also fails ~25%, the version is conclusively irrelevant and the pin's premise is dead on its own terms.
- **Emitter attribution:** a high-repeat on-device sweep of the emitted kernel on the RTX 4070, **at Fuel's exact failing geometry** — the allocation size, grid dims, and the `elem_count()` the failing run used — with enough repeats to catch a 25% intermittent. The experiment is **asymmetric**: a *failure* implicates the emitted side decisively; a *clean* result clears it **only under the reproduced geometry**. The candidate mechanism (a state-dependent write not landing in the read buffer) depends on allocation / grid / launch geometry, all **consumer-supplied**, so a sweep run under the emitter's own geometry cannot go red by construction and would be a green-without-red instrument. Any clean result must be stated as *"clean at N repeats under geometry G"*, not *"the emitted side is clear."* (Fuel's lane found `layouts[n_inputs].shape().elem_count()` sizing the grid from a **layout** rather than an **allocation** — the specific geometry to reproduce.) Baracuda has offered to run this whenever Fuel supplies the failing geometry.

## Provenance

- Baracuda gate 5/6: commit `7bd90baf` (on-device, RTX 4070).
- Fuel GAP-001 discriminator: Fuel lane `fuel-gap029-qwen2`, 2026-08-19 — 20 fresh-process repeats, NaN-prefill discriminator + mock-PTX in-run control. Fixture `relu(add(a,b))`, 7×f32, `ArchSku::Sm89`, expected `[3,0,0,0,2,0,4]`.
