# GAP-001 — `baracuda-cuda-emit` emitter exoneration (version bisect)

**Date:** 2026-08-19
**Owner:** Baracuda (emitter of record for the CUDA kernels under test)
**Why this file exists:** Fuel's `Cargo.toml` pinned `baracuda-kernelgen = "=0.0.1-alpha.77"` for eleven days with a comment naming `alpha.78` as an "all-zero-output regression," bisected `.76 PASS / .77 PASS / .78 FAIL`. That comment is being rewritten. A retraction that survives only in the artifact being retracted is not a record — so the emitter-owner attestation is preserved here, independently, where it outlives the pin comment.

## Scope

Originally scoped to the VERSION BISECT only; the intermittent was unattributed at authoring (§2). **UPDATE 2026-08-19: the intermittent's root cause was found and is entirely Fuel-side — the failing test never launched the Baracuda kernel at all (§Resolution). The emitter is fully cleared.**

## 1. No version boundary — the alpha.78-specific accusation is invalid

- The emitted `relu(add)` f32 kernel — scalar **and** vectorized — is **byte-identical across the 77→78 bump** (Baracuda on-device gate 5, commit `7bd90baf`, RTX 4070). No golden `.cu` changed; the `emit_scalar` signature is unchanged; the delta was a hoist refactor, behaviorally identical.
- The `count_unit` launch contract (`count_unit: elements`, class `elementwise`, `n = elem_count() = 7`) is **emitted correctly and read in the same unit** by the Fuel consumer. Emit and launch agree.
- Fuel's GAP-001 discriminator (Fuel lane `fuel-gap029-qwen2`, 2026-08-19) measured a **~25% NONDETERMINISTIC** failure at kernelgen `=0.0.1-alpha.78`: **20 fresh-process repeats, NaN-prefilled output → 15/20 clean, 5/20 all-7-elements-never-written**, with the **mock-PTX control passing 20/20** (a *sound* control — it launched on its own live device and correctly answered "does the loader/launcher work"; see §Resolution for why 20/20 didn't validate the subject).
- The **per-run failure rate is ≈ 0.25** — that is what 0.25 refers to, *not* the probability of the observed sequence. A lone FAIL is therefore ~25% likely at *any* single bisect point regardless of version, and the specific `PASS/PASS/FAIL` across three single runs arises by chance with probability 0.75 × 0.75 × 0.25 ≈ 0.14. Three single runs cannot place a boundary at 78. **There is no version boundary.**

## 2. The ~25% intermittent is a separate, currently UNATTRIBUTED defect

- Leading candidate by elimination (three of four candidate mechanisms — wrong count, marshaling mismatch, body defect — are deterministic by construction): **a state-dependent write not landing in the buffer that is read.**
- **Its origin — emitted code vs. consumer buffer-lifetime / grid handling — is OPEN.** Gate 6 proved on-device output-correctness in its runs but did **not** run enough repeats to rule out a ~25% intermittent in the emitted kernel.
- So the emitted kernel is cleared of the **bisect**, **not** of the **intermittent** — and neither is the consumer.

## 3. What closed it (both measured, 2026-08-19)

- **Version — CLOSED.** The same 20× repeat at `=0.0.1-alpha.77` failed **11/20 (55%)** — the *pinned "safe"* version fails **more** than the accused `.78` (5/20, 25%); pooled **16/40 (40%)**. Both versions fail; the version is conclusively irrelevant. (Not a claim that `.77` is worse: Fisher exact p=0.105 at n=20, and the `.78` arm ran under machine load while `.77` ran quiet — a race's rate is load-sensitive, so the 25/55 gap is not version-attributable. What survives: both fail, version irrelevant.)
- **Emitter attribution — RUN, RESULT: 20/20 clean.** A high-repeat on-device sweep of the emitted kernel on the RTX 4070 at Fuel's exact failing geometry (`OperandDesc::new(1,&[7],&[1],F32,4)`×3, grid from `elem_count()`=7, `ArchSku::Sm89`, Fuel's exact a/b), NaN-prefilled output, **correct-sync launch, 20 repeats** (`baracuda-kernels-bench/tests/alpha78_relu_add_ondevice.rs::gap001_relu_add_n7_fuel_geometry_sweep`): **20/20 clean, 0 all-unwritten, 0 wrong.** Sizing: 0/20 bounds the true rate **below ~14% at 95% confidence**; the excluded rate is the measured **~40%** — no overlap, so this is a real discrimination.
  - **Asymmetric bound.** A *failure* would have implicated the emitted side decisively; the *clean* result clears the kernel-body **only under a correct-sync launch at n=7**. It does **not** reproduce Fuel's launcher (grid-from-`layouts[n_inputs].shape().elem_count()`, buffer lifetime, sync ordering) and exercises **one geometry, not the space** — so it does not clear the emitted code generally.
  - **Where the intermittent lives, by this result:** the **consumer's grid / buffer / sync path**. The all-7-unwritten / never-partial signature plus Fuel's **load-sensitive** failure rate reads as a **timing/sync hazard** in the live-synthesis launch (module-load ordering, an async DtoH copy not synced before readback, stream sync) rather than a static out-of-bounds write in the kernel body — a static OOB fires at a layout-set rate, not a load-set one. The load-variation falsifier (rate climbs under GPU load → timing, not OOB) is Fuel's to run; the emitter sweep produced no failures to vary.

## Resolution (2026-08-19) — root cause found, emitter fully cleared

Fuel's lane added a launch-side probe recording which kernel each dispatch actually executed. In **both failing and passing runs**, the live Baracuda test launched `fuel_test_jit_relu_add_f32_scalar` — **Fuel's hand-written mock PTX** — not the synthesized `baracuda_gen_jit_relu_..._f32_scalar`. The Baracuda kernel was adopted, printed, placed in its own slot, and **never launched.**

Mechanism, all measured by Fuel (corrected 2026-08-19 against Fuel `origin/main`):

- The `FusedOpId(32768)` collision is **by design**: an earlier mock synthesizes the same `relu_add_region()` at f32 on `Cuda`, and Fuel dedups on `region_base_map_hash` (documented — same shape, different names → same id), treating co-id kernels as **alternatives** at one decision point. So the mock and the synthesized Baracuda kernel share id 32768, and **both are registered** — the Baracuda kernel as alternative 1.
- **Defect 1 — selection (Fuel GAP-213):** `alts.first()` (`fuel-dispatch/src/kernel.rs:1318`) always returns the **first** registration — the mock's slot-0 dispatcher — so the live Baracuda kernel, though registered, is never chosen. **Nothing is discarded; the wrong alternative is selected.**
- **Defect 2 — slot lifetime (Fuel GAP-214):** the mock's `CudaFunc` is bound to a per-test `CudaDevice` that is **already dropped** by then. Launching from a destroyed context is undefined — sometimes it writes correctly, sometimes it silently writes nothing. **That is the ~40–50%.**
- Both kernels compute `relu(add)` on f32, so when the stale launch happens to work it yields the *correct* answer and the assertion passes — a **vacuous test with a flake attached**, exercising the mock while naming the synthesizer.

**No version of `baracuda-kernelgen` / `baracuda-cuda-emit` was in the symptom or in the oracle.** Gate 5/6 and the 20/20 sweep were both correct and were both weighed against a measurement that was not measuring Baracuda. Fuel owns the fix: the id collision is intentional (documented dedup → per-decision-point alternatives, nothing discarded); the bugs are **selection** (GAP-213) and **slot lifetime** (GAP-214).

### The control was sound; the subject was misidentified

The mock-PTX control that passed 40/40 was **not** contaminated — it launched on its own live device and answered "does the loader/launcher work" correctly. The flaw was that the *subject* was misidentified: the "live" test never exercised live synthesis (it launched the mock via the id collision), so a sound instrument answered a question nobody was asking. **The instrument was sound and aimed where the property wasn't.** This is operationally distinct from a contaminated control: a contaminated control is caught by *varying the control*; a misidentified subject is caught only by *probing what actually executed* — no strengthening of the control reaches it.

### Generalizable lesson — a clean emitter sweep is not self-sufficient

The 20/20 sweep cleared the emitted kernel *"at n=7 under a correct-sync launch."* That bound held for a reason the sweep itself could not see: **the launcher never reached the kernel.** From the emitter's side, *"clean at geometry G"* and *"the code was never on the path"* are **indistinguishable** — only a **consumer-side launch probe** (which kernel actually executed) separates them. So a clean emitter-side sweep is **not** "exonerated by testing" until the consumer confirms the code was invoked. The next cross-project attribution should read a clean emitter sweep as *"consistent with correctness OR with never-invoked"* until the consumer closes that gap.

## Provenance

- Baracuda gate 5/6: commit `7bd90baf` (on-device, RTX 4070).
- Fuel GAP-001 discriminator: Fuel lane `fuel-gap029-qwen2`, 2026-08-19 — 20 fresh-process repeats, NaN-prefill discriminator + mock-PTX in-run control. Fixture `relu(add(a,b))`, 7×f32, `ArchSku::Sm89`, expected `[3,0,0,0,2,0,4]`.
