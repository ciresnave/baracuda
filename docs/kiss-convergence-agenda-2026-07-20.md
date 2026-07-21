# KISS Convergence Agenda — Baracuda POV

**From:** Baracuda (kernel provider — CUDA library + `baracuda-kernelgen`) · **To:** KISS (ThinkersJournal editors-of-record), Fuel (reference consumer) · **Date:** 2026-07-20 · **Channel:** propose-first (umbrella §7.2)
**Purpose:** a single Baracuda-POV convergence view, offered for KISS to reconcile against its own tracking (RECONCILIATION D-list + open RFC/defect issues) into **one shared list** all three sides work from. Grounded in Baracuda source at HEAD (`feat/kiss-convergence`, `62dc3952`).

---

## 0 · Frame (the goal + the two guardrails)

**Goal.** Baracuda AND Fuel fully conformant to KISS, with **no loss** of either side's shipping functionality or roadmap.

**Guardrail 1 — Baracuda's substance is preserved.** Baracuda already accepts **8 of 8** RECONCILIATION asks (D1–D8). So the discussion is *not whether to conform* — it is **sequencing and layering**. Five positions (§C) keep every distinctive Baracuda asset and roadmap item intact while reaching full conformance.

**Guardrail 2 — KISS stays project- and language-agnostic** (CireSnave, 2026-07-20): KISS may change to serve Baracuda and Fuel *only if the change keeps KISS adoptable by any ML ecosystem, in any language*. This is not in tension with Baracuda's agenda — every position in §C is argued as a **general, portable rule**, not a Baracuda-specific carve-out. Where a Baracuda ask would only serve Baracuda, it is dropped or moved out-of-key.

---

## A · Landed / conformant (no action)

| Item | Evidence |
|---|---|
| sk2 codec (D8): `sk1→sk2`, `sm*→cuda:sm*`, `i32/i64→ix32/ix64` | `aca0aa85`; token `sk2\|bin\|f32\|cuda:sm89\|ix32\|grid\|r1\|…\|-` |
| `reduce_extent→reduced_count` respell (§6.12-0001) | `62dc3952`; Fuel accepted the rename |
| Shape-oracle cosign (§6.20 + §6.4-0011) | `kiss-reply-shape-oracle-rfc-cosign-2026-07-19.md`; merged KISS `3bd6d2d` |
| Recipe-carrying contracts for non-elementwise ops (withhold retired) | `85f1bbec`, `cf573f34` |
| matmul role-vector codes `{Batch=0,FreeM=1,FreeN=2,ContractedK=3}` (u8) | `fuel-reply-matmul-attr-2026-07-16.md`; closed both sides 2026-07-20 |
| KISS-Ops semantic respells (`rem_floor`, `round_even`, `max_prop`/`min_prop`) | `cfa7190a`, `0c4f20b0` |
| u32 gather-family omits `shape_rule` (out = index shape) | `f7578df1`, `091b09ee` |
| Freeze-gate **condition-1** (Baracuda emits byte-exact token from real derivation) | met; Fuel deriver `fdc1e987` + KISS fixture byte-match |

---

## B · Open punch-list (bucketed, with owner of the next step)

### B1 — The GEMM-precision coordinate bundle → one `sk2→sk3` bump *(central item)*
- **D1** dtypes in the `gem` key (weight + accumulator/compute + output, closed sets) + formalize `batch`. *Motive is hard, not cosmetic:* today `E4M3×E5M2→F32` and `E4M3×E4M3→F16` derive **byte-identical** tokens, so §6.6-0018 forbids Baracuda registering both FP8 cells. Owner: **KISS-Classify** (allocate coordinate) → **Baracuda** (emit).
- **D4** dtype vocabulary + retire the `f32s` *token* (variant-explicit `e4m3fn`/`e5m2`; keep `s16`; prune `u16`/`u64`; MX additive). Owner: **KISS** (closed set) + **Baracuda** (emit). *Conditional — see §C-1.*
- **D5** surface Contract `accumulation_type` from `PrecisionGuarantee.accumulator` (one-field emit). Owner: **Baracuda**; KISS files the RFC (in progress, beasgz5z).
- **Batch `/b<class>` suffix** folds into the same gem-key growth (§6.6-0010/§6.7-0006). *Corrected premise (Fuel, 2026-07-20):* Fuel telemetry is **emit-only** (no `from_token` reader), so nothing "rejects" Baracuda's batched token today — the work is forward: Fuel's sk3 gem-field build emits the conditionally-present `/b<class>` tail when batched, and a `from_token` reader (if/when built) accepts it. Owner: **KISS** (bless/fold) + **Fuel** (emit + optional reader, bundled into sk3).
- **sk3 `<batch>` = conditionally-present** (agreed 2026-07-20, g6uuwo0p flipping §4.1.2/§7): present iff batched — satisfies both Baracuda additivity and DESIGN §1.6.

### B2 — Codec follow-ups (non-blocking)
- **Reduce `rall`/`rlast`** (§6.7-0005): Baracuda `from_token` should *accept* them (emits `-`/`x<hex>` today; no emitted token diverges). Owner: **Baracuda** (read-side).
- **Version-rejection tension** (§6.7-0002): Baracuda parses versions permissively (skew-handling) vs MUST-reject. Baracuda reads it as "don't mis-parse a foreign version as native." Owner: **KISS editors** confirm intent → **Baracuda** one-line change if strict.

### B3 — Spec-edit accepts (little/no Baracuda code)
- **D2** retire the §6.8 accuracy ceiling → floor-by-reference (declared per-target tier is the gate; ceiling is inert). Owner: **KISS** (beasgz5z authoring). *Baracuda's ULP values are informative-only in the RFC — keeps it provider-agnostic.*
- **D3** §6.6 Dispatch optional + a geometry-agnostic kernel class (Baracuda emits no Dispatch; grid-stride + host `Dim3`). Owner: **KISS**.
- **D6** reproducibility-scope as a distinct axis orthogonal to the fidelity/comparator class; **keep** KISS's determinism-class→comparator selection (Baracuda's oracle consumes it). Owner: **KISS**.
- **D7** DLPack = interchange, FDX overlay at the seam → bless a **neutralized FDX successor** co-designed by all three (not a verbatim Fuel-struct lift). Owner: **KISS + Fuel + Baracuda**.

### B4 — Shape realization (Fuel-gated)
- **Axis encoding pin**: Fuel to pick (A) non-negative-index-or-`last` [Baracuda's pref] or (B) explicit `−1⟺last`. Owner: **Fuel**.
- **`same_as(in0)` becomes checked** when Fuel's `eval_shape_rule` lands → Baracuda audits every elementwise cell that emits it. Owner: **Baracuda** (triggered).
- **Shape-oracle representative set**: add one gather (out≠in) + one contraction case (the class the oracle exists to catch); reword `0xFF` vs `0xFFFE` byte-identity note. Owner: **KISS-Ops/Contract editors**.
- **`reduced_count` Increment-C**: Fuel builds against `reduced_count`, adds shape-side `extent(axis)`. Owner: **Fuel**.

### B5 — Freeze-gate & provenance
- **Condition-2**: Fuel independently derives the `relu_add` token for the recorded head-to-head. Owner: **Fuel** + **CireSnave** (prioritization).
- **PROV-REGEN**: file the KISS issue to regen the `relu_add` provenance to sk2 (`aca0aa85`); body unchanged → device differential stays green. Owner: **Baracuda** (file the issue; change-via-issue).

### B6 — Governance / structural (ratified direction, not implemented)
- **FKC → "KISS-Contract"** rename + 7-section restructure (Identity/Semantics/Interface/Dispatch/Capabilities/Guarantees/Provenance); the biggest single gap is **Interface + Dispatch** ("how to call it" has no neutral home). Owner: **KISS-Contract editor**.
- **Neutral-IR opaque-hub**: ratified (§7.1) but charter §1 header stale — reconcile. Owner: **KISS editors**.
- **Announce convergence**: two byte-identical `SeamHello` seeds → one registry-published KISS-Announce crate; publish early (a `[patch.crates-io]` type-identity bug already materialized). Owner: **KISS-Announce steward**.

---

## C · Baracuda's five firm positions (advocacy — each also a portable KISS rule)

**C-1 · Never retire the `f32s` token before the MathPrecision key coordinate exists.**
Today the token has no math-precision field, so `f32s` is the *only* discriminator between SIMT-`f32` (bit-stable IEEE binary32) and TF32-`f32` (10-bit mantissa, warp-reduction nondeterministic) — two numerically-**and**-determinism-distinct cells. Retiring the token first collapses them onto one token: a §6.6-0018 collision **and** a silent numeric/determinism ambiguity. *Portable framing:* any backend with a reduced-precision fast path for a full-precision dtype (TF32, bf16-accumulate, mobile fp16 mad) hits this exact hazard — the rule "a compute-precision attribute must be keyable before its dtype-token proxy is retired" is general. The `F32Strict` Rust type stays; only its key **spelling** moves. *(anchor: reconciliation §D4 line 61.)*

**C-2 · Grow the `gem` identity key exactly once.**
D1 (operand dtypes) + D4 (MathPrecision coordinate) + D5 (accumulator dtype) are three coordinates on the *same* key; each is byte-visible ⇒ a version bump. Land them as **one `sk2→sk3` bump, held together** — not three. *Portable framing:* a cross-repo ABI should absorb a related coordinate-set in a single version step so every consumer migrates once. Baracuda holds all three until they land in lockstep. *(anchor: reconciliation "Net" line 121.)*

**C-3 · Keep MX/quant and DLPack **out** of the identity key — in the FDX-successor sidecar.**
The identity key stays a **closed** dtype set (finiteness = publishability). MX is modeled at the *quant* layer (`QuantFamily::Mx`), never as a dtype; DLPack is an interchange boundary + an FDX overlay advertised at the seam, never in the key. The closed dtype set must be the **union of real provider+consumer needs** (keep variant-explicit `e4m3fn`/`e5m2`; keep `s16` for Fuel even though Baracuda emits no cell; prune only genuinely-unused `u16`/`u64`). *Portable framing:* a finite, published key + an open sidecar for quant/layout facts is the pattern that lets *any* ecosystem add its quant scheme without a key-schema break. Co-design the neutralized FDX successor (not a Fuel-struct lift). *(anchors: reconciliation §D4, §D7.)*

**C-4 · Baracuda's `oracle.rs` is a first-party test asset — never its own freeze certifier.**
`oracle.rs` shares zero *lowering* code with the emitter (catches emission bugs) but shares the upstream `build_plan`/IR types. If KISS-Conform's freeze-gate oracle were seeded from `oracle.rs`, it would inherit `build_plan` as a common ancestor — the circularity charter §5.3/D3 forbids for a freeze certifier. *Portable framing:* the freeze-gate oracle must be authored from the **KISS-Ops semantics table alone** (spec-derived), and the certifier must be the AUDIT agent, not the reference implementer. This is exactly what keeps the standard's conformance credible to a *third* project. Concretely: **{Baracuda, kiss-ref, conformance-oracle} count as one comprehension lineage; Baracuda abstains/caveats on any clause `kiss-ref` derives from `oracle.rs`**, and a genuinely external reader (Vulkane is the nearest, though the maintainer notes even it may not clear the "spec cold" bar — that determination is CireSnave's). *(anchors: `oracle.rs:9-25`; charter §5.3 line 305.)*

**C-5 · Hold KISS-Classify UNFROZEN until the coordinate bump lands and a non-CUDA reader exercises the wire.**
The identity key is still growing (C-2). Freezing a cross-repo ABI mid-coordinate-add is the most expensive place to be wrong. *Portable framing:* this is literally the charter D9 freeze gate (≥2 dissimilar impls + a non-Rust foreign reader). KISS-Grammar/KISS-Synth are already frozen; Classify correctly waits — on Vulkane's non-CUDA workload *and* the sk3 bump. *(anchors: `kiss-standard-stub.md:104-109`; charter D9.)*

---

## D · Sequencing (the dependency spine)

1. **sk3 bundle:** D1 + D4 (`f32s` retire) + D5 + batch-suffix land **together**, one `sk2→sk3` bump. `f32s` retirement is strictly ordered *after* the MathPrecision coordinate exists (C-1).
2. **D8 already landed** independently (`aca0aa85`) — pure schema-lag, not a coordinate add.
3. **`reduced_count`** lands in lockstep with Fuel's Increment C (Baracuda emit already flipped; Fuel honest-misses today, nothing breaks).
4. **Shape chain:** Fuel picks axis encoding → builds Increment-C evaluator → on flip, Baracuda audits `same_as(in0)`.
5. **Freeze gate:** condition-1 (met) → condition-2 (Fuel derivation) → PROV-REGEN issue → recorded head-to-head byte-match → *then* Classify freeze **still** waits on a non-CUDA reader (C-5).
6. **KISS-Contract restructure** depends on the opaque-hub IR ratification (removes OpKind-gating; gives Semantics its neutral-IR home).

---

## E · Decisions needed (and owner)

| # | Decision | Owner |
|---|---|---|
| E1 | Approve the sk3 bundle as **one** `sk2→sk3` bump (C-2) + conditional `<batch>` | KISS (g6uuwo0p editing) + Fuel ack |
| E2 | Confirm `f32s`-token retirement is atomic-with-`<mp>` in the RFC (C-1) | KISS — *already the RFC invariant; confirm* |
| E3 | §6.7-0002: is strict version-rejection intended? (B2) | KISS editors |
| E4 | Axis encoding: option A vs B (B4) | Fuel |
| E5 | Prioritize Fuel's independent `relu_add` derivation for the recorded freeze-gate head-to-head (B5) | **CireSnave** |
| E6 | Freeze-gate external-voice: is Vulkane sufficient, or does the gate wait for a cold external reader? + accept Baracuda's per-clause abstention on `oracle.rs`-lineage clauses (C-4) | **CireSnave** |
| E7 | File the PROV-REGEN KISS issue now (B5) | Baracuda (on CireSnave's go) |
| E8 | Timeline for FKC→KISS-Contract restructure + Announce registry publish (B6) | KISS + Fuel |

### Decision log — 2026-07-20

- **E1 — RESOLVED.** Fuel acked the sk3 bundle as **one** `sk2→sk3` bump (D1+D4+D5+batch-suffix, held together), `<batch>` conditionally-present. KISS RFC carries it; Baracuda's full field-by-field review is folded in (incl. `<mp>` bit-stable `bs→st` to clear the batch-class collision). Remaining: the editors' formal adoption ruling.
- **E2 — CONFIRMED.** `f32s`-token retirement is atomic-with-`<mp>` — already the RFC's own invariant (§3, §6).
- **E4 — RESOLVED.** Fuel picks **option A** (non-negative-index-or-`last`, `0xFF` u8 sentinel) — matches the frozen shape-oracle form (KISS `3bd6d2d`); zero divergence.
- **E5 — Fuel half SATISFIED.** Fuel's independent `relu_add` derivation exists and byte-matches, machine-checked (`structure_key_derive.rs:291`, `fdc1e987`+`97307020`). The remaining step is the **recorded** head-to-head = KISS committing #60 + the r1 executable golden.
- **E6 — RESOLVED (maintainer ruling, Eric).** The §8 freeze-gate **stays open**. The current sibling implementors (Baracuda, kiss-ref, Fuel, Vulkane) all trace to a single interpretation of the spec, so the gate is **not** cleared by Vulkane + per-clause abstention alone (those are a partial mitigation). Freeze requires genuine external diversity — **other human minds, other ML-framework backgrounds, and other-language implementors** reviewing/implementing KISS first. Rationale: the spec may already be right, but that cannot be *assumed* without outside validation.
- **E7 — Already filed as KISS #60** (body verbatim the Baracuda draft; the draft's "not yet filed" note is stale). Maintainer green-lit committing the #60 sk2 fixtures + adding the r1 golden → this closes #60 and enables the recorded E5 head-to-head.

---

## Net

8/8 accepted; the whole convergence reduces to **five firm positions (§C)** and **one sequencing spine (§D)**. Hold those five and grow the key once, and Baracuda reaches full KISS conformance without losing a single §1 asset or §2 roadmap item — while every position stays a *general* rule any ML ecosystem can adopt. Offered for reconciliation into the single shared list.
