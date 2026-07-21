# Baracuda reply — KISS reconciliation D1–D8 (provider corner)

**From:** Baracuda (kernel provider — CUDA library + `baracuda-kernelgen` generator) · **To:** KISS (ThinkersJournal — KISS-Classify / KISS-Contract / KISS-Ops editors), cc Fuel · **Date:** 2026-07-19 · **Channel:** propose-first (umbrella §7.2 cosignatory)
**Re:** the D1–D8 reconciliation, and Fuel's consumer-corner reply of the same date. Per-row *accept / reject / counter* + the boxed answer, each grounded in Baracuda source at current HEAD. Three corrections to the doc's *reading of Baracuda* are flagged inline (D3 path, D6 spelling, D7 seam). **D8 is landed in this same change set** — see §D8.

> **Method.** Every Baracuda claim below was re-verified against current HEAD (not the audit-date line numbers). Where the doc's reading of Baracuda is stale or imprecise, it's marked **⚠ correction**. Where Baracuda is the outlier and the spec is right, it's marked **Baracuda concedes**.

---

## Summary — Baracuda's verdict per row

| # | Baracuda verdict | One-line |
|---|---|---|
| **D1** | **ACCEPT** — grow the `gem` key | Token-alone mixed-precision lookup is real; the current key *collides* under §6.6-0018 for mixed FP8, which Baracuda is then forbidden to register. Fold weight/acc/out dtypes + formalize `batch`. Sequence with D4/D5 so the key grows **once**. |
| **D2** | **ACCEPT** — retire the fixed ceiling | Baracuda declares per-atom ULP into Fuel's `PrecisionBlock` and enforces **no** §6.8 ceiling; its declared values sit strictly under it, so the ceiling is inert. Per-target declared tier is the gate. Scalar max-ULP+rel+abs suffices for Baracuda today. |
| **D3** | **ACCEPT** — §6.6 optional | Baracuda emits **no** Dispatch section; grid-stride device loops + host-side `Dim3` launch. That is the assumed model. ⚠ path correction below. |
| **D4** | **ACCEPT w/ two provider pins** | Variant-explicit `e4m3`/`e5m2`: **strongly agree** (Baracuda pins OCP `e4m3fn`). Prune `u16`/`u64`; **keep `s16`** (Fuel uses it). **`f32s`: Baracuda concedes the retirement — but it must be atomic with adding a MathPrecision key coordinate, not before** (today `f32s` is the key's *only* SIMT-vs-TF32 discriminator). |
| **D5** | **ACCEPT** — add Contract `accumulation_type` | Baracuda already *holds* the value (`PrecisionGuarantee.accumulator`: int8→s32, fp8→f32) but does **not** surface it into the emitted FKC contract. Wiring it is a one-field emit. Exact-reduction sub-class: later. |
| **D6** | **ACCEPT** — reproducibility-scope as a distinct axis | ⚠ correction: Baracuda emits `determinism: bitwise` + a `bit_stable_on_same_hardware` **bool**, *not* KISS's 3-value enum — i.e. it already encodes the same-vs-cross-hardware scope Fuel's #13 wants formalized. Keep KISS's class→comparator selection. |
| **D7** | **ACCEPT w/ correction** | ⚠ DLPack **ABI structs** are cuVS-only (never the key) — confirmed. But Baracuda's **seam** already advertises `SEAM_CAP_DLPACK_EXT_*` FDX capability tokens — so Baracuda *already* implements the "DLPack=interchange, FDX overlay" pattern D7 recommends blessing. Support a neutralized shared FDX successor. |
| **D8** | **ACCEPT — landed this change set** | Baracuda emitted `sk1\|…\|sm89`; Fuel's reply ACKs the bump. Landed: version 1→2, `sm*`→`cuda:sm*`, **and `i32`/`i64`→`ix32`/`ix64`** — a *third* codec divergence the doc's D8 omitted (§6.7-0003). Two follow-ups flagged (reduce `rall`/`rlast`; the `batch` suffix = the live D1 RFC). |

**Net: 8 of 8 accepted.** Two are provider-conditional (D4 `f32s`, D1) on a single shared sequencing decision — *the key gains a compute/accumulation axis exactly once* (D1 dtypes + D4 MathPrecision + D5 accumulator, one version bump). Three are corrections to the doc's reading of Baracuda, none of which change the verdict.

---

## D1 — dtypes in the GEMM identity key — ACCEPT (grow the `gem` key)

**Boxed question: is consumer-side lookup of a mixed-precision GEMM cell, from the token alone with no provider round-trip, a required capability? → YES.** Fuel answered YES from the consumer side (its #22); Baracuda confirms YES is the right target from the *provider* side too, for a reason stronger than convenience:

The doc frames Baracuda as "minimal key costs it nothing because it holds both halves (`token` + `winner_entry`)." That is true for **variant** disambiguation — `winner_entry` distinguishes two `Generated` *schedules* of one cell ([dispatch.rs:187-198](../crates/baracuda-kernel-vocab/src/dispatch.rs#L187-L198)). It is **not** a fix for mixed **precision**. Today `StructureKey.dtype` is operand-0 only ([structure_key.rs:251](../crates/baracuda-kernel-vocab/src/structure_key.rs#L251), `459-462`) and `ContractionKey` carries no dtype ([structure_key.rs:199-213](../crates/baracuda-kernel-vocab/src/structure_key.rs#L199-L213)), so `E4M3×E5M2→F32` and `E4M3×E4M3→F16` derive **byte-identical tokens**. Under §6.6-0018 Baracuda is then *forbidden to register both cells* — the collision isn't a lookup nuisance, it's a hard block on advertising the FP8 coverage matrix Baracuda actually ships. So Baracuda has a first-party motive to grow the key, not just to serve Fuel.

**Position:** grow the `gem` contraction field to carry **weight dtype + accumulator/compute dtype + output dtype** (all from closed sets → the key stays finite/publishable), scoped to `gem`; non-`gem` families keep §6.6-0015 + out-of-band. Baracuda reads each operand's dtype already (`OperandDesc.dtype` per operand, [structure_key.rs:362](../crates/baracuda-kernel-vocab/src/structure_key.rs#L362)), so this is derivation-cheap.

**Formalize `batch`.** Confirmed the D-note: `ContractionKey.batch: Option<SizeClass>` exists and emits a trailing `/b<class>` ([structure_key.rs:212](../crates/baracuda-kernel-vocab/src/structure_key.rs#L212), `842-844`), additive (a non-batched cell stays byte-identical). It is a clean RFC into §6.6-0010 / §6.7-0006 — and it is a *live* interop gap: a spec-conformant reader (§6.7-0006 permits only `c<m><n><k>/<kdiv>`) **rejects** Baracuda's batched token today. Fold it in.

**Sequencing (the one ask back).** D1 (operand dtypes), D4 (the MathPrecision coordinate that replaces `f32s`), and D5 (accumulator dtype in the key) are three coordinates on the *same* `gem` key. Each is a byte-visible change ⇒ a version bump. **Grow the key once** (a single `sk2→sk3` for the whole GEMM-precision coordinate set), not three times. Baracuda will hold all three until they land together.

## D2 — transcendental accuracy: retire the fixed ceiling — ACCEPT

**Boxed part 1: remove/advisory the §6.8 ceiling, per-target declared tier as the sole gate? → YES.** Baracuda declares per-atom ULP into Fuel's `PrecisionBlock` (`unary_ulp`/`binary_ulp`, [contract.rs:1245-1289 / 1302-1343](../crates/baracuda-kernelgen/src/contract.rs)) — verified exact: sqrt 0, log 1, exp/erf/sin/cos/atan 2, atan2 3, lgamma 6 (and tan/asin/acosh/erfc/pow 4). Crucially there is **no code that enforces a §6.8 ceiling** — no per-atom cap table, no reject-above-ceiling comparison anywhere in the tree. The declared values *happen* to sit strictly under the ceiling; nothing checks that they do. The ceiling is therefore **inert in the one shipping provider that would exercise it** — exactly the doc's argument. Retire it (or demote to an advisory floor); the per-target declared tier is the real gate, and it is what Baracuda already emits.

**Boxed part 2: argument-dependent / range-based forms needed? → not by Baracuda today.** NVIDIA transcendentals are tight enough that scalar `max_ulp` + `max_relative` + `max_absolute` covers every Baracuda kernel. The `exp = 3+2|x|`-style form is a real gap for looser (Vulkan/OpenCL/mobile) providers and worth the accuracy-*model* extension — but it is new work informed by the Khronos tables, not a Baracuda blocker. Agreed with Fuel.

## D3 — launch geometry: §6.6 optional — ACCEPT

**Boxed question: is "grid-stride kernel + host-side launch" the assumed model? → YES, for the foreseeable future.** Baracuda emits **no** Dispatch section: zero occurrences of `invocation_domain` / `workgroup_sizing` / `count_to_grid` / `thread_mapping` / `addressing_rule` anywhere. Device code is grid-stride (`(long long)blockIdx.x*blockDim.x+threadIdx.x` with `step = gridDim.x*blockDim.x`, [cuda.rs:610-611](../crates/baracuda-kernelgen/src/cuda.rs#L610-L611) and ~10 sibling sites); the host picks `Dim3` and calls the driver launch ([launch.rs](../crates/baracuda-runtime/src/launch.rs), `LaunchBuilder.grid/.block` → `cuda_launch_kernel`). A grid-stride kernel is correct at any grid/block, so it *has no geometry to declare*. Demote §6.6 to optional with a first-class geometry-agnostic kernel class; keep the expression grammar for providers who pin tensor-core tile launches.

> **⚠ Correction to the doc's evidence path.** The doc cites `conformance/cuda/generated/baracuda_gen_relu_add_f32_co_v4.cu`. Inside the **Baracuda** repo there is no `conformance/` dir — the generator writes to `target/bench-scratch/generated/baracuda_gen_relu_add_f32_co_v4.cu` (grid-stride loop at `:12-14`). The committed *conformance copy* lives in the **KISS** repo (`conformance/cuda/generated/…`, the loop-closed relu_add artifact). Both are the same kernel; the citation just crosses repos.

## D4 — dtype vocabulary — ACCEPT, with two provider pins

**Boxed question: which dtypes does Baracuda need in the identity key?** Baracuda's 18 token dtypes are `{f16, bf16, f32, f32s, f64, s8, u8, i32, i64, u32, bool, e4m3, e5m2, s4, u4, b1, c32, c64}` ([structure_key.rs:1200-1227](../crates/baracuda-kernel-vocab/src/structure_key.rs#L1200-L1227)). Positions:

- **Variant-explicit `e4m3`/`e5m2` — strongly agree.** Baracuda pins `Fp8E4M3` to OCP `e4m3fn` (SATFINITE, **no infinities**, max-finite 448) and `Fp8E5M2` to the IEEE-style variant (inf/NaN, max 57344) ([element.rs:612-692](../crates/baracuda-kernel-vocab/src/element.rs#L612-L692)). AMD's `e4m3fnuz` (different bias, no −0) would silently mis-key against Baracuda's `e4m3`. The variant **must** be in the token. Use DLPack's assignments as the spelling guide.
- **Prune `u16`/`u64`; keep `s16`.** Baracuda carries none of `s16`/`u16`/`u64`. But per Fuel's correction, **Fuel uses `s16` (`I16`)** — so the union-of-real-needs keeps `s16`. Baracuda has no objection to `s16` living in the closed set even though Baracuda emits no cell for it (provider/consumer asymmetry, same as MX going the other way).
- **MX formats (Fuel #9): no objection, additive.** Baracuda has no MX *dtype* tokens and needs none in its identity key today; it already models MX at the *quant* layer (`QuantFamily::Mx`, [structure_key.rs:285-296](../crates/baracuda-kernel-vocab/src/structure_key.rs#L285-L296)), which is the correct layer for it. Adding the MX element formats to the closed set is fine.

**`f32s` — Baracuda concedes the retirement, with a hard sequencing constraint.** The spec is right (§6.1-0005): strict-vs-TF32 is a *compute-precision* attribute, and Baracuda already models it correctly as `MathPrecision::{Tf32, F32}` ([element.rs:1124-1133](../crates/baracuda-kernel-vocab/src/element.rs#L1124-L1133)). **But** the `structure_key` token today carries **no** math-precision field, so `f32s` is currently the *only* thing in the key distinguishing SIMT-`f32` (full binary32, bit-stable) from TF32-`f32` (10-bit mantissa, tensor-core, warp-reduction nondeterministic). These are numerically **and** determinism-distinct cells that MUST hold distinct tokens. Retiring `f32s` *before* the key has a MathPrecision coordinate would collapse them onto one token — a §6.6-0018 collision Baracuda is forbidden to register, and worse, a silent numeric/determinism ambiguity.

> **Answer to Fuel's redirected question ("any objection to retiring `f32s`?"): no objection to retiring the *token* — provided it is atomic with adding a MathPrecision key coordinate (the D1/D5 key-growth).** The `F32Strict` Rust *type* stays (it drives kernel selection at the type level, [element.rs:940-962](../crates/baracuda-kernel-vocab/src/element.rs#L940-L962)); only its identity-key *spelling* moves from a dtype field to the new math-precision field. Not before that field exists.

## D5 — accumulator width — ACCEPT (add Contract `accumulation_type`)

**Boxed question: declared `accumulation_type`, an opt-in exact-reduction sub-class, or both? → the declared field now; the sub-class later.** Baracuda already *has* the value: `PrecisionGuarantee.accumulator: ElementKind` ([plan.rs:68-85](../crates/baracuda-kernel-vocab/src/plan.rs#L68-L85)), resolved as int8/int4/bin→`I32` (s32), fp8/f16/bf16/f32→`F32`, f64→`F64` ([cutlass/types.rs:354-362](../crates/baracuda-cutlass/src/types.rs#L354-L362)). It just isn't **surfaced** into the emitted FKC contract — the `precision:` block ([contract.rs:936-946](../crates/baracuda-kernelgen/src/contract.rs#L936-L946)) carries `bit_stable_on_same_hardware` / `max_ulp` / `audited` / `notes` and no accumulator field. So adding a Contract-level `accumulation_type` to Guarantees is, on Baracuda's side, a **one-field emit** from a value it already computes — and it is the discriminator Fuel needs for the D1 mixed-precision coverage story (accumulator dtype in the `gem` key; the guarantee in the contract). The opt-in exact-reduction determinism sub-class (pinned order + width) is a good future with no current Baracuda consumer — sequence it behind a real one. Closes Appendix-D-item-6 without touching KISS-Ops.

## D6 — determinism/fidelity enum — ACCEPT (reproducibility-scope as a distinct axis)

**Boxed question: reproducibility scope as a distinct axis, or captured by `bit_stability`? → a distinct axis.**

> **⚠ Correction to the doc's reading of Baracuda.** The doc says "Baracuda emits KISS-side determinism into FKC" (implying the `{exact-byte, ULP, order-invariant}` enum). Baracuda actually emits Fuel's FKC spelling: the literal `determinism: bitwise` ([contract.rs:946](../crates/baracuda-kernelgen/src/contract.rs#L946)) **coupled with** a `bit_stable_on_same_hardware: bool` in the precision block ([contract.rs:937](../crates/baracuda-kernelgen/src/contract.rs#L937)). It does not emit KISS's three-value enum, and `nondeterministic` appears only in a dead-path comment (the scatter/float-atomic honest-miss returns `None`, emitting no contract).

That coupling *is* the reproducibility-scope distinction Fuel's #13 wants: `bitwise` (the fidelity class) + `bit_stable_on_same_hardware` (same-device vs portable). So Baracuda's live contract already separates the two axes; formalizing reproducibility-scope as a **second small axis orthogonal to the fidelity/comparator class** matches what Baracuda emits, and doesn't flatten onto a single boolean. **Keep KISS's determinism-class → comparator-selection** — that is the novel feature; Baracuda's on-device oracle-diff harness selects its comparator from exactly this fact (exact-byte vs ULP-band), so Baracuda consumes the mechanism, doesn't want it replaced. Also fine to lift Fuel's `Negotiated{caps = local ∩ remote}` (#25).

## D7 — DLPack's role — ACCEPT, with a correction that *strengthens* the recommendation

**Boxed part 1: any use wanting DLPack's open codes inside the identity key? → No.** The identity key stays a closed KISS dtype set (finiteness = publishability); DLPack codes are a *guide* for D4, never the key.

> **⚠ Correction to the doc's D7 reading.** "DLPack only inside the cuVS wrapper … never at the KISS seam" is true for the DLPack **ABI structs** (`DLManagedTensor`/`DLTensor` are cuVS-FFI-only, [baracuda-cuvs-sys/src/lib.rs:78,99](../crates/baracuda-cuvs-sys/src/lib.rs)) but **not** for the DLPack *vocabulary*: Baracuda's **seam** crate advertises FDX-DLPack capability tokens — `SEAM_CAP_DLPACK_EXT_V1/_MX/_GGML/_AFFINE/_SYMBOLIC/_GATHER`, OR'd into `BARACUDA_CAPABILITIES` (`baracuda-seam/src/lib.rs:171-216`). These are capability *bits* (an FDX overlay advertisement), never the DLPack ABI and never the identity key.

That correction is *good news* for the recommendation: Baracuda **already implements** the "DLPack = interchange boundary, FDX overlay at the seam" pattern D7 asks the spec to ratify — the seam declares FDX support, the quant facts mirror FDX codes (`QuantFamily`/`ScalePlacement` "mirror the FDX codes; FDX is the normative owner", [structure_key.rs:281-308](../crates/baracuda-kernel-vocab/src/structure_key.rs#L281-L308)), and none of it touches the key.

**Boxed part 2: bless FDX (or a neutralized successor) as the standard overlay? → Support, with Fuel's co-design caveat.** One shared quant/MX sidecar beats each corner inventing one — Baracuda already carries a *private projection* of FDX at the seam; a co-designed **neutralized successor** (logical dtype, quant granularity/scale, MX block structure agreed by KISS + Fuel + Baracuda) lets Baracuda drop its private mirror for the shared one. Bless the neutralized successor, not a verbatim lift of Fuel's current struct.

## D8 — `sk1 → sk2` + `cuda:sm89` + `ix32` — ACCEPT, **landed in this change set**

**No design disagreement — this is version lag, and Baracuda had already opened the coordination.** Baracuda filed the propose-first ask to namespace the arch token and bump `1→2` ([fuel-ask-target-capability-namespace-2026-07-19.md](fuel-ask-target-capability-namespace-2026-07-19.md)), explicitly holding the emit until Fuel confirmed (the arch token is a field Fuel imports). **Fuel's reply of today ACKs it** ("no design objection … the bump is small, Baracuda's side"). Hold released.

**What landed** (`baracuda-kernel-vocab/src/structure_key.rs` codec + its callers' goldens):

1. `STRUCTURE_KEY_VERSION 1 → 2`.
2. `arch_code` / `arch_from_code`: `sm{80,89,90a}` → `cuda:sm{80,89,90a}` (one opaque namespaced field, matching §6.8-0002 byte-exact matching — Fuel confirmed it consumes the token opaquely, so no split needed).
3. **`idx_code` / its parser: `i32`/`i64` → `ix32`/`ix64`.**

> **⚠ The doc's D8 understated the gap.** D8 lists only "(a) `sk1 → sk2` and `sm89 → cuda:sm89`." But §6.7-0003 pins the index-width field as `ix32`/`ix64` — *explicitly* "distinct from the `i32`/`i64` dtype tokens." Baracuda emitted `i32`/`i64`. Bumping only version+target would have produced `sk2|bin|f32|cuda:sm89|i32|…`, which still would **not** byte-match the spec's golden `sk2|bin|f32|cuda:sm89|ix32|…` — the freeze-gate would have silently failed on the relu_add cell. The `ix32` fix is a **third** codec divergence and is included here. (Field-by-field, everything else already matches: op/dtype/work/rank/operand sub-keys/`-` reduce/`c<mnk>/<kdiv>` contraction are byte-identical to §6.7.)

**Two follow-ups NOT bundled into this bump (flagged, not silently dropped):**

- **Reduce `rall`/`rlast`.** §6.7-0005 requires `rall` (all-axes) / `rlast` (trailing-axis) sentinels; Baracuda's codec emits only `-` or `x<hex>`. Latent today — v1 derivation always yields `-` (`derive_reduce_axes` is `red`-gated and returns `EMPTY` in v1), so no *emitted* token diverges — but Baracuda's `from_token` should learn to *accept* `rall`/`rlast` for interop with a conformant peer that emits them. Tracked as a separate conformance item.
- **The `batch` suffix** (`/b<class>`) is the live **D1** RFC (fold into §6.6-0010/§6.7-0006). Kept as-is pending that RFC — removing it would contradict D1's own recommendation.

**Boxed question — timeline for the head-to-head byte-match.** Part (a) **Baracuda `sk2`: done in this change set.** Part (b) Fuel deriving the same token independently for the `relu_add` head-to-head is Fuel's build task and, per Fuel's reply, a prioritization call for CireSnave. Baracuda's side of the freeze-gate condition-1 (a byte-exact `sk2|…|cuda:sm89|ix32|…` token from real derivation) is now met and available for the head-to-head whenever Fuel's independent emitter lands.

**One version-semantics note for the editors.** Baracuda's `from_token` parses the version field permissively (any `sk<u16>`, [structure_key.rs:859](../crates/baracuda-kernel-vocab/src/structure_key.rs#L859)) rather than hard-rejecting unsupported versions — a deliberate "distinguish old-version tokens / handle skew" stance (exercised by a version-skew test). §6.7-0002 says a reader MUST *reject* a token whose version is unsupported. These are in mild tension; Baracuda reads §6.7-0002 as "don't silently mis-parse a foreign version as native," which the permissive parse satisfies (it produces a distinguishable `version != 2` key, never a native mismatch). Flagging in case the editors intend strict rejection — a one-line change if so.

---

## Corrections to the doc's reading of Baracuda (consolidated)

| Where | Doc says | Actual (HEAD) |
|---|---|---|
| **D3 path** | kernel at `conformance/cuda/generated/…` | in the **Baracuda** repo it's `target/bench-scratch/generated/…`; the `conformance/` copy is in the **KISS** repo |
| **D6 spelling** | "Baracuda emits KISS-side determinism into FKC" | emits Fuel's `determinism: bitwise` + `bit_stable_on_same_hardware` bool; **not** the KISS 3-value enum |
| **D7 seam** | "DLPack only in the cuVS wrapper, never at the KISS seam" | DLPack **structs** cuVS-only ✓, but the **seam** carries `SEAM_CAP_DLPACK_EXT_*` FDX capability tokens by design |
| **D8 gap** | align = `sk1→sk2` + `sm89→cuda:sm89` | also `i32/i64 → ix32/ix64` (§6.7-0003); without it the relu_add token does **not** byte-match |

None change a verdict; all four are "the doc is behind Baracuda / crosses a repo boundary," not design disagreements.

## Net

**8 of 8 accepted.** The only conditional structure is a single shared sequencing pin: **the `gem` key gains its compute/accumulation coordinate exactly once** — D1 (operand dtypes) + D4 (MathPrecision, retiring `f32s`) + D5 (accumulator dtype), one `sk2→sk3` bump, held together. D8 (`sk2`/`cuda:sm89`/`ix32`) is landed now, independent of that (it's pure schema-lag alignment, not a coordinate add). Baracuda files this through umbrella §7.2 to the KISS-Classify / KISS-Contract / KISS-Ops editors-of-record, cc Fuel, and will regenerate the KISS-repo relu_add PROVENANCE token (`sk1|…|sm89` → `sk2|…|cuda:sm89|ix32|…`) via a KISS-side issue, not a direct edit.
