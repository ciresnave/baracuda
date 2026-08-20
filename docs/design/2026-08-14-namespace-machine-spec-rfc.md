# RFC: Machine-readable namespace specifications for KISS `target_capability` annexes

**Status:** DRAFT for cross-project circulation (not yet a KISS issue)
**Date:** 2026-08-14
**Author:** Baracuda (cuda namespace owner)
**Deciding parties:** Baracuda (cuda), Vulkane (vulkan), Unpopped (neutral core / generator), KISS (the standard)
**Downstream consumer (not a decider):** Fuel (runtime selector)

## 1. Problem

KISS `target_capability` annexes (KISS-Classify §6.8) are currently **prose Markdown**
— e.g. Baracuda's `spec/namespaces/cuda.md`. The §6.8-0003 registry "records a pointer
to each vocabulary, never its content," and those pointers resolve to prose. A prose
annex is not machine-checkable: a conformance harness cannot consume it, and a producer's
tokens cannot be byte-validated against it. Each namespace needs a **machine-readable
specification** with a defined home, authoring model, and governance.

This is the **spec-side** counterpart to the Q4 decision (who owns the token-emitting
*code*): Q4 settled that the cuda vocabulary code leaves neutral `unpopped-vocab` for a
Baracuda-owned crate; this RFC settles where the normative machine-readable *spec* lives
and how it is kept honest. The two are independent — the spec lands in KISS regardless of
which crate emits the tokens.

## 2. Settled from Baracuda's side (pending party confirmation)

- **D1 — Placement: in KISS, alongside the prose annex.** A standard must be
  self-contained (conformance cannot require fetching N external repos at N pinned
  versions); co-locating prose + machine spec means drift is caught in one PR (the
  `tools/kiss_trace.py` clause↔test checker extends to clause↔spec-entry). This is also
  internally consistent with cuda.md §4's own rule: a claim is "held by a conformance
  assertion in this repository — not by prose, and not by a cross-repo reference." The
  link-out alternative reintroduces exactly the pointer-rot §6.8-0003 already has.

- **D2 — Generation: from each owner's reference crate.** The machine spec is *generated*
  from the owner's crate and byte-compared, never hand-authored — matching Vulkane's
  stated manifest rule ("generated from the reference crate and byte-compared, never
  hand-written") and the harness's existing POD byte-exact discipline.

- **D3 — cuda source crate: `baracuda-cuda-vocab`** (dedicated, minimal). Not the emitter:
  the vocabulary has multiple consumers (emitter, neutral codec, KISS harness, plausibly
  Fuel's selector), and embedding it in the 13k-line emitter forces all of them to depend
  on the code generator to answer a vocabulary question — re-creating the coupling Q4
  removed. A focused crate is also the honest target for the §6.8-0003
  `reference_implementation` pointer. Resulting DAG: `baracuda-cuda-vocab` ←
  `baracuda-cuda-emit`; `unpopped-vocab` (neutral) carries tokens opaquely and no longer
  bakes `ArchSku`; KISS harness depends on `baracuda-cuda-vocab` only to regenerate.

- **D4 — Governance: owner-PR, NOT KISS-auto-pull, with a mandatory owner-side drift-gate.**
  The owner opens a PR into KISS with the regenerated spec; KISS CI validates the committed
  artifact but does **not** build-depend on vendor crates. Deliberate review by all parties
  before a change is official is the point of a standard; "automatic" propagation is the
  thing a standard should not be. **The condition that makes this sound:** each owner's CI
  regenerates the spec from its crate and compares against the KISS-committed copy — a
  mismatch fails the *owner's* build and prompts the update PR. Without that owner-side
  check, owner-PR silently drifts. This drift-gate is a **stated obligation of each
  namespace owner** (Baracuda for cuda, Vulkane for vulkan), not a Baracuda courtesy.
  Two refinements from Vulkane's review: the gate compares against the **published ref**
  (`git show origin/main:<path>`), not the working-tree checkout — a stale local clone is
  otherwise indistinguishable from real drift and misblames the wrong party; and it fires
  in **both directions** (my-crate-changed/KISS-stale AND KISS-moved/mine-didn't, which
  happens on regeneration/merge/hand-edit), which a single equality-against-ref assertion
  gives for free.

- **D5 — Prose stays owner-authored; machine spec generated; both atomic in one PR.** The
  prose annex carries rationale a generator cannot produce, so it is inherently
  owner-authored. Auto-generation could only ever produce the machine half, risking prose
  and machine spec landing separately; the owner PR keeps them together, kept in lockstep
  by the `kiss_trace` clause mapping.

## 3. Open items for the parties

- **O1 (KISS to rule) — Format: reuse the §6.8-0008…-0013 capability-manifest format, or a
  distinct spec artifact?** KISS merged a manifest *format* with zero instances. Is a
  namespace's machine-readable vocabulary spec the *same artifact* as a capability manifest
  (in which case cuda below becomes the **first manifest instance**, closing the
  zero-instances gap and fixing the format now), or a distinct spec type? The prototype in
  §4 is a **format candidate**, deliberately not committing this.

- **O2 (Vulkane) — the list-bearing case.** cuda is the *simple* namespace: a single scalar
  token per kernel (cuda.md §4). Vulkan's vocabulary is a capability **set** (`<arith>` =
  `dot8, f16, i8, st16, st8`), so the format must express list-bearing capability-sets,
  dedup/sort/digest discipline, and admission over a set — none of which cuda exercises.
  Vulkane's confirmation of D4 for vulkan **and** its reaction to the format on the
  list-bearing case are both needed.

- **O3 (all) — file layout in the KISS tree + artifact naming**, and where the byte-compare
  runs in practice (owner CI pushing vs a KISS check that only validates internal
  consistency).

## 3b. Format requirements — the list-bearing case (Vulkane review, accepted)

My prototype (§4) is the **degenerate scalar case**. Vulkane, the list-bearing namespace,
showed the format must be designed for the *general* (generated) case with cuda falling out
as the trivial instance — not cuda-first-extended. Six accepted requirements:

1. **`tokens` is optional; a namespace declares its `kind`.** Vulkan's vocabulary is
   unbounded (`sg<N>` for any power-of-two N; arbitrary cooperative-matrix/vector tuple
   lists) — it cannot be enumerated, only validated. `kind` is an **open set** (KISS #199),
   not frozen at two.
2. **`set_shape` is per-FIELD, not per-namespace.** Vulkan's five fields have five shapes
   (`<subgroup>` = a constrained *predicate*, not a list; `<ops>` = fixed-width char
   alphabet; `<arith>` = variable-length sorted names; `<coop>` = 7-tuples; `<coopvec>` =
   5-tuples). A namespace-level shape is wrong about four of five — even the simplest field
   is not an enumeration.
3. **Canonicalization is named in the format, not implied** — `order`, `dedup`,
   `threshold`, and the **length-conditional digest** (canonical enumeration > 512 bytes →
   `fnv1a64-<hex16>` of that same string). A consumer holding the full grammar can still
   *produce* the wrong token → a silent §6.8-0002 byte-exact cache miss with nothing
   reporting. Per #199, for a `generated` namespace **the vectors are the contract, the
   field spec is documentation**, with vectors required to cover those four names.
4. **Admission is marked explicitly NON-MATCHING.** Admission ("can this device run this
   kernel" — a build/request decision) must not sit next to `tokens` inviting use as a
   lookup rule: §6.8-0002 forbids subset/implication logic in *matching*, which is
   byte-exact. Conflating them re-opens the merge hazard namespace registration exists to
   prevent.
5. **Per-member `axis` tag, not a storage/compute boolean.** Capabilities name different
   axes (`st16` = storage-access; `i8` = `shaderInt8` compute; `dot8` = accelerated-op
   class — three, not two). A tag generalizes; a boolean needs widening at the fourth axis.
6. **A representable third state: supported / unsupported / not-expressible-in-this-vocab-
   version.** A consumer must distinguish "device lacks this" from "this vocabulary has no
   name for it" (vulkan names `shaderInt8`, has no name for `shaderInt16`). If the format
   cannot say "we do not name this," every gap reads as a denial.

cuda reconciles as the degenerate instance: `kind: enumerable` (it *has* a `tokens` list),
one field, scalar shape, per-class admission marked non-matching, single axis. The general
format subsumes it without loss. This strongly informs **O1**: #199 already shaped much of
this (open `kind`, vectors-are-contract, `order`/`dedup`/`threshold`/`digest_input`), which
points toward the machine-spec **reusing the §6.8-0008…-0013 generated-namespace/manifest
format** rather than defining a new artifact.

## 4. Prototype — `cuda` machine spec (worked example, format-candidate)

A faithful machine encoding of cuda.md §1–§6, generated (in the target design) from
`baracuda-cuda-vocab`. Presented to make the O1 format debate concrete:

```json
{
  "namespace": "cuda",
  "vocab_version": "cuda-vocab v1",
  "maintainer": "baracuda",
  "reference_implementation": "baracuda-cuda-vocab",
  "encoding": {
    "token_grammar": "cuda:sm<digits>[a]",
    "charset_after_prefix": "[a-z0-9]",
    "comparison": "byte-exact",
    "capability_set_shape": "single-scalar"
  },
  "tokens": [
    { "token": "cuda:sm80",  "class": "base", "target": "sm_80",  "sm_number": 80, "note": "Ampere; forward-compat floor" },
    { "token": "cuda:sm89",  "class": "base", "target": "sm_89",  "sm_number": 89, "note": "Ada; adds FP8 over sm80" },
    { "token": "cuda:sm90",  "class": "base", "target": "sm_90",  "sm_number": 90, "note": "Hopper portable / PTX-forward" },
    { "token": "cuda:sm90a", "class": "a",    "target": "sm_90a", "sm_number": 90, "note": "Hopper arch-exclusive; SASS-locked" }
  ],
  "admission": {
    "device_sm_number": "major*10 + minor",
    "base_clause": "device D admits base token T iff T.sm_number <= D.sm_number",
    "a_clause":    "device D admits a-token T iff T.sm_number == D.sm_number"
  },
  "ordering": "sort by (sm_number ascending, variant: base < a)",
  "versioning": "adding/removing a token, altering an admission relation, or changing the encoding layer is a version bump"
}
```

Note the two independent admission clauses (base `<=`, `a` `==`) are the structural heart —
a namespace-wide `<=` would silently admit an `a`-token onto hardware that cannot decode
its SASS. The format MUST be able to carry per-class admission relations, not one blanket
rule.

## 5. Next step

Parties respond; once aligned (especially O1 format and O2 list-bearing), Baracuda
formalizes this as a KISS issue/PR per the "KISS change via PR, not edit" discipline, and
stands up `baracuda-cuda-vocab` + its owner-side drift-gate as the reference instance.

## 6. Update log — 2026-08-15: all three deciders responded (convergent)

- **O1 RULED (KISS Architect): SAME artifact.** The cuda spec is the **first instance** of
  the §6.8-0008 vocabulary manifest ("the machine-readable form of the annex"). No new type.
- **Prototype made §6.8-0008-conformant** (KISS required-field review): `schema:
  kiss-namespace-vocabulary-v1`; `vocabulary_version` an **integer**; `generated_from`
  added; `kind: enumerated` with `members` (was `tokens`); `grammar` hoisted top-level;
  `coverage_note` added.
- **`admission` → `device_admission`, fenced (KISS + Vulkane independently).** Governs
  token↔DEVICE (may this device RUN a kernel built for this token), NEVER token↔token —
  matching stays byte-exact under §6.8-0002 (no ordering/subset/prefix/implication). A
  consumer applying `<=` at cache lookup would BE the §6.8-0002 violation.
- **cuda is `kind: enumerated` permanently** — the sm-set is closed/finite (`sm100` = one
  more member by version bump), never a `generated` product, so §6.8-0013's vector
  obligations (order/dedup/threshold/digest) do not apply to cuda.
- **Drift-gate (D4) — three composed refinements:** compare the **committed file** (Unpopped:
  generator stdout can CRLF-differ by output path) at the **published ref** `git show
  origin/main:<path>` (Vulkane: a stale clone ≠ real drift), **both directions** (Vulkane).
  One equality-against-ref assertion delivers all three.
- **D3 sizing (Unpopped): the codec generalization already landed (task #22)** —
  `arch_code`/`arch_from_code` deleted, `StructureKey.arch` → `target: TargetId`; proven
  opaque by vulkan v3→v4 needing zero codec changes. Remaining surface = **4 non-test
  sites + 2 decisions**, not codec surgery. (1) DROP the `From<ArchSku>` reserved block →
  full CUDA eviction via `TargetId::parse` (owner call, made). (2) `KernelSku.arch` →
  `TargetId` if neutral, pending constructor check with Fuel.
- **`kiss_trace` clause↔spec-entry gate: accepted**, boundary stated in-tool (verifies a
  clause HAS an entry, not that the entry SAYS what the clause requires).
- **Open:** `generated_from` semantics — names the reference crate (D2's generator source)
  or the prose annex (§6.8-0011's wording)? Back to KISS.

Next: bring §2/§3/§4 to the conformant + fenced form above, resolve `generated_from`, then
formalize as a KISS issue and stand up `baracuda-cuda-vocab`.
