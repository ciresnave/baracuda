# KISS Design Charter (working draft v0.1)

**Destination:** `github.com/ThinkersJournal/KISS` (drafted here; to be moved).
**Status:** working draft — the umbrella skeleton every KISS sub-standard doc conforms to.
**Provenance:** synthesized from a 7-lens design-considerations pass (prior-art, conformance/testability, versioning, architecture/scope, governance/IP, adversarial, consumer/emitter), 2026-07-11.

This charter is **not** a sub-standard. It is the shared frame — conventions, doc
template, versioning/capability model, governance, and the freeze gate — that
every KISS sub-standard references and none restates. It also defines the
**design→validate→red-team agent process** used to author each sub-standard.

---

## 1. Marquee decision — status of the neutral IR ⟶ **PENDING USER RATIFICATION**

Everything about KISS-Consume and KISS-Emit blocks on this, because the IR is
literally their output/input type.

- **(a) Fully Unpopped-internal** — Consume/Emit are *not* separately conformable
  (just implementation detail).
- **(b) Full normative interchange format** (SPIR-V model) — freeze
  `OpDef`/`ScalarExpr`/`Access` as a versioned ABI with its own `IR_VERSION`.
- **(c) OPAQUE-HUB (recommended)** — the Rust IR types stay Unpopped-internal and
  **unfrozen**, but a **normative op-semantics table** (NaN propagation, signed
  zero, IEEE-`fmax` vs torch-`max`, wrapping-int, select-moves-raw-bits, …) **plus
  the existing serialized text form** (`op_to_text`/`op_from_text`) become the
  shared conformance currency, checked by **oracle round-trip**, not structural
  type equality.

**Why (c):** it is the only self-consistent position — Consume and Emit become
real, independently-implementable sub-standards (they agree on what an `OpDef`
*means*) **without** freezing the alpha-fluid `#[non_exhaustive]` Rust internals
ahead of Vulkane. It mirrors exactly how Fuel already treats `structure_key` as an
opaque join token. Avoids both MLIR's over-standardize-a-churning-IR trap and the
vacuous-contract trap of leaving the exchanged type undefined.

---

## 2. Cross-cutting decisions

| # | Decision | Recommendation | Who decides |
|---|---|---|---|
| D1 | **Neutral-IR scope** | Opaque-hub (§1) | **USER** |
| D2 | **Source of truth for POD tiers** (Announce/Classify/Grammar) | A **machine-readable canonical schema** (SPIR-V grammar-JSON / MLIR-ODS style) generates docs + Rust types + conformance vectors — kills drift. | Design-agent (user ratifies tooling cost) |
| D3 | **Spec/test/impl authority** | Written spec **normative**; Unpopped is *a* conformant impl with no privilege; KISS-Conform authored **independently** of the impl (the CPU oracle shares zero lowering code); "no test without a spec clause". | Settled |
| D4 | **How values are pinned** | **Bits / IEEE-754 semantics, endianness-pinned — never one language's surface spelling.** Reclassify `const_lit` as emitter-supplied; require full special-value round-trip (±inf, quiet/signaling NaN, ±0, subnormals) per dtype. | Settled |
| D5 | **Determinism/fidelity vocabularies** (3 today: emitter `VariantFidelity`, oracle `Fidelity`, Fuel FKC) | **One canonical enum + a normative mapping table**; every numeric clause declares its class so KISS-Conform auto-selects memcmp / tolerance / order-invariant. Emit **imports** Fuel's FKC strings + Rule-9, never re-forks them. | Emit design-agent + Synth editor (Fuel) |
| D6 | **Conformance shape** | **Claimable DAG-aligned subsets** + a **mandatory core** per sub-standard + a machine-checkable claim manifest (reuse the 16-profile cap + u64 capability bitset). Test both that the claim passes *and* that un-claimed inputs produce **typed declines, never panics**. | Settled |
| D7 | **Extensions / capability u64** | Reserve **core/experimental/vendor** ranges now (while single-ownership makes it free); **split the u64** into three axes (which sub-standards I speak / optional features / external FDX-DLPack tokens); PR-gated registry under ThinkersJournal; EXT→ARB→core promotion path. Reconcile the FDX-borrowed low-bit block against a written registry. | Design-agent + **USER** (FDX reconciliation, registrar) |
| D8 | **Licensing / patent / mark / custody** | Decide **now** (see §7). Retrofitting onto external signatories is near-impossible. | **USER (legal/IP)** |
| D9 | **Freeze gate** | Promote draft→frozen only after **≥2 structurally dissimilar implementations interoperated on golden vectors AND a non-Rust foreign reader consumed the wire** (endianness / pointer-width / `repr(C)` padding checked) AND the sub-standard's conformance suite exists+passes. | Settled (rule); certifier named per sub-standard |
| D10 | **Consume/Emit symmetry + input type** | Give Consume a **`Consumer` trait + name→Consumer registry** (it's bare free functions today); Emit's **normative input is `(OpDef + StructureKey)`, not the schedule-resolved `KernelPlan`** (which would drag Baracuda's scheduler into the ABI). Generalize typed declines to dtype/op/schedule. | Consume/Emit design-agents (need editors) |

---

## 3. The dual doc template (satisfies "overview **and** testable spec")

Every KISS sub-standard doc has two halves:

**Overview (informative) — §0–§5**
0. Front-matter: title, sub-standard ID, maturity stage (Draft / Frozen(date) / Deprecated / Retired), editor, steward, reference seed crate(s), DAG position.
1. Purpose & Scope + a one-line "KISS-⟨X⟩ is **NOT**: …" exclusion list (self-policing scope).
2. Overview / Rationale — the human mental model, worked examples, why the choices (may use lowercase must/should freely).
3. Terms & Definitions.
4. Normative References (Rust `repr(C)` layout guarantee, IEEE-754, byte-order; upstream KISS sub-standards **by version**, each DAG edge labeled **OPAQUE** vs **STRUCTURAL**).
5. Conventions — pointer to this charter's keyword + clause-ID rules (stated once here, referenced, never restated).

**Conformance spec (normative) — §6–§10 + appendices**
6. **Specification** — numbered atomic clauses, each a single MUST/MUST-NOT/SHALL with a stable ID; values pinned as bits/IEEE-754. For POD tiers, **generated** from the canonical schema.
7. Capability, Profile & Extension model — mandatory core; negotiable options; reserved ranges; the version-negotiation algorithm; hard-gate-vs-reserved-and-ignored per field.
8. Versioning & Lifecycle — the **two version axes** (wire/ABI schema version vs published-crate semver) with a bump-vs-no-bump rule table; maturity entry/exit criteria; the freeze gate (D9); retire-by-floor deprecation.
9. Conformance — the claim format, DAG prerequisite closure, and the clause-ID→KISS-Conform-test traceability matrix stub.
10. Governance — editor, ratifier, license, patent grant, mark-use tie.
A. Appendices (informative) — golden-vector references, migration recipes, examples.

**Testability convention (the VUID model):** normative §6+ uses **only uppercase
MUST/MUST-NOT/SHALL** (SHOULD/MAY reserved for governance, never byte-level wire
facts). Every atomic requirement carries a stable, append-only, machine-parseable
ID `KISS-<SUB>-<section>-<nnnn>` (e.g. `KISS-ANNOUNCE-6.1-0004`), allocated by that
sub-standard's editor, **never reused after retirement**. Each ID maps **1:1 to ≥1
named KISS-Conform test; the suite build FAILS on any untested normative clause**
(bidirectional traceability). Ban unquantified adjectives ("well-formed",
"reasonable", "neutral") in normative text. Clause IDs live in a machine-readable
sidecar (a `validusage.json` analog) kept in sync by a lint; for POD tiers both the
prose tables and the sidecar are **generated** from the canonical schema. Every
clause declares its determinism/fidelity class so KISS-Conform picks the right
comparator. Reference-crate debug asserts cite the clause ID they enforce
(Vulkan-validation-layer style).

---

## 4. What each sub-standard must nail (condensed)

- **KISS-Announce** — the 56-byte envelope re-specified **language-independently**
  (field order/offsets/sizes/padding/alignment/magic/version, little-endian) so a
  C/Slang/SPIR-V reader reproduces the exact bytes; **converge the two byte-identical
  `SeamHello` seeds to ONE canonical registry-published crate** (no-wire-change
  re-export shim; verify by golden hex, not struct equality); the version-negotiation
  algorithm as a testable procedure (highest mutual profile, hard-fail never panic on
  empty intersection); split the capability u64; POD readers MUST **reject** unknown
  layout; zero-dependency budget as a conformance check.
- **KISS-Classify** — decide the ABI is the **string token codec** (`sk2|bin|f32|cuda:sm89|ix32|…`),
  struct is reference impl; **resolve the CUDA-shaped `ArchSku`** (abstract target-capability
  class or remove — an NVIDIA-only SKU contradicts sharing with Vulkane); pin every
  primitive (dtype set, `MAX_RANK=8`, `MAX_OPERANDS=8`, signed-i64 stride model,
  `align_bytes`); the exhaustive-vs-`#[non_exhaustive]` freeze policy as a normative
  per-enum rule; **explicitly UNFROZEN** with the Vulkane freeze gate spelled out.
- **KISS-Grammar** — the Grammar↔IR-op mapping as a normative surface (every advertisable
  op has a named `OpTag`); how a **frozen** Grammar admits the still-growing IR op set;
  golden vectors at the 2026-07-04 shape; **retro-fit a conformance gate** ("frozen"
  must mean "has a passing gate" — it doesn't yet).
- **KISS-Synth** — never-panic as a **fuzz-testable** obligation; golden vectors for the
  two-step handover + `JitRequest`/`JitResponse`/`SynthArtifact`; **owns** the FKC
  determinism vocabulary Emit imports; the JIT-on-request trigger ↔ Announce capability
  bit; resolve variant/`launch_note` ownership (Synth vs Emit).
- **KISS-Consume** — a **`Consumer` trait + registry** (bare free fns today); residue as a
  normative refusal taxonomy (not-a-kernel / wrong-op-class / unrecognized-but-expressible /
  inexpressible-residue); **mandate CST-based recognition** (declare substring/keyword
  sniffing non-conforming; retire `lift.rs`); **relocate the shared taxonomy/tables**
  (`LiftError`/`Lifted`/`unary_fn`/`binary_fn` — which pin `fmaxf→FmaxIeee` not `Max`)
  before deleting the pilot; source languages/grammars are **out of scope**.
- **KISS-Emit** — a **complete partition** of every lowering decision into "neutral driver
  may spell it" vs "emitter MUST supply it" (const/non-finite spelling → emitter-supplied;
  and audit the "infix `+-*/` is universal" claim, not assume it); a **pre-freeze
  neutrality audit** (hunt `const_lit` siblings — the worked example); one typed
  decline taxonomy (dtype/op/schedule) answerable without panicking on the JIT path;
  normative input `(OpDef+StructureKey)`; the **emit↔lift round-trip in two tiers**
  (structural IR equality over a declared subset; numeric bit-identity **same-language
  on-device only** — do NOT overclaim cross-language numeric identity); pull the
  `output`/`input{K}` naming + harness conventions **into** the standard; `EMIT_ABI_VERSION`;
  freeze only after a non-C emitter (Slang, done) + a second consumer certify it.
- **KISS-Conform** — the **bidirectional traceability matrix** (build fails on untested MUST);
  four modalities (golden byte-vectors; the **independent** CPU-oracle differential harness;
  an IR-DAG fuzzer emitting to every backend; negative/decline vectors); determinism-class-aware
  comparators; conformance keyed per sub-standard **per version**; deprecation tested; the
  **adversarial-outsider checklist** (endianness / pointer-width / `repr(C)` padding /
  non-Rust reader) as a freeze precondition; **the reference impl runs the same public suite
  with no exemption**.

---

## 4a. Kernel provision & contracts — RATIFIED 2026-07-12

Surfaced by Fuel receiving a kernel and doing nothing with it. A kernel a consumer
can't learn *how to call* or *what it computes* is unusable, so KISS must guarantee
the contract is reachable. Ratified:

- **Every kernel MUST have an FKC contract.** Today's "honest miss" (reductions /
  scans / etc. emit *no* contract) is **not** a fundamental limit — it exists only
  because the current contract is tied to Fuel's named `OpKind` vocabulary. The
  opaque-hub IR (§1) removes it: the contract's **semantics field carries the neutral
  serialized IR** (the shared op-semantics currency), so a kernel is describable even
  to a consumer with no named op for it (a reduction → "reduction, op=Sum,
  body=Input(0)"). The **required core is universal for every kernel**: identity
  (accept-predicate = `structure_key`) + interface/ABI (operand signature, entry
  point, launch, alignment, in-place) + declared capabilities (dtype / precision-ULP
  / determinism class / cost). The semantics field is machine-checkable IR for
  *generated* kernels and degrades to a declared op-identity tag for *hand-written*
  escape-hatch kernels — still honest, never faking a named semantics it lacks.
  *Owner: the KISS-Contract sub-standard (the contract format). Level: **MUST**.*

- **Semantics is a HIERARCHICAL op DAG, resolved recursively — not fully lowered.** A
  kernel's semantics is a DAG of ops at *mixed* abstraction levels: a fused gelu-matmul
  is `matmul` + `gelu`, NOT the thousands of primitive fma/exp/poly ops it expands to.
  Every op that isn't primitive **has its own contract** giving its reference
  decomposition into strictly-lower-level ops; a consumer that doesn't know an op
  **queries that op's contract** and resolves recursively until it reaches ops it knows
  or the primitive floor. Strictly better than always-lowering: *compact* (two ops vs a
  primitive soup), preserves *meaningful matching* (a consumer with a native `gelu`
  matches the node directly instead of **raising** a primitive DAG — hard and lossy), and
  stays *fully resolvable* for consumers lacking the vocabulary. The fully-lowered
  primitive form is produced **on demand** (by resolving the DAG) as the oracle for
  bit-identical verification. Invariants:
  - a normative **PRIMITIVE FLOOR** (a mandatory-core op set every consumer MUST
    understand) is the termination guarantee — every chain bottoms out there;
  - **acyclic + strictly-decreasing level** (an op is defined only via lower-level ops) —
    resolution always terminates, never cycles;
  - **labeling a kernel with a high-level op ASSERTS conformance to that op's declared
    precision/determinism** (the consumer-verification SHOULD checks it — a cruder-than-
    `gelu` kernel can't be labeled `gelu` without declaring the delta).
- **Extend by DEFINITION, not by primitive — but never limit to a consumer's current
  vocabulary.** If a kernel isn't describable, we MUST add the op needed (we do NOT cap
  ourselves at Fuel's current `OpKind`s). But adding a **high-level op** is cheap +
  additive (a decomposition over the existing floor; no consumer is *required* to know it
  — they resolve it), while adding a **primitive** is a new *axiom* every consumer must
  eventually implement (a mandatory-core change). Rule: add high-level ops freely; add a
  primitive ONLY when genuinely inexpressible over the floor (a true atom — a hardware
  intrinsic with no decomposition). Keep the floor small + stable; guard its growth.
- **Hand-written / escape-hatch kernels get REAL semantics via decomposition, not a bare
  tag** — they are lifted (KISS-Consume) into the op DAG as far as they go; the un-liftable
  remainder is honest **residue**. So a hand-written kernel carries the richest semantics
  its decomposition affords. Note: **Fuel's own kernel-decomposer IS a KISS-Consume
  lifter** — Fuel breaking a kernel into an equivalent op sequence is the same capability
  as the generator's source→IR lift; standardizing the op vocabulary makes them target the
  same neutral ops, so either can produce a contract's semantics field.
- **New foundational sub-standard: KISS-Ops (the Op Vocabulary & Semantics).** DECIDED
  2026-07-12: the op vocabulary + per-op semantics table + per-op reference decompositions
  + the primitive floor get their own standalone sub-standard — the **concrete home of the
  opaque-hub op-semantics currency** (§1 committed to it normatively without a home; this is
  the home, not new scope). It contains: the op set (primitive floor + higher-level ops);
  each op's semantics (NaN-propagation / signed-zero / IEEE-fmax-vs-torch-max / wrapping-int
  / select-moves-raw-bits); each non-primitive op's reference decomposition (what makes the
  hierarchy resolvable). Referenced by KISS-Contract (semantics field), KISS-Grammar
  (advertisable ops), KISS-Consume (lift targets), KISS-Emit (lowering sources). **DAG
  position: foundational, beside KISS-Classify** — Classify is the DATA vocabulary
  (`OperandDesc`/`StructureKey`), KISS-Ops is the COMPUTATION vocabulary. **KISS-Grammar
  re-bases on it:** an `OpTag` becomes "a KISS-Ops op name + Fuel's pattern/synthesis
  attributes" (no parallel op list; Grammar and Ops version on their own cadences).
- **A hand-written kernel's contract completeness tracks the LIFT FRACTION.** The generator
  contracts any kernel it can lift (source → IR) identically to one it built; the lifter
  recognizes known idioms and refuses the rest as **residue**, so a fully-lifted kernel gets
  a full contract and one with residue gets a partial (honest) one. The hierarchical model
  raises the lift success rate too (recognizing a `gelu` and emitting the high-level op is
  easier + more robust than lifting its primitive expansion) — so "broaden the lift fraction"
  = "broaden contract coverage for hand-written kernels", the same work with two payoffs.

- **Provision is announce → query, not announce-with-contract** — and it is the
  **generalization of KISS-Synth**. Querying an existing kernel's contract and
  JIT-synthesizing a missing one are the SAME request/response: *"consumer asks the
  provider for a kernel by `structure_key`; provider returns `{artifact, contract}`,
  building it if it doesn't exist yet."* JIT is the build-on-miss case. So KISS-Synth
  becomes a **kernel-provision** protocol, and **every returned kernel carries its
  contract**. The three levels:
  1. **Provider handshake** (SeamHello, KISS-Announce) — provider-level capability
     negotiation only (KISS profiles/versions, "supports contract-query", "does JIT").
  2. **Kernel availability** — the provider announces kernels by **identity only**
     (`structure_key`, `revision_hash`) so a consumer distinguishes a cache hit from a miss.
  3. **Contract query** — on a miss, the consumer requests the contract for a
     `structure_key`; the provider returns the full contract (LSP negotiate-then-request).
  *Rationale: a provider may offer thousands of specialized cells; pushing a full
  contract per kernel in the announce is wasteful — fetch on cache-miss instead.*

- **The per-kernel announce carries NO capability — only identity (de-dup).** Per-kernel
  capability is the **contract's** job (single source of truth); duplicating it in the
  announce would drift. What SeamHello keeps is **provider-level handshake capability**
  (profiles / "supports contract-query" / "does JIT") — a different thing, not per-kernel,
  and absent from any kernel contract. So: SeamHello = provider handshake; availability =
  kernel-identity list; contract (queried) = per-kernel capability + usage + semantics.

- **A consumer SHOULD verify a received kernel against its contract** before trusting it —
  run it against the contract's declared precision (ULP bound / tolerance comparator),
  determinism class, and accept-predicate (= `structure_key`), reusing the KISS-Conform
  oracle-differential (D3) + determinism-class comparators (D5). Trust-but-verify.
  *Owner: KISS-Conform. Level: **SHOULD** (consumer-behavior governance, not a wire MUST).*

**Sub-standard impact:** KISS-Synth generalizes JIT-on-request → **provision**-on-request
(every provided kernel carries its contract); KISS-Announce adds the identity-availability +
contract-query protocol and **drops per-kernel capability**; a **KISS-Contract** facet owns
the universal contract format (identity + interface + capabilities + IR-semantics);
KISS-Conform adds the consumer-verification SHOULD.

**The KISS-Contract structure — SEVEN sections (from the 2026-07-12 completeness analysis).**
The current (F)KC is a *good Fuel feed but a poor source-of-truth*: right facts, wrong
architecture — a flat serde field list (accept/return/caps/cost/precision) where "how to
call it" is smeared across five places and semantics is a trailing fusion-only afterthought.
The neutral KISS-Contract gives every fact exactly one home:

1. **Identity** — contract kind+version, kernel name, revision hash, and the accept-predicate
   **unified with op-identity**. *(Correction to §4a's earlier framing: `structure_key` is an
   ADMISSIBILITY predicate over a layout/dtype/arch specialization CELL — coarse op *category*
   + operand-0 dtype, no extents — NOT the op's semantic identity. A consumer must match BOTH
   "which cell fits" AND "which op is this"; the identity section joins them into one match.)*
2. **Semantics** — the single mandatory, recursively-resolved **hierarchical op DAG over
   KISS-Ops** (primitive = a 1-node DAG; fusion = its DAG), each node carrying an **OpAttrs
   channel** (axis / OOB / permutation / reduce-axis) + edge-case policy + the human blurb.
   The spine, not a trailing field — mandatory + decomposition-backed is what fixes the
   honest-miss "semantics vanishes" bug.
3. **Interface (ABI)** — everything to mechanically call it, in ONE place: entry point +
   target, and the **full positional argument signature** — operand pointers AND the runtime
   launch scalars the ABI actually takes (extents/strides, `n`, base offsets, gather extents,
   workspace pointer+size, scalar params) with declared order; plus `count_unit`/`in_place`/
   `alignment` moved out of the `caps` grab-bag.
4. **Dispatch** — the **normative launch model**: invocation/index domain, workgroup/block
   sizing, the count-unit→grid derivation, thread→element mapping (grid-stride) — contract,
   not a doc hint.
5. **Capabilities** — the declared envelope that is NOT a per-call signature (supported dtype
   set, awkward-layout strategy, in-place-eligible variants, index-width).
6. **Guarantees** — precision (**reference function NAMED** + per-*backend* ULP tiers),
   determinism level, cost — unified, not scattered across four structural levels.
7. **Provenance** — origin/trust in one place (kernel source, revision base/hash, cost
   declared-vs-measured, audit status, negotiation metadata).

**Biggest single gap: Interface + Dispatch — the "how to call it" the contract exists FOR has
no home today.** A neutral SPIR-V/CPU consumer literally can't bind or launch: it gets a
`dlsym` symbol but no argument signature, none of the strided launch scalars (even though
strided cells DO get contracts), no gather extent, and the launch geometry is declared
"provider-internal." The real ABI-marshalling lives out-of-band in the sys-crate `KernelRef`,
shared tacitly Baracuda↔Fuel. **Keep as-is (already neutral, rename/re-home only):**
`structure_key` accept-predicate, `entry_point` + `revision_hash`, dtypes, `same_as(in0)`
shape rule, alignment, the `coeff*n` cost exprs. Remove the Fuel gates (`fuel_primitive_op_kind`
None-withhold, `FUEL_FUSED_OPS`) so contract existence is decided by the contract's OWN
semantics; replace the markdown-`fkc`-fence transport with a self-delimiting, strictly-schema'd
document (header declares kind+version) so a malformed contract fails LOUDLY, not empty. Freeze
the schema only after ≥2 dissimilar implementations can bind+launch+reason from the text alone.

---

## 5. Design-agent process (the later fleet)

1. **Draft this charter first** (done, v0.1) — every sub-standard references it.
2. **Per sub-standard, a three-role fleet, each adversarially distinct** so it is not rubber-stamp:
   - **DESIGN** — authors the dual-doc against the template; generates POD clauses from the canonical schema.
   - **VALIDATE** — checks every normative clause is atomic, testable, bit-pinned, and has a mapped KISS-Conform test; builds the traceability matrix; flags unquantified adjectives + informative-leaks-into-normative.
   - **AUDIT / RED-TEAM** — tries to **break** the spec: hunts C/CUDA-isms in "neutral" seams; runs the adversarial-outsider checklist; **attempts a second dissimilar implementation from the DOCUMENT ALONE** (not from Unpopped) and reports every ambiguity that let it drift.
3. **Oracle-independence as process** — the agent authoring conformance vectors MUST NOT read the reference-impl lowering code; vectors derive from the spec's op-semantics table. A test satisfiable only by matching Unpopped output is rejected as circular.
4. **Checklist gate per maturity transition** — traceability complete, red-team second impl succeeded from the doc alone, adversarial-outsider checklist passed, (for freeze) the D9 gate met. **The AUDIT agent signs the transition, not DESIGN.**
5. **Sequence by DAG + actionability** — Announce, Classify first (POD, drift already live), then Grammar (retro-fit its gate), then Synth, then Consume/Emit LAST and explicitly UNFROZEN. Emit's neutrality audit blocks any Emit freeze.
6. **Cross-sub-standard reconciliation pass** — a dedicated agent checks shared surfaces don't fork (determinism vocab; capability registry; IR op-semantics table; the `structure_key` opaque boundary) and that each DAG edge's opaque/structural label matches on both sides.
7. **Every surfaced ambiguity becomes a numbered RFC** in the ThinkersJournal RFC directory (import the existing bilateral ask-docs as RFC-0001…) — the fleet's findings become the public record.

---

## 6. Top traps (guardrails the docs must bake in)

- **Reference-impl-IS-the-spec** (CPython/TypeScript ossification) — with the sole author being the whole consortium, "whatever Unpopped does" silently becomes normative unless tests are spec-derived and a foreign implementation exercises the wire. The `const_lit` C-ism proves incidental impl choices already leak.
- **Happy-path golden vectors** — finite-const tests all passed while `const_lit` encoded a C-ism; you need adversarial/negative + dissimilar-backend differential vectors.
- **Freezing before a second *dissimilar* workload** — a frozen wrong cross-repo ABI is the most expensive place to be wrong.
- **Vacuous translation contracts** — standardizing Consume/Emit as trait signatures while leaving the IR "out of scope"; interop *is* agreement on what an `OpDef` means.
- **Overclaiming cross-language numeric identity** — round-trip is structural IR equality; Slang `tanh` ≠ CUDA `tanh`.
- **Coverage theatre** — prose-only MUSTs with no clause IDs and no clause→test map; untyped panics inside `lower()`.
- **Byte-identical copies ≠ convergence** — identity must come from ONE canonical crate (the `[patch.crates-io]` distinct-trait bug already materialized this).
- **Capability/profile bit sprawl** (the OpenGL swamp) — one flat u64, no reserved ranges, no mandatory core.
- **Non-profit-veneer neutrality** — a steward whose crate/GitHub org sits under a personal Evans account is not credibly independent; deferring the licenses/patent/mark past publish makes them near-impossible to retrofit.
- **Forward-compat direction backwards** — hard-gate POD (reject unknown), soft-skip JSON/text (ignore unknown); never the reverse.
- **Scope creep by silence** — every doc must explicitly exclude the neutral IR internals, the source languages, and Vulkane's SPIR-V dispatch.
- **Vendor/project-name leakage into normative text (the neutrality illusion, concrete).** The central artifact is still named **FKC = *Fuel* Kernel Contract**, and the reference impl carries `baracuda_gen_*` / Fuel `OpKind` shaping. *Ratified 2026-07-12:* normative KISS clauses (§6+) MUST be vendor-neutral — generic roles only (**provider** / **consumer** / **implementation** / **kernel** / **contract** / **target**), never "Fuel"/"Baracuda"/"CUDA-only". **Rename FKC → "Kernel Contract"** (sub-standard **KISS-Contract**; spell it out — no new opaque acronym). Project names appear ONLY in non-normative *examples*, *provenance/acknowledgments*, *reference-impl pointers* (§0 front-matter), and the *governance/signatory* record. The rename is the easy half; the hard half is neutralizing Fuel-shaped **assumptions** — OpKind-gated semantics (fixed by hierarchical KISS-Ops), the markdown-fenced transport (the headingless-block footgun), reliance on Fuel adopt/loader/Rule-9 semantics, the CUDA-shaped `ArchSku` — surfaced by the neutral-consumer analysis and to be stripped per sub-standard.

---

## 7. Decisions (RATIFIED 2026-07-11) — legal/IP + marquee

### Ratified by the user 2026-07-11

1. **IR = opaque-hub.** Internals don't matter so long as the consumer/emitter interfaces — including the op-**semantics** currency (what each op MEANS) — are fully documented.
2. **Licenses:** spec **CC0 1.0 Universal** (public-domain dedication; *supersedes the earlier CC-BY-4.0 plan* — the ThinkersJournal/KISS repo was set up with a CC0 `LICENSE` on 2026-07-12, and the umbrella §9.1 + Announce license notes were aligned to CC0; CC0 waives copyright only, so the §9.4 patent grant is retained separately); reference crates **MIT-OR-Apache-2.0**; conformance suite permissive-to-run with a mark policy forbidding a MODIFIED suite from backing a conformance claim.
3. **Patent:** royalty-free grant to essential claims on RFC contribution + defensive termination.
4. **Conformance-claim posture + custody:** Conformance is a **factual** property — pass the *unmodified* KISS-Conform suite — and the **steward's published registry is the authoritative record** of verified implementations. KISS does **not** police "KISS-conformant" claims: a false claim self-reveals (the software won't interoperate, and it isn't on the registry), so value accrues to being *listed*, not to the assertion. A registered certification mark is an OPTIONAL future lever (worth it only if the badge ever becomes a widely-relied-on purchasing signal at scale), **not a v1 requirement** — this keeps KISS simple. Custody: ThinkersJournal on the user's personal GitHub is fine while sole-member (GitHub Orgs are free — no paid tier needed); formalize by incorporating the non-profit + transferring org/crate ownership when external parties join.
5. **Governance = editor-of-record per sub-standard + interested-consignatory comment/vote.** Grammar/Synth = **Fuel**, Classify = **Baracuda**, **Consume/Emit = Unpopped** — each requests comment from affected consignatories (projects building consumers/emitters, or using Unpopped to generate/optimize/translate) before deciding.
6. **`ArchSku` → a generic, namespaced capability descriptor** `<namespace>:<capability-set>`, **open to ALL hardware targets** (CUDA, Vulkan, ROCm, Metal, Intel Arc, NPUs, TPUs, …) — not a CUDA enum, and **not gated on Vulkane**. The **steward registers namespaces**; each **namespace's maintainer** (the vendor or its supporters) owns that namespace's capability-sets (CUDA's are `sm80`/`sm89`/`sm90`, …). The token stays byte-exact (match on the full string). The DESIGN is decided now; FREEZING the exact per-namespace vocabulary still waits on real non-CUDA usage (Classify stays UNFROZEN). Open refinement for the Classify RFC: whether the namespace axis is best keyed on **ecosystem/compilation-target** (recommended — one manufacturer's hardware can be driven via multiple kernel paths, e.g. NVIDIA via CUDA vs Vulkan/SPIR-V, which need different kernels) or pure manufacturer.
7. **Conformance:** initial self-certification with published results **+** a steward-maintained registry — the steward queues self-certs that request certification and certifies them free (as resources permit), posting results behind a "steward-certified implementations" list.
8. **Vulkane SPIR-V loader/executor = OUT of scope.** In-ecosystem kernel loading is already standardized (Vulkan/CUDA); KISS governs kernel passing BETWEEN software, so KISS buys nothing by reinventing it.

### Original framing (for context)

These block or expire; agents cannot make them.

1. **IR scope** (D1) — opaque-hub / interchange-format / internal? *(recommend opaque-hub)*
2. **Three licenses** — spec text **CC-BY-4.0** (explicit right-to-implement); reference crates **MIT-OR-Apache-2.0** (Apache carries the patent grant MIT lacks — *note: the already-published `baracuda-kernel-vocab` inherited the workspace's `MIT OR Apache-2.0`, so the crate leg is effectively already this*); conformance suite permissive-to-run with a mark policy forbidding a **modified** suite from backing a conformance claim.
3. **Patent posture** — automatic royalty-free grant to essential claims on RFC contribution + defensive termination, bound now while you hold all rights?
4. **Trademark** — clearance on the overloaded name "KISS" (the principle, KISS FFT, KISS Linux, the band); if unenforceable, a more distinctive certification mark for "KISS-conformant". Confirm the mark + crates.io org + `ThinkersJournal/KISS` admin are held by the **steward**, never a personal/Evans account.
5. **Editors for Consume/Emit** — currently ownerless (Grammar/Synth=Fuel, Classify=Baracuda settled).
6. **`ArchSku` fate** — stays (abstracted to a target-capability class) or comes out of the neutral token?
7. **Conformance mechanism** — self-certification with published results (test262/WebGL) or a steward-maintained conformant-implementations registry (Khronos)?
8. **Vulkane's SPIR-V loader/executor** — its own sub-standard, or explicitly OUT of KISS's kernel-gen scope? *(recommend stating OUT so silence doesn't invite creep)*
9. **Contribution intake** — DCO sign-off vs CLA; a one-page signatory agreement (patent grant + mark + DCO) so joining is a form-sign, not a renegotiation?

---

## 8. Recommended first moves

1. **This charter** (v0.1) — ratify/edit; move to `ThinkersJournal/KISS`.
2. **User settles §7.1–§7.4** (IR-scope + the three legal/IP instruments) — they gate everything and can't be retrofitted.
3. **Draft KISS-Announce first** — lowest DAG tier, live drift (two `SeamHello` seeds), and its convergence to one canonical crate is a concrete v1 deliverable that exercises the whole template + testability convention end-to-end.
4. **Then KISS-Classify** — pin the primitives, decide codec-vs-struct, force the `ArchSku` resolution; keep it explicitly UNFROZEN pending Vulkane.
5. **Stand up KISS-Conform's scaffold alongside** (not last) — the traceability lint, golden-vector harness, oracle differential engine — so "frozen" can mean "has a passing gate".
6. **Set up the canonical-schema tooling** for the POD tiers before writing their §6, so clause tables / Rust types / vectors are generated, not hand-synced.
7. **Defer Consume/Emit docs** until the IR-scope verdict lands + editors are assigned; front-load Emit's neutrality audit as a blocking pre-freeze task; give Consume the trait+registry+CST upgrade before treating it as a standard.
