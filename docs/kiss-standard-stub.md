# KISS — Kernel Interface Standards Suite

**Draft v0.1 · stub for review · 2026-07-10**

> **Steward:** ThinkersJournal (a non-profit publishing free public standards).
> **Initial signatories:** Baracuda, Fuel, Vulkane (all Evans Laboratories projects).
> **Reference implementation:** the *Unpopped* project (Evans Laboratories) hosts
> the reference crates; the crates named below are the current pre-neutralization
> seeds living in the Baracuda/Fuel workspaces.
>
> This is a **stub for the signatories' agents to check**, not a ratified spec.
> Nothing here is frozen. The whole point of publishing it now — while the sole
> author is the whole consortium — is to fix the *structure and names* cheaply,
> and to freeze each sub-standard's *wire shape* only once a real independent
> workload (notably Vulkane's Vulkan/SPIR-V path) has exercised it.

---

## 1. Purpose & scope

KISS standardizes the **interface between ML libraries, compute libraries, and
kernel providers** — the seam across which a graph/runtime and a kernel source
negotiate *what kernels exist, what they can do, what data they accept, and how a
missing one gets built*. It is a **wire/ABI + protocol** standard, not a kernel
library and not a compiler.

Explicitly **in scope:** kernel availability announcement, capability sharing,
input/output data negotiation (count / shape / size / layout / dtype / alignment),
missing-kernel notification (the JIT-on-request trigger), and the shared
vocabulary that describes all of the above.

Explicitly **out of scope:** kernel *implementations*, the neutral compiler IR
(that is Unpopped's concern, not a wire standard), and language-specific source
(CUDA/Slang/…).

## 2. Why a *suite* (not one standard)

Different implementors speak different subsets. A hand-written CUDA provider that
only *announces* what it already has implements far less than a JIT generator. A
loader/executor (Vulkane today: load SPIR-V, dispatch under Fuel's command)
touches a different interface than a kernel *builder*. Bundling all of that into
one monolithic standard would force every implementor to adopt (and every version
bump to ripple through) contracts they do not use.

So KISS is a set of **interrelated sub-standards with a strict dependency DAG**.
An implementor conforms to the subset it needs.

## 3. Layering (the DAG)

```
KISS-Announce   (POD)   availability + capability handshake         ── everyone
      │
      ├── KISS-Classify (POD)   data-shape vocabulary                ── providers + selectors
      │
      └── KISS-Grammar  (POD)   code-region synthesis grammar        ── generators + selectors
                │
            KISS-Synth  (light) JIT request/response + Synthesizer    ── generators
                │
          (KISS-Conform)        conformance suite for all the above   ── everyone, as a gate
```

Two **distinct base vocabularies** sit at the bottom, and keeping them distinct is
load-bearing:

- **KISS-Classify** describes **data** — the operand shapes/strides/dtype/alignment
  a provider is keyed on. A provider announcing "I serve this class" needs this
  and *not* the code grammar.
- **KISS-Grammar** describes **code** — the op/region grammar a generator emits and
  a selector matches. Only generators and selectors touch it.

## 4. Sub-standards (summaries)

| Sub-standard | Covers | Reference seed (today) | Frozen? |
|---|---|---|---|
| **KISS-Announce** | 56-byte `#[repr(C)]` availability/capability handshake envelope; profile + capability bits | `baracuda-seam` (`SeamHello`) | envelope versioned; escape hatch exists |
| **KISS-Classify** | dtype tags, layout/op-family tags, `StructureKey` / `OperandDesc`, `structure_key` derivation | `baracuda-kernel-vocab` (carved 2026-07-10) | **not frozen** — see §6 |
| **KISS-Grammar** | op-tag + region grammar (`OpTag` / `OpAttrs` / `PatternNode`) | `fuel-kernel-seam-types` | frozen 2026-07-04 (Fuel-authored) |
| **KISS-Synth** | `JitRequest` / `JitResponse` / `Synthesizer` trait / `SynthArtifact`; missing-kernel negotiation; two-step `synthesize` + `take_kernel` handover; never-panic contract | `fuel-kernel-seam` | frozen 2026-07-04 |
| **KISS-Conform** | conformance tests each sub-standard is checked against, keyed to the relevant `*_VERSION` | *to be built* | — |

## 5. Reference-implementation mapping

The reference implementation lives in **Unpopped** once extracted; today the seeds
live in the Baracuda/Fuel workspaces under vendor-prefixed names. Renaming to
neutral KISS crate names is a **non-breaking, type-identity-preserving re-export
shim** step, deferred until the neutral host is real. The POD wire crates other
implementors depend on get neutral names; Unpopped-the-generator keeps its brand.

Any party may implement a sub-standard independently — the standard, not
Unpopped's crates, is the contract.

## 6. Governance & versioning

- **Steward:** ThinkersJournal hosts the written spec and the conformance suite.
- **Editor (who holds the pen):** single named editor per sub-standard, **not**
  design-by-committee. Today Fuel authors KISS-Grammar/KISS-Synth (it owns the
  region grammar and the `Synthesizer` trait); Baracuda authors KISS-Classify (it
  owns `OperandDesc`). This split ownership is preserved, not flattened.
- **Process:** propose-first. A change is floated to **all** signatories before it
  is wired; version bumps that are cross-party-visible (`STRUCTURE_KEY_VERSION`,
  `SEAM_ENVELOPE_VERSION`) are coordinated. While the consortium is effectively one
  party, this runs lightweight (reference-implementation-driven); it upgrades to a
  real multi-party RFC series when an external implementer joins.
- **Freezing:** ownership + names are declared now (cheap); a sub-standard's **wire
  shape is frozen only after a real independent workload exercises it.**
  KISS-Grammar/KISS-Synth are already frozen (2026-07-04). **KISS-Classify is
  deliberately NOT frozen** — Vulkane's real Vulkan/SPIR-V descriptor semantics may
  reshape it, and being wrong about a frozen cross-repo ABI is the most expensive
  place to be wrong.

## 7. Conformance

A crate/library conforms to a sub-standard if it (a) speaks exactly the declared
types/wire format for its version and (b) passes KISS-Conform's suite for that
sub-standard at that version. Type-identity note (Rust): a single canonical
published crate per POD sub-standard, consumed from the registry by all parties, is
what makes "one type everywhere" hold — mirror-and-convert reintroduces drift.

## 8. Open questions (for the signatories, incl. Fuel's agents)

1. Is **KISS** (Kernel Interface Standards Suite) the right name, and is the
   sub-standard breakdown in §3–§4 the right decomposition?
2. Is the ownership split correct — Fuel authors KISS-Grammar + KISS-Synth,
   Baracuda authors KISS-Classify — with a single editor per sub-standard?
3. Does **KISS-Announce = `SeamHello`** belong as the lowest tier, or does Fuel
   see availability/capability negotiation living elsewhere?
4. Anything Fuel builds from its stored Slang (kernel building, the planned Slang
   kernel-helpers library) that implies requirements on **KISS-Classify** or a
   future **KISS-Emit** sub-standard we have not captured?
5. Is there a loader/executor interface (Vulkane's SPIR-V load + dispatch) that
   warrants its own sub-standard, or does it stay out of KISS's kernel-gen scope?
