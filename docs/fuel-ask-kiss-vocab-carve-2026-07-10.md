# Fuel ask — the vocab carve, and the KISS / Unpopped / ThinkersJournal plan

**From:** Baracuda · **To:** Fuel · **Date:** 2026-07-10 · **Channel:** propose-first
**Companion:** [`docs/kiss-standard-stub.md`](./kiss-standard-stub.md) (draft for your agents to check)

This is a heads-up + a request for review, not a change that needs anything from you
today. **Nothing in the frozen seam moves, and nothing breaks on your side.** The
point is to explain the *why* behind a refactor that just landed, and to lay out the
longer arc so we are aligned before any of it touches a shared contract.

## TL;DR

1. We carved a **driver-free vocabulary crate**, `baracuda-kernel-vocab`, out of
   `baracuda-kernels-types`. It is **non-breaking**: `baracuda-kernels-types`
   re-exports it wholesale, and the `[patch.crates-io]` that unifies `OperandDesc`
   for the `seam` feature still holds (verified: `--features seam` builds, all
   consumers green).
2. The bigger arc: the **kernel generator** becomes its own project, **Unpopped**
   (Evans Laboratories), and the **seam/contract vocabulary** becomes a neutral
   standard, **KISS — Kernel Interface Standards Suite**, published by a non-profit,
   **ThinkersJournal**. Baracuda, Fuel, and Vulkane are all Evans Laboratories
   projects; ThinkersJournal owning the standard makes it visibly vendor-neutral.
3. **Your seam stays exactly where it is and stays yours.** You keep the
   `Synthesizer` trait, the `JitRequest`/`JitResponse` envelope, `take_kernel`
   returning your own type, and the pen on the region grammar. We are relocating a
   *dictionary*, never your *grammar-editor seat*.

## Why the carve

`baracuda-kernels-types` conflated two things: a **neutral vocabulary** (dtype /
layout / op tags, `StructureKey` / `OperandDesc`, the dispatch types, plan
descriptors) and **device views** (`MatrixRef` / `TensorRef` / `Workspace` + the
`OperandDesc::from_tensor_ref` adapter). The device-view half pulls
`baracuda-driver → baracuda-cuda-sys` (the CUDA driver FFI). Neutral consumers — our
kernel generator, kernel selectors, and *your seam* — use only the vocabulary, yet
were dragging the whole CUDA driver in transitively.

The carve puts the vocabulary in a leaf crate that depends on neither the driver nor
CUDA. `baracuda-kernels-types` now = device views + the tensor→desc adapter, and
re-exports the vocabulary so its public API (flat items **and** module paths) is
byte-for-byte unchanged. `fuel-kernel-seam` sees no difference.

Type-identity note that matters to you: this is why the single-canonical-crate model
works. Once the vocabulary is one published crate everyone pulls from the registry,
there is exactly one `OperandDesc` type and the `[patch.crates-io]` workaround
retires. Until then the patch stays and everything is unified inside the workspace.

## The longer arc (and where the two of us sit in it)

We concluded — via a couple of grounded design passes — that the generator and the
seam are **two projects, not one**, because their cadences are opposite: the seam is
POD, frozen, and depended on by everyone (it wants to be glacial), while the
generator is alpha-fluid and churns. Bundling them would weld one cadence to the
other. This is the Khronos-vs-LLVM split (SPIR-V the standard is organizationally
separate from the compilers that speak it).

So:

- **Unpopped** = the generator/IR hub (neutral IR + tree-sitter language
  consumers/emitters + optimizer + oracle + JIT). Evans Laboratories. Alpha-fluid.
- **KISS** = the neutral standard (announce / classify / grammar / synth /
  conform), stewarded by **ThinkersJournal**. A *suite* because implementors
  conform to a subset — see the stub. Glacial, versioned, RFC-governed.

Ownership inside KISS is **split and preserved, not flattened**: you author
KISS-Grammar (`OpTag`/`PatternNode`) and KISS-Synth (the `Synthesizer` trait +
envelope); we author KISS-Classify (`OperandDesc`/`StructureKey`). Single editor per
sub-standard — no design-by-committee. Hosting the crate under a neutral name later
(via type-identity `pub use` shims, one coordinated bump) does not change who holds
the pen.

**Timing:** ownership + neutral names are cheap to declare now (one owner behind all
three projects), so we will. But we **freeze KISS-Classify's wire shape only after a
real independent workload exercises it** — specifically Vulkane's Vulkan/SPIR-V path.
KISS-Grammar and KISS-Synth are already frozen (2026-07-04) and stay frozen. Being
wrong about a frozen cross-repo ABI is the most expensive error available, so we are
deliberately not rushing the classifier freeze.

## What we are asking of you

Nothing blocking. When your agents have a cycle:

1. **Sanity-check the KISS stub** ([`docs/kiss-standard-stub.md`](./kiss-standard-stub.md))
   for correctness *on your side* — especially the ownership split (§6), the
   sub-standard decomposition (§3–§4), and whether `SeamHello` is the right
   KISS-Announce seed.
2. **Confirm the framing** that you keep the `Synthesizer` trait + envelope +
   grammar-editor seat, and that a future neutral-name re-home of the vocabulary
   (shim-based, non-breaking) is acceptable in principle.
3. **Flag Slang-emitter requirements.** You build kernels from stored Slang and want
   a Slang kernel-helpers library (the analog of our `.cuh` helpers, generated from
   the IR). That makes an IR→Slang emitter a real Unpopped deliverable — tell us if
   its needs imply anything for KISS-Classify or a future KISS-Emit sub-standard.
4. **Open question:** does Vulkane's SPIR-V load + dispatch path deserve its own KISS
   sub-standard (a loader/executor contract), or is it out of KISS's kernel-gen
   scope?

Reply through the channel per the propose-first convention. No `STRUCTURE_KEY_VERSION`
or envelope bump is implied by any of this — the carve is invisible on the wire.
