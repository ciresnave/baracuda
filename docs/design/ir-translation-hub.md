# The IR as a universal kernel-translation hub — design + roadmap (2026-07-10)

**Status: in progress — several phases shipped.** This captures a multi-phase
initiative that grows `baracuda-kernelgen` from "a generator that emits
specialized CUDA kernels" into the **single source of truth** for every
IR-expressible optimization, with the neutral IR as a bidirectional translation
hub. It builds directly on the shipped pieces: the neutral IR
([`ir.rs`](../../crates/baracuda-kernelgen/src/ir.rs)), the `Backend` emitter
seam ([`backend.rs`](../../crates/baracuda-kernelgen/src/backend.rs)), the CPU
correctness oracle ([`oracle.md`](oracle.md)), and the ship-top-K bench-gate.

**Shipped so far** (device-validated on RTX 4070 / sm_89 / CUDA 13.3):
Phase 1 — freestanding-helper emission (`emit_coord_unravel_helper`; a generated
`.cuh` bit-identical to the hand-written one and **1.31× faster**, from a shared
`emit_unravel_decomp` that also backs all four inline strided emitters
byte-identically). Phase 2c (partial) — a second generated helper
(`emit_dtype_promote_helper`), exhaustively bit-exact across all 131072 f16+bf16
codes (a de-dup win, no speedup — the honest outcome). Phase 4 (spike) — the
CUDA→IR **`lift`** frontend ([`lift.rs`](../../crates/baracuda-kernelgen/src/lift.rs)):
recognizes a grid-stride elementwise CUDA kernel, parses its body to a
`ScalarExpr`/`OpDef`, refuses non-expressible constructs as residue, and re-emits
the same lifted IR to CUDA **and** portable-C — source → IR → every backend,
proven end-to-end.

## The shape

```
 {CUDA, Slang, …}                                    {CUDA, CpuC, Metal, SPIR-V, …}
        │                                                        ▲
        ▼                                                        │
   [ converters ]  ───────────►  neutral IR  ───────────►  [ emitters ]
  (source → IR frontends)      (ScalarExpr DAG          (IR → source backends)
                                + Access + OpDef)
                                     │
                       two emission modes off one IR:
                       ├─ fused specialized whole kernels   (IR-writers / op facade — peak perf via fusion)
                       └─ generic reusable helpers (.cuh/.h/.metal)  (source-writers — auto-improving primitives)
                                     │
                       validation spine (both directions):
                       ├─ CPU oracle        → bit-exact correctness
                       └─ ship-top-K bench  → per-arch performance
```

`N` source frontends × `M` target backends collapse to `N + M` translators
through one IR, instead of `N × M` hand-ported kernels. Every improvement to a
shared emitter routine (better vectorization, a new schedule, a new backend)
propagates to **every** output — generated kernels *and* generated helpers *and*
converted kernels.

## Two developer audiences — both first-class

1. **IR-writers** — describe the math in the IR and get a kernel in every
   backend. Supported *today* (see "Authoring the IR" below), just rough.
2. **Source-writers** — keep writing the language they know (CUDA, Slang). They
   either `#include` our **generated** helper `.cuh`/`.h` (and thereby inherit
   every generator improvement), or run their existing kernels through the
   **converters** to lift them into the IR and re-emit to every backend.

Neither audience is a second-class citizen; the IR serves both from one source
of truth.

## Component 1 — freestanding-helper emission mode (the enabler)

Today the emitter produces *specialized, fused whole kernels* only; its
optimization routines (`emit_block_reducers`, the coord-unravel emitter,
`scatter_combine_store`, …) inline into kernels. Add a mode that emits the same
logic as **generic composable device functions** — a standalone
`template<class T> __device__ T warp_reduce_sum(T)` etc. — into a `.cuh` (and a
`.h` for CpuC, a `.metal` for a future Metal backend).

- **Boundary (same rule as everywhere):** algorithmic helpers
  (`coord_unravel`, warp/block `reduce` + `scan`, `atomic` scatter, dtype
  `promote`, `cast`, `contiguize`) generate cleanly. Hardware-*technique*
  helpers (`cp_async`, shared-memory bank-tiling, sub-byte nibble packing) hit
  the "CUDA-shaped IR node" limit — they stay hand-written or become
  backend-specific emitter extensions, not neutral IR.
- **Performance bar: "≥ the hand-written `.cuh`."** A generic reusable helper
  has call overhead and no cross-op fusion, so it will *not* match the
  generator's own fused whole-kernel output — and it doesn't need to. Fusion is
  the IR-writer audience's win; the helper audience trades it for convenience +
  auto-improvement. The right comparison is against the hand-written helper it
  replaces, which is achievable.

## Component 2 — migrate the helpers to generated (oracle + bench gated)

For each hand-written algorithmic helper: generate the equivalent → prove it
bit-matches via the **oracle** → bench it against the hand-written one via the
**ship-top-K bench-gate** → replace only where the generated version wins-or-ties
→ delete the ones that stay dead. The correctness and performance gates already
exist, so "as good or better" is *measured*, not asserted — the migration is
principled and reversible, not a leap of faith.

(Context: an audit found 15 reusable-optimization helpers today, 7 live / 8 dead;
the same optimization can currently live in up to three drifting homes — the
dead `.cuh`, per-kernel hand-rolled copies, and the generator's inline emitter.
This component collapses them to one.)

## Component 3 — IR-authoring ergonomics (for the IR-writer audience)

Today the IR is a **Rust embedded-DSL**: build a body with the `Expr` builder,
wrap it in a named `Access` constructor, and generate —

```rust
let body   = /* Expr builder: e.g. input(0) * input(1), then .relu() */;
let op     = OpDef::elementwise("fused_mul_relu", 2, &[F32], body);
let key    = structure_key(OpCategory::BinaryElementwise, &[a, b, out], ArchSku::Sm89);
let kernel = generate(&op, &key, &Cuda);   // -> GeneratedKernel { name, source }
```

This works now for every supported shape (elementwise / pred / multi,
row_reduce, scan, window, row_sort/topk). What it lacks: **no text/serde form**
(so it can't be authored in a file, round-tripped, or diffed), thin authoring
docs, and no author-facing construction errors. Add those to make the IR-writer
path pleasant. (Optional: an authoring macro.) None of this is new core
capability — it's ergonomics on an already-working path.

## Component 4 — source-language → IR converters (the new frontier)

The symmetric frontends to the emitters. Parse existing kernel source and
**raise** what it can onto the IR; leave the rest as source-language residue.

- **Frontend:** a real language parser — `libclang` for CUDA C++, Slang's own
  frontend for Slang. (Raising off an AST, not text hacks.)
- **Raise recognizable idioms:** a grid-stride loop with a pure per-element body
  → `Access::Elementwise` + a `ScalarExpr`; a block/warp reduction →
  `Access::Reduction`; a prefix loop → `Access::Scan`; and so on. This is idiom
  recognition (decompiler-style raising) — tractable exactly for the shapes the
  IR already models.
- **Convert-what-you-can, leave-the-rest-as-residue** (the core design):
  - **NVIDIA-library calls** (cuBLAS/cuDNN/…) — recognized and left as
    library routes, never converted (reimplementing them is a non-goal). The
    concrete example: the converter pulls those call sites out into extracted
    `.cuh` residue from the original source rather than trying to lift them.
  - **Non-expressible device code** (custom PTX, `mma`/`cp.async`, exotic
    control flow / pointer arithmetic) — extracted as source-language `.cuh`
    residue the emitted kernel calls.
- **Output is a hybrid:** an IR core (re-emittable to every backend) plus
  source-language residue (anchored to its origin language).
- **Honesty — portability scales with lift fraction.** A kernel that lifts
  100% to IR re-emits to *every* backend. A kernel with CUDA residue re-emits
  *fully* only to CUDA; to other backends only its IR fraction ports and the
  residue is a hole. So the converter's payoff is highest on the IR-expressible
  subset — the same boundary as everything else here. This is stated, never
  hidden: a converted kernel reports its lift fraction and what stayed residue.
- **Validation — trust but verify against the original.** Convert CUDA→IR, emit
  CUDA back, and diff the re-emitted kernel against the *original* via the oracle
  / on-device comparison. A mis-lifted idiom surfaces as numeric divergence.
  The oracle is the converter's safety net exactly as it is the emitter's.
- **Use cases:** (a) authors write in the language they know and still generate
  our IR (and thus every backend); (b) "copy-pasters" take an existing/foreign
  kernel they didn't write, run it through a converter → IR → emitters, and get
  an equivalent kernel in every language we have an emitter for — to the extent
  it lifts.
- **Difficulty, honestly:** raising is harder than lowering; it needs a language
  frontend; general kernels mostly become residue. But value is incremental —
  the easy idioms (elementwise, reductions, norms) pay off on day one, and the
  hit rate grows as the IR and the recognizer grow. Treat it as research-grade
  and phased, not a single deliverable.

## Roadmap (sequenced; each phase is independently valuable)

- **Phase 1 — prototype that proves the thesis.** Add the freestanding-helper
  emission mode and generate a single helper: **`coord_unravel`** (dead today,
  purely algorithmic, currently duplicated three ways). Oracle-check the
  generated `.cuh` and bench it against the hand-written
  `baracuda_coord_unravel.cuh`. One small, self-contained deliverable that
  demonstrates single-source-of-truth + "as good or better" (measured) +
  improvement-propagation end-to-end.
- **Phase 2 — migrate the algorithmic helpers** (Component 2): generate the
  live 7 + revive the useful dead ones as generated helpers; delete the truly
  dead. Oracle + bench gated throughout.
- **Phase 3 — IR-authoring ergonomics** (Component 3): a text/serde IR form +
  authoring docs/examples. Unblocks the IR-writer audience as a first-class UX.
- **Phase 4 — CUDA→IR converter spike** (Component 4): a `libclang`-based
  frontend targeting the elementwise/reduction idioms first, with
  residue-extraction (NVIDIA-lib calls + non-expressible code → `.cuh`) and the
  oracle round-trip as the validator. Prove the hybrid-output + verify loop on a
  handful of real kernels.
- **Phase 5 — broaden.** Grow converter idiom coverage; add the Slang frontend;
  add more emitter backends (Metal / SPIR-V) so the hub has real fan-out on both
  sides.

## Invariant principles (hold across every phase)

- **The "CUDA-shaped IR node" rule.** If expressing something forces an IR node
  that only makes sense on one backend, it is residue / hand-written, not IR.
  This is the single boundary that keeps the hub neutral — it governs the
  emitter, the helpers, *and* the converters identically.
- **Never reimplement the vendor libraries.** cuBLAS/cuDNN/… are routes, both
  when emitting and when converting. The converter recognizes and preserves
  them; it never tries to lift them.
- **The oracle validates everything.** Every generated, migrated, or converted
  artifact is bit-checked against the independent CPU reference; every
  replacement is bench-gated. Correctness is a property of the pipeline, not of
  trust in any one translator.
- **Portability is proportional to lift fraction, and it's reported.** Residue
  is honest and visible, never silently dropped.

## What this is not

Not a plan to generate *every* kernel (peak-performance Tensor-Core GEMM / Flash
attention and the vendor-library ops stay hand-written / routed — see the op
boundary discussion). Not a promise that generic generated helpers match fused
generated kernels (they serve a different audience). Not a claim that arbitrary
CUDA lifts cleanly (most of a hard kernel is residue). It *is* a plan to make the
IR the one place IR-expressible optimization logic lives, translatable in both
directions, validated by the oracle, for both the describe-the-math and the
write-my-own-CUDA developer.
