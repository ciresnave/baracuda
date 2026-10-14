# Glossary

Terms this repository uses with a project-specific meaning, each defined once.
Where a term has a primary document, the entry links to it rather than
restating it. See [`ARCHITECTURE.md`](ARCHITECTURE.md) for how the pieces fit.

## The op facade

- **baracuda-kernels** — the single crate downstream callers import. It exposes
  every ML primitive through `Plan` types and decides internally whether a
  vendor library or a bespoke kernel serves the op.
- **Plan** — a typed, per-op-family object built by `Plan::select` from a
  descriptor. It owns the resolved SKU and any cached library handles, and is
  executed repeatedly.
- **Descriptor** — the plain-data description of one op instance (shapes,
  dtypes, layout flags) that `Plan::select` reads to pick a SKU.
- **SKU (`KernelSku`)** — the identity tuple of the kernel a plan selected:
  category, layout, arch and backend among others. `Plan::sku()` exposes it for
  telemetry; autotuner caches key on it. See `ARCHITECTURE.md`.
- **Wrapper crate** — a `baracuda-<lib>` crate giving idiomatic Rust access to
  one NVIDIA library (cuBLAS, cuDNN, cuFFT, …), usable stand-alone.
- **Bespoke kernel** — a hand-written `.cu` kernel shipped in
  `baracuda-kernels-sys` for ops no NVIDIA library covers, or covers poorly.
- **`sm_NN`** — a CUDA compute capability (for example `sm_89`, the RTX 4070
  this project validates on). `ArchSku` records the one a kernel was built for.

## Kernel generation

- **Neutral IR** — the backend-agnostic op description (a scalar-expression DAG
  plus an access shape such as elementwise, reduction or contraction) that
  emitters lower. See [`docs/design/ir-translation-hub.md`](docs/design/ir-translation-hub.md).
- **Emitter / backend** — a component that lowers a plan from the IR to source
  for one target (CUDA today; a portable-C emitter proves the IR is neutral).
- **StructureKey (sk)** — the structural admissibility key naming a schedule
  cell. It is versioned (`STRUCTURE_KEY_VERSION`, published by Unpopped); the
  version is stamped into emitted kernels and goldens, so a bump regenerates them.
- **Golden** — a checked-in emitted kernel (under `baracuda-cuda-emit`'s
  `src/goldens/`) that tests compare fresh emitter output against byte for byte.
- **Oracle** — an independent CPU interpreter that computes what a generated
  kernel should compute from the same IR but sharing no lowering code, so a
  spelling bug cannot hide in both. See [`docs/design/oracle.md`](docs/design/oracle.md).
- **`BitIdentical` / `MorePrecise`** — the two lowering policies for
  reductions and normalizers: the default, and a double-precision serial fold
  that is more accurate and bit-reproducible across IEEE-754 hardware.
- **Decline** — an emitter or lifter refusing an op it cannot lower faithfully,
  rather than emitting an approximation. A decline is a stated result, not an
  error.

## Sibling projects

- **Fuel** — the sibling ML library that remains the runtime kernel selector
  and consumes baracuda's output.
- **FKC contract / seam** — the negotiated kernel contract between baracuda and
  Fuel, and the boundary crates (`baracuda-seam`, `baracuda-cuda-seam`) that
  carry it. Only the seam is a stable contract; the IR and emitters are not.
- **Synthesizer** — the seam-side component that JIT-compiles a kernel for a
  contract on demand.
- **Unpopped** — the external project that owns the shared vocabulary
  (`unpopped-vocab`), the structure key and the CPU oracle that baracuda
  depends on.
- **KISS** — the specification repository whose namespace documents define the
  op vocabulary; `kiss-ref` is its logical reference implementation, against
  which `oracle.rs` is the physical counterpart.

## Release and CI

- **Lockstep family** — the crates in this workspace that share one version
  number and are bumped and published together (`0.0.1-alpha.N`).
- **alpha.N** — the workspace version. Every pushed change moves it; pre-1.0,
  the second number is the breaking-change axis.
- **Driver-free** — built and checked without a CUDA driver installed, which is
  what the `msrv-driver-free` CI job and the no-CUDA build prove.
- **Liveness workflow** — a scheduled workflow (`cuda-vocab-pin-liveness`,
  `pytorch-baseline-liveness`) that opens a single deduplicated issue when a
  pinned upstream reference has drifted.
