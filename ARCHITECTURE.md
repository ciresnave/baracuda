# baracuda architecture

This document describes the layered design of the `baracuda-kernels` ML
op facade and the supporting crates around it. It complements the
top-level [`README.md`](README.md): the README answers "what is this and
how do I use it"; this file answers "how is it put together and why".

The intended audience is contributors and downstream framework authors
who want to add a new op family, write a sibling plan for a new compute
capability, or wire a new NVIDIA library wrapper into the dispatcher.

## Layered design

```text
┌────────────────────────────────────────────────────────────┐
│   User code (Rust)                                         │
│                                                            │
│   use baracuda_kernels::SoftmaxPlan;                       │
│   let plan = SoftmaxPlan::select(&stream, &desc, prefs)?;  │
│   plan.run(&stream, workspace, args)?;                     │
└──────────────────────────┬─────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────┐
│   baracuda-kernels  (safe Rust plan API)                   │
│                                                            │
│   - Plan + Descriptor + Args triples, one per op family.   │
│   - Plan::select validates the descriptor and picks a SKU. │
│   - Plan::run launches with caller-supplied workspace.     │
│   - Plan::sku() surfaces the selected KernelSku for        │
│     telemetry / autotuner cache keys.                      │
└──────┬─────────────────┬──────────────┬────────────────────┘
       │                 │              │
       ▼                 ▼              ▼
┌─────────────┐   ┌─────────────┐   ┌────────────────────────┐
│ Library     │   │ CUTLASS     │   │ Bespoke .cu kernels    │
│ wrappers    │   │ via         │   │ via                    │
│             │   │ baracuda-   │   │ baracuda-kernels-sys   │
│ baracuda-   │   │   cutlass   │   │                        │
│   cublas,   │   │             │   │ (hand-rolled           │
│   cudnn,    │   │ Float GEMM, │   │  mma.sync, cp.async,   │
│   cufft,    │   │ int8 RCR    │   │  ldmatrix, warp-level  │
│   cusolver, │   │ GEMM,       │   │  primitives)           │
│   curand,   │   │ batched,    │   │                        │
│   cusparse  │   │ grouped     │   │                        │
└─────────────┘   └─────────────┘   └────────────────────────┘
       │                 │              │
       ▼                 ▼              ▼
┌────────────────────────────────────────────────────────────┐
│   baracuda-driver  +  CUDA driver / library shared objects │
│   (RAII over Context / Stream / Event / DeviceBuffer; raw  │
│    FFI in baracuda-cuda-sys and the per-library *-sys)     │
└────────────────────────────────────────────────────────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │  NVIDIA hardware │
                  └──────────────────┘
```

The shared **type vocabulary** (`Element`, `TensorRef`, `KernelSku`,
…) lives in `baracuda-kernels-types` and is depended on by every layer
above the raw `*-sys` FFI crates. Both the safe facade and the per-
library wrappers agree on one set of dtype tags, layout enums, and view
structs — no per-crate re-declarations.

## The Plan–Descriptor–Args triple

Every op family in `baracuda-kernels` exposes the same three structs:

- **`<Op>Descriptor`** — pure shape + dtype + tuning knobs. Caller-
  immutable after construction. No GPU handles, no lifetimes, `Copy +
  Debug`. The descriptor is what `select` reads to pick a SKU.

- **`<Op>Args`** — the actual per-call tensor handles + scalars. Carries
  device-resident `TensorRef` / `TensorMut` views, scalars (alpha /
  beta / dropout-p / temperature / …), and any auxiliary buffers
  (workspace pointers, saved-FW outputs for BW). Lifetimed.

- **`<Op>Plan`** — the selected kernel implementation. Owns the
  descriptor and the resolved `KernelSku` (plus any handles the chosen
  backend caches, e.g. a `cublasLtMatmulPreference_t`). Exposes:
  - `select(stream, &desc, prefs) -> Result<Self>` — validates the
    descriptor, runs whatever heuristic / library-side
    `can_implement` check applies, returns a usable plan or a typed
    error.
  - `query_workspace_size(&self) -> usize` (or `workspace_size`) —
    bytes of device scratch this plan needs at `run` time.
  - `run(&self, stream, workspace, args) -> Result<()>` — launches.
  - `sku(&self) -> KernelSku` — observability hook.

The three-struct split has a few load-bearing motivations:

1. **`select` is the choke point.** All shape / dtype / arch
   validation runs once at planning time; `run` is a thin launcher.
   This keeps the hot path narrow and pushes the cost of errors to
   construction time, where surfacing them is cheap.
2. **Plans are reusable.** Build the plan once for a fixed
   descriptor (a transformer layer's `[batch, seq, hidden]` softmax)
   and call `run` many times with different `Args`. The plan's
   resolved SKU + cached library handles amortize across calls.
3. **The descriptor is hashable.** A pure-data descriptor is a clean
   key for caller-side autotuner caches — `HashMap<MyDescriptor,
   MyPlan>` works without any handle-lifetime acrobatics.
4. **The args are minimal.** `run` takes the workspace + args alone;
   no need to thread shapes / dtypes through `run` again because the
   plan already captured them.

The pattern was lifted from `baracuda-cutlass::GemmPlan` and applied
uniformly across every op family (Phase 1 onward).

## `KernelSku` — the op identity tuple

`KernelSku` (defined in
[`baracuda-kernels-types::sku`](crates/baracuda-kernels-types/src/sku.rs))
is the structural identity of a selected kernel. It is what
`Plan::sku()` returns, what autotuner caches key on, and what
telemetry consumers serialize.

```rust
pub struct KernelSku {
    pub category: OpCategory,
    pub op: u16,
    pub element: ElementKind,
    pub aux_element: Option<ElementKind>,
    pub layout: Option<LayoutSku>,
    pub epilogue: Option<EpilogueKind>,
    pub arch: ArchSku,
    pub backend: BackendKind,
    pub precision_guarantee: PrecisionGuarantee,
}
```

Field by field:

- **`category: OpCategory`** — the top-level taxonomy a SKU belongs to.
  Mirrors the section letters of the comprehensive plan: `Gemm`,
  `UnaryElementwise`, `BinaryElementwise`, `TernaryElementwise`,
  `GatedActivation`, `Reduction`, `Scan`, `Normalization`, `Softmax`,
  `Convolution`, `Pooling`, `Attention`, `Indexing`, `Embedding`,
  `ShapeLayout`, `Sorting`, `Quantization`, `Random`, `Loss`,
  `SegmentOps`, `Image`, `Fft`, `Linalg`, `Moe`.

- **`op: u16`** — a category-local op discriminant. For
  `BinaryElementwise` it's `BinaryKind as u16`; for `Loss` it's
  `LossKind as u16`; for `Softmax` it's `SoftmaxKind as u16`. Surfacing
  it as a flat `u16` keeps `KernelSku`'s shape stable across categories
  so it can be hashed uniformly into autotuner caches.

- **`element: ElementKind`** — the primary scalar type the kernel
  operates on. `F32 | F16 | Bf16 | F64 | S8 | U8 | S4 | U4 | Bin |
  F32Strict | Fp8E4M3 | Fp8E5M2 | I32 | I64 | Bool | …`.

- **`aux_element: Option<ElementKind>`** — the auxiliary element type
  when meaningful. The bias element for a GEMM bias epilogue, the index
  element for gather / scatter, the accumulator type for ops where it
  diverges from the primary element. `None` when the op has no
  auxiliary element.

- **`layout: Option<LayoutSku>`** — `Some(Rcr)` or `Some(Rrr)` for
  matrix-multiply-shaped kernels; `None` for ops that don't have a
  row/col layout dimension (elementwise, reduce, scan, softmax, …).

- **`epilogue: Option<EpilogueKind>`** — `Some(Identity | Bias |
  BiasRelu | BiasGelu | BiasSilu)` for GEMM SKUs with a fused
  post-matmul chain; `None` otherwise.

- **`arch: ArchSku`** — the compute capability the selected kernel was
  compiled for: `Sm80 | Sm89 | Sm90a`. Future Blackwell adds `Sm100`.

- **`backend: BackendKind`** — which compute path served this SKU:
  `Bespoke | Cutlass | Cublas | Cudnn | Cufft | Cusparse | Cusolver |
  Curand | Cutensor | Npp | Cvcuda`. Surfaced so callers can branch
  diagnostics or autotuner choices on the backend.

- **`precision_guarantee: PrecisionGuarantee`** — numerical contract
  (see next section).

The struct is `Copy + Eq + Hash`, designed to be cheap to compare and
suitable as a `HashMap` key.

## `PrecisionGuarantee`

Every plan publishes a `PrecisionGuarantee` so callers can decide
whether a SKU satisfies their numerical contract without re-deriving
it from documentation:

```rust
pub struct PrecisionGuarantee {
    pub math_precision: MathPrecision,
    pub accumulator: ElementKind,
    pub bit_stable_on_same_hardware: bool,
    pub deterministic: bool,
}
```

- **`math_precision`** — the bit-precision used inside the math
  instruction. For tensor-core paths this is the multiplicand precision
  (`F16` / `BF16` / `TF32` / `F32` / `F64` / `Int8` / `Int4` / `B1` /
  `Fp8E4M3` / `Fp8E5M2`). For SIMT kernels it's the input scalar
  precision.

- **`accumulator`** — the element type of the multiply-accumulate
  accumulator. For F16 tensor-core GEMM this is `F32` (the standard
  upcast); for int8 it's `I32`. Callers verifying "I need an F32
  accumulator" against an unknown SKU check this field.

- **`bit_stable_on_same_hardware`** — `true` iff the kernel produces
  bit-identical results across runs on the same hardware with the same
  inputs. **False** for tensor-core kernels (F16, BF16, TF32) because
  the warp-level reduction order isn't fixed by the spec — adjacent
  runs can differ in the last bit. **True** for SIMT F32 and for
  integer-MMA kernels.

- **`deterministic`** — `true` iff the kernel produces bit-identical
  results across runs from a single thread within a process. **False**
  whenever the kernel uses `atomicAdd` for cross-block accumulation
  (e.g. the embedding backward, segment-sum backward, and some
  normalization backwards) — atomic-add ordering depends on block
  scheduling. **True** when the kernel routes through warp shuffles
  and a single-block reduction. Where determinism mattered we
  consciously chose the warp-shuffle path even at a perf cost — see
  the Phase 5 affine-BW for the worked example.

## The dispatcher

Today `Plan::select` is mostly per-plan hand-coded validation: each op
family knows what it supports and which backend serves the SKU. For
example, `IntGemmPlan` routes `LayoutSku::Rcr` to `baracuda-cutlass`
and `LayoutSku::Rrr` to the bespoke `baracuda-kernels-sys` kernels;
`SoftmaxPlan` routes unconditionally to bespoke; `Conv2dPlan` routes
unconditionally to cuDNN.

The dispatcher is intentionally *not* a single magical autotuner. Each
plan owns its own dispatch logic because the choices are different:

- **Some plans select between two SKUs at planning time** based on
  descriptor fields alone (the int8-GEMM layout split).
- **Some plans only have one SKU available today** but expose the
  selection point so a future sibling can plug in without breaking
  callers (every Phase 3+ unary / binary plan).
- **Some plans need handle setup that depends on the descriptor**
  (cuDNN's conv-algorithm picker, cuBLASLt's preference / heuristics).
  These cache the configured handle inside the plan.

Future work — tracked under Phases 10-12 — adds a **Fuel-style judge**
that picks at runtime between sibling plans (e.g.
`FlashSdpaPlan` vs `FlashSdpaSm89Plan`) by racing them at construction
time and caching the winner per descriptor. The first sibling plan
landed in Phase 10 (`FlashSdpaSm89Plan` for Ada); the judge that picks
between siblings lives downstream in the Fuel autotuner today.

## The workspace contract

Plans never own device memory in baracuda. Scratch is **caller-
supplied** at `run` time through `Workspace::Borrowed(DeviceSliceMut<u8>)`,
or `Workspace::None` for plans that report zero bytes needed.

```rust
let n = plan.query_workspace_size();
let scratch: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, n)?;
plan.run(&stream, Workspace::Borrowed(scratch.as_slice_mut()), args)?;
```

Why this design:

- **No hidden malloc inside `run`.** Every CUDA allocation is visible
  in the caller's code. This matters for capture-mode CUDA graphs
  (which can't allocate during capture) and for downstream frameworks
  that pool their own memory.
- **Predictable memory footprint.** The caller can pre-allocate a
  per-plan or per-stream scratch buffer and size it from
  `query_workspace_size()`. No GC churn, no surprise OOMs deep in a
  training loop.
- **Composable with arena allocators.** Frameworks that allocate from
  a fixed-size arena per step can carve out scratch from the arena
  without coordinating with baracuda.

The size returned from `query_workspace_size` is conservative — it's
the maximum the plan might consume for the descriptor it was
constructed with, not for the specific args passed to `run`. Passing a
larger scratch than required is always safe.

## Tensor types

`TensorRef<T, N>` and `TensorMut<T, N>` (in
[`baracuda-kernels-types::tensor`](crates/baracuda-kernels-types/src/tensor.rs))
are the standard borrowed views of device-resident tensors used by
every op family except GEMM (which uses the 2-D-specialized
`MatrixRef` / `MatrixMut` instead).

```rust
pub struct TensorRef<'a, T: DeviceRepr + Copy + 'static, const N: usize> {
    pub data: DeviceSlice<'a, T>,
    pub shape: [i32; N],
    pub stride: [i64; N],
}
```

- **Const-generic rank.** Same pattern `dfdx` uses. Rank mismatches
  are caught at compile time; the view struct is heap-free; pattern
  matching on rank is straightforward inside generic code.
- **Element stride along each axis.** `stride[i]` is the memory offset
  (in elements, not bytes) to advance one step along axis `i`. The
  default row-major contiguous stride for shape `[d0, …, dN-1]` is
  `[d1·d2·…·dN-1, …, dN-1, 1]`.
- **Broadcast convention.** A `stride[i] == 0` marks axis `i` as a
  broadcast operand — the kernel reads the same memory cell for every
  step along that axis. The plan layer decides which axes a kernel
  supports broadcasting on; some kernels accept only contiguous views,
  others fold broadcast directly into the index calculation.
- **Relaxed `T` bound.** Bounded by `DeviceRepr` (not `Element`) so
  the same view struct can carry any scalar payload — input, output,
  mask, index, auxiliary buffer. The per-op element-class check
  (`T: Element`, `T: IntElement`, `T: FpElement`) is enforced at the
  Plan layer.

For ops whose input and output rank differ (e.g. `meshgrid` from
rank-1 to rank-N, reductions from rank-N to rank-`N-D`), the Plan
exposes two const-generic rank parameters — one per operand.

## Element trait + `ElementKind`

Per-type identity is two-pronged:

- **Compile-time:** `T: Element` provides the trait-level vocabulary
  (associated constants, `KIND: ElementKind`, scalar projection
  helpers, bit-pattern conversions for FFI). Sub-traits — `IntElement`,
  `FpElement`, `BiasElement`, `BinElement` — narrow the set of valid
  scalar types per kernel family.
- **Runtime:** `ElementKind` is a flat enum (one variant per supported
  scalar type) used inside the descriptor, the SKU, and any cross-
  category code that needs to switch on dtype without monomorphizing.

The two are kept in sync by the requirement that every `Element` impl
sets `const KIND: ElementKind`. The descriptor's `element` field is
how a generic `Plan<T>::select` cross-checks "did the caller's `T` and
the descriptor's dtype agree?" — a mismatch returns
`Error::Unsupported` at planning time, not a kernel-launch failure
later.

## Error handling

The kernel facade unifies on `baracuda_cutlass::Error` (re-exported as
`baracuda_kernels::Error`) — historical reasons, will likely move into
`baracuda-kernels-types` post-1.0. The variants:

```rust
pub enum Error {
    Unsupported(&'static str),
    InvalidProblem(&'static str),
    MisalignedOperand,
    WorkspaceTooSmall { needed: usize, got: usize },
    BufferTooSmall { needed: usize, got: usize },
    CutlassInternal(i32),
    Driver(baracuda_driver::Error),
}
```

Status-code mapping from `.cu` kernels' `int32_t` returns:

| Code | Meaning | Mapped error |
| --- | --- | --- |
| 0 | success | `Ok(())` |
| 1 | misaligned operand | `Error::MisalignedOperand` |
| 2 | invalid problem (M, N, or K non-positive, or shape inconsistency) | `Error::InvalidProblem(...)` |
| 3 | not supported (this kernel doesn't implement the requested shape) | `Error::Unsupported(...)` |
| 4 | workspace too small or null when required | pre-checked at the safe layer; surfaces as `Error::WorkspaceTooSmall { needed, got }` with the actual byte counts. A status-4 return from a kernel after the safe layer pre-checked workspace indicates a plan-vs-runtime inconsistency and is surfaced as `Error::CutlassInternal(4)` so it's visible. |
| 5 | internal launch failure | `Error::CutlassInternal(5)` |
| n > 5 | reserved | `Error::CutlassInternal(n)` |

The `CutlassInternal` variant name is historical — the same status-
code ABI is used by every bespoke kernel under
`baracuda-kernels-sys/kernels/`, not only the CUTLASS-instantiated
ones. The variant will likely be renamed `KernelInternal` before 1.0.

## Sibling-plan pattern for arch-specific tuning

When an op has the same API on two compute capabilities but different
kernel implementations (different tile sizes, different async-copy
shapes), baracuda ships them as **sibling plans** with identical
descriptor and args shapes but distinct plan types.

The first instance lives in Phase 10 (alpha.25):

```rust
// sm_80 baseline — runs on Ampere; works as forward-compat on Ada / Hopper.
use baracuda_kernels::{FlashSdpaPlan, FlashSdpaDescriptor, FlashSdpaArgs};

// sm_89 Ada-specialized sibling — cp.async-double-buffered K/V loads,
// wider thread block for Ada's larger per-SM register file. f16 + bf16 only.
#[cfg(feature = "sm89")]
use baracuda_kernels::{FlashSdpaSm89Plan, FlashSdpaSm89Descriptor, FlashSdpaSm89Args};
```

The two plans:

- Have identical descriptor / args layout — a caller can substitute
  one for the other with a type alias.
- Live in the same module (`baracuda_kernels::attention`).
- Are observably distinct via `Plan::sku().arch`.

A future Fuel-style autotuner / judge will pick between siblings at
runtime by racing them once per descriptor and caching the winner.

## Vendoring convention

baracuda generally **wraps** rather than forks upstream code. Two
narrow exceptions where vendoring is the right answer:

1. **Upstream is drifting from our direction.** When an NVIDIA library
   or open-source kernel collection has architectural decisions that
   conflict with baracuda's contracts (caller-owned workspace, status-
   code error ABI, capture-safe launches), we vendor the kernel body
   and rewrap it.
2. **No upstream covers our needs.** The int8 RRR GEMM is the
   archetypal case: CUTLASS 4.2.0 doesn't ship the warp-iterator
   specialization the RRR layout needs, and two attempts at vendoring
   the missing piece into the CUTLASS template chain exceeded the
   upstream code they'd have reused. Hand-rolling at the PTX level was
   shorter and is more maintainable.

When we vendor, the convention is:

- The adapted `.cu` source carries an `SPDX-FileCopyrightText:` +
  `SPDX-License-Identifier:` header.
- The consolidated provenance lives in
  [`crates/baracuda-kernels-sys/LICENSE-thirdparty.md`](crates/baracuda-kernels-sys/LICENSE-thirdparty.md).
- Each adaptation summary documents what was changed and why (FFI
  shape, status codes, dtype handling on the Windows-x64 ABI).
- For larger fork-style vendoring (the `baracuda-forge` build crate),
  the upstream commit hash and any patches live in a per-crate
  `NOTICE` file.

Currently vendored kernels (Phase 8 → 70 vendoring track):

- **HuggingFace candle** CUDA elementwise set (cast / fill / affine).
- **llama.cpp `ggml-cuda`** GGUF block-format dequantization and
  MMVQ kernels (Q4_0 through Q8_K plus the k-quants; Phase 8 → 34).
- **`guoqingbao/attention.rs`** fused MoE expert kernels (FP WMMA,
  scalar GGUF, combined WMMA+GGUF; Phase 8.5).
- **Dao-AILab FlashAttention v2** (BSD-3) — Tier-1 vendor at
  `vendor/flash-attention/` (Phase 42; Phase 59 BW, varlen,
  sliding, softcap, ALiBi; Phase 60 head_dim expansion).
- **DeepSeek-AI mHC.cu** (MIT) — HyperConnection plan
  (`vendor/mhc/`; Phase 43).
- **ozIMMU clean-fork** (MIT) — Ozaki-scheme DGEMM
  (`crates/baracuda-ozimmu-sys/cuda/`; Phase 44 + 44b Windows port).
- **FlashInfer** (Apache-2.0) — paged-KV decode/prefill, cascade,
  sampling (Phase 46 + Phase 66 Tier-2 closure).
- **IST-DASLab Marlin** + **mit-han-lab llm-awq** (Phase 48) —
  W4A16 GEMM for symmetric (Marlin) + asymmetric (AWQ) int4 weight
  formats.
- **NVIDIA Apex** (BSD-3) — multi-tensor optimizer kernels
  (`baracuda-optim`; Phase 49).
- **Dao-AILab Mamba-2 + causal-conv1d** (Phase 50) — SSD chunk
  scan and causal 1d convolution.
- **bitsandbytes NF4** (MIT) — 4-bit non-uniform quantile QLoRA
  inference (Phase 53).
- **xFormers algorithmic reference** (BSD-3, clean-room hand-port)
  for block-sparse SDPA + 2:4 structured sparsity GEMM (Phase 54).
- **NVIDIA TransformerEngine** (Apache-2.0) — FP8 cast / dequant +
  delayed-scaling recipe (`baracuda-transformer-engine`; Phase 55).
- **Ring Attention** (Apache-2.0 reference) — sequence-parallel
  attention atop NCCL (Phase 56).

## Phase roadmap

The numbered phases through 10 followed an internal comprehensive
plan. From Phase 11 onward the work has been **downstream-driven** —
Fuel team integration feedback and follow-on asks set the agenda, so
the phase numbering no longer maps to the original Hopper / 1.0-freeze
roadmap. Each completed phase has a one-line entry in
[`CHANGELOG.md`](CHANGELOG.md); per-phase scopes and remaining work
are tracked in [`ROADMAP.md`](ROADMAP.md).

### Phases 0-10 — planned roadmap (complete)

- **Phase 0** — Crate scaffolding: `baracuda-kernels-types`,
  `baracuda-kernels-sys`, `baracuda-kernels`. Migrate the shared type
  vocabulary out of `baracuda-cutlass`.
- **Phase 1** — int8 GEMM RRR (Fuel-blocking). 18 SKUs:
  `{S8, U8} × Rrr × {Identity, Bias, BiasRelu, BiasGelu, BiasSilu} ×
  {f32 bias, i32 bias}`.
- **Phase 2** — FP8 (E4M3, E5M2), int4 (S4, U4), bin (B1) GEMM —
  filling out the sm_89 tensor-core dtype matrix.
- **Phase 3** — Elementwise + shape / layout (Categories B, B', C, C',
  D, N). Largest single phase by op count; ~120 ops × FW+BW.
- **Phase 4** — Reductions + scans + random (Categories E, F, Q).
  Warp/block primitives that pay dividends across the rest of the op
  set.
- **Phase 5** — Normalization + softmax + loss (Categories G, H, R).
  RMSNorm, LayerNorm, BatchNorm, GroupNorm, InstanceNorm, Softmax,
  LogSoftmax, GumbelSoftmax, Sparsemax, the full Tier 1 + Tier 2
  losses, CTCLoss.
- **Phase 6** — Attention + linalg + FFT (Categories K, Linalg, U).
  RoPE, ALiBi, KV-cache, SDPA + Flash SDPA, dense factorizations
  (Cholesky, LU, QR, SVD, batched variants, ormqr), FFT family.
- **Phase 7** — Convolution + pooling + indexing + embedding + segment
  (Categories I, J, L, M, S). Conv2d / Pool2d via cuDNN; gather /
  scatter / index_select / masked_fill / one_hot / nonzero bespoke;
  embedding + embedding_bag; segment sum/mean/max/min/prod.
- **Phase 8** — Quantization helpers + GGUF + MoE (Categories P, V).
  Per-tensor / per-channel / per-token / per-group quantize +
  dequantize + fake_quantize; GGUF block-format dequant + MMVQ for
  Q4_0..Q8_K + k-quants; fused MoE inference forward.
- **Phase 9** — Sort / topk / image (Categories O, T). Block-bitonic
  sort + topk + kthvalue + msort + argsort + searchsorted +
  bincount + histogram + unique; interpolate + grid_sample +
  affine_grid + pixel_shuffle + roi_align + roi_pool + NMS.
- **Phase 10** — sm_89 (Ada Lovelace) tuning sweep. Sibling plans for
  Flash Attention; bench harness; populated baseline table.

### Phases 11-14 — Fuel-driven (complete)

Phase numbering 11 onward was redirected by Fuel team's integration
feedback. The original "Phase 11 = sm_90a Hopper" and "Phase 12 =
1.0 freeze" items are now tracked as backlog rather than the next
work — they remain valid targets but are sequenced behind ongoing
downstream-driven work.

- **Phase 11 (alpha.27)** — Eight items from Fuel's alpha.26
  integration feedback: `ScalarType::ZERO/ONE/from_f32` ergonomics;
  Git-for-Windows fake-`link.exe` probe in `build.rs`; bf16 / f16
  `atomicAdd_via_cas` for indexing & segment BW; GGUF Q8_K MMVQ
  (bespoke — no llama.cpp upstream); i64 indices across the indexing
  family via new `IndexElement` sealed trait; Sparsemax row cap lifted
  64 → 1024 via `cub::BlockRadixSort` + `BlockScan`; Conv 1D / 3D /
  Transpose / depthwise fanout via cuDNN (Conv2dDescriptor gained
  `groups` field — breaking literal-init change); Pool 1D / 3D +
  Adaptive fanout (FractionalMaxPool / LpPool stubbed).
- **Phase 12 (alpha.28)** — PowI parameterized unary plan (FW + BW,
  4 fp dtypes) and ArgMaxDim / ArgMinDim u32 / i32 output dtypes via
  new `IndexOutputElement` sealed trait (default `i64` preserves
  source-compat).
- **Phase 13 (alpha.29)** — WriteSlice (KV-cache append fast path) +
  Contiguize (signed strides for Flip; retires Fuel's D2H → CPU →
  H2D fallback) + sub-byte cast paths (Fp8 / S4 / U4 ↔ fp / int —
  34 FFI symbols) + Triu / Tril plans (FW + BW; mask is self-adjoint
  so BW reuses FW kernel). Introduced the `T: DeviceRepr + Copy`
  trait-bound pragma for plans that need to cover sub-byte dtypes
  alongside `Element` types.
- **Phase 13.5 (alpha.30)** — `DeviceBuffer::zero()` + `zero_async()`
  for re-zeroing an existing allocation without realloc. README
  freshness fixes (absolute image URL for crates.io; status +
  regression badges bumped).
- **Phase 14 (alpha.31)** — Strided FFI siblings for Affine, PowI,
  Triu / Tril, RoPE + SDPA, and GGUF MMVQ (activation-strided +
  `w_start_byte_offset`). 56 new FFI symbols, all sibling — contig
  FFIs untouched so existing callers compile without change.
  Plan-layer dispatch routes canonical-contig inputs to the existing
  fast path; non-canonical inputs route to the new strided FFI.

### Phases 15-29 — Fuel-driven completion of the planned matrix

This band closed every remaining deferral from Phases 3-10 and added
the 1.0-freeze prerequisites surfaced during Phase 19.

- **Phase 15 (alpha.32)** — MMVQ alignment guard; OneHot / Nonzero
  i64 wrappers; MoE fixture race fix.
- **Phase 16 (alpha.33)** — Pool completion: bit-exact
  AdaptiveAvgPool/MaxPool {1,2,3}d (replaces cuDNN approximation),
  LpPool {1,2}d, FractionalMaxPool {2,3}d. 48 new FFI symbols.
- **Phase 17 (alpha.34)** — Flash SDPA sm_89 strided FW sibling +
  SDPA BW + GQA broadcast (template-bool `if constexpr` dispatch to
  `atomic::add<T>` for dK/dV). Caller-zero contract for dK/dV under
  broadcast.
- **Phase 18 (alpha.35)** — f16 / bf16 activations for GgufMmvqPlan
  across all 11 block formats × contig + strided = 44 new FFI symbols.
- **Phase 19 (alpha.36)** — Fuel retirement asks: non-adaptive pool
  FFI surface (48 symbols), Conv / ConvTranspose FFI (72), Upsample
  Nearest2d + Bilinear2d (12), im2col / col2im (12). 140 new FFI
  symbols. **Surfaced design correction**: all library-backed Rust
  plans need `baracuda-kernels-sys` FFI wrappers — 1.0-freeze prereq.
- **Phase 20 (alpha.37)** — MoE batched MMVQ × N-experts kernel
  family (36 new FFI symbols).
- **Phase 21 (alpha.38)** — Bilinear interpolate expansion
  (align_corners + scale_h/w_factor) + f16 / bf16 fanout.
- **Phase 22 (alpha.39)** — MMVQ ncols≥64 debug-build assertion +
  cuSOLVER FFI facade (10 plan families, ~50 C symbols).
- **Phase 23 (alpha.40)** — cuFFT + cuRAND FFI facade (32 C symbols).
- **Phase 24 (alpha.41)** — CUTLASS GEMM re-export FFI facade
  (210 thin trampolines). **Completes the Phase 19 1.0-freeze
  prereq** — every library-backed Rust plan now has a
  `baracuda-kernels-sys` FFI symbol.
- **Phase 25-26 (alpha.42)** — Segment + EmbeddingBag BW completion
  (9 new plans + 24 FFI symbols); BatchedOrmqrWy complex variants
  via bespoke WY-block kernels + cuBLAS C/Z gemmStridedBatched.
- **Phase 27** — Q8_1 MMVQ perf inspection (doc-only). Established
  the multi-M MMVQ port as the material opportunity for prefill
  (closed in Phase 33+34).
- **Phase 28 (alpha.43)** — API hygiene for 1.0 prep: NEW
  `KernelDtype` umbrella sealed trait; `#[non_exhaustive]` on 28
  enums.
- **Phase 29 (alpha.44)** — Cross-implementation bench suite vs
  cuBLAS / cuDNN. Established the f16/bf16 GEMM decode-regime gap
  closed by Phase 30.

### Phases 30-58 — vendor track + perf closure + training adjacency

The recent band added third-party vendoring (FA2, mHC, ozIMMU,
FlashInfer, Mamba, Marlin, AWQ, NF4, xFormers, TransformerEngine,
Ring Attention, Megatron-TP) alongside continued op-matrix backfill
and bench-driven perf wins. Highlights:

- **Phase 30 (alpha.45)** — `GemmPlan` cuBLAS-backed dispatch;
  3× speedup at M=32 f16 decode batch.
- **Phase 33-34 (alpha.48-49)** — Multi-M MMVQ for all 10 GGUF
  block formats. Q5_0 peaks at 17.32× at M=8 (target was 3-7×).
- **Phase 42 (alpha.56)** — Tri Dao FlashAttention v2 Tier-1
  vendor (head_dim=128, fp16+bf16, FW).
- **Phase 43-44 (alpha.56)** — mHC.cu + ozIMMU vendors.
- **Phase 46 (alpha.51)** — FlashInfer Tier-1 (paged decode,
  cascade, sampling).
- **Phase 48 (alpha.53)** — Marlin + AWQ W4A16 GEMM.
- **Phase 49-50 (alpha.54-55)** — Apex multi-tensor optimizers +
  Mamba-2 SSD.
- **Phase 53-55 (alpha.55-58)** — bitsandbytes NF4, xFormers
  sparse, TransformerEngine FP8.
- **Phase 56-57 (alpha.58)** — Ring Attention + Megatron-LM TP.
- **Phase 59 (alpha.59)** — FA2 BW + varlen.
- **Phase 60 (alpha.60)** — FA2 head_dim {160, 224, 512} expansion.
- **Phase 61-62 (alpha.61-62)** — Same-pointer in-place op
  family completion (contig + strided contracts).
- **Phase 63 (alpha.63)** — FA2 saved-tensor wiring for downstream
  autograd.

### Phases 64-74 — recent release history

Phases 64-71 shipped with alpha.64; Phases 72-73 shipped with
alpha.65; alpha.66 added the driver VRAM introspection (Fuel ask);
Phase 74 shipped with alpha.67 (see `CHANGELOG.md`).

- **Phase 64** — Extended in-place aliasing docs for Cast / Where /
  Triu / Tril / Activation BW / Fill (safe); Flip / Roll / Permute /
  RoPE explicitly marked NOT safe.
- **Phase 65a-d** — SMEM helper headers + SMEM-staged
  RMSNorm / LayerNorm / Softmax / LogSoftmax (in-place safe for
  f32/f16/bf16); BN / GN / IN proven in-place safe by construction
  (f64 covered too). 24 in-place proof tests.
- **Phase 66** — FlashInfer Tier-2 closure (paged prefill,
  spec-decode, FP8 KV, ragged prefill).
- **Phase 67a-f** — Reusable kernel-helper headers
  (`baracuda_smem_*`, fp_bits, cp_async).
- **Phase 68** — TensorRT vtable-dispatch C++ shim (`shim` feature).
- **Phase 69** — NVSHMEM host-side wrapper pair.
- **Phase 70** — nvImageCodec sys + safe wrapper (supersedes nvJPEG).
- **Phase 71** — RAPIDS cuVS GPU vector-search pair.
- **Phase 72** — Strided FFI siblings for the normalizer + shape-op
  families (88 new FFI symbols; alpha.65).
- **Phase 73** — Cross-impl bench follow-ups: `FlashDecodingPlan`
  (split-K seq_q=1 decode, 12-16×), warp-cooperative QKᵀ, concat +
  reduce perf closures (alpha.65).
- **Phase 74** — Fuel dense-FP-GEMM + reduce-to closure: NEW
  cuBLAS-backed `baracuda_kernels_gemm_dense_*` FFI family
  (f32/f64/f16/bf16 × RRR/RCR/CRR × strided-batch, 12 symbols) +
  `DenseGemmPlan<T>`; NEW `ReduceToPlan<T, N>` + `UnaryKind::Step`
  facades over the existing Phase 31/37 sys symbols; gelu flavor
  doc disambiguation. See
  `docs/fuel-reply-fp-gemm-reduce-to-2026-06-10.md`.

### Outstanding work

Outstanding items are tracked in [`ROADMAP.md`](ROADMAP.md). The
order in which we tackle them is driven by Fuel's near-term
integration needs in tension with the long-arc 1.0-freeze target.
The original "Phase 11 = Hopper sm_90a" and "Phase 12 = 1.0 freeze"
items remain valid targets but are sequenced behind ongoing
downstream-driven work.

The current published tag is **v0.0.1-alpha.67** (Phase 74 — Fuel
dense-FP-GEMM + reduce-to facade closure); consult `CHANGELOG.md`
for the release-by-release detail.
