# Changelog

All notable changes to baracuda are documented here. The project follows
[Semantic Versioning](https://semver.org/), though the `0.0.1-alpha.X`
series uses the alpha increment instead of patch/minor — every published
alpha represents one or more completed phases.

The phase numbering is Fuel-driven (Fuel is baracuda's primary downstream
consumer); see `ROADMAP.md` for the active phase board.

## 0.0.1-alpha.70 — 2026-06-28 (Fuel — device-load telemetry aliases)

Convenience aliases for Fuel's `DeviceLoadSelector` (Step E, Phase B2). Both
underlying reads already shipped in alpha.69 (`Stream::is_complete` /
`nvml::Device::utilization`); this adds thin, read-only aliases matching the
shape Fuel's load-balancer wants. No new dependency, no hot-path impact. See
[`docs/fuel-reply-device-load-telemetry-2026-06-28.md`](docs/fuel-reply-device-load-telemetry-2026-06-28.md).

### Added

- **`Stream::is_idle() -> Result<bool>`** on both `baracuda-driver` and
  `baracuda-runtime` — a load-probe alias of `is_complete()` (wraps
  `cuStreamQuery`/`cudaStreamQuery`; `Ok(true)` idle, `Ok(false)` busy).
  Read-only: never synchronizes the stream or perturbs scheduling. Reflects
  only this process's submissions to the stream.
- **`nvml::Device::gpu_utilization_percent() -> Option<u8>`** — GPU core
  busy-percent in the `Option<u8>` shape a load balancer wants, an honest
  `None` (never a fabricated value) when the driver can't answer. Convenience
  wrapper over `utilization()`. Captures *cross-process* device contention
  (the signal Fuel can't measure itself); pairs with the existing
  `compute_processes()`/`graphics_processes()` for an SM proxy.

## 0.0.1-alpha.69 — 2026-06-20 (Fuel — Profile v1 kernel-seam: handshake + structure_key + FKC generator)

Kernel-specialization / Baracuda↔Fuel **Profile v1** release. The cross-project
kernel-seam contract (FDX tensor ABI + FKC kernel contracts + telemetry + a
connect-time version handshake) was ratified by all parties (Fuel, Baracuda,
Vulkane); this ships Baracuda's conforming surface, reconciled against the FKC
fusion-patterns spec rev 4.

### Added

- **`baracuda-seam`** (new crate) — the Profile v1 kernel-seam handshake: the
  frozen 56-byte `#[repr(C)] SeamHello` negotiation envelope plus the out-param
  C-ABI entry point `baracuda_seam_hello()` that Fuel reads to negotiate a seam
  profile and capability set (Kernel-Seam Interop §3.1/§3.5). Capability bits
  reuse the FDX `BackendProbe` tokens.
- **`structure_key` + `structure_key_token`** (`baracuda-kernels-types`) — the
  canonical input/output layout-class join token, computed once by Baracuda from
  a minimal `OperandDesc` projection and called by Fuel for triple duty
  (telemetry tagging, FKC admissibility predicate, runtime dispatch), with a
  lossless string codec. `structure_key_token` is the one-call cross-boundary
  entry point.
- **`baracuda-kernelgen`** (dev tool, `publish = false`) — the AOT
  kernel-specialization generator: an abstract op IR lowered to CUDA, three
  schedules (vectorized `float4`/`double2` / scalar / strided+broadcast),
  f16/bf16/f32/f64 dtype breadth, and full **FKC contract emission** (provider
  front-matter + `accept`/`return`/`op_params`/`caps`/`cost`/`precision`/
  `determinism` + a declarative `pattern:` block with `extract:` for scalar
  params) plus the `link_registry` roster. Go/no-go validated at **2.03×** on
  sm_89; emitters audited conformant against FKC fusion-patterns rev 4 (incl. a
  GELU-flavor fix — exact-erf kernels now emit `GeluErf`, not the tanh-name
  `Gelu`).

### Fixed

- **`StructureKey.rank`** is the raw iteration rank (the widest operand rank),
  renamed from `eff_rank` so the field no longer implies the (deferred)
  contiguous-axis collapse.

## 0.0.1-alpha.68 — 2026-06-15 (Fuel — CUDA 13.3 / CCCL MSVC build fix)

Build-system release driven by the Fuel team's CUDA 13.3 upgrade. No kernel
or public API surface changes; the existing CUTLASS GEMM kernels were
verified end-to-end (102 GPU smoke tests each) against CUTLASS v4.2.0 and
4.5.2 on driver 610.47.

### Fixed — build

- **CUDA 13.3 CCCL traditional-preprocessor hard error.** CUDA 12.5+ / 13.x
  bundle a CCCL whose `<cuda/std/__cccl/preprocessor.h>` raises a hard
  `#error` (MSVC `C1189`) under cl.exe's legacy "traditional" preprocessor,
  breaking every CCCL-using `.cu` (CUTLASS, cub/thrust). `baracuda-forge` now
  passes `-Xcompiler /Zc:preprocessor` on MSVC hosts (in both `build_lib` and
  `build_ptx`), switching cl.exe to the standard-conforming preprocessor —
  NVIDIA's recommended fix. All forge consumers inherit it.
- **`cudart.lib` link-search.** `baracuda-cutlass-kernels-sys` now emits a
  `rustc-link-search` for the detected CUDA toolkit lib dir, so a binary that
  depends on `baracuda-cutlass` (but not `baracuda-kernels-sys`) links
  `cudart` without requiring the CUDA lib dir to already be on `LIB`.
- **`baracuda-build` Windows lib-dir detection.** `pick_lib_dir` returned the
  bare `lib` rather than `lib\x64` on Windows (where the import libraries
  live), contradicting `CudaInstall::lib`'s documented contract and making
  `CudaToolkit::detect()` disagree with `from_nvcc_path()`. It now prefers
  `lib\x64`; Linux ordering is unchanged. Regression-tested.

### Added

- **CUTLASS version guard** in `baracuda-cutlass-sys`: a `CUTLASS_DIR`
  checkout below v4.2.0 is rejected with a hard error, above v4.5.2 warns
  (the curated kernels are verified across the 4.2–4.5.2 range), and an
  in-range version is confirmed.

## 0.0.1-alpha.67 — 2026-06-10 (Phase 74 — Fuel dense FP GEMM + reduce-to closure)

### Added — kernels

- **Phase 74 — Fuel dense-FP-GEMM + reduce-to facade closure** (Fuel
  ask 2026-06-10; full consumer reply in
  `docs/fuel-reply-fp-gemm-reduce-to-2026-06-10.md`):
  - NEW `gemm_dense_cublas_facade` in `baracuda-kernels-sys` — **12
    flat C symbols** `baracuda_kernels_gemm_dense_{f32, f64, f16,
    bf16}_{run, can_implement, workspace_size}`, cuBLAS-backed
    (`cublasGemmEx` / newly-declared `cublasGemmStridedBatchedEx`).
    Row-major dense GEMM with runtime layout tag (RRR / RCR / CRR),
    flexible leading dims, strided-batch folded into the base symbol
    (`stride 0` = broadcast). f16/bf16 accumulate in f32; f32 is true
    IEEE binary32 (default math mode, NOT TF32); f64 = `COMPUTE_64F`.
    Lock-free context-keyed cuBLAS handle pool (hot-path deviation
    from the transient-handle facade convention; documented hazards).
  - NEW `DenseGemmPlan<T>` typed plan over the same symbols, with
    buffer-bounds (`BufferTooSmall`) validation at the safe layer and
    plan-local `DenseGemmLayout` (CRR has no `LayoutSku` variant yet).
  - NEW `ReduceToPlan<T, N>` facade (`{Sum, Max, Min, Prod} × {f32,
    f64, f16, bf16}`) + `ReduceToOp` in `baracuda-kernels-types` over
    the existing Phase 31/37 `reduce_*_to_*` FFI symbols (no new
    kernels — closes the sys-only facade gap that hid the capability
    from Fuel's alpha.66 audit).
  - `UnaryKind::Step` now dispatches through `UnaryPlan` (contig +
    strided × 4 dtypes; the kernels shipped in Phase 31).
  - Gelu flavor disambiguation docs on `unary_gelu_*` /
    `unary_gelu_erf_*` (bit-identical erf-exact twins) /
    `unary_gelu_tanh_*` (tanh approximation) and
    `UnaryKind::{Gelu, GeluTanh}`.
  - Tests: `dense_gemm_smoke` (3 layouts vs f64 CPU ref, padded lds,
    β-accumulate, strided batch + broadcast, all dtypes, direct-FFI
    binding-table shape, rejection matrix), `reduce_to_plan_smoke`,
    `unary_step_smoke` plan-level extension. All green on RTX 4070.

## 0.0.1-alpha.66 — 2026-06-07 (driver VRAM introspection)

- **`Device::vram_info` / `vram_free` / `vram_total`** wrapping
  `cuMemGetInfo_v2` (Fuel ask — powers Fuel's `BackendRuntime` and
  VRAM-pressure backend selection). Current-context caveat
  documented. See `ROADMAP.md` for detail.

## 0.0.1-alpha.65 — 2026-06-05 (Phases 72 + 73)

- **Phase 73 — Cross-impl bench follow-ups**: `FlashDecodingPlan`
  (split-K seq_q=1 decode, 12-16× vs the pre-fix path),
  warp-cooperative QKᵀ (2-2.6× at K≤2048), concat + reduce perf
  closures, `fa2` promoted to a default feature. See `ROADMAP.md`
  for the per-item detail.

### Added — kernels

- **Phase 72 — Strided FFI siblings for normalizer + shape ops**: closes
  the long-standing ROADMAP item. Authored explicit `_strided_run` /
  `_strided_can_implement` siblings for 7 op families:
  `rms_norm`/`layer_norm`/`softmax`/`log_softmax` (FW + BW × 4 dtypes
  each) + `flip`/`roll`/`permute` (FW × 4 dtypes each). **88 new FFI
  symbols** (44 `_strided_run` + 44 `_strided_can_implement`). Each
  sibling routes to the same underlying launcher as the non-strided
  `_run` — the existing exports already accepted stride arrays and the
  C kernels already honored them; the sibling exists so callers
  building explicit dispatch tables can pick the strided path by name
  (matches the Phase 14/18 convention for binary / unary-param ops).
  Test investment: 7 new direct-FFI smoke tests in
  `tests/strided_siblings_ffi_smoke.rs`, all 7/7 pass on RTX 4070 with
  non-contig fixtures (stride-2 over 2x-padded buffer for the
  norm/softmax/roll ops; transposed view for flip + permute).

## 0.0.1-alpha.64 — 2026-06-03 (Phases 64 + 65 + 66-prep + Tier-2 docs sweep)

The biggest single release in the alpha series — closes Phases 64–71
work + the full release-readiness audit (`_can_implement` companion
contract + Tier-2 cross-crate docs + workspace-wide `missing_docs`
sweep).

### Added — kernels

- **Phase 64 — In-place aliasing contract docs (extended)**: documented
  `x_ptr == y_ptr` safety for Cast / Where / Triu / Tril / Fill /
  Activation BW; explicitly marked Flip / Roll / Permute / RoPE as **not**
  in-place safe (write-before-read by construction).
- **Phase 65a — Reusable SMEM kernel helpers**: new
  `baracuda_smem_row_stager.cuh` (`smem_stage_row`, strided variants,
  `smem_budget_for_arch`) and `baracuda_smem_reduce.cuh`
  (`warp_reduce_{sum,max,min}_f32` + `f64`, cross-warp aggregation
  via `BARACUDA_MAX_WARPS = 32`). Lifts the SMEM staging pattern out
  of `baracuda_moe.cuh` into a reusable kernel-helper library at
  `crates/baracuda-kernels-sys/kernels/include/`.
- **Phase 65b — RMSNorm SMEM staging + in-place**: SMEM-staged
  `rms_norm_smem_kernel<T>` enables `y_ptr == x_ptr` aliasing for
  f32/f16/bf16. 3 in-place proof tests bit-exact vs non-aliased reference.
- **Phase 65c — LayerNorm / Softmax / LogSoftmax SMEM staging +
  in-place**: same pattern, three new `*_smem_kernel<T>` + dispatcher
  pairs. 9 in-place proof tests.
- **Phase 65d — BN / GN / IN in-place contract** (no kernel changes):
  documented the existing two-stage kernel (stage-1 stats-only,
  stage-2 single-read-then-write per cell) as in-place safe by
  construction. **f64 IS in-place safe here**. 12 in-place proof tests
  across f32/f16/bf16/f64.
- **Phase 65d-ext — f64 in-place SMEM normalizers**: added
  `block_reduce_{sum,max,min}_f64` to `baracuda_smem_reduce.cuh`;
  RMSNorm / LayerNorm / Softmax / LogSoftmax SMEM kernels gained `<f64>`
  specializations. f64 now uses the in-place fast path under the same
  contig + last-axis preconditions as f32. 4 new f64 in-place proof tests
  (16/16 in-place tests green on RTX 4070).
- **Phase 66-prep — `_can_implement` companion fanout (~2030 symbols)**:
  baracuda's stated FFI convention is one host-side validator per `_run`
  symbol. Audit found ~2026 `_run` symbols missing a `_can_implement`.
  Closed completely across four rounds — 59 C-side macros extended,
  ~2030 Rust extern declarations added. Every `_run` FFI symbol in
  `baracuda-kernels-sys` + `baracuda-transformer-engine-sys` now has
  a `_can_implement` validator.

### Added — new optional sibling crates (Phases 66-71, shipping with this alpha)

- **`baracuda-flashinfer`** + **`baracuda-flashinfer-sys`** — Phase 46+66.
  Safe wrapper for FlashInfer's inference-serving kernels (paged-KV decode
  + prefill + ragged prefill + append, cascade LSE-merge, sort-free top-K
  / top-P / min-P sampling, per-row sampling, FP8 KV decode, spec-decode
  via ChainSpeculativeSampling). Feature-gated behind `flashinfer` on
  `baracuda-kernels`.
- **`baracuda-nvshmem`** + **`baracuda-nvshmem-sys`** — Phase 69.
  Host-side NVSHMEM wrapper: symmetric-heap RDMA (sibling to NCCL's
  collectives). Feature-gated behind `nvshmem`.
- **`baracuda-nvimagecodec`** + **`baracuda-nvimagecodec-sys`** — Phase 70.
  Unified GPU image-codec wrapper (JPEG / JPEG2000 / PNG / TIFF / WebP);
  supersedes the standalone `baracuda-nvjpeg`.
- **`baracuda-cuvs`** + **`baracuda-cuvs-sys`** — Phase 71.
  RAPIDS cuVS vector-search wrapper (IVF-Flat + brute-force k-NN; L2 /
  cosine / inner-product distance). Linux-only.
- **`baracuda-tensorrt` shim feature** — Phase 68.
  Bundled vtable-dispatch C++ shim required to actually call TensorRT
  (NVIDIA ships no flat C ABI). Gated behind the `shim` feature.

### Documentation

- **README** — cargo-features table 4 → 18, workspace-layout +14 missing
  crates, ARCHITECTURE rewritten Phase 14-71, OP-MATRIX swept 11
  false-deferred claims (Pool family, Segment BWs, EmbeddingBag Max,
  BatchedOrmqrWy complex, GgufMmvq f16/bf16, paged FlashAttention, etc.),
  ROADMAP updated.
- **Per-crate READMEs** authored for 8 crates that previously fell back
  to the workspace README on crates.io (flashinfer{,-sys}, megatron,
  optim, ozimmu{,-sys}, transformer-engine{,-sys}).
- **5 stale crate-level `//!` docs** rewritten — kernels, nvjpeg, cudnn,
  cublas, nccl all had Phase-0/Phase-5-era "v0.1 covers X" preambles
  that didn't match the alpha.63 surface; now accurate.
- **13 `-sys` crate "safe-wrapper" pointer** added — every `-sys` crate
  now points downstream readers at its safe-wrapper sibling.
- **~35 executable doctests** added across cublas / cudnn / cufft /
  cusolver / cusparse / cutensor / cuvs / curand / runtime / types.
- **8 examples/ directories** created for the major safe wrappers
  (cublas/cudnn/cufft/cusolver/curand/nccl/runtime/driver).
- **`[workspace.lints]` clause** added to root Cargo.toml + per-crate
  `[lints] workspace = true` propagation (70 crates).
- **Workspace-wide `missing_docs` sweep** (~6900 one-line docs) closed
  every undocumented public item across 13 `-sys` crates. Workspace lint
  promoted from `warn` → `deny` as a regression guard.

### Added — tests

- 12 in-place proof tests for BN/GN/IN (Phase 65d) + 4 new f64 in-place
  proof tests for RMSNorm/LayerNorm/Softmax/LogSoftmax (Phase 65d-ext).
- `bincount_smoke.rs` (4 tests) — direct-FFI coverage for
  `baracuda_kernels_bincount_{i32,i64}_run`.
- `paged_kv_append_smoke.rs` (4 tests) — direct-FFI lifecycle for
  FlashInfer paged-KV append.
- `baracuda-cudf/tests/cudf_smoke.rs` (9 tests) — closed the zero-test
  red flag.
- `baracuda-types-derive` (21 tests, 10 integration + 11 unit) — proc-macro
  positive paths + rejection paths.

### Fixed

- 95 `clippy -D warnings` CI failures (stylistic + unused imports + Rust
  2024 `unsafe_op_in_unsafe_fn` migration via `cargo fix`).
- 10 GGUF-K-block test names allowed via `#[allow(non_snake_case)]`.
- `map_status_pub` + `alibi_dispatch` `#[allow(dead_code)]` for
  feature-gated callers (`xformers_sparse24` / `fa2` respectively).
- `FusedLinearCrossEntropyBackwardPlan::desc` `#[allow(dead_code)]` (BW
  pass is a pure dy-scalar broadcast that doesn't read the descriptor).
- Pre-existing `baracuda-cuvs` crate-level doctest used wrong Result error
  type — would have always failed if anyone ran it; fixed to
  `Box<dyn std::error::Error>`.

---

## Unreleased (queued for next alpha)

(Empty — alpha.64 just shipped.)

---

## 0.0.1-alpha.63 — 2026-05-30 (Phase 63)

### Added

- **FA2 saved-tensor wiring for downstream autograd**: new FFI symbol
  `baracuda_kernels_fa2_sdpa_lse_size` (dense sibling of the varlen
  helper from Phase 59b) + `# LSE saved-tensor contract` sections on
  FW + BW trailblazers. NEW docs guide
  `docs/guides/fa2-saved-tensor-contract.md`.
- 12 new tests (3 host `_lse_size` sanity + 5 FW→BW roundtrip + 4 BW
  feature backfill).

### Notes

- LSE has been a documented FW output since alpha.56 (Phase 42); Phase
  63 just added the size helper and wiring clarity that downstream
  autograd integrators were asking for.

## 0.0.1-alpha.62 — 2026-05-30 (Phase 62)

### Added

- **Strided in-place op support**: lifts the in-place aliasing contract
  from contig-only (Phase 61) to strided. 11 new `affine_inplace` FFI
  symbols (4 contig int backfill + 7 strided fp/int). Same-pointer
  aliasing on strided trailblazers documented as a stable public
  contract under `stride_x == stride_y`.
- NEW `baracuda_kernels_types::strides_equal` host helper.
- 39 new tests (14 host-only + 17 GPU smoke + 8 aliasing-contract proof);
  2229+/0 regression on RTX 4070.

## 0.0.1-alpha.61 — 2026-05-30 (Phase 61)

### Added

- **In-place op family completion (contig)**: 2 new `bf16` / `f16`
  `affine_inplace` FFI symbols with f32 scalars. Same-pointer aliasing
  safety documented as a stable public contract on unary / binary /
  ternary trailblazers (covers ~30 unary + 20 binary + parameterized
  unary launchers across all dtypes).
- Unblocks Fuel's planned in-place op fanout (16+ unary + 4 binary +
  Clamp + PowI families) with zero new symbols per family.

## 0.0.1-alpha.60 — 2026-05-29 (Phase 60)

### Added

- **FA2 head_dim {160, 224, 512} expansion via EricLBuehler/candle
  fork**: 2184/0 regression. Closes the FA2 FW head_dim gap relative to
  upstream FlashAttention v2.8.3 + GQA / ALiBi / sliding / softcap.

## 0.0.1-alpha.59 — 2026-05-28 (Phase 59)

### Added

- **Phase 59a**: FA2 FW expansion to full v2.8.3 head_dim set + GQA /
  ALiBi / sliding / softcap.
- **Phase 59b**: FA2 BW + varlen; closes Fuel's FA2-retirement
  requirements.
- **Phase 59c**: Flash SMEM race fix; regression re-validation.

## 0.0.1-alpha.58 → 0.0.1-alpha.45

Phase 30–58 release-by-release detail is preserved in `ROADMAP.md`'s
phase log. Major milestones in this band:

- **Phase 57** — Megatron-LM TP primitives (composition-only).
- **Phase 56** — Ring Attention (Apache-2.0 reference).
- **Phase 55** — TransformerEngine FP8 cast + delayed-scaling recipe.
- **Phase 54** — xFormers block-sparse SDPA + 2:4 structured sparsity.
- **Phase 53** — bitsandbytes NF4 dequant + GEMV.
- **Phase 50–52** — Mamba-2 SSD + causal-conv1d; vendor cleanup.
- **Phase 49** — NVIDIA Apex multi-tensor optimizer kernels.
- **Phase 48** — Marlin + AWQ W4A16 GEMM (both, side by side).
- **Phase 46** — FlashInfer Tier-1 (paged decode, cascade, sampling).
- **Phase 44/44b** — ozIMMU Ozaki DGEMM (with Windows port).
- **Phase 43** — DeepSeek-AI mHC.cu HyperConnection.
- **Phase 42** — Dao-AILab FlashAttention v2 Tier-1.
- **Phase 30** — `GemmPlan` gains cuBLAS-backed dispatch (3× speedup
  at M=32 f16 decode batch).

## Pre-Phase-30

See `ROADMAP.md` "Historical phase log" for the alpha.1 → alpha.45 band
covering the original kernel matrix build-out (Phases 1–29).

---

## Conventions

- Each phase is a self-contained scope completed before publish (no
  partial phases per release). Unfinished phases accumulate in
  "Unreleased" above until the next published alpha rolls them up.
- **Memory entries** live under
  `~/.claude/projects/c--Users-cires-OneDrive-Documents-projects-baracuda/memory/`
  with one `project_phase<N>_complete.md` per phase — they carry
  load-bearing context (gotchas, surprises) that doesn't fit a
  user-facing changelog.
- **Feature gates** are called out per release — see `README.md`'s
  cargo-features table for what each feature enables.
