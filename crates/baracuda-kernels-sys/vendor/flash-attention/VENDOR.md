# Vendored: Dao-AILab Flash Attention v2 (Phase 42)

This directory contains a curated subset of the Flash Attention v2
CUDA source tree, vendored into baracuda as part of Phase 42 to give
callers a backend-choice "best of both" between baracuda's existing
bespoke `FlashSdpaPlan` and FA2's long-context-optimized kernels.

## Provenance

- **Upstream**: <https://github.com/Dao-AILab/flash-attention>
- **Tag**: `v2.8.3`
- **Commit**: `060c9188beec3a8b62b33a3bfa6d5d2d44975fab`
- **License**: BSD 3-Clause (see `LICENSE` next to this file).
- **Vendored**: 2026-05-28.

## License attribution

The verbatim upstream `LICENSE` and `AUTHORS` files are checked in
alongside this README. **Do not modify them.** Per-file copyright
headers in every `src/*.{h,cuh,cu}` file are also preserved verbatim.
baracuda's own license (dual MIT / Apache-2.0) sits alongside; the
vendored FA2 sources retain BSD-3-Clause independently.

The top-level `README.md` of baracuda's workspace lists Flash
Attention v2 under its third-party attribution section.

## Scope: what we kept

`src/` contains the headers and `.cu` instantiations needed for the
**Phase 42 Tier-1 integration** PLUS the **Phase 59a FW expansion** to
the full upstream forward head_dim set PLUS the **Phase 59b BW
expansion + varlen FW/BW path** PLUS the **Phase 60 head_dim
{160, 224, 512} expansion** (sourced from the Candle fork — see
Attribution / Phase 60 sources section below).

**Headers** (full FA2 src headers, ~19 files):
- `flash.h` — `Flash_fwd_params` / `Flash_bwd_params` struct definitions.
- `flash_fwd_kernel.h` — main `compute_attn` / `compute_attn_splitkv` device code.
- `flash_fwd_launch_template.h` — host-side dispatcher + grid setup (FW).
- `flash_bwd_kernel.h` — BW dq/dk/dv device code (Phase 59b).
- `flash_bwd_launch_template.h` — BW host-side dispatcher (Phase 59b).
- `flash_bwd_preprocess_kernel.h` — BW preprocess (dot(do, o), dQaccum clear; Phase 59b).
- `kernel_traits.h` — block / warp / SMEM tile sizing.
- `mask.h`, `softmax.h`, `dropout.h`, `alibi.h`, `rotary.h` — algorithm pieces.
- `block_info.h`, `static_switch.h`, `utils.h`, `philox.cuh`, `philox_unpack.cuh` — utilities.
- `hardware_info.h` — CUDA device cap query.
- `namespace_config.h` — `FLASH_NAMESPACE` macro.

**Source `.cu` files** — Phase 42 (Tier 1) + Phase 59a + Phase 59b + Phase 60:
- `flash_fwd_hdim{32,64,96,128,192,256}_{fp16,bf16}_{,causal}_sm80.cu`
  (24 files; Phase 42 + 59a — from upstream FA2 v2.8.3)
- `flash_fwd_hdim{160,224}_{fp16,bf16}_{,causal}_sm80.cu` (8 files;
  Phase 60 — from `EricLBuehler/candle@main/candle-flash-attn/kernels/`,
  originally vendored to Candle by Laurent Mazare in PR #245
  (huggingface/candle@2ce5f125, 2023-07-26); hd224 restored after a
  prior upgrade had removed it by Michael Feil in PR #2688
  (huggingface/candle@71cd6d55, 2024-12-31).)
- `flash_fwd_hdim512_{fp16,bf16}_{,causal}_sm80.cu` (4 files;
  Phase 60 — from
  `huggingface/candle@5430d32c97c687973c53a4e65fac318d9be2a834/candle-flash-attn/kernels/`,
  added by Eric Buehler in PR #3417 (merged 2026-03-28). PR #3417 also
  added the `cudaDeviceGetAttribute(cudaDevAttrMaxSharedMemoryPerBlockOptin)`
  SMEM opt-in pattern in `run_mha_fwd_hdim512()` (lines 363-385 of the
  post-patch `flash_fwd_launch_template.h`) and updated the splitkv
  block-size formula to `kBlockN = Headdim <= 64 ? 256 : (Headdim <= 128
  ? 128 : (Headdim <= 256 ? 64 : 32))`. Both adopted verbatim by
  baracuda Phase 60.)
- `flash_bwd_hdim{32,64,96,128,192,256}_{fp16,bf16}_{,causal}_sm80.cu`
  (24 files; **Phase 59b** — from upstream FA2 v2.8.3)

**Phase 60 BW expansion NOT shipped — see "hd160/hd224/hd512 BW caveat"
below.** Phase 60 attempted to extend BW to the 3 new head_dims and
discovered the limitation is structural to FA2's BW algorithm, not a
missing file. The attempt + reasoning + nvcc evidence is preserved in
code comments at `build.rs` (lines 349-369) and in
`crates/baracuda-kernels/src/attention/flash_sdpa_backward.rs` docstring.

Total: 60 `.cu` files (36 FW + 24 BW).

**Varlen** does NOT have a separate .cu file family — upstream FA2
v2.8.3 dispatches varlen via a runtime `params.cu_seqlens_q != nullptr`
check inside `flash_{fwd,bwd}_launch_template.h`. Phase 59b plumbs
varlen through `kernels/attention/fa2_varlen_launcher.cu` (FW)
and the existing `fa2_backward_launcher.cu` (BW) using the same .cu
instantiations as the dense path. No additional vendoring needed.

## Scope: what we removed

- **Python bindings** (`flash_attn/` directory) — we expose FA2 through
  `baracuda_kernels_fa2_sdpa_<dt>_run{,_v2}` FFI symbols, not Python.
- **PyTorch C++ extension glue** (`csrc/flash_attn/flash_api.cpp`) —
  thin wrapper expecting `torch::Tensor` arguments; replaced by
  baracuda's own launcher (`kernels/attention/fa2_launcher.cu`).
- **Hopper / sm_90 paths** (`hopper/`) — FA3 supersedes FA2 on Hopper;
  hardware not in scope.
- **Composable Kernel ROCm path** (`csrc/composable_kernel/`,
  `csrc/flash_attn_ck/`) — AMD target, not applicable.
- **fused_dense_lib**, **layer_norm**, **benchmarks/**, **tests/**,
  **training/**, **examples/**, **assets/**, **Makefile**, **setup.py**,
  **MANIFEST.in**, **usage.md**.
- **Vendored CUTLASS submodule** (`csrc/cutlass/`) — baracuda already
  carries CUTLASS through `baracuda-cutlass-sys`. Build script reuses
  the same include path. See **CUTLASS version note** below.
- **Split-KV forward `.cu` files** (`flash_fwd_split_hdim*.cu`) —
  paged-attention dispatch path; Phase 46's FlashInfer cherry-pick
  covers paged attention.

### Head dimension coverage (post-Phase 60)

baracuda's vendored FA2 supports:

- **FW**: `head_dim ∈ {32, 64, 96, 128, 160, 192, 224, 256, 512}` (9 dims)
- **BW**: `head_dim ∈ {32, 64, 96, 128, 192, 256}` (6 dims)

Provenance:
- Upstream FA2 v2.8.3 ships `.cu` files only for {32, 64, 96, 128, 192,
  256} — that's the Phase 42 + 59a + 59b set, for both FW and BW.
- Head dimensions {160, 224, 512} **FW** are NOT in upstream FA2 v2.8.3
  releases. Phase 60 added them via the Candle fork (which carried
  them since 2023 and 2026 respectively — see Phase 60 sources above).
- Head dimensions {160, 224, 512} **BW** are not supported by FA2's
  BW algorithm at all (see caveat sections below). Phase 60 verified
  this experimentally and documented the limitation.

**Earlier baracuda releases (alpha.59 and prior) incorrectly claimed
160/224/512 were permanently out-of-scope FOR FW.** That claim was based
on an upstream-tag check by the Phase 59a agent and overlooked that the
Candle fork had already extended the head_dim set. Phase 60 corrects
the FW error; BW remains genuinely out-of-scope for these three.

### hd160/hd224 BW caveat (kernel-traits limitation)

FA2's BW kernel requires `kBlockKSmem == 64`, which translates to
`kHeadDim % 64 == 0`. hd160 and hd224 don't satisfy this and route
through a kBlockKSmem=32 path that the BW kernel's atom_layout
doesn't implement. Upstream FA2 v2.8.3 and Candle confirm the
limitation by not shipping BW for these dims either.

baracuda's FA2 BW heuristic (`FA2_BW_SUPPORTED_HEAD_DIMS`) excludes
hd160 and hd224; callers needing BW at those head_dims get the
bespoke `SdpaBackwardPlan` path automatically. FW works fine for
both (different kernel path).

### hd512 BW caveat (kBlockM static-assert)

FA2's BW kernel_traits static-asserts `kBlockM >= 64`. hd512 would
require `kBlockM = 32` to fit any reasonable SMEM budget — even with
the 228 KiB sm_90 opt-in cap, `kBlockM = 64 × kHeadDim = 512` exceeds
viable BW tile sizes. Phase 60 attempted `kBlockM = 32` and got 7
static_assert failures from `Flash_bwd_kernel_traits`. Upstream FA2
v2.8.3 and Candle confirm the limitation by not shipping hd512 BW.

baracuda's FA2 BW heuristic (`FA2_BW_SUPPORTED_HEAD_DIMS`) excludes
hd512; callers needing BW at hd512 get the bespoke `SdpaBackwardPlan`
path automatically. FW works fine (different kernel path; Buehler's
PR #3417 added the SMEM opt-in pattern for FW only).

## PyTorch dependency stubs

FA2 v2.8.3 was authored as a PyTorch C++ extension. Three transitive
includes leak through into the FA2 kernel sources:

1. `<ATen/cuda/CUDAGeneratorImpl.h>` — provides `at::PhiloxCudaState`,
   the RNG state struct embedded in `Flash_fwd_params::philox_args`.
2. `<ATen/cuda/detail/UnpackRaw.cuh>` — provides
   `at::cuda::philox::unpack`, called from `flash_fwd_kernel.h` to
   materialize a `(seed, offset)` pair for the dropout RNG.
3. `<c10/cuda/CUDAException.h>` — provides `C10_CUDA_CHECK` and
   `C10_CUDA_KERNEL_LAUNCH_CHECK` macros wrapping `cudaError_t`.

baracuda ships **PyTorch-free shim headers** at
`crates/baracuda-kernels-sys/vendor/flash-attention/shim/` that satisfy
these includes with a minimal `at::PhiloxCudaState` struct, a stub
`at::cuda::philox::unpack` returning `{0, 0}`, and trivial
`C10_CUDA_*` macros that print + abort on CUDA error. These are
acceptable because:

- The philox path is only exercised when `Is_dropout == true` — Phase 42
  ships `Is_dropout = false` only, so `unpack` is dead code at runtime.
- `C10_CUDA_CHECK` is a launch-time assertion; substitute behaviour
  (fprintf + exit) matches the upstream macro's intent.

The shim headers are documented at
`shim/README.md`.

## CUTLASS version note

FA2 v2.8.3 pins NVIDIA CUTLASS commit `dc4817921edda44a549197ff3a9dcf5df0636e7b`,
which corresponds to release **v4.0.0**. baracuda's
`baracuda-cutlass-sys` defaults to **v4.2.0**, the next minor release
on the same 4.x line. The CUTE-3 templates and `cute/tensor.hpp`
public API are source-compatible across v4.0 → v4.2; FA2 v2.8.3 compiles
clean against v4.2.0 in our build (verified on Tier-1 sources).

If a future CUTLASS release breaks FA2's templates, the workaround is
to set `BARACUDA_CUTLASS_COMMIT=dc4817921edda44a549197ff3a9dcf5df0636e7b`
or `CUTLASS_DIR=/path/to/cutlass-v4.0.0` in the environment before
`cargo build`. See `crates/baracuda-cutlass-sys/build.rs` for the
override knobs.

## Future scope

- **Phase 59b — BW path + varlen**: vendor `flash_bwd_hdim*_{fp16,bf16}_{,causal}_sm80.cu`
  + `flash_bwd_*.h` headers + varlen forward / backward `.cu` files.
  Expose `baracuda_kernels_fa2_sdpa_backward_<dt>_run` + varlen entry
  points (cu_seqlens_q / cu_seqlens_k).
- **FA3 / Hopper**: separate effort; hardware-blocked on the RTX 4070
  dev box.

## Phase 59b — BW + varlen (completed)

Closes the FA2-retirement requirements on the backward + varlen
fronts (the second-and-final piece needed for Fuel to drop their
FA2 vendor entirely):

- Vendored 24 new BW `.cu` files (full head_dim set × {fp16, bf16}
  × {causal, non-causal}; mirrors the FW set).
- Vendored 3 new BW headers (`flash_bwd_kernel.h`,
  `flash_bwd_launch_template.h`, `flash_bwd_preprocess_kernel.h`).
- Added `kernels/attention/fa2_backward_launcher.cu` — dispatches
  the BW kernels for the 6 head_dims via runtime switch; populates
  `Flash_bwd_params` with dQ/dK/dV outputs, dQaccum + dsoftmax_d
  scratch (workspace-supplied), the f32 LSE input, and full Phase
  59a feature plumbing (ALiBi, sliding window, softcap).
- Added `kernels/attention/fa2_varlen_launcher.cu` — varlen FW path
  (FA2 v2.8.3 dispatches varlen via runtime `cu_seqlens_*` check
  inside the existing FW launch template, so no separate .cu file
  family is needed).
- FFI surface: 12 new symbols (BW × 2 dtypes × {run, can_implement} +
  workspace_size; varlen FW × 2 dtypes × {run, can_implement} +
  lse_size; varlen BW × 2 dtypes × {run, can_implement} +
  workspace_size).
- API: NEW `FlashSdpaVarlenPlan` / `FlashSdpaVarlenBackwardPlan`
  families. `FlashSdpaBackwardPlan` extended (additive) with
  `BackendChoice::FlashAttentionV2` arm, ALiBi / sliding-window
  / softcap plumbing on the descriptor (`#[non_exhaustive]` +
  builder pattern), and an optional `lse_f32` arg on the args
  bundle (FA2 stores LSE in f32 regardless of operand dtype).
- BW workspace contract: `dq_accum + dsoftmax_sum` packed
  back-to-back; sizes returned by
  `baracuda_kernels_fa2_sdpa_backward_workspace_size`. Launcher
  zero-fills via `cudaMemsetAsync` before launch.
- Smoke tests: 12+ new tests covering eligibility / workspace
  sizing / e2e BW for all 6 head_dims × 2 dtypes × 2 causal
  modes, plus varlen FW (multi-sequence), varlen BW, and
  varlen × GQA.

## Phase 59a — FW expansion (completed)

Closes Fuel's "still needs upstream FA2" gap on the forward side:

- Vendored 20 new `.cu` files (head_dims 32, 64, 96, 192, 256 × {fp16, bf16}
  × {causal, non-causal}); upstream-supported head_dim=128 stays from
  Phase 42.
- Extended the `fa2_launcher.cu` to dispatch all 6 supported head_dims
  via a runtime switch on `params.d`.
- Added `..._run_v2` + `..._can_implement_v2` FFI entry points that
  expose the full Phase 59a feature set: GQA (via `num_heads_k`
  parameter), ALiBi slopes, sliding window, and softcap. v1 entry
  points preserved for backwards compatibility.
- Lifted `should_use_fa2` heuristic to accept any FA2-supported head_dim
  + GQA-divisible head counts.
- Added `#[non_exhaustive]` + `::new()` builder pattern to
  `FlashSdpaDescriptor` (per Phase 32 convention).
- Added new smoke tests: head_dim fanout (5 new head_dims × dtypes),
  GQA, ALiBi, sliding window, softcap.

## Pruning script

If you want to refresh from upstream, the kept-file allowlist is
authoritative in this README. To re-vendor, clone the upstream
release tag, `cp` only the files enumerated under "Scope: what we
kept", then re-verify by running
`cargo build -p baracuda-kernels-sys --features sm80,fa2`.
