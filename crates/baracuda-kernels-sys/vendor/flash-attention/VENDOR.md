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
the full upstream forward head_dim set.

**Headers** (full FA2 src headers, ~16 files):
- `flash.h` — `Flash_fwd_params` / `Flash_bwd_params` struct definitions.
- `flash_fwd_kernel.h` — main `compute_attn` / `compute_attn_splitkv` device code.
- `flash_fwd_launch_template.h` — host-side dispatcher + grid setup.
- `kernel_traits.h` — block / warp / SMEM tile sizing.
- `mask.h`, `softmax.h`, `dropout.h`, `alibi.h`, `rotary.h` — algorithm pieces.
- `block_info.h`, `static_switch.h`, `utils.h`, `philox.cuh`, `philox_unpack.cuh` — utilities.
- `hardware_info.h` — CUDA device cap query.
- `namespace_config.h` — `FLASH_NAMESPACE` macro.

**Source `.cu` files** — Phase 42 (Tier 1) + Phase 59a:
- `flash_fwd_hdim32_{fp16,bf16}_{,causal}_sm80.cu` (4 files, Phase 59a)
- `flash_fwd_hdim64_{fp16,bf16}_{,causal}_sm80.cu` (4 files, Phase 59a)
- `flash_fwd_hdim96_{fp16,bf16}_{,causal}_sm80.cu` (4 files, Phase 59a)
- `flash_fwd_hdim128_{fp16,bf16}_{,causal}_sm80.cu` (4 files, Phase 42)
- `flash_fwd_hdim192_{fp16,bf16}_{,causal}_sm80.cu` (4 files, Phase 59a)
- `flash_fwd_hdim256_{fp16,bf16}_{,causal}_sm80.cu` (4 files, Phase 59a)

Total: 24 forward `.cu` files (6 head_dims × 2 dtypes × 2 causal/non-causal).

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
- **Backward `.cu` files** (`flash_bwd_*`) — Tier-2 deferral (Phase 59b).
- **Backward kernel + preprocess headers** (`flash_bwd_kernel.h`,
  `flash_bwd_launch_template.h`, `flash_bwd_preprocess_kernel.h`) —
  Tier-2 deferral (not referenced from any FW source).
- **Split-KV forward `.cu` files** (`flash_fwd_split_hdim*.cu`) —
  paged-attention dispatch path; Phase 46's FlashInfer cherry-pick
  covers paged attention.
- **Varlen forward path** — Phase 59b territory.

### Head dimensions NOT supported

Upstream FA2 v2.8.3 ships ONLY head_dim ∈ {32, 64, 96, 128, 192, 256}.
Head dimensions **160, 224, and 512 are NOT supported** by upstream and
are therefore permanently out-of-scope for this vendor — there are no
`.cu` source files to copy. Callers requiring those exotic head_dims
must use baracuda's bespoke `FlashSdpaPlan` (which supports d_k ≤ 128
natively) or fall back to the naive `SdpaPlan`.

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
