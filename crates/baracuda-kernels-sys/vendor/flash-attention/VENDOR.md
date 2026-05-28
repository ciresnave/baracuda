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

`src/` contains the minimum set of headers and `.cu` instantiations
needed for the **Tier-1 integration** (Phase 42):

**Headers** (full FA2 src headers, ~16 files):
- `flash.h` — `Flash_fwd_params` / `Flash_bwd_params` struct definitions.
- `flash_fwd_kernel.h` — main `compute_attn` / `compute_attn_splitkv` device code.
- `flash_fwd_launch_template.h` — host-side dispatcher + grid setup.
- `kernel_traits.h` — block / warp / SMEM tile sizing.
- `mask.h`, `softmax.h`, `dropout.h`, `alibi.h`, `rotary.h` — algorithm pieces.
- `block_info.h`, `static_switch.h`, `utils.h`, `philox.cuh`, `philox_unpack.cuh` — utilities.
- `hardware_info.h` — CUDA device cap query.
- `namespace_config.h` — `FLASH_NAMESPACE` macro.

**Source `.cu` files** — Tier 1 only:
- `flash_fwd_hdim128_fp16_sm80.cu`
- `flash_fwd_hdim128_fp16_causal_sm80.cu`
- `flash_fwd_hdim128_bf16_sm80.cu`
- `flash_fwd_hdim128_bf16_causal_sm80.cu`

## Scope: what we removed

- **Python bindings** (`flash_attn/` directory) — we expose FA2 through
  `baracuda_kernels_fa2_sdpa_<dt>_run` FFI symbols, not Python.
- **PyTorch C++ extension glue** (`csrc/flash_attn/flash_api.cpp`) —
  thin wrapper expecting `torch::Tensor` arguments; replaced by
  baracuda's own launcher (`kernels/attention/fa2_launcher.cu`).
- **Hopper / sm_90 paths** (`hopper/`) — FA3 supersedes FA2 on Hopper;
  hardware not in scope for Phase 42.
- **Composable Kernel ROCm path** (`csrc/composable_kernel/`,
  `csrc/flash_attn_ck/`) — AMD target, not applicable.
- **fused_dense_lib**, **layer_norm**, **benchmarks/**, **tests/**,
  **training/**, **examples/**, **assets/**, **Makefile**, **setup.py**,
  **MANIFEST.in**, **usage.md**.
- **Vendored CUTLASS submodule** (`csrc/cutlass/`) — baracuda already
  carries CUTLASS through `baracuda-cutlass-sys`. Build script reuses
  the same include path. See **CUTLASS version note** below.
- **Backward `.cu` files** (`flash_bwd_*`) — Tier-2 deferral.
- **Backward kernel + preprocess headers** (`flash_bwd_kernel.h`,
  `flash_bwd_launch_template.h`, `flash_bwd_preprocess_kernel.h`) —
  Tier-2 deferral (not referenced from any Tier-1 source).
- **Split-KV forward `.cu` files** (`flash_fwd_split_hdim*.cu`) —
  paged-attention dispatch path; not in scope for Phase 42.
- **Forward `.cu` files for hdim ≠ 128** — Tier-3 deferral
  (head_dim ∈ {32, 64, 96, 192, 256}).

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

- **Tier 2**: vendor `flash_bwd_hdim128_{fp16,bf16}_{,causal}_sm80.cu` +
  `flash_bwd_*.h` headers, expose `baracuda_kernels_fa2_sdpa_backward_<dt>_run`.
- **Tier 3**: vendor remaining head dimensions (32, 64, 96, 192, 256)
  + varlen path + GQA verification.
- **FA3 / Hopper**: separate effort, not Phase 42.

## Pruning script

If you want to refresh from upstream, the kept-file allowlist is
authoritative in this README. To re-vendor, clone the upstream
release tag, `cp` only the files enumerated under "Scope: what we
kept", then re-verify by running
`cargo build -p baracuda-kernels-sys --features sm80,fa2`.
