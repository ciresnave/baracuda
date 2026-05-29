# Vendored: IST-DASLab/marlin (Phase 48)

This directory contains a curated subset of the Marlin (Mixed-Precision
Auto-Regressive Parallel Inference of Large Language Models) CUDA
source tree, vendored into baracuda as part of Phase 48 to give callers
a state-of-the-art W4A16 (4-bit weights, 16-bit activations) GEMM kernel
for fast LLM decode.

Marlin reports ~3.87x speedup over FP16 GEMM at batch sizes 1-32 on
Ampere / Ada GPUs. Goal A of Phase 48.

## Provenance

- **Upstream**: <https://github.com/IST-DASLab/marlin>
- **Pinned commit**: `1f25790bdd49fba53106164a24666dade68d7c90`
  ("Update README.md to point to arxiv paper", 2024-09-04)
- **License**: Apache-2.0 (see `LICENSE` next to this file).
- **Vendored**: 2026-05-28.

## License attribution — Apache-2.0 patent grant

The verbatim upstream `LICENSE` file is checked in alongside this
README. **Do not modify it.** Marlin is licensed under the Apache
License, Version 2.0 — this license includes an **explicit patent
grant under Section 3** ("Grant of Patent License") with a
termination-on-litigation clause. Preserve this grant: do not
re-license the vendored Marlin sources under a non-Apache license.

baracuda's own license is dual MIT / Apache-2.0; the Apache-2.0 arm of
baracuda is compatible with the vendored Marlin sources (the same
patent terms flow through). Downstream consumers using only the
MIT arm of baracuda still receive Marlin under Apache-2.0 — they
**cannot** strip its license header or its patent terms.

Per-file copyright headers in `src/marlin_cuda_kernel.cu` are
preserved verbatim from upstream. The top-level `README.md` of
baracuda's workspace lists Marlin under its third-party attribution
section.

The original `AUTHORS` file did not exist in the upstream tree —
`AUTHORS` here captures the author attribution that the upstream
README + LICENSE convey.

## Paper citation

```bibtex
@article{frantar2024marlin,
  title={MARLIN: Mixed-Precision Auto-Regressive Parallel Inference of
         Large Language Models},
  author={Frantar, Elias and Castro, Roberto L. and Chen, Jiale and
          Hoefler, Torsten and Alistarh, Dan},
  journal={arXiv preprint arXiv:2408.11743},
  year={2024}
}
```

## Scope: what we kept

`src/` contains a single self-contained CUDA kernel TU:

- `marlin_cuda_kernel.cu` — the Marlin W4A16 GEMM kernel. Pure
  CUDA + `cuda_fp16.h` + `<iostream>`; no PyTorch or other heavy
  dependencies. ~822 LOC.

The kernel exposes a single host entry point — `marlin_cuda(...)` —
which is the same function the upstream PyTorch C++ wrapper calls.
We replace the PyTorch wrapper (`marlin_cuda.cpp`) with a baracuda
C-ABI launcher at `kernels/quantization/marlin_launcher.cu`.

## Scope: what we removed

- **PyTorch C++ extension wrapper** (`marlin_cuda.cpp`) — replaced
  by `kernels/quantization/marlin_launcher.cu`, which exposes the
  `baracuda_kernels_int4_marlin_gemm_*` C symbols.
- **Python package** (`marlin/__init__.py`, `setup.py`, `bench.py`,
  `test.py`) — out of scope for a Rust crate. The pack-time
  permutation that `__init__.py` computes is re-implemented in
  pure Rust at `crates/baracuda-kernels/src/quantize/gptq_to_marlin.rs`
  (Goal C of Phase 48).
- **GPTQ pre-quantization scripts** (`gptq/`) — calibration-time
  Python pipeline, not relevant for inference.
- **assets** — images / README artifacts.

## Hardware support

Marlin targets **sm_80 / sm_86 / sm_89** (Ampere + Ada Lovelace). The
kernel makes heavy use of `mma.sync.m16n8k16` tensor-core
instructions plus async `cp.async.cg.shared.global` copies (Ampere+
only). **sm_90 (Hopper) is NOT supported** by the kernel as written
— Hopper requires WGMMA-class kernels for peak throughput, which
the upstream Marlin author has indicated is the domain of a Marlin
v2 / Sparse-Marlin follow-up.

When built with `--features sm90a` on baracuda-kernels-sys, the
Marlin launcher path is compiled only if `sm80` or `sm89` is also
enabled. Otherwise `MarlinGemmPlan::select` returns
`Error::Unsupported`.

## Kernel contract

Marlin is **symmetric int4** (no per-group zero-points; the zero-point
is fused into the dequant by subtracting 8 from the unsigned
4-bit value at dequant time). This is fundamentally incompatible
with the asymmetric int4 used by AWQ / GPTQ; converting GPTQ-quantized
weights to the Marlin layout requires absorbing the zero-point into
the scale, which is the algorithmic work of Goal C
(`gptq_to_marlin_repack`).

- **Activations** : `[M, K]` row-major `half` (fp16).
- **Weights** : pre-shuffled tensor-core-fragment-aligned int4,
  packed 8 int4 per int32 word, with offline permutation applied
  by the pack-time utility. Shape `[K/16, N*16/8]` int32.
- **Scales** : `[K/groupsize, N]` half, with offline permutation
  along the N axis (`scale_perm` / `scale_perm_single` in the
  upstream Python). The Rust repack utility re-implements both
  permutations.
- **Output** : `[M, N]` row-major half.
- **Workspace** : int32 buffer with `>= N/128 * max_par` entries,
  zero-initialised. Marlin uses this as a per-tile lock array.
- **Group size** : `-1` (per-channel) or `128` (the primary
  supported setting; Marlin's permutation is hardcoded to assume
  groupsize is either K or 128).
- **K alignment** : K must be divisible by 128.
- **N alignment** : N must be divisible by 256 (the kernel's
  `thread_n` x parallel-tile lower bound).

These are the same constraints the upstream `Layer.__init__` enforces.

## Build integration

The vendored kernel is compiled when the `marlin` cargo feature on
`baracuda-kernels-sys` is enabled. The build script
(`crates/baracuda-kernels-sys/build.rs`) adds the include paths and
compiles both `vendor/marlin/src/marlin_cuda_kernel.cu` and the
baracuda launcher `kernels/quantization/marlin_launcher.cu`. Marlin
has zero external dependencies — no CUTLASS, no cuBLAS, no cuDNN.
This is the cleanest vendor port in baracuda.

## Future scope

- **Sparse-Marlin (2:4 structured sparsity)** — a follow-up Marlin
  variant that exploits the 2:4 sparse tensor cores on Ampere+. Phase
  N+1 add-on.
- **bf16 path** — Marlin upstream is fp16-only. Adding bf16 would
  require dequantize.cuh-style intrinsic changes (the `0x64006400`
  magic-number trick is fp16-specific). Out of scope for Phase 48.
- **sm_90 / Hopper** — would need a WGMMA rewrite (Marlin v2 territory).

## Pruning script

To re-vendor from upstream:

```bash
git clone https://github.com/IST-DASLab/marlin
cd marlin
git checkout 1f25790bdd49fba53106164a24666dade68d7c90
cp marlin/marlin_cuda_kernel.cu <baracuda>/vendor/marlin/src/
cp LICENSE <baracuda>/vendor/marlin/
```
