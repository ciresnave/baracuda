# Vendored: Fuel Q8_1 staging kernels (for inspection only)

This directory contains a snapshot of Fuel's Q8_1 staging + MMQ
kernels from `fuel-cuda-kernels/src/quantized.cu`, vendored
**for inspection only** as part of Phase 19 (alpha.36).

## Status: not built, not linked

These kernels are **NOT** included in baracuda's build (no
`build.rs` entry). They sit here purely so baracuda can evaluate
them for shape-specific perf wins versus baracuda's existing
MMVQ path.

## Why vendored

Fuel is retiring `fuel-cuda-kernels` and migrating CUDA QMatMul
to baracuda's MMVQ-everywhere path (mirroring their Vulkan
migration in `b7360fbc`). The Q8_1 staging path
(`quantize_q8_1` → per-format `mul_mat_vec_q*_q8_1_cuda`) is
specialized for high-M prefill shapes — sourced originally from
llama.cpp / vLLM lineage.

Before deleting these kernels on the Fuel side, the Fuel team
asked baracuda to preserve them for evaluation. If we surface a
shape class where the Q8_1 staging path beats baracuda's current
MMVQ implementation, the win is worth absorbing.

## Plan

See `ROADMAP.md` (Phase 19.4): inspect these against baracuda's
alpha.35 MMVQ kernels (with the f16/bf16 activation paths) for
performance characteristics at:

- **High-M prefill** shapes (the original specialization).
- **Block formats** that share lineage (Q4_0, Q4_1, Q5_0, Q5_1,
  Q8_0, Q2_K..Q6_K).

If any block format × shape combo shows a meaningful win for the
Q8_1 path, port the relevant inner loop / SMEM-tile structure
into the corresponding `baracuda_kernels_mmvq_<qtype>_run` kernel
in `kernels/include/baracuda_gguf.cuh`.

If no win surfaces, this directory can be deleted in a future
release.

## Provenance

- **Source**: `fuel-cuda-kernels/src/quantized.cu` from the Fuel
  repository.
- **Lineage**: derived from llama.cpp and vLLM CUDA kernels;
  Fuel's copy was further specialized.
- **License**: MIT (matches Fuel's licensing; compatible with
  baracuda's dual MIT / Apache-2.0).
- **Vendored**: 2026-05-25 as part of Phase 19.
- **Vendored commit**: see git log of this directory.
