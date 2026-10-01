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
llama.cpp (see "Provenance" below for the precise per-section
lineage; this is NOT from `vllm-project/vllm`, despite an earlier,
now-corrected version of this doc saying "vLLM").

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
- **Vendored**: 2026-05-25 as part of Phase 19.
- **Vendored commit**: see git log of this directory.

This file has two distinct upstream lineages, **both MIT**, corrected
2026-09-29 (the original "License: MIT (lineage: llama.cpp and vLLM)"
claim below was checked against the actual upstreams and found wrong
in a specific, correctable way — see "What was wrong" below):

### 1. `quantized.cu` lines 1-4319 — llama.cpp `ggml-cuda.cu`

Per the file's own top-of-file comment ("Kernels adapted from
llama.cpp ggml-cuda.cu"). Covers the block-format structs
(`block_q4_0` … `block_q8_K`), the type-0/1 and k-quant dequantize
kernels, the FP-activation MMVQ path, and the q8_1-staging MMQ tile
matmul family (`mul_mat_q`, `load_tiles_q*`, `mul_mat_vec_q*_q8_1_cuda*`).

- **Upstream**: <https://github.com/ggml-org/llama.cpp> (formerly
  `ggerganov/llama.cpp`; GitHub redirects).
- **License**: MIT. Verbatim text + the vintage-matching copyright
  line ("Copyright (c) 2023 Georgi Gerganov") in `LICENSE-llama.cpp`
  next to this file.
- **Approximate dating, not a byte-exact commit match**: the
  `MMQ_X_*_RDNA2` / `_RDNA1` / `_AMPERE` / `_PASCAL` per-architecture
  tuning macros and `K_QUANTS_PER_ITERATION` date this to llama.cpp's
  single-file `ggml-cuda.cu` era, ~September–October 2023 (matches
  commit `0a5eebb45d` "CUDA: mul_mat_q RDNA2 tunings (#2910)",
  2023-09-13, and its neighbors). Fuel's copy is **not a verbatim
  diff match against any single upstream commit** — it strips the
  AMD/ROCm branches (`RDNA1`/`RDNA2`/`RDNA3`, `__launch_bounds__`
  guards) that the real `0a5eebb45d` file still carries, so it's a
  pruned derivative, not a straight copy. `ggml-cuda.cu` itself no
  longer exists upstream as a single file — it was split into
  `ggml/src/ggml-cuda/*.cu` per-op files in the "reorganize source
  code" commit `f3f65429c4` (2024-06-26), so there is no current
  single-file equivalent to diff against either.

### 2. `quantized.cu` lines 4334-4538 — `indexed_moe_forward`

The templated `indexed_moe_forward<>` fused indexed-MoE matvec device
function and its per-qtype `extern "C"` entry points
(`indexed_moe_forward_<qtype>_q8_1`). Per the function's own doc
comment: `@author Guoqing Bao`, `Part of the project:
https://github.com/guoqingbao/vllm.rs/`.

- **Upstream**: that repository has since been renamed to
  <https://github.com/guoqingbao/xinfer> (GitHub redirects the old
  `vllm.rs` URL there; confirmed via `gh repo view`).
- **License**: MIT. Verbatim text + copyright ("Copyright (c) 2026
  Guoqing Bao", from the current `LICENSE.txt` at the head of that
  repo) in `LICENSE-xinfer` next to this file.
- **This is a different, unrelated project from `vllm-project/vllm`**
  (the well-known Apache-2.0 serving engine) despite the name — see
  "What was wrong" below. It is also a **different Guoqing Bao
  project** from `guoqingbao/attention.rs`, the separate upstream
  baracuda's actually-built MoE kernels come from (documented in
  `crates/baracuda-kernels-sys/LICENSE-thirdparty.md`). Both of his
  projects are MIT; they are not the same code.

### What was wrong in the original claim

The original text here said "License: MIT" with "Lineage: derived
from llama.cpp and vLLM CUDA kernels" in one blanket line. The
llama.cpp half of that was correct. The "vLLM" half was not: **there
is no code from `vllm-project/vllm` (Apache-2.0) in this file.** The
`indexed_moe_forward` function's own doc comment always named its
actual origin precisely — `guoqingbao/vllm.rs`, a Rust reimplementation
project by an unrelated individual author, now renamed `xinfer` — but
the README's summary line collapsed that specific, correctly-named
project into "vLLM," which reads as the Apache-2.0 project of that
name. Coincidentally both real upstreams turn out to be MIT, so the
license conclusion ("MIT") was right by accident; the attribution
(who actually wrote it, under what project) was wrong. Per
CireSnave's provenance rule, get the "who" right even when the "what
license" answer happens to still work out.

No Apache-2.0 NOTICE file is needed — neither upstream is Apache-2.0.
