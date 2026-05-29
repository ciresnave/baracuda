# Vendored: bitsandbytes-foundation/bitsandbytes (Phase 53)

This directory contains a curated subset of the bitsandbytes
[NF4 (NormalFloat 4-bit)](https://arxiv.org/abs/2305.14314)
quantization kernels, vendored into baracuda as part of Phase 53 to
natively support QLoRA-trained models on the Hugging Face Hub
(Llama / Mistral / Qwen prebuilts published with NF4 weights).

## Provenance

- **Upstream**: <https://github.com/bitsandbytes-foundation/bitsandbytes>
- **License**: MIT (Tim Dettmers et al., see `LICENSE` next to this
  file). MIT — **no explicit patent grant**, same as the sibling AWQ
  vendor (`vendor/awq/`, also MIT).
- **Vendored**: 2026-05-28 (Phase 53).

## Why NF4 ≠ GGUF Q4_0 ≠ AWQ

These are three genuinely different 4-bit formats, each owned by a
different inference ecosystem:

- **GGUF Q4_0** (llama.cpp ecosystem; baracuda Phase 8) — symmetric
  int4 × scale, 32 elements per block, integer multiply-add via DP4A.
- **AWQ int4** (mit-han-lab/llm-awq; baracuda Phase 48) — asymmetric
  int4 with explicit per-group zero-points, fp16 dequant via magic-
  number bit-twiddling.
- **NF4** (bitsandbytes; this phase) — **non-uniform quantile codebook**
  derived from the inverse CDF of a Normal distribution. Dequant is a
  16-entry lookup, NOT arithmetic. Produces better accuracy than
  symmetric int4 for normally-distributed weights (which neural-network
  weights approximately are).

NF4 is the dominant 4-bit format for **QLoRA-trained** Llama / Mistral /
Qwen prebuilts on HuggingFace. The Phase 53 plan family parallels
baracuda's existing GGUF / AWQ machinery — separate descriptor,
separate plan, separate FFI symbols.

## Scope — Phase 53 minimum viable for QLoRA inference

`src/nf4_kernel.cuh` contains the kernel-level primitives:

- The **16-entry NF4 codebook** (Section 3.1 of Dettmers et al. 2023),
  hardcoded as a constant `__device__` `float` table. Verbatim — these
  16 values come from the inverse CDF of `N(0, 1)` evaluated at the
  16-quantile midpoints with the zero-quantile pinned to exactly 0.
- The **NF4 unpack helper** `nf4_lookup<T>(idx, codebook)`: maps a
  4-bit code (0..15) to a fp32 / fp16 / bf16 value, multiplied by the
  per-block absmax scale.
- The **block decoder** for the standard NF4 storage layout: pack two
  4-bit codes per byte; per-`block_size`-element absmax scale; weight
  matrix stored as `[N/2, K]` (caller convention; the kernel views the
  same memory as `[N, K]` of nibbles).

`src/nf4_gemv.cuh` contains the kernel template instantiations:

- `nf4_gemv_m1<TAct>` — single-vector decode (M=1) MMVQ. One thread
  block per output row; cooperative-warp reduction across K.
- `nf4_gemv_multi_m<TAct, M>` — batched-decode (M ∈ {2, 4, 8}) MMVQ
  with weight reuse across the M activation rows. Parallels the Phase
  33 GGUF multi-M pattern.

## Scope: what we did NOT vendor

- **8-bit optimizers** (`Adam8bit`, `Lion8bit`) — baracuda's Phase 49
  Apex multi-tensor optimizers already cover the optimizer-step
  surface; bitsandbytes' 8-bit optimizers are an orthogonal axis we
  defer.
- **LLM.int8()** vector-wise W8A8 with FP16 outlier path — obsoleted
  by SmoothQuant (baracuda Phase 45) + Phase 8 int8 GEMM.
- **FP4** — different format from NF4 (different codebook); a
  separate phase if/when a caller asks.
- **Block-wise activation quantization** — the Phase 53 kernels expect
  caller-pre-quantized inputs.
- **Double quantization** of scales — Tier-2 follow-up; the Phase 53
  plan reads `absmax[N/block_size]` from device memory directly.
- **PyTorch ATen wrappers** — bitsandbytes is a Python C extension.
  All PyTorch-binding glue is stripped; we expose a clean
  device-pointer C-ABI matching baracuda's existing GGUF MMVQ pattern.

## Kernel contract

NF4 weights are packed `[N/2, K]` bytes (two 4-bit codes per byte,
upper nibble = code for row `2i+1`, lower nibble = code for row `2i`,
matching bitsandbytes' upstream pack layout). Per-block scale
`absmax: [N/block_size]`, block size typically 64.

- **Activations** (`y`): `[M, K]` row-major in `T` ∈ {`f16`, `bf16`}.
- **Weights** (`W_q`): `[N/2, K]` `u8` (packed). Indexed in-kernel as
  `[N, K]` of 4-bit codes; row `n` lives at byte `(n/2) * K + k_byte`,
  nibble `(n & 1) ? high : low`.
- **Absmax scale** (`absmax`): `[N * K / block_size]` `f32` (per
  weight block; block_size=64 → one scale per 64-element span of K
  within a single output row).
- **Output**: `[M, N]` row-major in the same dtype as activations
  (PyTorch convention).

The accumulator stays `f32` for every variant — only the activation
load and the destination store cast to/from `T`.

## License attribution — MIT (no patent grant)

The verbatim upstream `LICENSE` file is checked in alongside this
VENDOR.md. **Do not modify it.** bitsandbytes is licensed under the
**MIT License**, which **does NOT include an explicit patent grant**.

baracuda's own license is dual MIT / Apache-2.0 — both arms are
compatible with the vendored bitsandbytes MIT sources.

The `AUTHORS` file next to this VENDOR.md captures the author
attribution that the upstream README + LICENSE convey (upstream does
not maintain a separate `AUTHORS` file).

## Paper citation

```bibtex
@inproceedings{dettmers2023qlora,
  title={{QLoRA}: Efficient Finetuning of Quantized {LLMs}},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and
          Zettlemoyer, Luke},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```

## Build integration

Compiled when the `bnb_nf4` cargo feature on `baracuda-kernels-sys`
is enabled. The build script adds the vendored include directory and
compiles `kernels/quantize/nf4_launcher.cu` (which template-
instantiates the kernels from `vendor/bitsandbytes/src/`).

## Re-vendor / upgrade

The NF4 codebook is fixed by Section 3.1 of arXiv:2305.14314 — it does
not change across upstream versions. The launcher layout (pair-packed
4-bit codes, per-block absmax) has been stable since bitsandbytes
0.40.0 (2023). To re-vendor from a fresher upstream:

```bash
git clone https://github.com/bitsandbytes-foundation/bitsandbytes
cd bitsandbytes
# Codebook lives in `csrc/kernels.cu` (search for `nf4_dequantize`
# or the 16 hardcoded fp32 constants). Pair-packed layout assumptions
# are baked into bitsandbytes' Python `Linear4bit` wrapper —
# verify the bit packing matches by quantize/dequant roundtrip.
```
