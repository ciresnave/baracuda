# Vendored: facebookresearch/xformers (Phase 54)

This directory contains the attribution + license for the **algorithmic
reference** baracuda's Phase 54 cherry-picks from xFormers. **No
upstream source files are vendored verbatim** — the implementation
under `crates/baracuda-kernels-sys/kernels/{gemm,attention}/` is a
clean-room hand-port. This file documents the provenance, the in-scope
algorithm families, and what was deliberately dropped.

## Provenance

- **Upstream**: <https://github.com/facebookresearch/xformers>
- **License**: BSD 3-Clause (see `LICENSE` next to this file).
- **Algorithmic-reference date**: 2026-05-28.

## License attribution

The verbatim upstream `LICENSE` is checked in alongside this README.
**Do not modify it.** Phase 54's hand-port preserves the BSD-3-Clause
attribution by:

- Listing xFormers in the workspace `README.md` third-party section.
- Carrying this `VENDOR.md` + `LICENSE` + `AUTHORS` triple under
  `vendor/xformers/`, even though no source code lives next to them.

## Scope: what we cherry-picked (algorithmic reference)

### Goal A — BlockSparseAttention

**Source pages (read for algorithm; not copied):**
- `xformers/components/attention/blocksparse.py` — Python-side block
  pattern representation + dispatch.
- `xformers/components/attention/csrc/` — CUDA kernel files for the
  block-sparse SDPA path.

**baracuda implementation**: `crates/baracuda-kernels-sys/kernels/include/baracuda_sdpa_block_sparse.cuh`
+ `crates/baracuda-kernels-sys/kernels/attention/sdpa_block_sparse_fp.cu`
+ `crates/baracuda-kernels/src/attention/sdpa_block_sparse.rs`.

**What we kept (algorithmic):**
- Per-block boolean pattern `[B, H, num_blocks_q × num_blocks_k]`.
- Iterate **only the active blocks** in the K loop (vs the dense
  flash-attention all-pairs loop) — the value-prop is real wall-clock
  speedup on long-context attention with known sparse patterns.
- Block-aligned tile structure — tile size matches block size so the
  pattern check is a single boolean lookup per inner loop iteration.

**What we dropped:**
- The Triton kernel path (`xformers/components/attention/feature_maps/triton_*`)
  — baracuda has no Triton toolchain. We hand-port to C++/CUDA.
- The pre-existing `SparseTensorAttention` / `LowerTriangular` helper
  classes — baracuda's plan API supplies the block pattern as a raw
  device tensor.
- The Python `BlockSparseTensor` wrapper — baracuda's `TensorRef`
  carries the block pattern.

### Goal B — 2:4 Structured Sparsity GEMM

**Source pages (read for algorithm; not copied):**
- `xformers/sparse24/` — Python + CUDA sources for the 2:4 GEMM path.
- xFormers documents the pattern: in every 4 consecutive weights, at
  most 2 are non-zero; offline-compressed weight is `[M, K/2]` plus a
  `[M, K/8]` metadata tensor (`u16`) describing which 2 of every 4
  positions are non-zero.

**baracuda implementation**: `crates/baracuda-kernels-sys/kernels/include/baracuda_gemm_sparse24.cuh`
+ `crates/baracuda-kernels-sys/kernels/gemm/gemm_sparse24_fp.cu`
+ `crates/baracuda-kernels/src/gemm/sparse24.rs`.

**What we kept (algorithmic):**
- The 2:4 weight compression format (`W_compressed: [M, K/2]` + 
  `W_metadata: [M, K/8]` u16).
- The metadata encoding (2 bits per 4-wide group identify which 2 of
  the 4 positions are non-zero).
- The Plan signature `(W_compressed, W_metadata, X) -> Y` matching
  xFormers' `sparse24.SparseSemiStructuredTensor.matmul`.

**What we dropped:**
- The Triton 2:4 GEMM kernel (`xformers/sparse24/triton/`) — we have
  no Triton toolchain.
- cuSPARSELt back-end glue (`xformers/sparse24/cusparselt/`) — pulling
  cuSPARSELt as a dep would mirror cuDNN's "separate NVIDIA download"
  burden. Phase 54 ships a bespoke `mma.sp.sync` path instead;
  cuSPARSELt back-end deferred to Phase 54b if needed.
- The CUTLASS sparse template machinery (`xformers/sparse24/cutlass/`)
  — would duplicate CUTLASS-heavy templates baracuda already
  vendors elsewhere. The bespoke kernel reads `mma.sp.sync.aligned`
  PTX directly.
- The PyTorch `nn.Linear` drop-in subclass — baracuda exposes
  `GemmSparse24Plan` at the kernel level; framework integration is
  caller-side.
- `xformers.sparse24.compress` (offline weight compression) — baracuda
  consumes pre-compressed weights. The compression step is a one-off
  offline transform that has no per-step performance dimension;
  callers can use xFormers' Python utility, or write the trivial
  per-row 2:4 sparsification themselves.

## Out of scope — what we deliberately did NOT vendor from xFormers

xFormers' broader surface heavily duplicates baracuda's existing
families. We did **not** cherry-pick:

- **Memory-efficient attention** (`xformers/ops/fmha.py`) — overlaps
  with baracuda's existing `SdpaPlan` (Phase 6.2) and `FlashSdpaPlan`
  (Phase 6.6) + the Phase 42 FA2 vendor.
- **Fused biases / dropout** (`xformers/components/feedforward/`) —
  baracuda's elementwise + activation families cover this.
- **Rotary embeddings** (`xformers/components/positional_embedding/`)
  — overlaps with baracuda's Phase 6.1 + Phase 36 + Phase 41 RoPE
  surface.
- **LayerNorm fusions** (`xformers/triton/layer_norm.py`) — overlaps
  with baracuda's Phase 5 norm family.
- **Submodule deps**: xFormers pulls flash-attention as a submodule —
  baracuda already vendors FA2 (Phase 42), no need for a second copy.

## Build integration

The bespoke kernels Phase 54 ships are compiled when their respective
cargo features are enabled on `baracuda-kernels-sys`:

- `xformers_blocksparse` — compiles
  `kernels/attention/sdpa_block_sparse_fp.cu` (block-sparse SDPA FW).
- `xformers_sparse24` — compiles `kernels/gemm/gemm_sparse24_fp.cu`
  (2:4 sparse GEMM).

Both features are **off by default** and **independent** (enable one
without the other). The vendored `LICENSE` / `AUTHORS` / `VENDOR.md`
files are unconditional — they're attribution, not source.

## Tier-2 deferrals

- **Sparse-tensor-core path** for `GemmSparse24Plan` using
  `cusparseLt` library (would unlock the "2:4 is faster than dense"
  perf claim at sm_89; today's bespoke `mma.sp.sync` path is slower
  than dense cuBLAS at most shapes — the API surface exists,
  speed-of-light deferred).
- **BlockSparseAttention BW pass** (training-time; xFormers has it,
  Phase 54 ships FW only — same Tier-1 cadence as Phase 42 FA2).
- **GQA broadcast** for BlockSparse — would need a strided FFI
  sibling like Phase 14's SDPA strided path.

## Re-vendoring (in case xFormers' algorithm changes)

To re-read xFormers as an algorithmic reference for an update:

```bash
git clone https://github.com/facebookresearch/xformers
cd xformers
# Algorithm files of interest:
ls xformers/components/attention/blocksparse.py
ls xformers/components/attention/csrc/
ls xformers/sparse24/
# Cross-check baracuda's hand-port against any algorithm changes.
```
