// baracuda-kernels Phase 54 — BlockSparseAttention FW (FW only,
// xFormers algorithmic reference, clean-room hand-port).
//
// Routes through the block-sparse Flash-style tile kernel in
// `kernels/include/baracuda_sdpa_block_sparse.cuh`. Tier-1 dtype set
// mirrors the existing flash_sdpa family: {f32, f16, bf16, f64}.
//
// Gated at build time behind the `xformers_blocksparse` cargo feature
// on `baracuda-kernels-sys`.

#include "../include/baracuda_sdpa_block_sparse.cuh"

BARACUDA_KERNELS_SDPA_BLOCK_SPARSE_INSTANTIATE(sdpa_f32,  float)
BARACUDA_KERNELS_SDPA_BLOCK_SPARSE_INSTANTIATE(sdpa_f16,  __half)
BARACUDA_KERNELS_SDPA_BLOCK_SPARSE_INSTANTIATE(sdpa_bf16, __nv_bfloat16)
BARACUDA_KERNELS_SDPA_BLOCK_SPARSE_INSTANTIATE(sdpa_f64,  double)
