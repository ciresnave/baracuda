// baracuda-kernels Phase 54 — 2:4 Structured Sparsity GEMM (FW only,
// xFormers algorithmic reference, clean-room hand-port).
//
// Inflate-then-reference-GEMM Tier-1 path. Sparse-tensor-core
// (`mma.sp.sync.aligned`) hardware speedup deferred to Tier 2 alongside
// cuSPARSELt integration (see VENDOR.md for rationale).
//
// Gated at build time behind the `xformers_sparse24` cargo feature on
// `baracuda-kernels-sys`.

#include "../include/baracuda_gemm_sparse24.cuh"

BARACUDA_KERNELS_GEMM_SPARSE24_INSTANTIATE(gemm_f32,  float)
BARACUDA_KERNELS_GEMM_SPARSE24_INSTANTIATE(gemm_f16,  __half)
BARACUDA_KERNELS_GEMM_SPARSE24_INSTANTIATE(gemm_bf16, __nv_bfloat16)
