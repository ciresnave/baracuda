// baracuda-kernels Phase 7 Milestone 7.5 — `embedding` FW.

#include "../include/baracuda_embedding.cuh"

BARACUDA_KERNELS_EMBEDDING_INSTANTIATE(embedding_f32, float)
BARACUDA_KERNELS_EMBEDDING_INSTANTIATE(embedding_f64, double)
BARACUDA_KERNELS_EMBEDDING_INSTANTIATE(embedding_f16, __half)
BARACUDA_KERNELS_EMBEDDING_INSTANTIATE(embedding_bf16, __nv_bfloat16)
