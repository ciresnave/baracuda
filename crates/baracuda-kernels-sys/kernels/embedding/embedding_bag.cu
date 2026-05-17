// baracuda-kernels Phase 7 Milestone 7.5 — `embedding_bag` FW
// (Sum / Mean modes shared via a runtime `mode` parameter).

#include "../include/baracuda_embedding.cuh"

BARACUDA_KERNELS_EMBEDDING_BAG_INSTANTIATE(embedding_bag_f32, float)
BARACUDA_KERNELS_EMBEDDING_BAG_INSTANTIATE(embedding_bag_f64, double)
BARACUDA_KERNELS_EMBEDDING_BAG_INSTANTIATE(embedding_bag_f16, __half)
BARACUDA_KERNELS_EMBEDDING_BAG_INSTANTIATE(embedding_bag_bf16, __nv_bfloat16)
