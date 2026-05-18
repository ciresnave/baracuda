// baracuda-kernels Phase 7 Milestone 7.5 — `embedding_bag` FW
// (Sum / Mean modes shared via a runtime `mode` parameter).
//
// Phase 11.5: adds i64 index instantiations.

#include "../include/baracuda_embedding.cuh"

BARACUDA_KERNELS_EMBEDDING_BAG_INSTANTIATE(embedding_bag_f32,  float,         int32_t)
BARACUDA_KERNELS_EMBEDDING_BAG_INSTANTIATE(embedding_bag_f64,  double,        int32_t)
BARACUDA_KERNELS_EMBEDDING_BAG_INSTANTIATE(embedding_bag_f16,  __half,        int32_t)
BARACUDA_KERNELS_EMBEDDING_BAG_INSTANTIATE(embedding_bag_bf16, __nv_bfloat16, int32_t)

BARACUDA_KERNELS_EMBEDDING_BAG_INSTANTIATE(embedding_bag_i64idx_f32,  float,         int64_t)
BARACUDA_KERNELS_EMBEDDING_BAG_INSTANTIATE(embedding_bag_i64idx_f64,  double,        int64_t)
BARACUDA_KERNELS_EMBEDDING_BAG_INSTANTIATE(embedding_bag_i64idx_f16,  __half,        int64_t)
BARACUDA_KERNELS_EMBEDDING_BAG_INSTANTIATE(embedding_bag_i64idx_bf16, __nv_bfloat16, int64_t)
