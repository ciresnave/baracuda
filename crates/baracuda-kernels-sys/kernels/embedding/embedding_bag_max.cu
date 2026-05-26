// baracuda-kernels Phase 25 — `embedding_bag` Max mode FW.
//
// Per-feature argmax tracking: writes both the max value AND the
// per-(b, d) `indices[k]` that contributed it (used by the BW kernel
// for sparse scatter).
//
// Dtype coverage matches the other embedding_bag FWs: f32, f64, f16,
// bf16. Indices: i32 + i64 (Phase 11.5 contract).

#include "../include/baracuda_embedding.cuh"

BARACUDA_KERNELS_EMBEDDING_BAG_MAX_INSTANTIATE(embedding_bag_max_f32,  float,         int32_t)
BARACUDA_KERNELS_EMBEDDING_BAG_MAX_INSTANTIATE(embedding_bag_max_f64,  double,        int32_t)
BARACUDA_KERNELS_EMBEDDING_BAG_MAX_INSTANTIATE(embedding_bag_max_f16,  __half,        int32_t)
BARACUDA_KERNELS_EMBEDDING_BAG_MAX_INSTANTIATE(embedding_bag_max_bf16, __nv_bfloat16, int32_t)

BARACUDA_KERNELS_EMBEDDING_BAG_MAX_INSTANTIATE(embedding_bag_max_i64idx_f32,  float,         int64_t)
BARACUDA_KERNELS_EMBEDDING_BAG_MAX_INSTANTIATE(embedding_bag_max_i64idx_f64,  double,        int64_t)
BARACUDA_KERNELS_EMBEDDING_BAG_MAX_INSTANTIATE(embedding_bag_max_i64idx_f16,  __half,        int64_t)
BARACUDA_KERNELS_EMBEDDING_BAG_MAX_INSTANTIATE(embedding_bag_max_i64idx_bf16, __nv_bfloat16, int64_t)
