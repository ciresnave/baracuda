// baracuda-kernels Phase 7 Milestone 7.5 — `embedding_bag` BW (atomicAdd).
//
// Phase 11.5: adds i64 index instantiations.

#include "../include/baracuda_embedding.cuh"

BARACUDA_KERNELS_EMBEDDING_BAG_BACKWARD_INSTANTIATE(embedding_bag_backward_f32, float,  int32_t)
BARACUDA_KERNELS_EMBEDDING_BAG_BACKWARD_INSTANTIATE(embedding_bag_backward_f64, double, int32_t)

BARACUDA_KERNELS_EMBEDDING_BAG_BACKWARD_INSTANTIATE(embedding_bag_backward_i64idx_f32, float,  int64_t)
BARACUDA_KERNELS_EMBEDDING_BAG_BACKWARD_INSTANTIATE(embedding_bag_backward_i64idx_f64, double, int64_t)
