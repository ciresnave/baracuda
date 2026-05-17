// baracuda-kernels Phase 7 Milestone 7.5 — `embedding_bag` BW (atomicAdd).

#include "../include/baracuda_embedding.cuh"

BARACUDA_KERNELS_EMBEDDING_BAG_BACKWARD_INSTANTIATE(embedding_bag_backward_f32, float)
BARACUDA_KERNELS_EMBEDDING_BAG_BACKWARD_INSTANTIATE(embedding_bag_backward_f64, double)
