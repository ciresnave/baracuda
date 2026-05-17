// baracuda-kernels Phase 7 Milestone 7.5 — `embedding` BW (atomicAdd).

#include "../include/baracuda_embedding.cuh"

BARACUDA_KERNELS_EMBEDDING_BACKWARD_INSTANTIATE(embedding_backward_f32, float)
BARACUDA_KERNELS_EMBEDDING_BACKWARD_INSTANTIATE(embedding_backward_f64, double)
