// baracuda-kernels Phase 7 Milestone 7.5 — `embedding` BW (atomicAdd).
//
// Phase 11.5 (Fuel team feedback #7): adds i64 index instantiations.
// atomicAdd into dweight goes through the CAS-based unified helper
// from baracuda_atomic.cuh (Phase 11.3) for half / bf16, native
// intrinsic for f32 / f64.

#include "../include/baracuda_embedding.cuh"

BARACUDA_KERNELS_EMBEDDING_BACKWARD_INSTANTIATE(embedding_backward_f32, float,  int32_t)
BARACUDA_KERNELS_EMBEDDING_BACKWARD_INSTANTIATE(embedding_backward_f64, double, int32_t)

BARACUDA_KERNELS_EMBEDDING_BACKWARD_INSTANTIATE(embedding_backward_i64idx_f32, float,  int64_t)
BARACUDA_KERNELS_EMBEDDING_BACKWARD_INSTANTIATE(embedding_backward_i64idx_f64, double, int64_t)
