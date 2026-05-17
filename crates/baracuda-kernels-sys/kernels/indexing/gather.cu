// baracuda-kernels Phase 7 Milestone 7.3 — `gather` FW + BW kernels.
//
// Trailblazer dtype coverage: f32, f64, i32 for FW; f32, f64 for BW
// (BW uses atomicAdd — restrict to FP for native support).

#include "../include/baracuda_indexing.cuh"

BARACUDA_KERNELS_GATHER_INSTANTIATE(gather_f32, float)
BARACUDA_KERNELS_GATHER_INSTANTIATE(gather_f64, double)
BARACUDA_KERNELS_GATHER_INSTANTIATE(gather_i32, int32_t)

BARACUDA_KERNELS_GATHER_BACKWARD_INSTANTIATE(gather_backward_f32, float)
BARACUDA_KERNELS_GATHER_BACKWARD_INSTANTIATE(gather_backward_f64, double)
