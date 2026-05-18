// baracuda-kernels Phase 7 Milestone 7.3 — `gather` FW + BW kernels.
//
// Trailblazer dtype coverage: f32, f64, i32 for FW; f32, f64 for BW
// (BW uses atomicAdd — restrict to FP for native support).
//
// Phase 11.5 (Fuel team feedback #7): adds i64 index instantiations
// alongside the original i32 ones. PyTorch defaults to int64 for
// indices, so the i64 variants spare callers a cast pass. Legacy
// `_f32 / _f64 / _i32` names keep the i32-index ABI; `_i64idx_*`
// suffix names are the new i64 entry points.

#include "../include/baracuda_indexing.cuh"

// i32 index — legacy / default surface (kept under the original names).
BARACUDA_KERNELS_GATHER_INSTANTIATE(gather_f32, float,   int32_t)
BARACUDA_KERNELS_GATHER_INSTANTIATE(gather_f64, double,  int32_t)
BARACUDA_KERNELS_GATHER_INSTANTIATE(gather_i32, int32_t, int32_t)

// i64 index — Phase 11.5 additions.
BARACUDA_KERNELS_GATHER_INSTANTIATE(gather_i64idx_f32, float,   int64_t)
BARACUDA_KERNELS_GATHER_INSTANTIATE(gather_i64idx_f64, double,  int64_t)
BARACUDA_KERNELS_GATHER_INSTANTIATE(gather_i64idx_i32, int32_t, int64_t)

// Backward — atomicAdd into dsrc, FP-only.
BARACUDA_KERNELS_GATHER_BACKWARD_INSTANTIATE(gather_backward_f32, float,  int32_t)
BARACUDA_KERNELS_GATHER_BACKWARD_INSTANTIATE(gather_backward_f64, double, int32_t)

BARACUDA_KERNELS_GATHER_BACKWARD_INSTANTIATE(gather_backward_i64idx_f32, float,  int64_t)
BARACUDA_KERNELS_GATHER_BACKWARD_INSTANTIATE(gather_backward_i64idx_f64, double, int64_t)
