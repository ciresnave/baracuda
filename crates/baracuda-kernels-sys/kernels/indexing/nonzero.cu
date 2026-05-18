// baracuda-kernels Phase 7 Milestone 7.3 — `nonzero` FW (no BW —
// non-differentiable indices output).
//
// Trailblazer dtype coverage: f32, f64, i32, bool. The output index
// dtype is i32 (legacy) or i64 (Phase 11.5 / Fuel team feedback #7).
// PyTorch's `torch.nonzero` returns int64 indices.
//
// Output ordering is NOT row-major (atomic-counter races among blocks);
// callers that need sorted output should sort the [counter, rank]
// coords on the host or with a follow-up sort kernel.

#include "../include/baracuda_indexing.cuh"

// i32 output indices — legacy.
BARACUDA_KERNELS_NONZERO_INSTANTIATE(nonzero_f32,  float,   int32_t)
BARACUDA_KERNELS_NONZERO_INSTANTIATE(nonzero_f64,  double,  int32_t)
BARACUDA_KERNELS_NONZERO_INSTANTIATE(nonzero_i32,  int32_t, int32_t)
BARACUDA_KERNELS_NONZERO_INSTANTIATE(nonzero_bool, uint8_t, int32_t)

// i64 output indices — Phase 11.5.
BARACUDA_KERNELS_NONZERO_INSTANTIATE(nonzero_i64idx_f32,  float,   int64_t)
BARACUDA_KERNELS_NONZERO_INSTANTIATE(nonzero_i64idx_f64,  double,  int64_t)
BARACUDA_KERNELS_NONZERO_INSTANTIATE(nonzero_i64idx_i32,  int32_t, int64_t)
BARACUDA_KERNELS_NONZERO_INSTANTIATE(nonzero_i64idx_bool, uint8_t, int64_t)
