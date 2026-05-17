// baracuda-kernels Phase 7 Milestone 7.3 — `nonzero` FW (no BW —
// non-differentiable indices output).
//
// Trailblazer dtype coverage: f32, f64, i32, bool (u8). The output is
// always i32 coordinates and an i32 counter; the input element type is
// generic over the dtype family.
//
// Output ordering is NOT row-major (atomic-counter races among blocks);
// callers that need sorted output should sort the [counter, rank]
// coords on the host or with a follow-up sort kernel.

#include "../include/baracuda_indexing.cuh"

BARACUDA_KERNELS_NONZERO_INSTANTIATE(nonzero_f32, float)
BARACUDA_KERNELS_NONZERO_INSTANTIATE(nonzero_f64, double)
BARACUDA_KERNELS_NONZERO_INSTANTIATE(nonzero_i32, int32_t)
BARACUDA_KERNELS_NONZERO_INSTANTIATE(nonzero_bool, uint8_t)
