// baracuda-kernels Phase 9 Category O — searchsorted FW.
//
// Trailblazer dtype coverage: f32, f64, i32, i64. 1-D sorted_seq
// shared across all queries; batched-per-row variant is a follow-up.

#include "../include/baracuda_searchsorted.cuh"

BARACUDA_KERNELS_SEARCHSORTED_INSTANTIATE(searchsorted_f32, float)
BARACUDA_KERNELS_SEARCHSORTED_INSTANTIATE(searchsorted_f64, double)
BARACUDA_KERNELS_SEARCHSORTED_INSTANTIATE(searchsorted_i32, int32_t)
BARACUDA_KERNELS_SEARCHSORTED_INSTANTIATE(searchsorted_i64, int64_t)
