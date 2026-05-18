// baracuda-kernels Phase 9 Category O — unique_consecutive FW.
//
// Trailblazer dtype coverage: f32, f64, i32. (Caller pre-sorts for
// the un-consecutive `unique` op.)

#include "../include/baracuda_unique.cuh"

BARACUDA_KERNELS_UNIQUE_CONSECUTIVE_INSTANTIATE(unique_consecutive_f32, float)
BARACUDA_KERNELS_UNIQUE_CONSECUTIVE_INSTANTIATE(unique_consecutive_f64, double)
BARACUDA_KERNELS_UNIQUE_CONSECUTIVE_INSTANTIATE(unique_consecutive_i32, int32_t)
