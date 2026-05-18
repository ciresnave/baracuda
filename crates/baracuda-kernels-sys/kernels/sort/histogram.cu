// baracuda-kernels Phase 9 Category O — histogram + bincount FW.
//
// Trailblazer dtype coverage:
//   * histogram: f32, f64 input → i32 counts.
//   * bincount:  i32, i64 input → i32 counts.

#include "../include/baracuda_histogram.cuh"

BARACUDA_KERNELS_HISTOGRAM_INSTANTIATE(histogram_f32, float)
BARACUDA_KERNELS_HISTOGRAM_INSTANTIATE(histogram_f64, double)

BARACUDA_KERNELS_BINCOUNT_INSTANTIATE(bincount_i32, int32_t)
BARACUDA_KERNELS_BINCOUNT_INSTANTIATE(bincount_i64, int64_t)
