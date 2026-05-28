// baracuda-kernels Phase 40 (Fuel ask Gap 6b) — multi-block argsort
// for `row_len > 1024`.
//
// Trailblazer dtype coverage: f32, f64, i32, i64.
//
// Algorithm: CUB segmented radix sort (`SortPairs` for ascending,
// `SortPairsDescending` for descending) with caller-supplied workspace.
// See `baracuda_sort_big.cuh` for the full design + memory layout.
//
// FFI surface per dtype:
//   * `baracuda_kernels_argsort_<dt>_big_run`
//   * `baracuda_kernels_argsort_<dt>_big_can_implement`
//   * `baracuda_kernels_argsort_<dt>_big_workspace_size`
//
// Each `_run` rejects `row_len <= 1024` with status 3 (caller dispatches
// to the block-bitonic kernel at that size).

#include "../include/baracuda_sort_big.cuh"

BARACUDA_KERNELS_ARGSORT_BIG_INSTANTIATE(argsort_f32_big, float)
BARACUDA_KERNELS_ARGSORT_BIG_INSTANTIATE(argsort_f64_big, double)
BARACUDA_KERNELS_ARGSORT_BIG_INSTANTIATE(argsort_i32_big, int32_t)
BARACUDA_KERNELS_ARGSORT_BIG_INSTANTIATE(argsort_i64_big, int64_t)
