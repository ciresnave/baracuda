// baracuda-kernels Phase 9 Category O — sort / argsort / msort FW + BW.
//
// Trailblazer dtype coverage:
//   * sort / argsort / msort FW: f32, f64, i32, i64.
//   * sort / msort BW: f32, f64 (FP grads only).
//
// Lineage: bitonic-network sort vendored / adapted from llama.cpp's
// argsort.cu via fuel-cuda-kernels (see kernels/include/baracuda_sort.cuh
// for the adaptation log).

#include "../include/baracuda_sort.cuh"

// ---------- sort FW (values + indices, both ascending and descending) ----------
BARACUDA_KERNELS_SORT_INSTANTIATE(sort_f32, float)
BARACUDA_KERNELS_SORT_INSTANTIATE(sort_f64, double)
BARACUDA_KERNELS_SORT_INSTANTIATE(sort_i32, int32_t)
BARACUDA_KERNELS_SORT_INSTANTIATE(sort_i64, int64_t)

// ---------- argsort FW (indices only) ----------
BARACUDA_KERNELS_ARGSORT_INSTANTIATE(argsort_f32, float)
BARACUDA_KERNELS_ARGSORT_INSTANTIATE(argsort_f64, double)
BARACUDA_KERNELS_ARGSORT_INSTANTIATE(argsort_i32, int32_t)
BARACUDA_KERNELS_ARGSORT_INSTANTIATE(argsort_i64, int64_t)

// ---------- msort FW (stable; values + indices) ----------
BARACUDA_KERNELS_MSORT_INSTANTIATE(msort_f32, float)
BARACUDA_KERNELS_MSORT_INSTANTIATE(msort_f64, double)
BARACUDA_KERNELS_MSORT_INSTANTIATE(msort_i32, int32_t)
BARACUDA_KERNELS_MSORT_INSTANTIATE(msort_i64, int64_t)

// ---------- sort BW (scatter dy via saved indices; FP only) ----------
BARACUDA_KERNELS_SORT_BACKWARD_INSTANTIATE(sort_backward_f32, float)
BARACUDA_KERNELS_SORT_BACKWARD_INSTANTIATE(sort_backward_f64, double)
// msort BW shares the same scatter kernel as sort BW — distinct symbol
// name kept for FFI / telemetry parity.
BARACUDA_KERNELS_SORT_BACKWARD_INSTANTIATE(msort_backward_f32, float)
BARACUDA_KERNELS_SORT_BACKWARD_INSTANTIATE(msort_backward_f64, double)
