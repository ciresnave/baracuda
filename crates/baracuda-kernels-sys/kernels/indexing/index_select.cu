// baracuda-kernels Phase 7 Milestone 7.3 — `index_select` FW + BW.
//
// Phase 11.5 (Fuel team feedback #7): adds i64 index instantiations.

#include "../include/baracuda_indexing.cuh"

BARACUDA_KERNELS_INDEX_SELECT_INSTANTIATE(index_select_f32, float,   int32_t)
BARACUDA_KERNELS_INDEX_SELECT_INSTANTIATE(index_select_f64, double,  int32_t)
BARACUDA_KERNELS_INDEX_SELECT_INSTANTIATE(index_select_i32, int32_t, int32_t)

BARACUDA_KERNELS_INDEX_SELECT_INSTANTIATE(index_select_i64idx_f32, float,   int64_t)
BARACUDA_KERNELS_INDEX_SELECT_INSTANTIATE(index_select_i64idx_f64, double,  int64_t)
BARACUDA_KERNELS_INDEX_SELECT_INSTANTIATE(index_select_i64idx_i32, int32_t, int64_t)

BARACUDA_KERNELS_INDEX_SELECT_BACKWARD_INSTANTIATE(index_select_backward_f32, float,  int32_t)
BARACUDA_KERNELS_INDEX_SELECT_BACKWARD_INSTANTIATE(index_select_backward_f64, double, int32_t)

BARACUDA_KERNELS_INDEX_SELECT_BACKWARD_INSTANTIATE(index_select_backward_i64idx_f32, float,  int64_t)
BARACUDA_KERNELS_INDEX_SELECT_BACKWARD_INSTANTIATE(index_select_backward_i64idx_f64, double, int64_t)
