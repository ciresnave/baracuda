// baracuda-kernels Phase 7 Milestone 7.3 — `index_select` FW + BW.

#include "../include/baracuda_indexing.cuh"

BARACUDA_KERNELS_INDEX_SELECT_INSTANTIATE(index_select_f32, float)
BARACUDA_KERNELS_INDEX_SELECT_INSTANTIATE(index_select_f64, double)
BARACUDA_KERNELS_INDEX_SELECT_INSTANTIATE(index_select_i32, int32_t)

BARACUDA_KERNELS_INDEX_SELECT_BACKWARD_INSTANTIATE(index_select_backward_f32, float)
BARACUDA_KERNELS_INDEX_SELECT_BACKWARD_INSTANTIATE(index_select_backward_f64, double)
