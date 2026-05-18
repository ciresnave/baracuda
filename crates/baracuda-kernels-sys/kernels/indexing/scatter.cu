// baracuda-kernels Phase 7 Milestone 7.3 — `scatter_add` FW (BW is
// gather forward — see gather_*.cu — so no separate BW kernel here).
//
// Phase 11.5 (Fuel team feedback #7): adds i64 index instantiations.

#include "../include/baracuda_indexing.cuh"

BARACUDA_KERNELS_SCATTER_ADD_INSTANTIATE(scatter_add_f32, float,  int32_t)
BARACUDA_KERNELS_SCATTER_ADD_INSTANTIATE(scatter_add_f64, double, int32_t)

BARACUDA_KERNELS_SCATTER_ADD_INSTANTIATE(scatter_add_i64idx_f32, float,  int64_t)
BARACUDA_KERNELS_SCATTER_ADD_INSTANTIATE(scatter_add_i64idx_f64, double, int64_t)
