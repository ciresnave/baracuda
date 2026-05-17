// baracuda-kernels Phase 7 Milestone 7.3 — `scatter_add` FW (BW is
// gather forward — see gather_*.cu — so no separate BW kernel here).

#include "../include/baracuda_indexing.cuh"

BARACUDA_KERNELS_SCATTER_ADD_INSTANTIATE(scatter_add_f32, float)
BARACUDA_KERNELS_SCATTER_ADD_INSTANTIATE(scatter_add_f64, double)
