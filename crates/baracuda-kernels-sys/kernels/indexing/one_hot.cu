// baracuda-kernels Phase 7 Milestone 7.3 — `one_hot` FW (no BW —
// non-differentiable; class index is i32).
//
// Output dtype is selected by the caller: f32, f64, i32, bool (u8).
// Input is always i32.

#include "../include/baracuda_indexing.cuh"

BARACUDA_KERNELS_ONE_HOT_INSTANTIATE(one_hot_f32, float, 1.0f, 0.0f)
BARACUDA_KERNELS_ONE_HOT_INSTANTIATE(one_hot_f64, double, 1.0, 0.0)
BARACUDA_KERNELS_ONE_HOT_INSTANTIATE(one_hot_i32, int32_t, 1, 0)
BARACUDA_KERNELS_ONE_HOT_INSTANTIATE(one_hot_bool, uint8_t, (uint8_t)1, (uint8_t)0)
