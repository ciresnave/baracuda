// baracuda-kernels Phase 7 Milestone 7.3 — `one_hot` FW (no BW —
// non-differentiable; class index is i32 or i64).
//
// Output dtype is selected by the caller: f32, f64, i32, bool (u8).
// Index dtype is i32 (legacy) or i64 (Phase 11.5).

#include "../include/baracuda_indexing.cuh"

BARACUDA_KERNELS_ONE_HOT_INSTANTIATE(one_hot_f32,  float,   int32_t, 1.0f,        0.0f)
BARACUDA_KERNELS_ONE_HOT_INSTANTIATE(one_hot_f64,  double,  int32_t, 1.0,         0.0)
BARACUDA_KERNELS_ONE_HOT_INSTANTIATE(one_hot_i32,  int32_t, int32_t, 1,           0)
BARACUDA_KERNELS_ONE_HOT_INSTANTIATE(one_hot_bool, uint8_t, int32_t, (uint8_t)1,  (uint8_t)0)

// Phase 11.5 — i64 index variants.
BARACUDA_KERNELS_ONE_HOT_INSTANTIATE(one_hot_i64idx_f32,  float,   int64_t, 1.0f,       0.0f)
BARACUDA_KERNELS_ONE_HOT_INSTANTIATE(one_hot_i64idx_f64,  double,  int64_t, 1.0,        0.0)
BARACUDA_KERNELS_ONE_HOT_INSTANTIATE(one_hot_i64idx_i32,  int32_t, int64_t, 1,          0)
BARACUDA_KERNELS_ONE_HOT_INSTANTIATE(one_hot_i64idx_bool, uint8_t, int64_t, (uint8_t)1, (uint8_t)0)
