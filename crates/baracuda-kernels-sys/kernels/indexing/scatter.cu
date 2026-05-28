// baracuda-kernels Phase 7 Milestone 7.3 — `scatter_add` FW (BW is
// gather forward — see gather_*.cu — so no separate BW kernel here).
//
// Phase 11.5 (Fuel team feedback #7): adds i64 index instantiations.
//
// Phase 39 (Fuel 6c.4 Gap 5): adds the pure-assign `scatter_*` family
// (NO accumulation; last writer wins on duplicate-target races). FP4
// dtype fanout — f32 / f64 / f16 / bf16 × i32 / i64 idx = 8 symbols.

#include "../include/baracuda_indexing.cuh"

BARACUDA_KERNELS_SCATTER_ADD_INSTANTIATE(scatter_add_f32, float,  int32_t)
BARACUDA_KERNELS_SCATTER_ADD_INSTANTIATE(scatter_add_f64, double, int32_t)

BARACUDA_KERNELS_SCATTER_ADD_INSTANTIATE(scatter_add_i64idx_f32, float,  int64_t)
BARACUDA_KERNELS_SCATTER_ADD_INSTANTIATE(scatter_add_i64idx_f64, double, int64_t)

// Phase 39 — pure-assign scatter (no accumulation).
BARACUDA_KERNELS_SCATTER_INSTANTIATE(scatter_f32,  float,          int32_t)
BARACUDA_KERNELS_SCATTER_INSTANTIATE(scatter_f64,  double,         int32_t)
BARACUDA_KERNELS_SCATTER_INSTANTIATE(scatter_f16,  __half,         int32_t)
BARACUDA_KERNELS_SCATTER_INSTANTIATE(scatter_bf16, __nv_bfloat16,  int32_t)

BARACUDA_KERNELS_SCATTER_INSTANTIATE(scatter_i64idx_f32,  float,          int64_t)
BARACUDA_KERNELS_SCATTER_INSTANTIATE(scatter_i64idx_f64,  double,         int64_t)
BARACUDA_KERNELS_SCATTER_INSTANTIATE(scatter_i64idx_f16,  __half,         int64_t)
BARACUDA_KERNELS_SCATTER_INSTANTIATE(scatter_i64idx_bf16, __nv_bfloat16,  int64_t)

// Phase 40 (Fuel 6c.4 Gap 6b spillover) — integer value-dtype fanout
// for pure-assign `scatter` (last-writer-wins; no atomics needed).
// Coverage: value ∈ {u8, i8, u16, i16, u32, i32, i64} × idx ∈ {i32, i64}
// = 14 symbols.
BARACUDA_KERNELS_SCATTER_INSTANTIATE(scatter_u8,  uint8_t,  int32_t)
BARACUDA_KERNELS_SCATTER_INSTANTIATE(scatter_i8,  int8_t,   int32_t)
BARACUDA_KERNELS_SCATTER_INSTANTIATE(scatter_u16, uint16_t, int32_t)
BARACUDA_KERNELS_SCATTER_INSTANTIATE(scatter_i16, int16_t,  int32_t)
BARACUDA_KERNELS_SCATTER_INSTANTIATE(scatter_u32, uint32_t, int32_t)
BARACUDA_KERNELS_SCATTER_INSTANTIATE(scatter_i32, int32_t,  int32_t)
BARACUDA_KERNELS_SCATTER_INSTANTIATE(scatter_i64, int64_t,  int32_t)

BARACUDA_KERNELS_SCATTER_INSTANTIATE(scatter_i64idx_u8,  uint8_t,  int64_t)
BARACUDA_KERNELS_SCATTER_INSTANTIATE(scatter_i64idx_i8,  int8_t,   int64_t)
BARACUDA_KERNELS_SCATTER_INSTANTIATE(scatter_i64idx_u16, uint16_t, int64_t)
BARACUDA_KERNELS_SCATTER_INSTANTIATE(scatter_i64idx_i16, int16_t,  int64_t)
BARACUDA_KERNELS_SCATTER_INSTANTIATE(scatter_i64idx_u32, uint32_t, int64_t)
BARACUDA_KERNELS_SCATTER_INSTANTIATE(scatter_i64idx_i32, int32_t,  int64_t)
BARACUDA_KERNELS_SCATTER_INSTANTIATE(scatter_i64idx_i64, int64_t,  int64_t)
