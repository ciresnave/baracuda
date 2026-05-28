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

// Phase 40 (Fuel 6c.4 Gap 6b spillover) — integer value-dtype fanout
// for `index_select` (read-only; no atomics needed). Coverage:
//   value ∈ {u8, i8, u16, i16, u32, i64} × idx ∈ {i32, i64} = 12 symbols.
// (i32 value is already covered above.)
BARACUDA_KERNELS_INDEX_SELECT_INSTANTIATE(index_select_u8,  uint8_t,  int32_t)
BARACUDA_KERNELS_INDEX_SELECT_INSTANTIATE(index_select_i8,  int8_t,   int32_t)
BARACUDA_KERNELS_INDEX_SELECT_INSTANTIATE(index_select_u16, uint16_t, int32_t)
BARACUDA_KERNELS_INDEX_SELECT_INSTANTIATE(index_select_i16, int16_t,  int32_t)
BARACUDA_KERNELS_INDEX_SELECT_INSTANTIATE(index_select_u32, uint32_t, int32_t)
BARACUDA_KERNELS_INDEX_SELECT_INSTANTIATE(index_select_i64, int64_t,  int32_t)

BARACUDA_KERNELS_INDEX_SELECT_INSTANTIATE(index_select_i64idx_u8,  uint8_t,  int64_t)
BARACUDA_KERNELS_INDEX_SELECT_INSTANTIATE(index_select_i64idx_i8,  int8_t,   int64_t)
BARACUDA_KERNELS_INDEX_SELECT_INSTANTIATE(index_select_i64idx_u16, uint16_t, int64_t)
BARACUDA_KERNELS_INDEX_SELECT_INSTANTIATE(index_select_i64idx_i16, int16_t,  int64_t)
BARACUDA_KERNELS_INDEX_SELECT_INSTANTIATE(index_select_i64idx_u32, uint32_t, int64_t)
BARACUDA_KERNELS_INDEX_SELECT_INSTANTIATE(index_select_i64idx_i64, int64_t,  int64_t)
