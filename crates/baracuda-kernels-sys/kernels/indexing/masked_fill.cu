// baracuda-kernels Phase 7 Milestone 7.3 — `masked_fill` FW + BW.
//
// Trailblazer dtype coverage: f32, f64, i32, bool (u8 storage) for both
// FW and BW. The kernel is workspace-free elementwise selection — no
// atomics, no reductions.
//
// The fill value travels through the FFI as a "bits" payload (i64 or
// f64) which the macro reinterprets into the element type T. The host
// side encodes the value into the appropriate bit pattern for its
// dtype.

#include "../include/baracuda_indexing.cuh"

BARACUDA_KERNELS_MASKED_FILL_INSTANTIATE(masked_fill_f32, float, int64_t)
BARACUDA_KERNELS_MASKED_FILL_INSTANTIATE(masked_fill_f64, double, int64_t)
BARACUDA_KERNELS_MASKED_FILL_INSTANTIATE(masked_fill_i32, int32_t, int64_t)
BARACUDA_KERNELS_MASKED_FILL_INSTANTIATE(masked_fill_bool, uint8_t, int64_t)

BARACUDA_KERNELS_MASKED_FILL_BACKWARD_INSTANTIATE(masked_fill_backward_f32, float)
BARACUDA_KERNELS_MASKED_FILL_BACKWARD_INSTANTIATE(masked_fill_backward_f64, double)
BARACUDA_KERNELS_MASKED_FILL_BACKWARD_INSTANTIATE(masked_fill_backward_i32, int32_t)
BARACUDA_KERNELS_MASKED_FILL_BACKWARD_INSTANTIATE(masked_fill_backward_bool, uint8_t)
