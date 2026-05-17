// baracuda-kernels Phase 5.1 Category G: BatchNorm BW for FP types.
//
// Three-stage kernel:
//   1) per-group sum_dxh / sum_dxhxh reduction (deterministic, no atomic).
//   2) per-cell dx.
//   3) per-channel dgamma / dbeta.
//
// Workspace required: 2 * c_extent * sizeof(float) bytes for the stage-1
// sums (always f32 regardless of T — f16/bf16 already detour through
// f32; f64 takes a tiny precision loss on the partials, acceptable for
// the trailblazer).

#include "../include/baracuda_norm.cuh"

BARACUDA_KERNELS_BN_GN_BACKWARD_INSTANTIATE(batch_norm_backward_f32, float)
BARACUDA_KERNELS_BN_GN_BACKWARD_INSTANTIATE(batch_norm_backward_f16, __half)
BARACUDA_KERNELS_BN_GN_BACKWARD_INSTANTIATE(batch_norm_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_BN_GN_BACKWARD_INSTANTIATE(batch_norm_backward_f64, double)
