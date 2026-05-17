// baracuda-kernels Phase 5.1 Category G: GroupNorm BW for FP types.
//
// Same three-stage scheme as BatchNorm BW (per-group sums, per-cell dx,
// per-channel affine grads), with group structure indexed by
// `(sample, g_within)` instead of just channel.

#include "../include/baracuda_norm.cuh"

BARACUDA_KERNELS_BN_GN_BACKWARD_INSTANTIATE(group_norm_backward_f32, float)
BARACUDA_KERNELS_BN_GN_BACKWARD_INSTANTIATE(group_norm_backward_f16, __half)
BARACUDA_KERNELS_BN_GN_BACKWARD_INSTANTIATE(group_norm_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_BN_GN_BACKWARD_INSTANTIATE(group_norm_backward_f64, double)
