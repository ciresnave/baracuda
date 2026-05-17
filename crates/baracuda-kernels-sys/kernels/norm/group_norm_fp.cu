// baracuda-kernels Phase 5.1 Category G: GroupNorm FW for FP types.
//
// Splits the channel axis into `num_groups` groups and normalizes per
// `(sample, group, *spatial*)`. `num_groups == c_extent` recovers
// InstanceNorm; that case is dispatched through the same kernel by the
// thin `InstanceNormPlan` wrapper. group_kind=1 selects the GN dispatch
// in the shared BN/GN kernel.

#include "../include/baracuda_norm.cuh"

BARACUDA_KERNELS_BN_GN_INSTANTIATE(group_norm_f32, float)
BARACUDA_KERNELS_BN_GN_INSTANTIATE(group_norm_f16, __half)
BARACUDA_KERNELS_BN_GN_INSTANTIATE(group_norm_bf16, __nv_bfloat16)
BARACUDA_KERNELS_BN_GN_INSTANTIATE(group_norm_f64, double)
