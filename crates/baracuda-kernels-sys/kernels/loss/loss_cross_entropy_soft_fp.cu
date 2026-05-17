// baracuda-kernels Milestone 5.2 Category R: Soft-target CrossEntropy FW.
//
// `y[n] = -Σ_c target[n,c] · log_softmax(input)[n,c]` (fused, stable).

#include "../include/baracuda_loss.cuh"

BARACUDA_KERNELS_LOSS_CROSS_ENTROPY_SOFT_FW_INSTANTIATE(loss_cross_entropy_soft_f32, float)
BARACUDA_KERNELS_LOSS_CROSS_ENTROPY_SOFT_FW_INSTANTIATE(loss_cross_entropy_soft_f16, __half)
BARACUDA_KERNELS_LOSS_CROSS_ENTROPY_SOFT_FW_INSTANTIATE(loss_cross_entropy_soft_bf16, __nv_bfloat16)
BARACUDA_KERNELS_LOSS_CROSS_ENTROPY_SOFT_FW_INSTANTIATE(loss_cross_entropy_soft_f64, double)
