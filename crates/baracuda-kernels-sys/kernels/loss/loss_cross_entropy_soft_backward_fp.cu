// baracuda-kernels Milestone 5.2 Category R: Soft-target CrossEntropy BW.
//
// `dinput[n, c] = (softmax(input)[n, c] - target[n, c]) · scale`.

#include "../include/baracuda_loss.cuh"

BARACUDA_KERNELS_LOSS_CROSS_ENTROPY_SOFT_BW_INSTANTIATE(loss_cross_entropy_soft_backward_f32, float)
BARACUDA_KERNELS_LOSS_CROSS_ENTROPY_SOFT_BW_INSTANTIATE(loss_cross_entropy_soft_backward_f16, __half)
BARACUDA_KERNELS_LOSS_CROSS_ENTROPY_SOFT_BW_INSTANTIATE(loss_cross_entropy_soft_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_LOSS_CROSS_ENTROPY_SOFT_BW_INSTANTIATE(loss_cross_entropy_soft_backward_f64, double)
