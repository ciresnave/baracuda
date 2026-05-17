// baracuda-kernels Milestone 5.2 Category R: Huber loss BW for FP types.

#include "../include/baracuda_loss.cuh"

BARACUDA_KERNELS_LOSS_PARAM_BW_INSTANTIATE(loss_huber_backward_f32, float, huber_backward_kernel)
BARACUDA_KERNELS_LOSS_PARAM_BW_INSTANTIATE(loss_huber_backward_f16, __half, huber_backward_kernel)
BARACUDA_KERNELS_LOSS_PARAM_BW_INSTANTIATE(loss_huber_backward_bf16, __nv_bfloat16, huber_backward_kernel)
BARACUDA_KERNELS_LOSS_PARAM_BW_INSTANTIATE(loss_huber_backward_f64, double, huber_backward_kernel)
