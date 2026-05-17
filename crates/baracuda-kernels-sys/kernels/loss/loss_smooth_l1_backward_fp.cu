// baracuda-kernels Milestone 5.2 Category R: SmoothL1 loss BW for FP types.

#include "../include/baracuda_loss.cuh"

BARACUDA_KERNELS_LOSS_PARAM_BW_INSTANTIATE(loss_smooth_l1_backward_f32, float, smooth_l1_backward_kernel)
BARACUDA_KERNELS_LOSS_PARAM_BW_INSTANTIATE(loss_smooth_l1_backward_f16, __half, smooth_l1_backward_kernel)
BARACUDA_KERNELS_LOSS_PARAM_BW_INSTANTIATE(loss_smooth_l1_backward_bf16, __nv_bfloat16, smooth_l1_backward_kernel)
BARACUDA_KERNELS_LOSS_PARAM_BW_INSTANTIATE(loss_smooth_l1_backward_f64, double, smooth_l1_backward_kernel)
