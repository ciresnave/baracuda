// baracuda-kernels Milestone 5.2 Category R: L1 loss BW for FP types.
//
// `dpred = sign(pred - target) · scale`; subgradient at 0 = 0.

#include "../include/baracuda_loss.cuh"

BARACUDA_KERNELS_LOSS_ELEMENTWISE_BW_INSTANTIATE(loss_l1_backward_f32, float, l1_backward_kernel)
BARACUDA_KERNELS_LOSS_ELEMENTWISE_BW_INSTANTIATE(loss_l1_backward_f16, __half, l1_backward_kernel)
BARACUDA_KERNELS_LOSS_ELEMENTWISE_BW_INSTANTIATE(loss_l1_backward_bf16, __nv_bfloat16, l1_backward_kernel)
BARACUDA_KERNELS_LOSS_ELEMENTWISE_BW_INSTANTIATE(loss_l1_backward_f64, double, l1_backward_kernel)
