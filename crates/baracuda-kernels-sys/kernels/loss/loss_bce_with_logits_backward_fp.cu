// baracuda-kernels Milestone 5.2 Category R: BCEWithLogits loss BW.
//
// `dlogits = (sigmoid(x) - target) · scale` using numerically stable sigmoid.

#include "../include/baracuda_loss.cuh"

BARACUDA_KERNELS_LOSS_ELEMENTWISE_BW_INSTANTIATE(loss_bce_with_logits_backward_f32, float, bce_with_logits_backward_kernel)
BARACUDA_KERNELS_LOSS_ELEMENTWISE_BW_INSTANTIATE(loss_bce_with_logits_backward_f16, __half, bce_with_logits_backward_kernel)
BARACUDA_KERNELS_LOSS_ELEMENTWISE_BW_INSTANTIATE(loss_bce_with_logits_backward_bf16, __nv_bfloat16, bce_with_logits_backward_kernel)
BARACUDA_KERNELS_LOSS_ELEMENTWISE_BW_INSTANTIATE(loss_bce_with_logits_backward_f64, double, bce_with_logits_backward_kernel)
