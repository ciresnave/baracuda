// baracuda-kernels Milestone 5.2 Category R: BCEWithLogits loss FW.
//
// Numerically stable BCE for raw logits:
//   `term = max(x, 0) - x · target + log(1 + exp(-|x|))`.

#include "../include/baracuda_loss.cuh"

BARACUDA_KERNELS_LOSS_ELEMENTWISE_FW_INSTANTIATE(loss_bce_with_logits_f32, float, bce_with_logits_per_cell_kernel)
BARACUDA_KERNELS_LOSS_ELEMENTWISE_FW_INSTANTIATE(loss_bce_with_logits_f16, __half, bce_with_logits_per_cell_kernel)
BARACUDA_KERNELS_LOSS_ELEMENTWISE_FW_INSTANTIATE(loss_bce_with_logits_bf16, __nv_bfloat16, bce_with_logits_per_cell_kernel)
BARACUDA_KERNELS_LOSS_ELEMENTWISE_FW_INSTANTIATE(loss_bce_with_logits_f64, double, bce_with_logits_per_cell_kernel)
