// baracuda-kernels Phase 5 Category R: BCE loss BW for FP types.
//
// `dpred = (pred - target) / (pred·(1-pred)) · scale`.

#include "../include/baracuda_loss.cuh"

BARACUDA_KERNELS_LOSS_ELEMENTWISE_BW_INSTANTIATE(loss_bce_backward_f32, float, bce_backward_kernel)
BARACUDA_KERNELS_LOSS_ELEMENTWISE_BW_INSTANTIATE(loss_bce_backward_f16, __half, bce_backward_kernel)
BARACUDA_KERNELS_LOSS_ELEMENTWISE_BW_INSTANTIATE(loss_bce_backward_bf16, __nv_bfloat16, bce_backward_kernel)
BARACUDA_KERNELS_LOSS_ELEMENTWISE_BW_INSTANTIATE(loss_bce_backward_f64, double, bce_backward_kernel)
