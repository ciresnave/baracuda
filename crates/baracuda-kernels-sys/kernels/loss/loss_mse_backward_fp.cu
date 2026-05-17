// baracuda-kernels Phase 5 Category R: MSE loss BW for FP types.
//
// `dpred = 2·(pred - target) · scale` where scale = dy/N (Mean), dy (Sum),
// or dy[i] per-cell (None). dtarget is symmetric (-dpred) — host handles
// the negation if it needs both gradients.

#include "../include/baracuda_loss.cuh"

BARACUDA_KERNELS_LOSS_ELEMENTWISE_BW_INSTANTIATE(loss_mse_backward_f32, float, mse_backward_kernel)
BARACUDA_KERNELS_LOSS_ELEMENTWISE_BW_INSTANTIATE(loss_mse_backward_f16, __half, mse_backward_kernel)
BARACUDA_KERNELS_LOSS_ELEMENTWISE_BW_INSTANTIATE(loss_mse_backward_bf16, __nv_bfloat16, mse_backward_kernel)
BARACUDA_KERNELS_LOSS_ELEMENTWISE_BW_INSTANTIATE(loss_mse_backward_f64, double, mse_backward_kernel)
