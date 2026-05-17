// baracuda-kernels Phase 5 Category R: MSE loss FW for FP types.
//
// `y = mean((pred - target)²)` (or sum / per-cell). Per-cell pass writes
// to workspace[numel] (for Mean/Sum) or directly to out (for None);
// single-block tree reduction collapses to a scalar.

#include "../include/baracuda_loss.cuh"

BARACUDA_KERNELS_LOSS_ELEMENTWISE_FW_INSTANTIATE(loss_mse_f32, float, mse_per_cell_kernel)
BARACUDA_KERNELS_LOSS_ELEMENTWISE_FW_INSTANTIATE(loss_mse_f16, __half, mse_per_cell_kernel)
BARACUDA_KERNELS_LOSS_ELEMENTWISE_FW_INSTANTIATE(loss_mse_bf16, __nv_bfloat16, mse_per_cell_kernel)
BARACUDA_KERNELS_LOSS_ELEMENTWISE_FW_INSTANTIATE(loss_mse_f64, double, mse_per_cell_kernel)
