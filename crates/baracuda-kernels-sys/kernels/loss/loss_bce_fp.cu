// baracuda-kernels Phase 5 Category R: BCE loss FW for FP types.
//
// `y = -mean(target·log(pred) + (1-target)·log(1-pred))`. Caller ensures
// pred ∈ (0, 1) — kernel doesn't guard against log(0).

#include "../include/baracuda_loss.cuh"

BARACUDA_KERNELS_LOSS_ELEMENTWISE_FW_INSTANTIATE(loss_bce_f32, float, bce_per_cell_kernel)
BARACUDA_KERNELS_LOSS_ELEMENTWISE_FW_INSTANTIATE(loss_bce_f16, __half, bce_per_cell_kernel)
BARACUDA_KERNELS_LOSS_ELEMENTWISE_FW_INSTANTIATE(loss_bce_bf16, __nv_bfloat16, bce_per_cell_kernel)
BARACUDA_KERNELS_LOSS_ELEMENTWISE_FW_INSTANTIATE(loss_bce_f64, double, bce_per_cell_kernel)
