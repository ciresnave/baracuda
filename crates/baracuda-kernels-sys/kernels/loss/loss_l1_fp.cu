// baracuda-kernels Milestone 5.2 Category R: L1 loss FW for FP types.
//
// `y = |pred - target|` per-cell; mean / sum / none reduction.

#include "../include/baracuda_loss.cuh"

BARACUDA_KERNELS_LOSS_ELEMENTWISE_FW_INSTANTIATE(loss_l1_f32, float, l1_per_cell_kernel)
BARACUDA_KERNELS_LOSS_ELEMENTWISE_FW_INSTANTIATE(loss_l1_f16, __half, l1_per_cell_kernel)
BARACUDA_KERNELS_LOSS_ELEMENTWISE_FW_INSTANTIATE(loss_l1_bf16, __nv_bfloat16, l1_per_cell_kernel)
BARACUDA_KERNELS_LOSS_ELEMENTWISE_FW_INSTANTIATE(loss_l1_f64, double, l1_per_cell_kernel)
