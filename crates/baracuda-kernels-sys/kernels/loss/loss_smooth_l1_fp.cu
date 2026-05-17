// baracuda-kernels Milestone 5.2 Category R: SmoothL1 loss FW for FP types.
//
// piecewise: `0.5·(x/β)²·β` if `|x|<β`, else `|x| - 0.5·β`.

#include "../include/baracuda_loss.cuh"

BARACUDA_KERNELS_LOSS_PARAM_FW_INSTANTIATE(loss_smooth_l1_f32, float, smooth_l1_per_cell_kernel)
BARACUDA_KERNELS_LOSS_PARAM_FW_INSTANTIATE(loss_smooth_l1_f16, __half, smooth_l1_per_cell_kernel)
BARACUDA_KERNELS_LOSS_PARAM_FW_INSTANTIATE(loss_smooth_l1_bf16, __nv_bfloat16, smooth_l1_per_cell_kernel)
BARACUDA_KERNELS_LOSS_PARAM_FW_INSTANTIATE(loss_smooth_l1_f64, double, smooth_l1_per_cell_kernel)
