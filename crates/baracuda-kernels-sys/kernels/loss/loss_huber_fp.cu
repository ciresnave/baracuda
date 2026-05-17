// baracuda-kernels Milestone 5.2 Category R: Huber loss FW for FP types.
//
// piecewise: `0.5·x²` if `|x|<δ`, else `δ·(|x| - 0.5·δ)`.

#include "../include/baracuda_loss.cuh"

BARACUDA_KERNELS_LOSS_PARAM_FW_INSTANTIATE(loss_huber_f32, float, huber_per_cell_kernel)
BARACUDA_KERNELS_LOSS_PARAM_FW_INSTANTIATE(loss_huber_f16, __half, huber_per_cell_kernel)
BARACUDA_KERNELS_LOSS_PARAM_FW_INSTANTIATE(loss_huber_bf16, __nv_bfloat16, huber_per_cell_kernel)
BARACUDA_KERNELS_LOSS_PARAM_FW_INSTANTIATE(loss_huber_f64, double, huber_per_cell_kernel)
