// baracuda-kernels Milestone 5.2 Category R: GaussianNLL loss FW.
//
// `term = 0.5 · (log(max(var, eps)) + (input - target)² / max(var, eps))`

#include "../include/baracuda_loss.cuh"

BARACUDA_KERNELS_LOSS_GAUSSIAN_NLL_FW_INSTANTIATE(loss_gaussian_nll_f32, float)
BARACUDA_KERNELS_LOSS_GAUSSIAN_NLL_FW_INSTANTIATE(loss_gaussian_nll_f16, __half)
BARACUDA_KERNELS_LOSS_GAUSSIAN_NLL_FW_INSTANTIATE(loss_gaussian_nll_bf16, __nv_bfloat16)
BARACUDA_KERNELS_LOSS_GAUSSIAN_NLL_FW_INSTANTIATE(loss_gaussian_nll_f64, double)
