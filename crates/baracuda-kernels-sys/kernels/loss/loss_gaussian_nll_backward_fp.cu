// baracuda-kernels Milestone 5.2 Category R: GaussianNLL loss BW.
//
// `dinput = (input - target) / max(var, eps) · scale` (no grad to var/target).

#include "../include/baracuda_loss.cuh"

BARACUDA_KERNELS_LOSS_GAUSSIAN_NLL_BW_INSTANTIATE(loss_gaussian_nll_backward_f32, float)
BARACUDA_KERNELS_LOSS_GAUSSIAN_NLL_BW_INSTANTIATE(loss_gaussian_nll_backward_f16, __half)
BARACUDA_KERNELS_LOSS_GAUSSIAN_NLL_BW_INSTANTIATE(loss_gaussian_nll_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_LOSS_GAUSSIAN_NLL_BW_INSTANTIATE(loss_gaussian_nll_backward_f64, double)
