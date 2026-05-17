// baracuda-kernels Milestone 5.2 Category R: PoissonNLL loss BW.
//
// log_input=true:  dinput = (exp(input) - target) · scale
// log_input=false: dinput = (1 - target/input) · scale

#include "../include/baracuda_loss.cuh"

BARACUDA_KERNELS_LOSS_POISSON_NLL_BW_INSTANTIATE(loss_poisson_nll_backward_f32, float)
BARACUDA_KERNELS_LOSS_POISSON_NLL_BW_INSTANTIATE(loss_poisson_nll_backward_f16, __half)
BARACUDA_KERNELS_LOSS_POISSON_NLL_BW_INSTANTIATE(loss_poisson_nll_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_LOSS_POISSON_NLL_BW_INSTANTIATE(loss_poisson_nll_backward_f64, double)
