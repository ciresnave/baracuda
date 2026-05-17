// baracuda-kernels Milestone 5.2 Category R: PoissonNLL loss FW.
//
// log_input=true:  term = exp(input) - target · input  (default)
// log_input=false: term = input - target · log(input)

#include "../include/baracuda_loss.cuh"

BARACUDA_KERNELS_LOSS_POISSON_NLL_FW_INSTANTIATE(loss_poisson_nll_f32, float)
BARACUDA_KERNELS_LOSS_POISSON_NLL_FW_INSTANTIATE(loss_poisson_nll_f16, __half)
BARACUDA_KERNELS_LOSS_POISSON_NLL_FW_INSTANTIATE(loss_poisson_nll_bf16, __nv_bfloat16)
BARACUDA_KERNELS_LOSS_POISSON_NLL_FW_INSTANTIATE(loss_poisson_nll_f64, double)
