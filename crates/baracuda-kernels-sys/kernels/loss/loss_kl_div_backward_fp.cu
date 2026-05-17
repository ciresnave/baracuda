// baracuda-kernels Phase 5 Category R: KLDiv loss BW for FP types.
//
// `dinput = -target · scale`.

#include "../include/baracuda_loss.cuh"

BARACUDA_KERNELS_LOSS_KL_DIV_BW_INSTANTIATE(loss_kl_div_backward_f32, float)
BARACUDA_KERNELS_LOSS_KL_DIV_BW_INSTANTIATE(loss_kl_div_backward_f16, __half)
BARACUDA_KERNELS_LOSS_KL_DIV_BW_INSTANTIATE(loss_kl_div_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_LOSS_KL_DIV_BW_INSTANTIATE(loss_kl_div_backward_f64, double)
