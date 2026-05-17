// baracuda-kernels Phase 5 Category R: KLDiv loss FW for FP types.
//
// `y = mean(target·(log(target) - input))`. PyTorch convention: `input`
// is already log-prob. Zeros-target cells contribute 0 (avoids log(0)).

#include "../include/baracuda_loss.cuh"

BARACUDA_KERNELS_LOSS_ELEMENTWISE_FW_INSTANTIATE(loss_kl_div_f32, float, kl_div_per_cell_kernel)
BARACUDA_KERNELS_LOSS_ELEMENTWISE_FW_INSTANTIATE(loss_kl_div_f16, __half, kl_div_per_cell_kernel)
BARACUDA_KERNELS_LOSS_ELEMENTWISE_FW_INSTANTIATE(loss_kl_div_bf16, __nv_bfloat16, kl_div_per_cell_kernel)
BARACUDA_KERNELS_LOSS_ELEMENTWISE_FW_INSTANTIATE(loss_kl_div_f64, double, kl_div_per_cell_kernel)
