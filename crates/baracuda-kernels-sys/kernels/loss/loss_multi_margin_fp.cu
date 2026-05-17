// baracuda-kernels Milestone 5.3 — MultiMargin FW for FP types.
#include "../include/baracuda_loss.cuh"
BARACUDA_KERNELS_LOSS_MULTI_MARGIN_FW_INSTANTIATE(loss_multi_margin_f32, float)
BARACUDA_KERNELS_LOSS_MULTI_MARGIN_FW_INSTANTIATE(loss_multi_margin_f16, __half)
BARACUDA_KERNELS_LOSS_MULTI_MARGIN_FW_INSTANTIATE(loss_multi_margin_bf16, __nv_bfloat16)
BARACUDA_KERNELS_LOSS_MULTI_MARGIN_FW_INSTANTIATE(loss_multi_margin_f64, double)
