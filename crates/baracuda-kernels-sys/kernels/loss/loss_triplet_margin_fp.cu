// baracuda-kernels Milestone 5.3 — TripletMargin FW for FP types.
#include "../include/baracuda_loss.cuh"
BARACUDA_KERNELS_LOSS_TRIPLET_MARGIN_FW_INSTANTIATE(loss_triplet_margin_f32, float)
BARACUDA_KERNELS_LOSS_TRIPLET_MARGIN_FW_INSTANTIATE(loss_triplet_margin_f16, __half)
BARACUDA_KERNELS_LOSS_TRIPLET_MARGIN_FW_INSTANTIATE(loss_triplet_margin_bf16, __nv_bfloat16)
BARACUDA_KERNELS_LOSS_TRIPLET_MARGIN_FW_INSTANTIATE(loss_triplet_margin_f64, double)
