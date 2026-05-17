// baracuda-kernels Milestone 5.3 — MarginRanking FW for FP types.
#include "../include/baracuda_loss.cuh"
BARACUDA_KERNELS_LOSS_MARGIN_RANKING_FW_INSTANTIATE(loss_margin_ranking_f32, float)
BARACUDA_KERNELS_LOSS_MARGIN_RANKING_FW_INSTANTIATE(loss_margin_ranking_f16, __half)
BARACUDA_KERNELS_LOSS_MARGIN_RANKING_FW_INSTANTIATE(loss_margin_ranking_bf16, __nv_bfloat16)
BARACUDA_KERNELS_LOSS_MARGIN_RANKING_FW_INSTANTIATE(loss_margin_ranking_f64, double)
