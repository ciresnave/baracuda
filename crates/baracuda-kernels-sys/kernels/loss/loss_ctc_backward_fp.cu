// baracuda-kernels Milestone 5.5: CTCLoss BW for FP types.
//
// Runs β (backward DP) over the CTC lattice, combines with the saved α
// to compute the CTC gradient. Reuses the per-sample loss and alpha
// workspace saved by the FW pass.

#include "../include/baracuda_ctc.cuh"

BARACUDA_KERNELS_LOSS_CTC_BW_INSTANTIATE_F32_ACC(loss_ctc_backward_f32, float)
BARACUDA_KERNELS_LOSS_CTC_BW_INSTANTIATE_F32_ACC(loss_ctc_backward_f16, __half)
BARACUDA_KERNELS_LOSS_CTC_BW_INSTANTIATE_F32_ACC(loss_ctc_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_LOSS_CTC_BW_INSTANTIATE_F64_ACC(loss_ctc_backward_f64, double)
