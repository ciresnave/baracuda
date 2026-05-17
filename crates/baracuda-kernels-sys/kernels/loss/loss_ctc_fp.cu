// baracuda-kernels Milestone 5.5: CTCLoss FW for FP types.
//
// Connectionist Temporal Classification loss. Forward dynamic programming
// over the CTC lattice — one CUDA block per batch sample, threads
// cooperate on the per-time-step recurrence over the extended target
// sequence of length L = 2·S + 1.
//
// f16 / bf16 / f32 use an f32 accumulator path; f64 uses native double.

#include "../include/baracuda_ctc.cuh"

BARACUDA_KERNELS_LOSS_CTC_FW_INSTANTIATE_F32_ACC(loss_ctc_f32, float)
BARACUDA_KERNELS_LOSS_CTC_FW_INSTANTIATE_F32_ACC(loss_ctc_f16, __half)
BARACUDA_KERNELS_LOSS_CTC_FW_INSTANTIATE_F32_ACC(loss_ctc_bf16, __nv_bfloat16)
BARACUDA_KERNELS_LOSS_CTC_FW_INSTANTIATE_F64_ACC(loss_ctc_f64, double)
