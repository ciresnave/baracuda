// baracuda-kernels Phase 5 Category R: CrossEntropy loss FW for FP types.
//
// Fused LogSoftmax + NLL: per-row two-pass (find max, sum-of-exp), then
// `term[i] = -(input[i, t] - max - log(Σ exp(input[i, j] - max)))`.
// Class-index target only (i64); soft-target CE is reserved for a future
// fanout. Numerically stable via max subtraction.

#include "../include/baracuda_loss.cuh"

BARACUDA_KERNELS_LOSS_CROSS_ENTROPY_FW_INSTANTIATE(loss_cross_entropy_f32, float)
BARACUDA_KERNELS_LOSS_CROSS_ENTROPY_FW_INSTANTIATE(loss_cross_entropy_f16, __half)
BARACUDA_KERNELS_LOSS_CROSS_ENTROPY_FW_INSTANTIATE(loss_cross_entropy_bf16, __nv_bfloat16)
BARACUDA_KERNELS_LOSS_CROSS_ENTROPY_FW_INSTANTIATE(loss_cross_entropy_f64, double)
