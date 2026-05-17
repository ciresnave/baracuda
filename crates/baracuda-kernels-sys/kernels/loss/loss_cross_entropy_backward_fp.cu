// baracuda-kernels Phase 5 Category R: CrossEntropy loss BW for FP types.
//
// `dinput[i, c] = (softmax(input)[i, c] - 1{c == target[i]}) · scale`,
// where scale = (dy / N) for Mean, dy for Sum, dy[i] per-row for None.
// Computes softmax inline (same stable two-pass scheme) for each row.

#include "../include/baracuda_loss.cuh"

BARACUDA_KERNELS_LOSS_CROSS_ENTROPY_BW_INSTANTIATE(loss_cross_entropy_backward_f32, float)
BARACUDA_KERNELS_LOSS_CROSS_ENTROPY_BW_INSTANTIATE(loss_cross_entropy_backward_f16, __half)
BARACUDA_KERNELS_LOSS_CROSS_ENTROPY_BW_INSTANTIATE(loss_cross_entropy_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_LOSS_CROSS_ENTROPY_BW_INSTANTIATE(loss_cross_entropy_backward_f64, double)
