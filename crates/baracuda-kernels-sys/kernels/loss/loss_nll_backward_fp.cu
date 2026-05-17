// baracuda-kernels Phase 5 Category R: NLL loss BW for FP types.
//
// `dinput[i, c] = -dy/N if c == target[i] else 0` (Mean reduction).
// The launcher pre-zeros dinput, then writes the active cell per row.

#include "../include/baracuda_loss.cuh"

BARACUDA_KERNELS_LOSS_NLL_BW_INSTANTIATE(loss_nll_backward_f32, float)
BARACUDA_KERNELS_LOSS_NLL_BW_INSTANTIATE(loss_nll_backward_f16, __half)
BARACUDA_KERNELS_LOSS_NLL_BW_INSTANTIATE(loss_nll_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_LOSS_NLL_BW_INSTANTIATE(loss_nll_backward_f64, double)
