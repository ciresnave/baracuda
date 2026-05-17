// baracuda-kernels Phase 5 Category R: NLL loss FW for FP types.
//
// Per-row gather: `term[i] = -input[i, target[i]]`. Then reduce to scalar
// for Mean / Sum, or output per-row for None. Heterogeneous-dtype:
// input is `T`, target is `i64`.

#include "../include/baracuda_loss.cuh"

BARACUDA_KERNELS_LOSS_NLL_FW_INSTANTIATE(loss_nll_f32, float)
BARACUDA_KERNELS_LOSS_NLL_FW_INSTANTIATE(loss_nll_f16, __half)
BARACUDA_KERNELS_LOSS_NLL_FW_INSTANTIATE(loss_nll_bf16, __nv_bfloat16)
BARACUDA_KERNELS_LOSS_NLL_FW_INSTANTIATE(loss_nll_f64, double)
