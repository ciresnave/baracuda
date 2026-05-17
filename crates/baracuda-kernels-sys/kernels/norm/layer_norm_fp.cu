// baracuda-kernels Phase 5 Category G: LayerNorm FW for FP types.
//
// `y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta` (gamma/beta
// optional, applied independently). Biased ("population") variance —
// matches PyTorch's LayerNorm convention.

#include "../include/baracuda_norm.cuh"

BARACUDA_KERNELS_LAYER_NORM_INSTANTIATE(layer_norm_f32, float)
BARACUDA_KERNELS_LAYER_NORM_INSTANTIATE(layer_norm_f16, __half)
BARACUDA_KERNELS_LAYER_NORM_INSTANTIATE(layer_norm_bf16, __nv_bfloat16)
BARACUDA_KERNELS_LAYER_NORM_INSTANTIATE(layer_norm_f64, double)
