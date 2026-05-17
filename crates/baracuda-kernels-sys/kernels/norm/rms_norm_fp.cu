// baracuda-kernels Phase 5 Category G: RMSNorm FW for FP types.
//
// `y = x / sqrt(mean(x², dim=norm_axis) + eps) * gamma` (gamma optional).
// f16 / bf16 accumulate in f32; f32 / f64 use native precision throughout.

#include "../include/baracuda_norm.cuh"

BARACUDA_KERNELS_RMS_NORM_INSTANTIATE(rms_norm_f32, float)
BARACUDA_KERNELS_RMS_NORM_INSTANTIATE(rms_norm_f16, __half)
BARACUDA_KERNELS_RMS_NORM_INSTANTIATE(rms_norm_bf16, __nv_bfloat16)
BARACUDA_KERNELS_RMS_NORM_INSTANTIATE(rms_norm_f64, double)
