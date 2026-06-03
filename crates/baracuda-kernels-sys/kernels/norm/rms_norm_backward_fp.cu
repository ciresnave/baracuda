// baracuda-kernels Phase 5 Category G: RMSNorm BW for FP types.
//
//   dx[..., i] = (dy[..., i] · gamma[i]) / rms
//              - x[..., i] · (Σ_j dy[..., j] · gamma[j] · x[..., j])
//                / (rms³ · N)
//   dgamma[i] = Σ over all non-norm-axis cells dy[..., i] · (x[..., i] / rms)
//
// Single launcher fires two kernels: the per-cell dx kernel and (when
// dgamma is non-null) a one-block-per-feature reduction kernel for dgamma.

#include "../include/baracuda_norm.cuh"

BARACUDA_KERNELS_RMS_NORM_BACKWARD_INSTANTIATE(rms_norm_backward_f32, float)
BARACUDA_KERNELS_RMS_NORM_BACKWARD_INSTANTIATE(rms_norm_backward_f16, __half)
BARACUDA_KERNELS_RMS_NORM_BACKWARD_INSTANTIATE(rms_norm_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_RMS_NORM_BACKWARD_INSTANTIATE(rms_norm_backward_f64, double)

// Phase 72 strided-sibling FFI exports — same underlying launcher.
BARACUDA_KERNELS_RMS_NORM_BACKWARD_INSTANTIATE_STRIDED(rms_norm_backward_f32, float)
BARACUDA_KERNELS_RMS_NORM_BACKWARD_INSTANTIATE_STRIDED(rms_norm_backward_f16, __half)
BARACUDA_KERNELS_RMS_NORM_BACKWARD_INSTANTIATE_STRIDED(rms_norm_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_RMS_NORM_BACKWARD_INSTANTIATE_STRIDED(rms_norm_backward_f64, double)
