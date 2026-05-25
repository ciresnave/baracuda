// baracuda-kernels Phase 9 Category T — interpolate (bilinear 2D) FW + BW.
// Phase 21 — align_corners + scale_factor params + f16 / bf16 fanout.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "../include/baracuda_interpolate.cuh"

// ---------- Forward (4 fp dtypes) ----------

BARACUDA_KERNELS_INTERPOLATE_BILINEAR_2D_INSTANTIATE(interpolate_bilinear_2d_f32, float)
BARACUDA_KERNELS_INTERPOLATE_BILINEAR_2D_INSTANTIATE(interpolate_bilinear_2d_f64, double)
BARACUDA_KERNELS_INTERPOLATE_BILINEAR_2D_INSTANTIATE(interpolate_bilinear_2d_f16, __half)
BARACUDA_KERNELS_INTERPOLATE_BILINEAR_2D_INSTANTIATE(interpolate_bilinear_2d_bf16, __nv_bfloat16)

// ---------- Backward (4 fp dtypes) ----------

BARACUDA_KERNELS_INTERPOLATE_BILINEAR_2D_BACKWARD_INSTANTIATE(
    interpolate_bilinear_2d_backward_f32, float)
BARACUDA_KERNELS_INTERPOLATE_BILINEAR_2D_BACKWARD_INSTANTIATE(
    interpolate_bilinear_2d_backward_f64, double)
BARACUDA_KERNELS_INTERPOLATE_BILINEAR_2D_BACKWARD_INSTANTIATE(
    interpolate_bilinear_2d_backward_f16, __half)
BARACUDA_KERNELS_INTERPOLATE_BILINEAR_2D_BACKWARD_INSTANTIATE(
    interpolate_bilinear_2d_backward_bf16, __nv_bfloat16)
