// baracuda-kernels Phase 9 Category T — interpolate (bilinear 2D) FW + BW.

#include "../include/baracuda_interpolate.cuh"

BARACUDA_KERNELS_INTERPOLATE_BILINEAR_2D_INSTANTIATE(interpolate_bilinear_2d_f32, float)
BARACUDA_KERNELS_INTERPOLATE_BILINEAR_2D_INSTANTIATE(interpolate_bilinear_2d_f64, double)

BARACUDA_KERNELS_INTERPOLATE_BILINEAR_2D_BACKWARD_INSTANTIATE(
    interpolate_bilinear_2d_backward_f32, float)
BARACUDA_KERNELS_INTERPOLATE_BILINEAR_2D_BACKWARD_INSTANTIATE(
    interpolate_bilinear_2d_backward_f64, double)
