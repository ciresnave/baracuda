// baracuda-kernels Phase 9 Category T — grid_sample + affine_grid (2D).

#include "../include/baracuda_grid_sample.cuh"

BARACUDA_KERNELS_GRID_SAMPLE_2D_INSTANTIATE(grid_sample_2d_f32, float)
BARACUDA_KERNELS_GRID_SAMPLE_2D_INSTANTIATE(grid_sample_2d_f64, double)

BARACUDA_KERNELS_GRID_SAMPLE_2D_BACKWARD_INSTANTIATE(grid_sample_2d_backward_f32, float)
BARACUDA_KERNELS_GRID_SAMPLE_2D_BACKWARD_INSTANTIATE(grid_sample_2d_backward_f64, double)

BARACUDA_KERNELS_AFFINE_GRID_2D_INSTANTIATE(affine_grid_2d_f32, float)
BARACUDA_KERNELS_AFFINE_GRID_2D_INSTANTIATE(affine_grid_2d_f64, double)
