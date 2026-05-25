// Phase 19.3 — im2col / im2col1d / col2im1d bespoke kernels.
//
// One templated body each (FW for 2-D, FW for 1-D, BW-like scatter
// for 1-D), instantiated for the four FP dtypes (f32 / f64 / f16 /
// bf16). See `kernels/include/baracuda_im2col.cuh` for the design
// notes and op semantics.
//
// 12 FFI symbols total: 3 ops × 4 dtypes.

#include "../include/baracuda_im2col.cuh"

// im2col_2d (NCHW → 2-D matrix)
BARACUDA_KERNELS_IM2COL_2D_INSTANTIATE(f32,  float)
BARACUDA_KERNELS_IM2COL_2D_INSTANTIATE(f64,  double)
BARACUDA_KERNELS_IM2COL_2D_INSTANTIATE(f16,  __half)
BARACUDA_KERNELS_IM2COL_2D_INSTANTIATE(bf16, __nv_bfloat16)

// im2col_1d (NCL → 2-D matrix)
BARACUDA_KERNELS_IM2COL_1D_INSTANTIATE(f32,  float)
BARACUDA_KERNELS_IM2COL_1D_INSTANTIATE(f64,  double)
BARACUDA_KERNELS_IM2COL_1D_INSTANTIATE(f16,  __half)
BARACUDA_KERNELS_IM2COL_1D_INSTANTIATE(bf16, __nv_bfloat16)

// col2im_1d (inverse of im2col_1d, atomicAdd scatter)
BARACUDA_KERNELS_COL2IM_1D_INSTANTIATE(f32,  float)
BARACUDA_KERNELS_COL2IM_1D_INSTANTIATE(f64,  double)
BARACUDA_KERNELS_COL2IM_1D_INSTANTIATE(f16,  __half)
BARACUDA_KERNELS_COL2IM_1D_INSTANTIATE(bf16, __nv_bfloat16)
