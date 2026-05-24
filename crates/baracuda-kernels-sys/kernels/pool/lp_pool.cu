// Phase 16.2 — LpPool 1d/2d fused bespoke kernels.
//
// One templated body each for FW and BW, instantiated for the four FP
// dtypes (f32 / f64 / f16 / bf16) — matches the rest of the pool plan
// family's coverage. See `kernels/include/baracuda_lp_pool.cuh` for
// the design notes and op semantics.

#include "../include/baracuda_lp_pool.cuh"

// 1-D FW
BARACUDA_KERNELS_LP_POOL_1D_FW_INSTANTIATE(f32,  float)
BARACUDA_KERNELS_LP_POOL_1D_FW_INSTANTIATE(f64,  double)
BARACUDA_KERNELS_LP_POOL_1D_FW_INSTANTIATE(f16,  __half)
BARACUDA_KERNELS_LP_POOL_1D_FW_INSTANTIATE(bf16, __nv_bfloat16)

// 1-D BW
BARACUDA_KERNELS_LP_POOL_1D_BW_INSTANTIATE(f32,  float)
BARACUDA_KERNELS_LP_POOL_1D_BW_INSTANTIATE(f64,  double)
BARACUDA_KERNELS_LP_POOL_1D_BW_INSTANTIATE(f16,  __half)
BARACUDA_KERNELS_LP_POOL_1D_BW_INSTANTIATE(bf16, __nv_bfloat16)

// 2-D FW
BARACUDA_KERNELS_LP_POOL_2D_FW_INSTANTIATE(f32,  float)
BARACUDA_KERNELS_LP_POOL_2D_FW_INSTANTIATE(f64,  double)
BARACUDA_KERNELS_LP_POOL_2D_FW_INSTANTIATE(f16,  __half)
BARACUDA_KERNELS_LP_POOL_2D_FW_INSTANTIATE(bf16, __nv_bfloat16)

// 2-D BW
BARACUDA_KERNELS_LP_POOL_2D_BW_INSTANTIATE(f32,  float)
BARACUDA_KERNELS_LP_POOL_2D_BW_INSTANTIATE(f64,  double)
BARACUDA_KERNELS_LP_POOL_2D_BW_INSTANTIATE(f16,  __half)
BARACUDA_KERNELS_LP_POOL_2D_BW_INSTANTIATE(bf16, __nv_bfloat16)
