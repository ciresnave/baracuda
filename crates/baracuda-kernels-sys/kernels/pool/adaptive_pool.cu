// Phase 16.1 — bit-exact PyTorch adaptive pooling (Avg / Max, 1D / 2D
// / 3D, FW + BW). Single rank-agnostic kernel parameterized on spatial
// rank; per-dtype instantiations land here.
//
// Replaces the cuDNN-approximation path the Phase 11.8 adaptive-pool
// plans previously used (uniform `kernel = ceil(in/out)` / `stride =
// floor(in/out)`). The bespoke kernels implement PyTorch's bit-exact
// non-uniform per-output-cell window convention:
//   start_i = floor(i * in / out)
//   end_i   = ceil((i + 1) * in / out)
//
// See `kernels/include/baracuda_adaptive_pool.cuh` for the full
// algorithmic / dtype / determinism story.

#include "../include/baracuda_adaptive_pool.cuh"

// AvgPool FW — 4 dtypes.
BARACUDA_KERNELS_ADAPTIVE_AVG_POOL_FW_INSTANTIATE(f16,  __half)
BARACUDA_KERNELS_ADAPTIVE_AVG_POOL_FW_INSTANTIATE(bf16, __nv_bfloat16)
BARACUDA_KERNELS_ADAPTIVE_AVG_POOL_FW_INSTANTIATE(f32,  float)
BARACUDA_KERNELS_ADAPTIVE_AVG_POOL_FW_INSTANTIATE(f64,  double)

// AvgPool BW — 4 dtypes.
BARACUDA_KERNELS_ADAPTIVE_AVG_POOL_BW_INSTANTIATE(f16,  __half)
BARACUDA_KERNELS_ADAPTIVE_AVG_POOL_BW_INSTANTIATE(bf16, __nv_bfloat16)
BARACUDA_KERNELS_ADAPTIVE_AVG_POOL_BW_INSTANTIATE(f32,  float)
BARACUDA_KERNELS_ADAPTIVE_AVG_POOL_BW_INSTANTIATE(f64,  double)

// MaxPool FW — 4 dtypes.
BARACUDA_KERNELS_ADAPTIVE_MAX_POOL_FW_INSTANTIATE(f16,  __half)
BARACUDA_KERNELS_ADAPTIVE_MAX_POOL_FW_INSTANTIATE(bf16, __nv_bfloat16)
BARACUDA_KERNELS_ADAPTIVE_MAX_POOL_FW_INSTANTIATE(f32,  float)
BARACUDA_KERNELS_ADAPTIVE_MAX_POOL_FW_INSTANTIATE(f64,  double)

// MaxPool BW — 4 dtypes.
BARACUDA_KERNELS_ADAPTIVE_MAX_POOL_BW_INSTANTIATE(f16,  __half)
BARACUDA_KERNELS_ADAPTIVE_MAX_POOL_BW_INSTANTIATE(bf16, __nv_bfloat16)
BARACUDA_KERNELS_ADAPTIVE_MAX_POOL_BW_INSTANTIATE(f32,  float)
BARACUDA_KERNELS_ADAPTIVE_MAX_POOL_BW_INSTANTIATE(f64,  double)
