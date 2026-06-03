// baracuda-kernels Phase 3 Category N: materialized `permute` for FP types.
//
// `y = x.permute(dims)` — output axis d is input axis `dims[d]`. The
// kernel iterates input cells and writes to the permuted output position.
//
// Today only f32 is wired (the trailblazer); other dtypes are
// mechanical fanout. The kernel template is fully generic in T.

#include "../include/baracuda_elementwise.cuh"

BARACUDA_KERNELS_PERMUTE_INSTANTIATE(permute_f32, float)
BARACUDA_KERNELS_PERMUTE_INSTANTIATE(permute_f16, __half)
BARACUDA_KERNELS_PERMUTE_INSTANTIATE(permute_bf16, __nv_bfloat16)
BARACUDA_KERNELS_PERMUTE_INSTANTIATE(permute_f64, double)

// Phase 72 strided-sibling FFI exports — same underlying launcher.
BARACUDA_KERNELS_PERMUTE_INSTANTIATE_STRIDED(permute_f32, float)
BARACUDA_KERNELS_PERMUTE_INSTANTIATE_STRIDED(permute_f16, __half)
BARACUDA_KERNELS_PERMUTE_INSTANTIATE_STRIDED(permute_bf16, __nv_bfloat16)
BARACUDA_KERNELS_PERMUTE_INSTANTIATE_STRIDED(permute_f64, double)
