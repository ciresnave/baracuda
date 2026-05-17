// baracuda-kernels Phase 3 Category N: `flip` (reverse along axes).
//
// `y = flip(x, dims)`. Output shape == input shape; per-axis mask
// `flip_axes` selects which axes to reverse. Today only f32 is wired
// (the trailblazer); other dtypes follow as single-INSTANTIATE fanout.

#include "../include/baracuda_elementwise.cuh"

BARACUDA_KERNELS_FLIP_INSTANTIATE(flip_f32, float)
BARACUDA_KERNELS_FLIP_INSTANTIATE(flip_f16, __half)
BARACUDA_KERNELS_FLIP_INSTANTIATE(flip_bf16, __nv_bfloat16)
BARACUDA_KERNELS_FLIP_INSTANTIATE(flip_f64, double)
