// baracuda-kernels Phase 3 Category N: `roll` (cyclic shift along axes).
//
// `y = roll(x, shifts)`. Output shape == input shape; per-axis shift
// amounts in `shifts` (positive or negative). Today only f32 is wired
// (the trailblazer); other dtypes follow as single-INSTANTIATE fanout.

#include "../include/baracuda_elementwise.cuh"

BARACUDA_KERNELS_ROLL_INSTANTIATE(roll_f32, float)
BARACUDA_KERNELS_ROLL_INSTANTIATE(roll_f16, __half)
BARACUDA_KERNELS_ROLL_INSTANTIATE(roll_bf16, __nv_bfloat16)
BARACUDA_KERNELS_ROLL_INSTANTIATE(roll_f64, double)
