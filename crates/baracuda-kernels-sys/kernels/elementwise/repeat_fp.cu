// baracuda-kernels Phase 3 Category N: `repeat` (per-axis tile).
//
// `output[d] = input.shape[d] * repeats[d]`. For each output cell, the
// input coord is `c'[d] = c[d] % input.shape[d]`. All four FP dtypes
// (f32 trailblazer + f16 / bf16 / f64 fanout) share the same template
// — Repeat is a pure copy + modular coord transform, no arithmetic, so
// the dtype only affects the `T` instantiation.

#include "../include/baracuda_elementwise.cuh"

BARACUDA_KERNELS_REPEAT_INSTANTIATE(repeat_f32, float)
BARACUDA_KERNELS_REPEAT_INSTANTIATE(repeat_f16, __half)
BARACUDA_KERNELS_REPEAT_INSTANTIATE(repeat_bf16, __nv_bfloat16)
BARACUDA_KERNELS_REPEAT_INSTANTIATE(repeat_f64, double)
