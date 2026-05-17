// baracuda-kernels Phase 4.5: bernoulli + affine-inplace helpers.
//
// `y[i] = (rand[i] < p) ? 1 : 0`, Bool output. `rand` is a caller-
// generated uniform-rand `float` buffer (one sample per output cell).
//
// Also ships the `affine_inplace_{f32,f64}` helper used by the safe-plan
// layer to remap a cuRAND uniform-(0, 1] buffer into Uniform(low, high]
// in place.

#include "../include/baracuda_random.cuh"

BARACUDA_KERNELS_BERNOULLI_INSTANTIATE()
BARACUDA_KERNELS_AFFINE_INPLACE_INSTANTIATE(f32, float)
BARACUDA_KERNELS_AFFINE_INPLACE_INSTANTIATE(f64, double)
