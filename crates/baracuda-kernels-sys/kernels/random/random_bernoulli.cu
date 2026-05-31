// baracuda-kernels Phase 4.5 + Phase 61: bernoulli + affine-inplace helpers.
//
// `y[i] = (rand[i] < p) ? 1 : 0`, Bool output. `rand` is a caller-
// generated uniform-rand `float` buffer (one sample per output cell).
//
// Also ships the `affine_inplace_{f32,f64,bf16,f16}` helper used by:
//   - Phase 4.5: the safe-plan layer to remap a cuRAND uniform-(0, 1]
//     buffer into Uniform(low, high] in place.
//   - Phase 61: Fuel's in-place op family (INPLACE_AFFINE) for
//     Op::AddScalar / Op::MulScalar / weight-decay scaling on
//     contiguous tensors. bf16/f16 take f32 scalars (matches the
//     forward affine_{f16,bf16}_run convention).

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "../include/baracuda_random.cuh"

BARACUDA_KERNELS_BERNOULLI_INSTANTIATE()
BARACUDA_KERNELS_AFFINE_INPLACE_INSTANTIATE(f32, float)
BARACUDA_KERNELS_AFFINE_INPLACE_INSTANTIATE(f64, double)
BARACUDA_KERNELS_AFFINE_INPLACE_F32SCALAR_INSTANTIATE(bf16, __nv_bfloat16)
BARACUDA_KERNELS_AFFINE_INPLACE_F32SCALAR_INSTANTIATE(f16, __half)
