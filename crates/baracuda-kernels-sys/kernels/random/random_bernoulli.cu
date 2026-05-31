// baracuda-kernels Phase 4.5 + Phase 61 + Phase 62: bernoulli + affine-inplace helpers.
//
// `y[i] = (rand[i] < p) ? 1 : 0`, Bool output. `rand` is a caller-
// generated uniform-rand `float` buffer (one sample per output cell).
//
// Also ships the `affine_inplace_<dtype>{,_strided}` helper used by:
//   - Phase 4.5: the safe-plan layer to remap a cuRAND uniform-(0, 1]
//     buffer into Uniform(low, high] in place (contig f32 / f64 only).
//   - Phase 61: Fuel's in-place op family (INPLACE_AFFINE) for
//     Op::AddScalar / Op::MulScalar / weight-decay scaling on
//     contiguous tensors. bf16/f16 take f32 scalars (matches the
//     forward affine_{f16,bf16}_run convention).
//   - Phase 62: int dtype contig backfill (i32/i64/u8/i8 — matching
//     the forward affine_<int>_run dtype set) + strided in-place
//     variants for the full 7-dtype set matching forward
//     affine_<dtype>_strided_run. Strided in-place requires the
//     caller to enforce `stride_x == stride_y` if same-pointer reuse
//     is intended (Rust-side helper: `kernel_types::strides_equal`).

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "../include/baracuda_random.cuh"

BARACUDA_KERNELS_BERNOULLI_INSTANTIATE()

// ----- Contig in-place — alpha.55 fp baseline + Phase 61 half-precision -----
BARACUDA_KERNELS_AFFINE_INPLACE_INSTANTIATE(f32, float)
BARACUDA_KERNELS_AFFINE_INPLACE_INSTANTIATE(f64, double)
BARACUDA_KERNELS_AFFINE_INPLACE_F32SCALAR_INSTANTIATE(bf16, __nv_bfloat16)
BARACUDA_KERNELS_AFFINE_INPLACE_F32SCALAR_INSTANTIATE(f16, __half)

// ----- Phase 62 — contig in-place int dtype backfill -----
// Matches the forward `affine_<int>_run` dtype set (i32 / i64 / u8 / i8).
BARACUDA_KERNELS_AFFINE_INPLACE_INSTANTIATE(i32, int32_t)
BARACUDA_KERNELS_AFFINE_INPLACE_INSTANTIATE(i64, int64_t)
BARACUDA_KERNELS_AFFINE_INPLACE_INSTANTIATE(u8,  uint8_t)
BARACUDA_KERNELS_AFFINE_INPLACE_INSTANTIATE(i8,  int8_t)

// ----- Phase 62 — strided in-place, full dtype set matching forward strided -----
// Forward `affine_<dtype>_strided_run` ships f32 / f64 / i32 / i64 / u8 / f16 / bf16
// (no i8 in strided forward; we match that 7-dtype set).
BARACUDA_KERNELS_AFFINE_INPLACE_STRIDED_INSTANTIATE(f32, float)
BARACUDA_KERNELS_AFFINE_INPLACE_STRIDED_INSTANTIATE(f64, double)
BARACUDA_KERNELS_AFFINE_INPLACE_STRIDED_INSTANTIATE(i32, int32_t)
BARACUDA_KERNELS_AFFINE_INPLACE_STRIDED_INSTANTIATE(i64, int64_t)
BARACUDA_KERNELS_AFFINE_INPLACE_STRIDED_INSTANTIATE(u8,  uint8_t)
BARACUDA_KERNELS_AFFINE_INPLACE_STRIDED_F32SCALAR_INSTANTIATE(bf16, __nv_bfloat16)
BARACUDA_KERNELS_AFFINE_INPLACE_STRIDED_F32SCALAR_INSTANTIATE(f16, __half)
