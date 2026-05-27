// SPDX-FileCopyrightText: 2026 Eric Evans and the baracuda contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Phase 37 Gap 1b — argmax / argmin axis reductions for integer inputs.
//
// Output dtype is `_i32` (int32_t) or `_i64` (int64_t) — explicit suffix.
// Ties broken by FIRST occurrence (smallest index wins) — matches the
// existing FP fanout (`arg_reduce_fp.cu`).
//
// Coverage: 6 input dtypes × 2 ops × 2 idx dtypes = 24 SKUs.

#include "../include/baracuda_elementwise.cuh"

#include <cstdint>

namespace baracuda { namespace elementwise {

// Integer argmax/argmin policies. Same shape as the FP `ArgmaxPolicy /
// ArgminPolicy` in `arg_reduce_fp.cu`; templated comparators work
// natively for integer types.
template <typename T>
struct ArgmaxPolicyInt {
    __device__ __forceinline__ bool prefer(
        T new_v, int64_t /*new_i*/, T best_v, int64_t /*best_i*/) const
    {
        return new_v > best_v;
    }
};

template <typename T>
struct ArgminPolicyInt {
    __device__ __forceinline__ bool prefer(
        T new_v, int64_t /*new_i*/, T best_v, int64_t /*best_i*/) const
    {
        return new_v < best_v;
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations — 6 int dtypes × 2 ops × 2 idx dtypes = 24.
// =============================================================================

// -----------------------------------------------------------------------------
// i32 idx output.
// -----------------------------------------------------------------------------
BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmax_u8_i32,  uint8_t,  baracuda::elementwise::ArgmaxPolicyInt<uint8_t>,  int32_t)
BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmin_u8_i32,  uint8_t,  baracuda::elementwise::ArgminPolicyInt<uint8_t>,  int32_t)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmax_i8_i32,  int8_t,   baracuda::elementwise::ArgmaxPolicyInt<int8_t>,   int32_t)
BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmin_i8_i32,  int8_t,   baracuda::elementwise::ArgminPolicyInt<int8_t>,   int32_t)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmax_u32_i32, uint32_t, baracuda::elementwise::ArgmaxPolicyInt<uint32_t>, int32_t)
BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmin_u32_i32, uint32_t, baracuda::elementwise::ArgminPolicyInt<uint32_t>, int32_t)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmax_i16_i32, int16_t,  baracuda::elementwise::ArgmaxPolicyInt<int16_t>,  int32_t)
BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmin_i16_i32, int16_t,  baracuda::elementwise::ArgminPolicyInt<int16_t>,  int32_t)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmax_i32_i32, int32_t,  baracuda::elementwise::ArgmaxPolicyInt<int32_t>,  int32_t)
BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmin_i32_i32, int32_t,  baracuda::elementwise::ArgminPolicyInt<int32_t>,  int32_t)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmax_i64_i32, int64_t,  baracuda::elementwise::ArgmaxPolicyInt<int64_t>,  int32_t)
BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmin_i64_i32, int64_t,  baracuda::elementwise::ArgminPolicyInt<int64_t>,  int32_t)

// -----------------------------------------------------------------------------
// i64 idx output.
// -----------------------------------------------------------------------------
BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmax_u8_i64,  uint8_t,  baracuda::elementwise::ArgmaxPolicyInt<uint8_t>,  int64_t)
BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmin_u8_i64,  uint8_t,  baracuda::elementwise::ArgminPolicyInt<uint8_t>,  int64_t)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmax_i8_i64,  int8_t,   baracuda::elementwise::ArgmaxPolicyInt<int8_t>,   int64_t)
BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmin_i8_i64,  int8_t,   baracuda::elementwise::ArgminPolicyInt<int8_t>,   int64_t)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmax_u32_i64, uint32_t, baracuda::elementwise::ArgmaxPolicyInt<uint32_t>, int64_t)
BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmin_u32_i64, uint32_t, baracuda::elementwise::ArgminPolicyInt<uint32_t>, int64_t)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmax_i16_i64, int16_t,  baracuda::elementwise::ArgmaxPolicyInt<int16_t>,  int64_t)
BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmin_i16_i64, int16_t,  baracuda::elementwise::ArgminPolicyInt<int16_t>,  int64_t)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmax_i32_i64, int32_t,  baracuda::elementwise::ArgmaxPolicyInt<int32_t>,  int64_t)
BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmin_i32_i64, int32_t,  baracuda::elementwise::ArgminPolicyInt<int32_t>,  int64_t)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmax_i64_i64, int64_t,  baracuda::elementwise::ArgmaxPolicyInt<int64_t>,  int64_t)
BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmin_i64_i64, int64_t,  baracuda::elementwise::ArgminPolicyInt<int64_t>,  int64_t)
