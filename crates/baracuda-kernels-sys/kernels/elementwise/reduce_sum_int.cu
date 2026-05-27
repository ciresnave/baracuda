// SPDX-FileCopyrightText: 2026 Eric Evans and the baracuda contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Phase 37 Gap 1b — axis-sum for integer dtypes (widened accumulator).
//
// `y = sum(x, dim=k)` with keepdim=true. Accumulator is i64 (signed)
// or u64 (unsigned); store-time narrows back to the input dtype with
// wrap-on-overflow. Matches Fuel's CPU reference contract.
//
// Coverage: u8, i8, u32, i16, i32, i64.

#include "../include/baracuda_reduce_int.cuh"

BARACUDA_KERNELS_REDUCE_INT_SUM_INSTANTIATE(reduce_sum_u8,  uint8_t)
BARACUDA_KERNELS_REDUCE_INT_SUM_INSTANTIATE(reduce_sum_i8,  int8_t)
BARACUDA_KERNELS_REDUCE_INT_SUM_INSTANTIATE(reduce_sum_u32, uint32_t)
BARACUDA_KERNELS_REDUCE_INT_SUM_INSTANTIATE(reduce_sum_i16, int16_t)
BARACUDA_KERNELS_REDUCE_INT_SUM_INSTANTIATE(reduce_sum_i32, int32_t)
BARACUDA_KERNELS_REDUCE_INT_SUM_INSTANTIATE(reduce_sum_i64, int64_t)
