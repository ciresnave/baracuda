// SPDX-FileCopyrightText: 2026 Eric Evans and the baracuda contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Phase 37 Gap 1b — axis-product for integer dtypes (widened accumulator).
//
// `y = prod(x, dim=k)` with keepdim=true. Accumulator is i64 (signed)
// or u64 (unsigned); store-time narrows back to the input dtype with
// wrap-on-overflow. Matches Fuel's CPU reference contract.
//
// Coverage: u8, i8, u32, i16, i32, i64.

#include "../include/baracuda_reduce_int.cuh"

BARACUDA_KERNELS_REDUCE_INT_PROD_INSTANTIATE(reduce_prod_u8,  uint8_t)
BARACUDA_KERNELS_REDUCE_INT_PROD_INSTANTIATE(reduce_prod_i8,  int8_t)
BARACUDA_KERNELS_REDUCE_INT_PROD_INSTANTIATE(reduce_prod_u32, uint32_t)
BARACUDA_KERNELS_REDUCE_INT_PROD_INSTANTIATE(reduce_prod_i16, int16_t)
BARACUDA_KERNELS_REDUCE_INT_PROD_INSTANTIATE(reduce_prod_i32, int32_t)
BARACUDA_KERNELS_REDUCE_INT_PROD_INSTANTIATE(reduce_prod_i64, int64_t)
