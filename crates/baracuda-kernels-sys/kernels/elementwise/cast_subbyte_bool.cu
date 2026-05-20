// SPDX-FileCopyrightText: 2026 Eric Evans and the baracuda contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Phase 13.3 — Bool ↔ T cast kernel instantiations.
//
// Bool storage is one byte per element. Truthiness convention:
//   - Bool → T: 0 → 0.0/0, any non-zero byte → 1.0/1
//   - T → Bool: x != 0 → 1, x == 0 → 0
// Output of T → Bool is always strictly 0 or 1 (not a copy of the
// source non-zero byte).

#include "../include/baracuda_cast_subbyte.cuh"

// ----------------------------------------------------------------------------
// Bool -> { i32, i64, f32, f16, bf16 }
// ----------------------------------------------------------------------------
BARACUDA_KERNELS_CAST_BOOL_TO_T_INSTANTIATE(cast_bool_i32,  int32_t)
BARACUDA_KERNELS_CAST_BOOL_TO_T_INSTANTIATE(cast_bool_i64,  int64_t)
BARACUDA_KERNELS_CAST_BOOL_TO_T_INSTANTIATE(cast_bool_f32,  float)
BARACUDA_KERNELS_CAST_BOOL_TO_T_INSTANTIATE(cast_bool_f16,  __half)
BARACUDA_KERNELS_CAST_BOOL_TO_T_INSTANTIATE(cast_bool_bf16, __nv_bfloat16)

// ----------------------------------------------------------------------------
// { i32, i64, f32, f16, bf16 } -> Bool
// ----------------------------------------------------------------------------
BARACUDA_KERNELS_CAST_T_TO_BOOL_INSTANTIATE(cast_i32_bool,  int32_t)
BARACUDA_KERNELS_CAST_T_TO_BOOL_INSTANTIATE(cast_i64_bool,  int64_t)
BARACUDA_KERNELS_CAST_T_TO_BOOL_INSTANTIATE(cast_f32_bool,  float)
BARACUDA_KERNELS_CAST_T_TO_BOOL_INSTANTIATE(cast_f16_bool,  __half)
BARACUDA_KERNELS_CAST_T_TO_BOOL_INSTANTIATE(cast_bf16_bool, __nv_bfloat16)
