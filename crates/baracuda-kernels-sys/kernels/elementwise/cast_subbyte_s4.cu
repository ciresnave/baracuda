// SPDX-FileCopyrightText: 2026 Eric Evans and the baracuda contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Phase 13.3 — S4 ↔ {i32, i64, f32} cast kernel instantiations.
// Packed-pair nibble storage (low nibble = even index, high nibble =
// odd index). UNPACK direction sign-extends each nibble to s32; PACK
// direction saturates the input to [-8, +7] before nibble-masking.
// `numel` is the element count and must be even.

#include "../include/baracuda_cast_subbyte.cuh"

// ----------------------------------------------------------------------------
// S4 -> { i32, i64, f32 } — UNPACK
// ----------------------------------------------------------------------------
BARACUDA_KERNELS_CAST_INT4_UNPACK_INSTANTIATE(
    cast_s4_i32, int32_t, baracuda::cast_subbyte::s4_nibble_to_t<int32_t>)
BARACUDA_KERNELS_CAST_INT4_UNPACK_INSTANTIATE(
    cast_s4_i64, int64_t, baracuda::cast_subbyte::s4_nibble_to_t<int64_t>)
BARACUDA_KERNELS_CAST_INT4_UNPACK_INSTANTIATE(
    cast_s4_f32, float,   baracuda::cast_subbyte::s4_nibble_to_t<float>)

// ----------------------------------------------------------------------------
// { i32, i64, f32 } -> S4 — PACK (saturates to [-8, +7])
// ----------------------------------------------------------------------------
BARACUDA_KERNELS_CAST_INT4_PACK_INSTANTIATE(
    cast_i32_s4, int32_t, baracuda::cast_subbyte::t_to_s4_nibble<int32_t>)
BARACUDA_KERNELS_CAST_INT4_PACK_INSTANTIATE(
    cast_i64_s4, int64_t, baracuda::cast_subbyte::t_to_s4_nibble<int64_t>)
BARACUDA_KERNELS_CAST_INT4_PACK_INSTANTIATE(
    cast_f32_s4, float,   baracuda::cast_subbyte::t_to_s4_nibble<float>)
