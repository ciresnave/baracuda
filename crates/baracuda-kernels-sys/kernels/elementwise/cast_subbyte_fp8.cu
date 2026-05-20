// SPDX-FileCopyrightText: 2026 Eric Evans and the baracuda contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Phase 13.3 — Fp8E4M3 / Fp8E5M2 ↔ {f32, f16, bf16} cast kernel
// instantiations. All conversions route through f32 via NVIDIA's
// `__nv_cvt_fp8_to_halfraw` / `__nv_cvt_float_to_fp8` intrinsics
// (saturating semantics on the wide → narrow direction).

#include "../include/baracuda_cast_subbyte.cuh"

// ----------------------------------------------------------------------------
// Fp8E4M3 -> { f32, f16, bf16 }
// ----------------------------------------------------------------------------
BARACUDA_KERNELS_CAST_FP8_TO_T_INSTANTIATE(cast_fp8e4m3_f32,  float,         e4m3_to_f32)
BARACUDA_KERNELS_CAST_FP8_TO_T_INSTANTIATE(cast_fp8e4m3_f16,  __half,        e4m3_to_f32)
BARACUDA_KERNELS_CAST_FP8_TO_T_INSTANTIATE(cast_fp8e4m3_bf16, __nv_bfloat16, e4m3_to_f32)

// ----------------------------------------------------------------------------
// { f32, f16, bf16 } -> Fp8E4M3
// ----------------------------------------------------------------------------
BARACUDA_KERNELS_CAST_T_TO_FP8_INSTANTIATE(cast_f32_fp8e4m3,  float,         f32_to_e4m3)
BARACUDA_KERNELS_CAST_T_TO_FP8_INSTANTIATE(cast_f16_fp8e4m3,  __half,        f32_to_e4m3)
BARACUDA_KERNELS_CAST_T_TO_FP8_INSTANTIATE(cast_bf16_fp8e4m3, __nv_bfloat16, f32_to_e4m3)

// ----------------------------------------------------------------------------
// Fp8E5M2 -> { f32, f16, bf16 }
// ----------------------------------------------------------------------------
BARACUDA_KERNELS_CAST_FP8_TO_T_INSTANTIATE(cast_fp8e5m2_f32,  float,         e5m2_to_f32)
BARACUDA_KERNELS_CAST_FP8_TO_T_INSTANTIATE(cast_fp8e5m2_f16,  __half,        e5m2_to_f32)
BARACUDA_KERNELS_CAST_FP8_TO_T_INSTANTIATE(cast_fp8e5m2_bf16, __nv_bfloat16, e5m2_to_f32)

// ----------------------------------------------------------------------------
// { f32, f16, bf16 } -> Fp8E5M2
// ----------------------------------------------------------------------------
BARACUDA_KERNELS_CAST_T_TO_FP8_INSTANTIATE(cast_f32_fp8e5m2,  float,         f32_to_e5m2)
BARACUDA_KERNELS_CAST_T_TO_FP8_INSTANTIATE(cast_f16_fp8e5m2,  __half,        f32_to_e5m2)
BARACUDA_KERNELS_CAST_T_TO_FP8_INSTANTIATE(cast_bf16_fp8e5m2, __nv_bfloat16, f32_to_e5m2)
