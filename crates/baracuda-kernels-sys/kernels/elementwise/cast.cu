// SPDX-FileCopyrightText: 2024 Eric Holscher and the candle / fuel-cuda-kernels contributors
// SPDX-FileCopyrightText: 2026 Eric Evans and the baracuda contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Per-dtype-pair instantiations of the cast kernel template.
//
// Vendored / adapted from `fuel-cuda-kernels/src/cast.cu`. The SKU
// matrix below covers the same cross-dtype pairs Fuel exposes for the
// non-FP8 dtypes (FP8 endpoints land in a future Phase 2-FP8 fanout).
// Each pair emits the standard `extern "C"
// baracuda_kernels_cast_<sin>_<sout>_run` + `_can_implement` symbols.
//
// Phase 31 (Fuel Phase 6c.2 storage.rs unblock): u32 + i16 added as
// new dtypes — total matrix is now 10×10 (was 8×8). The 4 self-pairs
// (u32→u32, i16→i16) live on the diagonal; the 36 new off-diagonal
// cells are split as 18 "outbound to {u32, i16}" + 18 "inbound from
// {u32, i16}" + 2 new cross pairs (u32↔i16). The `cast_value<>`
// template already handles i16 / uint32_t endpoints via the half-
// precision detour specs in baracuda_cast.cuh — no kernel changes,
// just new instantiations.

#include "../include/baracuda_cast.cuh"

// ----------------------------------------------------------------------------
// f32 -> *
// ----------------------------------------------------------------------------
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_f32_f32,  float,    float)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_f32_f64,  float,    double)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_f32_f16,  float,    __half)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_f32_bf16, float,    __nv_bfloat16)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_f32_i32,  float,    int32_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_f32_i64,  float,    int64_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_f32_u8,   float,    uint8_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_f32_i8,   float,    int8_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_f32_u32,  float,    uint32_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_f32_i16,  float,    int16_t)

// ----------------------------------------------------------------------------
// f64 -> *
// ----------------------------------------------------------------------------
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_f64_f32,  double,   float)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_f64_f64,  double,   double)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_f64_f16,  double,   __half)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_f64_bf16, double,   __nv_bfloat16)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_f64_i32,  double,   int32_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_f64_i64,  double,   int64_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_f64_u8,   double,   uint8_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_f64_i8,   double,   int8_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_f64_u32,  double,   uint32_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_f64_i16,  double,   int16_t)

// ----------------------------------------------------------------------------
// f16 -> *
// ----------------------------------------------------------------------------
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_f16_f32,  __half,   float)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_f16_f64,  __half,   double)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_f16_f16,  __half,   __half)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_f16_bf16, __half,   __nv_bfloat16)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_f16_i32,  __half,   int32_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_f16_i64,  __half,   int64_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_f16_u8,   __half,   uint8_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_f16_i8,   __half,   int8_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_f16_u32,  __half,   uint32_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_f16_i16,  __half,   int16_t)

// ----------------------------------------------------------------------------
// bf16 -> *
// ----------------------------------------------------------------------------
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_bf16_f32,  __nv_bfloat16, float)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_bf16_f64,  __nv_bfloat16, double)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_bf16_f16,  __nv_bfloat16, __half)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_bf16_bf16, __nv_bfloat16, __nv_bfloat16)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_bf16_i32,  __nv_bfloat16, int32_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_bf16_i64,  __nv_bfloat16, int64_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_bf16_u8,   __nv_bfloat16, uint8_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_bf16_i8,   __nv_bfloat16, int8_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_bf16_u32,  __nv_bfloat16, uint32_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_bf16_i16,  __nv_bfloat16, int16_t)

// ----------------------------------------------------------------------------
// i32 -> *
// ----------------------------------------------------------------------------
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i32_f32,  int32_t,  float)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i32_f64,  int32_t,  double)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i32_f16,  int32_t,  __half)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i32_bf16, int32_t,  __nv_bfloat16)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i32_i32,  int32_t,  int32_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i32_i64,  int32_t,  int64_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i32_u8,   int32_t,  uint8_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i32_i8,   int32_t,  int8_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i32_u32,  int32_t,  uint32_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i32_i16,  int32_t,  int16_t)

// ----------------------------------------------------------------------------
// i64 -> *
// ----------------------------------------------------------------------------
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i64_f32,  int64_t,  float)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i64_f64,  int64_t,  double)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i64_f16,  int64_t,  __half)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i64_bf16, int64_t,  __nv_bfloat16)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i64_i32,  int64_t,  int32_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i64_i64,  int64_t,  int64_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i64_u8,   int64_t,  uint8_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i64_i8,   int64_t,  int8_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i64_u32,  int64_t,  uint32_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i64_i16,  int64_t,  int16_t)

// ----------------------------------------------------------------------------
// u8 -> *
// ----------------------------------------------------------------------------
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_u8_f32,  uint8_t, float)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_u8_f64,  uint8_t, double)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_u8_f16,  uint8_t, __half)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_u8_bf16, uint8_t, __nv_bfloat16)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_u8_i32,  uint8_t, int32_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_u8_i64,  uint8_t, int64_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_u8_u8,   uint8_t, uint8_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_u8_i8,   uint8_t, int8_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_u8_u32,  uint8_t, uint32_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_u8_i16,  uint8_t, int16_t)

// ----------------------------------------------------------------------------
// i8 -> *
// ----------------------------------------------------------------------------
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i8_f32,  int8_t, float)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i8_f64,  int8_t, double)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i8_f16,  int8_t, __half)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i8_bf16, int8_t, __nv_bfloat16)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i8_i32,  int8_t, int32_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i8_i64,  int8_t, int64_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i8_u8,   int8_t, uint8_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i8_i8,   int8_t, int8_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i8_u32,  int8_t, uint32_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i8_i16,  int8_t, int16_t)

// ----------------------------------------------------------------------------
// Phase 31 — u32 -> *
// ----------------------------------------------------------------------------
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_u32_f32,  uint32_t, float)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_u32_f64,  uint32_t, double)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_u32_f16,  uint32_t, __half)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_u32_bf16, uint32_t, __nv_bfloat16)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_u32_i32,  uint32_t, int32_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_u32_i64,  uint32_t, int64_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_u32_u8,   uint32_t, uint8_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_u32_i8,   uint32_t, int8_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_u32_u32,  uint32_t, uint32_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_u32_i16,  uint32_t, int16_t)

// ----------------------------------------------------------------------------
// Phase 31 — i16 -> *
// ----------------------------------------------------------------------------
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i16_f32,  int16_t, float)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i16_f64,  int16_t, double)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i16_f16,  int16_t, __half)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i16_bf16, int16_t, __nv_bfloat16)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i16_i32,  int16_t, int32_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i16_i64,  int16_t, int64_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i16_u8,   int16_t, uint8_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i16_i8,   int16_t, int8_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i16_u32,  int16_t, uint32_t)
BARACUDA_KERNELS_CAST_INSTANTIATE(cast_i16_i16,  int16_t, int16_t)
