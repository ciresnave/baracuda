// SPDX-FileCopyrightText: 2024 Eric Holscher and the candle / fuel-cuda-kernels contributors
// SPDX-FileCopyrightText: 2026 Eric Evans and the baracuda contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Per-dtype instantiations of the affine kernel template.
//
// Vendored / adapted from `fuel-cuda-kernels/src/affine.cu`. Each
// instantiation emits the standard `extern "C"
// baracuda_kernels_affine_<dtype>_run` + `_can_implement` symbols.

#include "../include/baracuda_affine.cuh"

// Same-dtype-compute variants — a / b arrive in scalar type T.
BARACUDA_KERNELS_AFFINE_INSTANTIATE(affine_f32, float)
BARACUDA_KERNELS_AFFINE_INSTANTIATE(affine_f64, double)
BARACUDA_KERNELS_AFFINE_INSTANTIATE(affine_i32, int32_t)
BARACUDA_KERNELS_AFFINE_INSTANTIATE(affine_i64, int64_t)
BARACUDA_KERNELS_AFFINE_INSTANTIATE(affine_u8,  uint8_t)
BARACUDA_KERNELS_AFFINE_INSTANTIATE(affine_i8,  int8_t)

// Half-precision compute-in-f32 / store-as-half variants — a / b
// arrive as `float`. Matches the rest of the elementwise family's
// f32-accumulator precision-guarantee contract.
BARACUDA_KERNELS_AFFINE_INSTANTIATE_F16(affine_f16)
BARACUDA_KERNELS_AFFINE_INSTANTIATE_BF16(affine_bf16)

// =============================================================================
// Strided sibling — Phase 14.1.
// =============================================================================
//
// One thread per output element; thread decomposes its output linear
// index into a multi-coord and dots with the per-axis input / output
// strides to derive source / dest element offsets. Covers Fuel's
// 7-dtype request: f32, f64, f16, bf16, i32, i64, u8.

BARACUDA_KERNELS_AFFINE_STRIDED_INSTANTIATE(affine_f32, float)
BARACUDA_KERNELS_AFFINE_STRIDED_INSTANTIATE(affine_f64, double)
BARACUDA_KERNELS_AFFINE_STRIDED_INSTANTIATE(affine_i32, int32_t)
BARACUDA_KERNELS_AFFINE_STRIDED_INSTANTIATE(affine_i64, int64_t)
BARACUDA_KERNELS_AFFINE_STRIDED_INSTANTIATE(affine_u8,  uint8_t)
BARACUDA_KERNELS_AFFINE_STRIDED_INSTANTIATE_F16(affine_f16)
BARACUDA_KERNELS_AFFINE_STRIDED_INSTANTIATE_BF16(affine_bf16)
