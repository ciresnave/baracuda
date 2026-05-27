// SPDX-FileCopyrightText: 2024 Eric Holscher and the candle / fuel-cuda-kernels contributors
// SPDX-FileCopyrightText: 2026 Eric Evans and the baracuda contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Per-dtype instantiations of the fill kernel template.
//
// Vendored / adapted from `fuel-cuda-kernels/src/fill.cu`. Each
// instantiation emits the standard `extern "C"
// baracuda_kernels_fill_<dtype>_run` + `_can_implement` symbols.

#include "../include/baracuda_fill.cuh"

BARACUDA_KERNELS_FILL_INSTANTIATE(fill_f32, float)
BARACUDA_KERNELS_FILL_INSTANTIATE(fill_f64, double)
BARACUDA_KERNELS_FILL_INSTANTIATE(fill_i32, int32_t)
BARACUDA_KERNELS_FILL_INSTANTIATE(fill_i64, int64_t)
BARACUDA_KERNELS_FILL_INSTANTIATE(fill_u8,  uint8_t)
BARACUDA_KERNELS_FILL_INSTANTIATE(fill_i8,  int8_t)

// f16 / bf16 transport `value` as a raw uint16 bit pattern over the FFI;
// the device-side launcher memcpy's it into the half-precision wrapper
// type before launching the kernel.
BARACUDA_KERNELS_FILL_INSTANTIATE_HALF(fill_f16,  __half)
BARACUDA_KERNELS_FILL_INSTANTIATE_HALF(fill_bf16, __nv_bfloat16)

// Phase 36 (Fuel ask Gap 4) — additional integer dtypes + FP8 E4M3.
// `fill_u32` / `fill_i16` follow the standard `T value` ABI; `fill_fp8e4m3`
// uses raw `uint8_t` storage (FP8 E4M3 is bit-identical to `uint8_t` at
// the storage layer — the caller passes the raw 8-bit encoded value
// directly).
BARACUDA_KERNELS_FILL_INSTANTIATE(fill_u32,     uint32_t)
BARACUDA_KERNELS_FILL_INSTANTIATE(fill_i16,     int16_t)
BARACUDA_KERNELS_FILL_INSTANTIATE(fill_fp8e4m3, uint8_t)

// Strided fill — one launcher per dtype. Writes `y[lin]` where
// `lin = Σ_axis coord[axis] * stride_y[axis]`. Shape and stride arrays
// live on the HOST (the launcher copies them into a small kernel-side
// param block). `numel` must equal the product of `shape[0..rank]`.
BARACUDA_KERNELS_FILL_STRIDED_INSTANTIATE(fill_f32_strided, float)
BARACUDA_KERNELS_FILL_STRIDED_INSTANTIATE(fill_f64_strided, double)
BARACUDA_KERNELS_FILL_STRIDED_INSTANTIATE(fill_i32_strided, int32_t)
BARACUDA_KERNELS_FILL_STRIDED_INSTANTIATE(fill_i64_strided, int64_t)
BARACUDA_KERNELS_FILL_STRIDED_INSTANTIATE(fill_u8_strided,  uint8_t)
BARACUDA_KERNELS_FILL_STRIDED_INSTANTIATE(fill_i8_strided,  int8_t)
BARACUDA_KERNELS_FILL_STRIDED_INSTANTIATE(fill_u32_strided, uint32_t)
BARACUDA_KERNELS_FILL_STRIDED_INSTANTIATE(fill_i16_strided, int16_t)
BARACUDA_KERNELS_FILL_STRIDED_INSTANTIATE(fill_fp8e4m3_strided, uint8_t)
BARACUDA_KERNELS_FILL_STRIDED_INSTANTIATE_HALF(fill_f16_strided,  __half)
BARACUDA_KERNELS_FILL_STRIDED_INSTANTIATE_HALF(fill_bf16_strided, __nv_bfloat16)
