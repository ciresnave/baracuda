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
