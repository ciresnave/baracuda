// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Epilogue building blocks for the FP8 bespoke kernels.
//
// Re-uses everything from `baracuda_epilogue_int8.cuh`:
//   * the `Activation` enum (`None / Bias / BiasRelu / BiasGelu / BiasSilu`),
//   * the `has_bias<Act>()` compile-time predicate,
//   * the activation primitives (`apply_relu_f32`, `apply_gelu_f32`,
//     `apply_silu_f32`, dispatcher `apply_activation_f32`),
//   * the `bias_to_f32<BiasT>(b)` broadcast (for FP8 today BiasT is
//     always `float`; the int family also surfaces `int32_t`).
//
// Adds the FP8-specific pieces that the int8 family doesn't need:
//   * an `Fp8Encoding` tag (E4M3 vs E5M2),
//   * a `sat_cast_fp8_from_f32<Enc>(x)` trampoline,
//   * a `dequant_fp8_to_f32<Enc>(bits)` round-trip used by the `C` add
//     (the `beta * C[i,j]` term in the epilogue).
//
// The chain in the kernel is exactly
//
//     acc(f32) → alpha * acc
//             + (beta != 0 ? beta * dequant_fp8_to_f32<Enc>(C[i,j]) : 0)
//             + (has_bias<Act>() ? bias_to_f32<BiasT>(bias[j]) : 0)
//     z' = apply_activation_f32<Act>(z)
//     D[i,j] = sat_cast_fp8_from_f32<Enc>(z')

#pragma once

#include <cstdint>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "baracuda_dtype.cuh"
#include "baracuda_epilogue_int8.cuh"

namespace baracuda {

// FP8 encoding selector. Encoded as a compile-time enum so the kernel
// can `if constexpr`-dispatch into the right MMA / cast path with no
// runtime branching.
enum class Fp8Encoding : int {
    E4M3 = 0,
    E5M2 = 1,
};

// Saturating f32 → fp8 cast trampoline. Specialized on `Enc` so the
// underlying NVIDIA helper (`__nv_cvt_float_to_fp8`) lives in one place
// (baracuda_dtype.cuh).
template <Fp8Encoding Enc>
__device__ __forceinline__ uint8_t sat_cast_fp8_from_f32(float x);

template <>
__device__ __forceinline__ uint8_t sat_cast_fp8_from_f32<Fp8Encoding::E4M3>(float x) {
    return sat_cast_f32_to_e4m3(x);
}

template <>
__device__ __forceinline__ uint8_t sat_cast_fp8_from_f32<Fp8Encoding::E5M2>(float x) {
    return sat_cast_f32_to_e5m2(x);
}

// fp8 (raw 8-bit storage) → f32 dequant. Uses
// `__nv_cvt_fp8_to_halfraw(_, __NV_E{4M3,5M2})` then `__half2float`,
// matching the host-side `float8::F8E*::to_f32` round-trip exactly.
template <Fp8Encoding Enc>
__device__ __forceinline__ float dequant_fp8_to_f32(uint8_t bits);

template <>
__device__ __forceinline__ float dequant_fp8_to_f32<Fp8Encoding::E4M3>(uint8_t bits) {
    __half_raw h = __nv_cvt_fp8_to_halfraw(bits, __NV_E4M3);
    return __half2float(*reinterpret_cast<__half*>(&h));
}

template <>
__device__ __forceinline__ float dequant_fp8_to_f32<Fp8Encoding::E5M2>(uint8_t bits) {
    __half_raw h = __nv_cvt_fp8_to_halfraw(bits, __NV_E5M2);
    return __half2float(*reinterpret_cast<__half*>(&h));
}

} // namespace baracuda
