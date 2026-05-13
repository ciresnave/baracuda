// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Dtype helpers shared across baracuda-kernels-sys kernels.
//
// Today: int8 saturating-cast helpers (host-style rounded saturate to
// S8 / U8) used by the int-GEMM epilogues, plus the FP8 E4M3 sat-cast
// used by Phase 2 FP8 GEMM. The header grows as new dtype families
// land (E5M2 + int4 + bin in later Phase 2 sessions; bfloat16 helpers
// in Phase 3+).

#pragma once

#include <cstdint>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace baracuda {

// Saturating round-to-nearest-int cast from f32 to s8. Matches the
// CUTLASS `NumericConverterClamp<int8_t, float, RoundStyle::NearestEven>`
// semantics: round-half-to-even via `__float2int_rn` then clamp to
// [-128, 127].
__device__ __forceinline__ int8_t sat_cast_f32_to_s8(float x) {
    int32_t r = __float2int_rn(x);
    r = max(r, -128);
    r = min(r, 127);
    return static_cast<int8_t>(r);
}

// Saturating round-to-nearest-int cast from f32 to u8.
__device__ __forceinline__ uint8_t sat_cast_f32_to_u8(float x) {
    int32_t r = __float2int_rn(x);
    r = max(r, 0);
    r = min(r, 255);
    return static_cast<uint8_t>(r);
}

// Saturating round-half-to-even cast from f32 to FP8 E4M3. Returns the
// raw 8-bit storage (`__nv_fp8_storage_t` = `unsigned char`). Matches
// NVIDIA's `__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3)`:
// clamps `|x|` to the E4M3 max-finite (448.0) instead of producing
// infinities — appropriate because E4M3 has no infinity encoding.
__device__ __forceinline__ uint8_t sat_cast_f32_to_e4m3(float x) {
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(
        x, __NV_SATFINITE, __NV_E4M3));
}

// Saturating round-half-to-even cast from f32 to FP8 E5M2. Returns the
// raw 8-bit storage. Matches NVIDIA's
// `__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E5M2)`: clamps `|x|`
// to the E5M2 max-finite (57344.0). Unlike E4M3, E5M2 has IEEE-style
// infinity / NaN encodings — `SATFINITE` rounds those toward
// max-finite rather than letting them through.
__device__ __forceinline__ uint8_t sat_cast_f32_to_e5m2(float x) {
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(
        x, __NV_SATFINITE, __NV_E5M2));
}

// ============================================================================
// int4 helpers — packed-pair nibble storage
// ============================================================================
//
// One byte holds two int4 values: low nibble = element at even index
// along the leading axis, high nibble = element at odd index. The on-
// host wrappers `baracuda_kernels_types::S4` / `::U4` use this same
// packing convention so device storage is bit-compatible with host
// `Vec<u8>` packed inputs.

// Sign-extend the low 4 bits of `nibble` to an int32 in [-8, +7].
__device__ __forceinline__ int32_t s4_nibble_to_s32(uint8_t nibble) {
    return (int32_t)((int8_t)(nibble << 4) >> 4);
}

// Zero-extend the low 4 bits of `nibble` to an int32 in [0, 15].
__device__ __forceinline__ int32_t u4_nibble_to_s32(uint8_t nibble) {
    return (int32_t)(nibble & 0x0F);
}

// Unpack the two int4 values stored in `byte` (low nibble + high nibble)
// into a [low, high] pair of s32 values.
__device__ __forceinline__ void unpack_s4_byte(
    uint8_t byte, int32_t &lo, int32_t &hi)
{
    lo = s4_nibble_to_s32(byte & 0x0F);
    hi = s4_nibble_to_s32((byte >> 4) & 0x0F);
}

__device__ __forceinline__ void unpack_u4_byte(
    uint8_t byte, int32_t &lo, int32_t &hi)
{
    lo = u4_nibble_to_s32(byte & 0x0F);
    hi = u4_nibble_to_s32((byte >> 4) & 0x0F);
}

// Saturating round-to-nearest-int cast from f32 to s4 nibble. Matches
// the int8 family's `NumericConverterClamp` semantics: round-half-to-
// even via `__float2int_rn` then clamp to `[-8, +7]`. Returns the raw
// 4-bit value in the low nibble (high nibble is zero).
__device__ __forceinline__ uint8_t sat_cast_f32_to_s4(float x) {
    int32_t r = __float2int_rn(x);
    r = max(r, -8);
    r = min(r, 7);
    return static_cast<uint8_t>(r & 0x0F);
}

// Saturating round-to-nearest-int cast from f32 to u4 nibble. Returns
// the raw 4-bit value in the low nibble.
__device__ __forceinline__ uint8_t sat_cast_f32_to_u4(float x) {
    int32_t r = __float2int_rn(x);
    r = max(r, 0);
    r = min(r, 15);
    return static_cast<uint8_t>(r & 0x0F);
}

// Pack two nibbles `lo` and `hi` (each masked to 4 bits) into one
// packed-pair byte. Used by the epilogue to write a pair of adjacent
// output cells in one store.
__device__ __forceinline__ uint8_t pack_int4_pair(
    uint8_t lo, uint8_t hi)
{
    return (uint8_t)((lo & 0x0F) | ((hi & 0x0F) << 4));
}

} // namespace baracuda
