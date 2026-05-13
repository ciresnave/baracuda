// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Dtype helpers shared across baracuda-kernels-sys kernels.
//
// Today: just int8 saturating-cast helpers (host-style rounded saturate
// to S8 / U8) used by the int-GEMM epilogues. The header grows as new
// dtype families land (FP8, int4, bin in Phase 2; bfloat16 helpers in
// Phase 3+).

#pragma once

#include <cstdint>
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

} // namespace baracuda
