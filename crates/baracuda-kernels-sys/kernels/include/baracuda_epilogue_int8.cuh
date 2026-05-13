// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Epilogue building blocks shared across the int8 RRR bespoke kernels.
//
// Today: the bias + activation chain that lives between the int32
// accumulator and the saturating-cast-to-int8 store. The chain is
//
//     z(f32) = alpha * (acc as f32) + beta * (C as f32) + bias_to_f32(bias[j])
//     z'     = activation(z)
//     D[i,j] = sat_cast<{s8,u8}>(z')
//
// activation ∈ {Identity, Bias, BiasRelu, BiasGelu, BiasSilu}; the
// first two skip activation, the last three apply the indicated
// scalar function. `Identity` (a.k.a. `Activation::None`) also skips
// the bias-broadcast — it's a true plain-GEMM epilogue.
//
// Match the CPU reference (crates/baracuda-kernels/tests/int8_rrr_smoke.rs):
//   * gelu = exact erf-based (`0.5 * x * (1 + erf(x / sqrt(2)))`).
//   * silu = `x / (1 + exp(-x))`.
//   * relu = `max(x, 0)`.
// erf / exp via the standard CUDA libdevice intrinsics. Reference uses
// the Abramowitz-Stegun 5-term erf approximation; we use `erff` from
// libdevice — both give ≤ 1.5e-7 relative error, well within int8
// sat-cast tolerance.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace baracuda {

// Activation / bias selector tag. Encoded as a compile-time enum so
// the kernel can `if constexpr`-dispatch into the no-bias / no-activation
// fast paths without any runtime branching.
enum class Activation : int {
    None      = 0,  // Identity — no bias, no activation.
    Bias      = 1,  // Bias broadcast, no activation.
    BiasRelu  = 2,
    BiasGelu  = 3,
    BiasSilu  = 4,
};

template <Activation Act>
__device__ __forceinline__ constexpr bool has_bias() {
    return Act != Activation::None;
}

// Activation primitives. Each one operates on an f32 scalar and returns
// an f32; the caller is responsible for the final sat-cast.

__device__ __forceinline__ float apply_relu_f32(float x) {
    return fmaxf(x, 0.0f);
}

// Exact (erf-based) Gaussian Error Linear Unit, matching PyTorch's
// default `nn.GELU()` and the smoke-test reference.
__device__ __forceinline__ float apply_gelu_f32(float x) {
    // 1 / sqrt(2)
    constexpr float inv_sqrt2 = 0.70710678118654752440f;
    return 0.5f * x * (1.0f + erff(x * inv_sqrt2));
}

// SiLU / Swish: `x * sigmoid(x) = x / (1 + exp(-x))`.
__device__ __forceinline__ float apply_silu_f32(float x) {
    return x / (1.0f + __expf(-x));
}

template <Activation Act>
__device__ __forceinline__ float apply_activation_f32(float x) {
    if constexpr (Act == Activation::BiasRelu) return apply_relu_f32(x);
    else if constexpr (Act == Activation::BiasGelu) return apply_gelu_f32(x);
    else if constexpr (Act == Activation::BiasSilu) return apply_silu_f32(x);
    else return x;
}

// BiasT-agnostic broadcast. The bias vector is length-N (one per
// output column); each output row sees the same bias element. BiasT is
// `float` or `int32_t`; both convert to f32 before the bias-add.
template <typename BiasT>
__device__ __forceinline__ float bias_to_f32(BiasT b) {
    return static_cast<float>(b);
}

} // namespace baracuda
