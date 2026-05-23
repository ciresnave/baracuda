// baracuda-kernels Phase 12.1 — elementwise integer-power-of-x (PowI).
//
// Forward: `y = x^n` where `n` is a runtime integer exponent threaded
// through the param ABI via `p0` (cast to `int` at kernel entry). The
// second parameter slot `p1` is ignored — kept for ABI parity with the
// rest of the `unary_param_*` family.
//
// We use power-by-squaring (O(log n) multiplies) rather than going
// through `__powf(x, (float)n)` because:
//   1. Integer-exponent power is well-defined for negative `x` (real
//      `pow(-1.5, 2) = 2.25`); the libm `pow` path returns NaN for
//      negative-base / non-integer-exponent pairs and even for negative
//      base with integer exponent it routes through `__expf(n*__logf(x))`
//      which NaNs.
//   2. No `__expf`/`__logf` round-trip — the result is the literal
//      product of |n|-1 multiplies of `x`. For `n = 2` that's exactly
//      `x*x` (the same as `Square`), bit-identical.
//   3. Faster for the typical |n| ≤ 8 cases (one or two multiplies).
//
// For `n < 0`: compute `pow_i(1/x, |n|)`. For `n == 0`: return `T(1)`
// regardless of `x` (including `x == 0` — matches `f32::powi` and IEEE
// 754 `pow(±0, 0) = 1`).
//
// f16 / bf16 do the entire product chain in f32, then cast once at the
// end — same precision convention as the rest of the unary_param family.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

// Power-by-squaring at the math-precision type. Always f32 for f16/bf16,
// native otherwise.
template <typename Acc>
__device__ __forceinline__ Acc powi_kernel(Acc x, int n) {
    if (n == 0) return Acc(1);
    int absn = n < 0 ? -n : n;
    Acc base = (n < 0) ? (Acc(1) / x) : x;
    Acc result = Acc(1);
    while (absn > 0) {
        if (absn & 1) result = result * base;
        base = base * base;
        absn >>= 1;
    }
    return result;
}

template <typename T>
struct PowIFunctor {
    __device__ __forceinline__ T operator()(T x, float p0, float /*p1*/) const {
        // Default specialization unused — explicit specs below.
        int n = static_cast<int>(p0);
        return powi_kernel<T>(x, n);
    }
};

template <>
struct PowIFunctor<float> {
    __device__ __forceinline__ float operator()(float x, float p0, float /*p1*/) const {
        int n = static_cast<int>(p0);
        return powi_kernel<float>(x, n);
    }
};

template <>
struct PowIFunctor<double> {
    __device__ __forceinline__ double operator()(double x, float p0, float /*p1*/) const {
        int n = static_cast<int>(p0);
        return powi_kernel<double>(x, n);
    }
};

template <>
struct PowIFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x, float p0, float /*p1*/) const {
        int n = static_cast<int>(p0);
        float fx = __half2float(x);
        float fy = powi_kernel<float>(fx, n);
        return __float2half(fy);
    }
};

template <>
struct PowIFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x, float p0, float /*p1*/) const {
        int n = static_cast<int>(p0);
        float fx = __bfloat162float(x);
        float fy = powi_kernel<float>(fx, n);
        return __float2bfloat16(fy);
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_PARAM_INSTANTIATE(
    unary_powi_f32,
    float,
    baracuda::elementwise::PowIFunctor<float>)

BARACUDA_KERNELS_UNARY_PARAM_INSTANTIATE(
    unary_powi_f16,
    __half,
    baracuda::elementwise::PowIFunctor<__half>)

BARACUDA_KERNELS_UNARY_PARAM_INSTANTIATE(
    unary_powi_bf16,
    __nv_bfloat16,
    baracuda::elementwise::PowIFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_PARAM_INSTANTIATE(
    unary_powi_f64,
    double,
    baracuda::elementwise::PowIFunctor<double>)

// Phase 14.2 — strided sibling. Emits `*_strided_run` for the same
// (op, dtype) matrix as the contig path above.
BARACUDA_KERNELS_UNARY_PARAM_INSTANTIATE_STRIDED(
    unary_powi_f32,
    float,
    baracuda::elementwise::PowIFunctor<float>)

BARACUDA_KERNELS_UNARY_PARAM_INSTANTIATE_STRIDED(
    unary_powi_f16,
    __half,
    baracuda::elementwise::PowIFunctor<__half>)

BARACUDA_KERNELS_UNARY_PARAM_INSTANTIATE_STRIDED(
    unary_powi_bf16,
    __nv_bfloat16,
    baracuda::elementwise::PowIFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_PARAM_INSTANTIATE_STRIDED(
    unary_powi_f64,
    double,
    baracuda::elementwise::PowIFunctor<double>)
