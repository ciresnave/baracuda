// baracuda-kernels Phase 3 unary fanout: elementwise sign for FP types.
//
// Implements `y = sign(x)`:
//   sign(x) = +1 if x > 0
//             0 if x == 0
//            -1 if x < 0
//
// f32 / f64 use direct piecewise comparison. f16 / bf16 need explicit
// specialization because `T(1) / T(-1) / T(0)` literals don't generalize
// across `__half` — the half-precision specializations precompute the
// three constants via `__float2half` / `__float2bfloat16` and select via
// host-style comparison (PTX emits proper half-precision compares).

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct SignFunctor {
    __device__ __forceinline__ T operator()(T x) const {
        return (x > T(0)) ? T(1) : ((x < T(0)) ? T(-1) : T(0));
    }
};

template <>
struct SignFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const {
        return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
    }
};

template <>
struct SignFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const {
        return (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0);
    }
};

template <>
struct SignFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        const __half pos = __float2half(1.0f);
        const __half neg = __float2half(-1.0f);
        const __half zero = __float2half(0.0f);
        // `__hgt` / `__hlt` are the canonical half-precision compares.
        if (__hgt(x, zero)) return pos;
        if (__hlt(x, zero)) return neg;
        return zero;
    }
};

template <>
struct SignFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        const __nv_bfloat16 pos = __float2bfloat16(1.0f);
        const __nv_bfloat16 neg = __float2bfloat16(-1.0f);
        const __nv_bfloat16 zero = __float2bfloat16(0.0f);
        if (__hgt(x, zero)) return pos;
        if (__hlt(x, zero)) return neg;
        return zero;
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_sign_f32,
    float,
    baracuda::elementwise::SignFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_sign_f32,
    float,
    baracuda::elementwise::SignFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_sign_f16,
    __half,
    baracuda::elementwise::SignFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_sign_f16,
    __half,
    baracuda::elementwise::SignFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_sign_bf16,
    __nv_bfloat16,
    baracuda::elementwise::SignFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_sign_bf16,
    __nv_bfloat16,
    baracuda::elementwise::SignFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_sign_f64,
    double,
    baracuda::elementwise::SignFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_sign_f64,
    double,
    baracuda::elementwise::SignFunctor<double>)
