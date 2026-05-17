// baracuda-kernels Phase 3 unary fanout: elementwise softshrink for FP types.
//
// Implements `y = x - λ if x > λ; x + λ if x < -λ; else 0`, with λ
// hardcoded to 0.5 (PyTorch default `nn.Softshrink(lambd=0.5)`). When
// the parameterized-unary plan ships, this kernel is re-emitted with λ
// as a runtime parameter. Pure piecewise arithmetic — no
// transcendentals. f16 / bf16 use the f32-detour pattern.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct SoftshrinkFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct SoftshrinkFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const {
        if (x > 0.5f)  return x - 0.5f;
        if (x < -0.5f) return x + 0.5f;
        return 0.0f;
    }
};

template <>
struct SoftshrinkFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const {
        if (x > 0.5)  return x - 0.5;
        if (x < -0.5) return x + 0.5;
        return 0.0;
    }
};

template <>
struct SoftshrinkFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        float f = __half2float(x);
        float y;
        if (f > 0.5f)       y = f - 0.5f;
        else if (f < -0.5f) y = f + 0.5f;
        else                y = 0.0f;
        return __float2half(y);
    }
};

template <>
struct SoftshrinkFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float f = __bfloat162float(x);
        float y;
        if (f > 0.5f)       y = f - 0.5f;
        else if (f < -0.5f) y = f + 0.5f;
        else                y = 0.0f;
        return __float2bfloat16(y);
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_softshrink_f32,
    float,
    baracuda::elementwise::SoftshrinkFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_softshrink_f32,
    float,
    baracuda::elementwise::SoftshrinkFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_softshrink_f16,
    __half,
    baracuda::elementwise::SoftshrinkFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_softshrink_f16,
    __half,
    baracuda::elementwise::SoftshrinkFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_softshrink_bf16,
    __nv_bfloat16,
    baracuda::elementwise::SoftshrinkFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_softshrink_bf16,
    __nv_bfloat16,
    baracuda::elementwise::SoftshrinkFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_softshrink_f64,
    double,
    baracuda::elementwise::SoftshrinkFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_softshrink_f64,
    double,
    baracuda::elementwise::SoftshrinkFunctor<double>)
