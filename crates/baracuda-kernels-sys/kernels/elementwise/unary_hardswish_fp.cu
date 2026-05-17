// baracuda-kernels Phase 3 unary fanout: elementwise Hardswish for FP types.
//
// Implements `y = x * relu6(x + 3) / 6` where `relu6(z) = min(max(z, 0), 6)`.
// Piecewise linear approximation of swish/silu. f32 uses `fmaxf`/`fminf`;
// f64 uses `fmax`/`fmin`. f16 / bf16 use the f32 detour.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct HardswishFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct HardswishFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const {
        float z = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
        return x * z * (1.0f / 6.0f);
    }
};

template <>
struct HardswishFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const {
        double z = fmin(fmax(x + 3.0, 0.0), 6.0);
        return x * z * (1.0 / 6.0);
    }
};

template <>
struct HardswishFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        float f = __half2float(x);
        float z = fminf(fmaxf(f + 3.0f, 0.0f), 6.0f);
        return __float2half(f * z * (1.0f / 6.0f));
    }
};

template <>
struct HardswishFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float f = __bfloat162float(x);
        float z = fminf(fmaxf(f + 3.0f, 0.0f), 6.0f);
        return __float2bfloat16(f * z * (1.0f / 6.0f));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_hardswish_f32,
    float,
    baracuda::elementwise::HardswishFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_hardswish_f32,
    float,
    baracuda::elementwise::HardswishFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_hardswish_f16,
    __half,
    baracuda::elementwise::HardswishFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_hardswish_f16,
    __half,
    baracuda::elementwise::HardswishFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_hardswish_bf16,
    __nv_bfloat16,
    baracuda::elementwise::HardswishFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_hardswish_bf16,
    __nv_bfloat16,
    baracuda::elementwise::HardswishFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_hardswish_f64,
    double,
    baracuda::elementwise::HardswishFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_hardswish_f64,
    double,
    baracuda::elementwise::HardswishFunctor<double>)
