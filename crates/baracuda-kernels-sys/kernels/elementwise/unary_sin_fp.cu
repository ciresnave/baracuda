// baracuda-kernels Phase 3 unary fanout: elementwise sin for FP types.
//
// Implements `y = sin(x)` over contig + strided. f32 uses `sinf`; f64
// uses `sin`. f16 / bf16 use the f32-detour pattern.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct SinFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct SinFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const { return sinf(x); }
};

template <>
struct SinFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const { return sin(x); }
};

template <>
struct SinFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        return __float2half(sinf(__half2float(x)));
    }
};

template <>
struct SinFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        return __float2bfloat16(sinf(__bfloat162float(x)));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_sin_f32,
    float,
    baracuda::elementwise::SinFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_sin_f32,
    float,
    baracuda::elementwise::SinFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_sin_f16,
    __half,
    baracuda::elementwise::SinFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_sin_f16,
    __half,
    baracuda::elementwise::SinFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_sin_bf16,
    __nv_bfloat16,
    baracuda::elementwise::SinFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_sin_bf16,
    __nv_bfloat16,
    baracuda::elementwise::SinFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_sin_f64,
    double,
    baracuda::elementwise::SinFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_sin_f64,
    double,
    baracuda::elementwise::SinFunctor<double>)
