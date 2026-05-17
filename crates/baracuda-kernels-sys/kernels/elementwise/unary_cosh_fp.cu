// baracuda-kernels Phase 3 unary fanout: elementwise cosh for FP types.
//
// Implements `y = cosh(x)` over contig + strided. f32 uses `coshf`; f64
// uses `cosh`. f16 / bf16 use the f32-detour pattern.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct CoshFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct CoshFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const { return coshf(x); }
};

template <>
struct CoshFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const { return cosh(x); }
};

template <>
struct CoshFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        return __float2half(coshf(__half2float(x)));
    }
};

template <>
struct CoshFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        return __float2bfloat16(coshf(__bfloat162float(x)));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_cosh_f32,
    float,
    baracuda::elementwise::CoshFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_cosh_f32,
    float,
    baracuda::elementwise::CoshFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_cosh_f16,
    __half,
    baracuda::elementwise::CoshFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_cosh_f16,
    __half,
    baracuda::elementwise::CoshFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_cosh_bf16,
    __nv_bfloat16,
    baracuda::elementwise::CoshFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_cosh_bf16,
    __nv_bfloat16,
    baracuda::elementwise::CoshFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_cosh_f64,
    double,
    baracuda::elementwise::CoshFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_cosh_f64,
    double,
    baracuda::elementwise::CoshFunctor<double>)
