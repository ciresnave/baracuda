// baracuda-kernels Phase 3 unary fanout: elementwise erf for FP types.
//
// Implements `y = erf(x)` (Gauss error function). f32 uses `erff`; f64
// uses `erf` (CUDA libdevice). f16 / bf16 use the f32 detour pattern.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct ErfFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct ErfFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const { return erff(x); }
};

template <>
struct ErfFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const { return erf(x); }
};

template <>
struct ErfFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        return __float2half(erff(__half2float(x)));
    }
};

template <>
struct ErfFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        return __float2bfloat16(erff(__bfloat162float(x)));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_erf_f32,
    float,
    baracuda::elementwise::ErfFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_erf_f32,
    float,
    baracuda::elementwise::ErfFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_erf_f16,
    __half,
    baracuda::elementwise::ErfFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_erf_f16,
    __half,
    baracuda::elementwise::ErfFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_erf_bf16,
    __nv_bfloat16,
    baracuda::elementwise::ErfFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_erf_bf16,
    __nv_bfloat16,
    baracuda::elementwise::ErfFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_erf_f64,
    double,
    baracuda::elementwise::ErfFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_erf_f64,
    double,
    baracuda::elementwise::ErfFunctor<double>)
