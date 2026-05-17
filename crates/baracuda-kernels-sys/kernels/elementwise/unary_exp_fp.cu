// baracuda-kernels Phase 3 unary fanout: elementwise exp for FP types.
//
// Implements `y = exp(x)` over contig + strided. f32 uses `expf`; f64
// uses `exp`. f16 / bf16 use the f32-detour pattern.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct ExpFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct ExpFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const { return expf(x); }
};

template <>
struct ExpFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const { return exp(x); }
};

template <>
struct ExpFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        return __float2half(expf(__half2float(x)));
    }
};

template <>
struct ExpFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        return __float2bfloat16(expf(__bfloat162float(x)));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_exp_f32,
    float,
    baracuda::elementwise::ExpFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_exp_f32,
    float,
    baracuda::elementwise::ExpFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_exp_f16,
    __half,
    baracuda::elementwise::ExpFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_exp_f16,
    __half,
    baracuda::elementwise::ExpFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_exp_bf16,
    __nv_bfloat16,
    baracuda::elementwise::ExpFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_exp_bf16,
    __nv_bfloat16,
    baracuda::elementwise::ExpFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_exp_f64,
    double,
    baracuda::elementwise::ExpFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_exp_f64,
    double,
    baracuda::elementwise::ExpFunctor<double>)
