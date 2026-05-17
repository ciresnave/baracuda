// baracuda-kernels Phase 3 unary fanout: elementwise lgamma for FP types.
//
// Implements `y = lgamma(x) = ln(|Γ(x)|)` (log-gamma). f32 uses
// `lgammaf`; f64 uses `lgamma` (CUDA libdevice). f16 / bf16 use the f32
// detour pattern. Domain: `x` not a non-positive integer (poles); the
// kernel is value-faithful to the libdevice routines so callers are
// responsible for keeping inputs in-domain.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct LgammaFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct LgammaFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const { return lgammaf(x); }
};

template <>
struct LgammaFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const { return lgamma(x); }
};

template <>
struct LgammaFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        return __float2half(lgammaf(__half2float(x)));
    }
};

template <>
struct LgammaFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        return __float2bfloat16(lgammaf(__bfloat162float(x)));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_lgamma_f32,
    float,
    baracuda::elementwise::LgammaFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_lgamma_f32,
    float,
    baracuda::elementwise::LgammaFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_lgamma_f16,
    __half,
    baracuda::elementwise::LgammaFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_lgamma_f16,
    __half,
    baracuda::elementwise::LgammaFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_lgamma_bf16,
    __nv_bfloat16,
    baracuda::elementwise::LgammaFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_lgamma_bf16,
    __nv_bfloat16,
    baracuda::elementwise::LgammaFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_lgamma_f64,
    double,
    baracuda::elementwise::LgammaFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_lgamma_f64,
    double,
    baracuda::elementwise::LgammaFunctor<double>)
