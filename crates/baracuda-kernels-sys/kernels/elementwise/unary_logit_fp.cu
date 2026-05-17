// baracuda-kernels Phase 3 unary fanout: elementwise logit for FP types.
//
// Implements `y = log(x / (1 - x))` (inverse of sigmoid). f32 uses
// `logf`; f64 uses `log` (CUDA libdevice). f16 / bf16 use the f32
// detour pattern. Domain: `(0, 1)`; at the endpoints the routine
// produces ±inf (and undefined behaviour outside `[0, 1]`).

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct LogitFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct LogitFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const {
        return logf(x / (1.0f - x));
    }
};

template <>
struct LogitFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const {
        return log(x / (1.0 - x));
    }
};

template <>
struct LogitFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        float f = __half2float(x);
        return __float2half(logf(f / (1.0f - f)));
    }
};

template <>
struct LogitFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float f = __bfloat162float(x);
        return __float2bfloat16(logf(f / (1.0f - f)));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_logit_f32,
    float,
    baracuda::elementwise::LogitFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_logit_f32,
    float,
    baracuda::elementwise::LogitFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_logit_f16,
    __half,
    baracuda::elementwise::LogitFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_logit_f16,
    __half,
    baracuda::elementwise::LogitFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_logit_bf16,
    __nv_bfloat16,
    baracuda::elementwise::LogitFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_logit_bf16,
    __nv_bfloat16,
    baracuda::elementwise::LogitFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_logit_f64,
    double,
    baracuda::elementwise::LogitFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_logit_f64,
    double,
    baracuda::elementwise::LogitFunctor<double>)
