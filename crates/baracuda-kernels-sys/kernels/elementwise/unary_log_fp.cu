// baracuda-kernels Phase 3 unary fanout: elementwise log (natural) for FP types.
//
// Implements `y = ln(x)` over contig + strided. f32 uses `logf`; f64
// uses `log`. f16 / bf16 use the f32-detour pattern.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct LogFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct LogFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const { return logf(x); }
};

template <>
struct LogFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const { return log(x); }
};

template <>
struct LogFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        return __float2half(logf(__half2float(x)));
    }
};

template <>
struct LogFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        return __float2bfloat16(logf(__bfloat162float(x)));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_log_f32,
    float,
    baracuda::elementwise::LogFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_log_f32,
    float,
    baracuda::elementwise::LogFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_log_f16,
    __half,
    baracuda::elementwise::LogFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_log_f16,
    __half,
    baracuda::elementwise::LogFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_log_bf16,
    __nv_bfloat16,
    baracuda::elementwise::LogFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_log_bf16,
    __nv_bfloat16,
    baracuda::elementwise::LogFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_log_f64,
    double,
    baracuda::elementwise::LogFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_log_f64,
    double,
    baracuda::elementwise::LogFunctor<double>)
