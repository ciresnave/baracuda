// baracuda-kernels Phase 3 unary fanout: elementwise log1p for FP types.
//
// Implements `y = ln(1 + x)` over contig + strided. The dedicated
// `log1pf` / `log1p` intrinsics preserve precision near zero (vs the
// naive `logf(1.0f + x)` form). f16 / bf16 use the f32-detour through
// `log1pf`.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct Log1pFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct Log1pFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const { return log1pf(x); }
};

template <>
struct Log1pFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const { return log1p(x); }
};

template <>
struct Log1pFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        return __float2half(log1pf(__half2float(x)));
    }
};

template <>
struct Log1pFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        return __float2bfloat16(log1pf(__bfloat162float(x)));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_log1p_f32,
    float,
    baracuda::elementwise::Log1pFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_log1p_f32,
    float,
    baracuda::elementwise::Log1pFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_log1p_f16,
    __half,
    baracuda::elementwise::Log1pFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_log1p_f16,
    __half,
    baracuda::elementwise::Log1pFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_log1p_bf16,
    __nv_bfloat16,
    baracuda::elementwise::Log1pFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_log1p_bf16,
    __nv_bfloat16,
    baracuda::elementwise::Log1pFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_log1p_f64,
    double,
    baracuda::elementwise::Log1pFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_log1p_f64,
    double,
    baracuda::elementwise::Log1pFunctor<double>)
