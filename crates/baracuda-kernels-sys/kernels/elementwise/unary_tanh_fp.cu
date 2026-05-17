// baracuda-kernels Phase 3 unary fanout: elementwise tanh for FP types.
//
// Implements `y = tanh(x)` over contig + strided. f32 uses `tanhf`; f64
// uses `tanh`. f16 / bf16 use the f32-detour pattern. Output is
// bounded in [-1, 1] so overflow is not a concern.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct TanhFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct TanhFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const { return tanhf(x); }
};

template <>
struct TanhFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const { return tanh(x); }
};

template <>
struct TanhFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        return __float2half(tanhf(__half2float(x)));
    }
};

template <>
struct TanhFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        return __float2bfloat16(tanhf(__bfloat162float(x)));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_tanh_f32,
    float,
    baracuda::elementwise::TanhFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_tanh_f32,
    float,
    baracuda::elementwise::TanhFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_tanh_f16,
    __half,
    baracuda::elementwise::TanhFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_tanh_f16,
    __half,
    baracuda::elementwise::TanhFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_tanh_bf16,
    __nv_bfloat16,
    baracuda::elementwise::TanhFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_tanh_bf16,
    __nv_bfloat16,
    baracuda::elementwise::TanhFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_tanh_f64,
    double,
    baracuda::elementwise::TanhFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_tanh_f64,
    double,
    baracuda::elementwise::TanhFunctor<double>)
