// baracuda-kernels Phase 3 unary fanout: elementwise rsqrt for FP types.
//
// Implements `y = 1 / sqrt(x)` over contig + strided. f32 uses the
// dedicated CUDA intrinsic `rsqrtf` (one PTX `rsqrt.approx.f32`); f64
// has no libm `rsqrt(double)` so we compose as `1.0 / sqrt(x)`. f16 /
// bf16 use the f32-detour through `rsqrtf`.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct RsqrtFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct RsqrtFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const { return rsqrtf(x); }
};

template <>
struct RsqrtFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const { return 1.0 / sqrt(x); }
};

template <>
struct RsqrtFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        return __float2half(rsqrtf(__half2float(x)));
    }
};

template <>
struct RsqrtFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        return __float2bfloat16(rsqrtf(__bfloat162float(x)));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_rsqrt_f32,
    float,
    baracuda::elementwise::RsqrtFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_rsqrt_f32,
    float,
    baracuda::elementwise::RsqrtFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_rsqrt_f16,
    __half,
    baracuda::elementwise::RsqrtFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_rsqrt_f16,
    __half,
    baracuda::elementwise::RsqrtFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_rsqrt_bf16,
    __nv_bfloat16,
    baracuda::elementwise::RsqrtFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_rsqrt_bf16,
    __nv_bfloat16,
    baracuda::elementwise::RsqrtFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_rsqrt_f64,
    double,
    baracuda::elementwise::RsqrtFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_rsqrt_f64,
    double,
    baracuda::elementwise::RsqrtFunctor<double>)
