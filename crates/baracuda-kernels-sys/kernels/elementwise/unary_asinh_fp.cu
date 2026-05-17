// baracuda-kernels Phase 3 unary fanout: elementwise asinh for FP
// types.
//
// Implements `y = asinh(x)` over both contiguous tensors (fast path) and
// arbitrary strided views. f32 uses the matching libm intrinsic; f64
// uses the double-precision intrinsic. f16 / bf16 follow the universal
// "f32-detour" pattern — convert up, compute in f32, convert back.
// This is the cleanest approach for transcendental / rounding math at
// half precision and avoids relying on the spotty libdevice
// half-precision intrinsics.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct AsinhFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct AsinhFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const { return asinhf(x); }
};

template <>
struct AsinhFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const { return asinh(x); }
};

template <>
struct AsinhFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        return __float2half(asinhf(__half2float(x)));
    }
};

template <>
struct AsinhFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        return __float2bfloat16(asinhf(__bfloat162float(x)));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_asinh_f32,
    float,
    baracuda::elementwise::AsinhFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_asinh_f32,
    float,
    baracuda::elementwise::AsinhFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_asinh_f16,
    __half,
    baracuda::elementwise::AsinhFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_asinh_f16,
    __half,
    baracuda::elementwise::AsinhFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_asinh_bf16,
    __nv_bfloat16,
    baracuda::elementwise::AsinhFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_asinh_bf16,
    __nv_bfloat16,
    baracuda::elementwise::AsinhFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_asinh_f64,
    double,
    baracuda::elementwise::AsinhFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_asinh_f64,
    double,
    baracuda::elementwise::AsinhFunctor<double>)