// baracuda-kernels Phase 3 unary fanout: elementwise asin for FP
// types.
//
// Implements `y = asin(x)` over both contiguous tensors (fast path) and
// arbitrary strided views. f32 uses the matching libm intrinsic; f64
// uses the double-precision intrinsic. f16 / bf16 follow the universal
// "f32-detour" pattern — convert up, compute in f32, convert back.
// This is the cleanest approach for transcendental / rounding math at
// half precision and avoids relying on the spotty libdevice
// half-precision intrinsics.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct AsinFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct AsinFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const { return asinf(x); }
};

template <>
struct AsinFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const { return asin(x); }
};

template <>
struct AsinFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        return __float2half(asinf(__half2float(x)));
    }
};

template <>
struct AsinFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        return __float2bfloat16(asinf(__bfloat162float(x)));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_asin_f32,
    float,
    baracuda::elementwise::AsinFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_asin_f32,
    float,
    baracuda::elementwise::AsinFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_asin_f16,
    __half,
    baracuda::elementwise::AsinFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_asin_f16,
    __half,
    baracuda::elementwise::AsinFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_asin_bf16,
    __nv_bfloat16,
    baracuda::elementwise::AsinFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_asin_bf16,
    __nv_bfloat16,
    baracuda::elementwise::AsinFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_asin_f64,
    double,
    baracuda::elementwise::AsinFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_asin_f64,
    double,
    baracuda::elementwise::AsinFunctor<double>)