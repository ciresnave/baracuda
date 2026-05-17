// baracuda-kernels Phase 3 unary fanout: elementwise atan for FP
// types.
//
// Implements `y = atan(x)` over both contiguous tensors (fast path) and
// arbitrary strided views. f32 uses the matching libm intrinsic; f64
// uses the double-precision intrinsic. f16 / bf16 follow the universal
// "f32-detour" pattern — convert up, compute in f32, convert back.
// This is the cleanest approach for transcendental / rounding math at
// half precision and avoids relying on the spotty libdevice
// half-precision intrinsics.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct AtanFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct AtanFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const { return atanf(x); }
};

template <>
struct AtanFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const { return atan(x); }
};

template <>
struct AtanFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        return __float2half(atanf(__half2float(x)));
    }
};

template <>
struct AtanFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        return __float2bfloat16(atanf(__bfloat162float(x)));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_atan_f32,
    float,
    baracuda::elementwise::AtanFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_atan_f32,
    float,
    baracuda::elementwise::AtanFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_atan_f16,
    __half,
    baracuda::elementwise::AtanFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_atan_f16,
    __half,
    baracuda::elementwise::AtanFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_atan_bf16,
    __nv_bfloat16,
    baracuda::elementwise::AtanFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_atan_bf16,
    __nv_bfloat16,
    baracuda::elementwise::AtanFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_atan_f64,
    double,
    baracuda::elementwise::AtanFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_atan_f64,
    double,
    baracuda::elementwise::AtanFunctor<double>)