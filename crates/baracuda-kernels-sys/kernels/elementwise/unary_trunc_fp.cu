// baracuda-kernels Phase 3 unary fanout: elementwise trunc for FP
// types.
//
// Implements `y = trunc(x) (truncate toward zero)` over both contiguous tensors (fast path) and
// arbitrary strided views. f32 uses the matching libm intrinsic; f64
// uses the double-precision intrinsic. f16 / bf16 follow the universal
// "f32-detour" pattern — convert up, compute in f32, convert back.
// This is the cleanest approach for transcendental / rounding math at
// half precision and avoids relying on the spotty libdevice
// half-precision intrinsics.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct TruncFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct TruncFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const { return truncf(x); }
};

template <>
struct TruncFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const { return trunc(x); }
};

template <>
struct TruncFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        return __float2half(truncf(__half2float(x)));
    }
};

template <>
struct TruncFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        return __float2bfloat16(truncf(__bfloat162float(x)));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_trunc_f32,
    float,
    baracuda::elementwise::TruncFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_trunc_f32,
    float,
    baracuda::elementwise::TruncFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_trunc_f16,
    __half,
    baracuda::elementwise::TruncFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_trunc_f16,
    __half,
    baracuda::elementwise::TruncFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_trunc_bf16,
    __nv_bfloat16,
    baracuda::elementwise::TruncFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_trunc_bf16,
    __nv_bfloat16,
    baracuda::elementwise::TruncFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_trunc_f64,
    double,
    baracuda::elementwise::TruncFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_trunc_f64,
    double,
    baracuda::elementwise::TruncFunctor<double>)