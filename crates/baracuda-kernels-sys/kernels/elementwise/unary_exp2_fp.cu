// baracuda-kernels Phase 3 unary fanout: elementwise exp2 for FP
// types.
//
// Implements `y = 2^x` over both contiguous tensors (fast path) and
// arbitrary strided views. f32 uses the matching libm intrinsic; f64
// uses the double-precision intrinsic. f16 / bf16 follow the universal
// "f32-detour" pattern — convert up, compute in f32, convert back.
// This is the cleanest approach for transcendental / rounding math at
// half precision and avoids relying on the spotty libdevice
// half-precision intrinsics.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct Exp2Functor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct Exp2Functor<float> {
    __device__ __forceinline__ float operator()(float x) const { return exp2f(x); }
};

template <>
struct Exp2Functor<double> {
    __device__ __forceinline__ double operator()(double x) const { return exp2(x); }
};

template <>
struct Exp2Functor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        return __float2half(exp2f(__half2float(x)));
    }
};

template <>
struct Exp2Functor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        return __float2bfloat16(exp2f(__bfloat162float(x)));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_exp2_f32,
    float,
    baracuda::elementwise::Exp2Functor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_exp2_f32,
    float,
    baracuda::elementwise::Exp2Functor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_exp2_f16,
    __half,
    baracuda::elementwise::Exp2Functor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_exp2_f16,
    __half,
    baracuda::elementwise::Exp2Functor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_exp2_bf16,
    __nv_bfloat16,
    baracuda::elementwise::Exp2Functor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_exp2_bf16,
    __nv_bfloat16,
    baracuda::elementwise::Exp2Functor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_exp2_f64,
    double,
    baracuda::elementwise::Exp2Functor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_exp2_f64,
    double,
    baracuda::elementwise::Exp2Functor<double>)