// baracuda-kernels Phase 3 unary fanout: elementwise log2 for FP
// types.
//
// Implements `y = log_2(x)` over both contiguous tensors (fast path) and
// arbitrary strided views. f32 uses the matching libm intrinsic; f64
// uses the double-precision intrinsic. f16 / bf16 follow the universal
// "f32-detour" pattern — convert up, compute in f32, convert back.
// This is the cleanest approach for transcendental / rounding math at
// half precision and avoids relying on the spotty libdevice
// half-precision intrinsics.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct Log2Functor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct Log2Functor<float> {
    __device__ __forceinline__ float operator()(float x) const { return log2f(x); }
};

template <>
struct Log2Functor<double> {
    __device__ __forceinline__ double operator()(double x) const { return log2(x); }
};

template <>
struct Log2Functor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        return __float2half(log2f(__half2float(x)));
    }
};

template <>
struct Log2Functor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        return __float2bfloat16(log2f(__bfloat162float(x)));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_log2_f32,
    float,
    baracuda::elementwise::Log2Functor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_log2_f32,
    float,
    baracuda::elementwise::Log2Functor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_log2_f16,
    __half,
    baracuda::elementwise::Log2Functor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_log2_f16,
    __half,
    baracuda::elementwise::Log2Functor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_log2_bf16,
    __nv_bfloat16,
    baracuda::elementwise::Log2Functor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_log2_bf16,
    __nv_bfloat16,
    baracuda::elementwise::Log2Functor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_log2_f64,
    double,
    baracuda::elementwise::Log2Functor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_log2_f64,
    double,
    baracuda::elementwise::Log2Functor<double>)