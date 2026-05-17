// baracuda-kernels Phase 3 unary fanout: elementwise acosh for FP
// types.
//
// Implements `y = acosh(x)` over both contiguous tensors (fast path) and
// arbitrary strided views. f32 uses the matching libm intrinsic; f64
// uses the double-precision intrinsic. f16 / bf16 follow the universal
// "f32-detour" pattern — convert up, compute in f32, convert back.
// This is the cleanest approach for transcendental / rounding math at
// half precision and avoids relying on the spotty libdevice
// half-precision intrinsics.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct AcoshFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct AcoshFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const { return acoshf(x); }
};

template <>
struct AcoshFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const { return acosh(x); }
};

template <>
struct AcoshFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        return __float2half(acoshf(__half2float(x)));
    }
};

template <>
struct AcoshFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        return __float2bfloat16(acoshf(__bfloat162float(x)));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_acosh_f32,
    float,
    baracuda::elementwise::AcoshFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_acosh_f32,
    float,
    baracuda::elementwise::AcoshFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_acosh_f16,
    __half,
    baracuda::elementwise::AcoshFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_acosh_f16,
    __half,
    baracuda::elementwise::AcoshFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_acosh_bf16,
    __nv_bfloat16,
    baracuda::elementwise::AcoshFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_acosh_bf16,
    __nv_bfloat16,
    baracuda::elementwise::AcoshFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_acosh_f64,
    double,
    baracuda::elementwise::AcoshFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_acosh_f64,
    double,
    baracuda::elementwise::AcoshFunctor<double>)