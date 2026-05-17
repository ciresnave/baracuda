// baracuda-kernels Phase 3 unary fanout: elementwise acos for FP
// types.
//
// Implements `y = acos(x)` over both contiguous tensors (fast path) and
// arbitrary strided views. f32 uses the matching libm intrinsic; f64
// uses the double-precision intrinsic. f16 / bf16 follow the universal
// "f32-detour" pattern — convert up, compute in f32, convert back.
// This is the cleanest approach for transcendental / rounding math at
// half precision and avoids relying on the spotty libdevice
// half-precision intrinsics.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct AcosFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct AcosFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const { return acosf(x); }
};

template <>
struct AcosFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const { return acos(x); }
};

template <>
struct AcosFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        return __float2half(acosf(__half2float(x)));
    }
};

template <>
struct AcosFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        return __float2bfloat16(acosf(__bfloat162float(x)));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_acos_f32,
    float,
    baracuda::elementwise::AcosFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_acos_f32,
    float,
    baracuda::elementwise::AcosFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_acos_f16,
    __half,
    baracuda::elementwise::AcosFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_acos_f16,
    __half,
    baracuda::elementwise::AcosFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_acos_bf16,
    __nv_bfloat16,
    baracuda::elementwise::AcosFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_acos_bf16,
    __nv_bfloat16,
    baracuda::elementwise::AcosFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_acos_f64,
    double,
    baracuda::elementwise::AcosFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_acos_f64,
    double,
    baracuda::elementwise::AcosFunctor<double>)