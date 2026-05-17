// baracuda-kernels Phase 3 unary fanout: elementwise reciprocal for
// FP types.
//
// Implements `y = 1 / x` over both contiguous tensors (fast path) and
// arbitrary strided views. f32 / f64 use straight `1.0 / x`. The
// half-precision specializations use the f32 detour
// (`__float2half(1.0f / __half2float(x))`) — the result is within
// 1 ULP of the correctly-rounded f16 reciprocal, which is well within
// the relative tolerance the smoke tests accept.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct ReciprocalFunctor {
    __device__ __forceinline__ T operator()(T x) const { return T(1) / x; }
};

template <>
struct ReciprocalFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const { return 1.0f / x; }
};

template <>
struct ReciprocalFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const { return 1.0 / x; }
};

template <>
struct ReciprocalFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        return __float2half(1.0f / __half2float(x));
    }
};

template <>
struct ReciprocalFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        return __float2bfloat16(1.0f / __bfloat162float(x));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_reciprocal_f32,
    float,
    baracuda::elementwise::ReciprocalFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_reciprocal_f32,
    float,
    baracuda::elementwise::ReciprocalFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_reciprocal_f16,
    __half,
    baracuda::elementwise::ReciprocalFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_reciprocal_f16,
    __half,
    baracuda::elementwise::ReciprocalFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_reciprocal_bf16,
    __nv_bfloat16,
    baracuda::elementwise::ReciprocalFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_reciprocal_bf16,
    __nv_bfloat16,
    baracuda::elementwise::ReciprocalFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_reciprocal_f64,
    double,
    baracuda::elementwise::ReciprocalFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_reciprocal_f64,
    double,
    baracuda::elementwise::ReciprocalFunctor<double>)
