// baracuda-kernels Phase 3 unary fanout: elementwise sqrt for FP types.
//
// Implements `y = sqrt(x)` over both contiguous tensors (fast path) and
// arbitrary strided views. f32 uses `sqrtf`; f64 uses `sqrt`. f16 / bf16
// follow the universal "f32-detour" pattern — convert up, compute in
// f32, convert back. This is the cleanest approach for transcendental
// math at half precision and avoids relying on the spotty libdevice
// half-precision sqrt intrinsics.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct SqrtFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct SqrtFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const { return sqrtf(x); }
};

template <>
struct SqrtFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const { return sqrt(x); }
};

template <>
struct SqrtFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        return __float2half(sqrtf(__half2float(x)));
    }
};

template <>
struct SqrtFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        return __float2bfloat16(sqrtf(__bfloat162float(x)));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_sqrt_f32,
    float,
    baracuda::elementwise::SqrtFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_sqrt_f32,
    float,
    baracuda::elementwise::SqrtFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_sqrt_f16,
    __half,
    baracuda::elementwise::SqrtFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_sqrt_f16,
    __half,
    baracuda::elementwise::SqrtFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_sqrt_bf16,
    __nv_bfloat16,
    baracuda::elementwise::SqrtFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_sqrt_bf16,
    __nv_bfloat16,
    baracuda::elementwise::SqrtFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_sqrt_f64,
    double,
    baracuda::elementwise::SqrtFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_sqrt_f64,
    double,
    baracuda::elementwise::SqrtFunctor<double>)
