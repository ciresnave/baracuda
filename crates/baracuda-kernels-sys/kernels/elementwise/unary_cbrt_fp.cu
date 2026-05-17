// baracuda-kernels Phase 3 unary fanout: elementwise cube-root for FP
// types.
//
// Implements `y = cbrt(x)` over both contiguous tensors (fast path)
// and arbitrary strided views. f32 uses `cbrtf`; f64 uses `cbrt`. f16
// / bf16 follow the universal "f32-detour" pattern — convert up,
// compute in f32, convert back. This is the cleanest approach for
// transcendental math at half precision and avoids relying on the
// spotty libdevice half-precision intrinsics.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct CbrtFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct CbrtFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const { return cbrtf(x); }
};

template <>
struct CbrtFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const { return cbrt(x); }
};

template <>
struct CbrtFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        return __float2half(cbrtf(__half2float(x)));
    }
};

template <>
struct CbrtFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        return __float2bfloat16(cbrtf(__bfloat162float(x)));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_cbrt_f32,
    float,
    baracuda::elementwise::CbrtFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_cbrt_f32,
    float,
    baracuda::elementwise::CbrtFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_cbrt_f16,
    __half,
    baracuda::elementwise::CbrtFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_cbrt_f16,
    __half,
    baracuda::elementwise::CbrtFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_cbrt_bf16,
    __nv_bfloat16,
    baracuda::elementwise::CbrtFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_cbrt_bf16,
    __nv_bfloat16,
    baracuda::elementwise::CbrtFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_cbrt_f64,
    double,
    baracuda::elementwise::CbrtFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_cbrt_f64,
    double,
    baracuda::elementwise::CbrtFunctor<double>)
