// baracuda-kernels Phase 3 unary backward: sin backward.
//
// Forward: `y = sin(x)`. Backward: `dx = dy * cos(x)`. Saved-x.
// f32 uses `cosf`; f64 uses `cos`; f16 / bf16 use the f32-detour
// pattern, mirroring the forward Sin kernel.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct SinBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T x) const {
        return dy * x;  // placeholder — explicit specializations below are used.
    }
};

template <>
struct SinBackwardFunctor<float> {
    __device__ __forceinline__ float operator()(float dy, float x) const {
        return dy * cosf(x);
    }
};

template <>
struct SinBackwardFunctor<double> {
    __device__ __forceinline__ double operator()(double dy, double x) const {
        return dy * cos(x);
    }
};

template <>
struct SinBackwardFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half dy, __half x) const {
        return __float2half(__half2float(dy) * cosf(__half2float(x)));
    }
};

template <>
struct SinBackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 dy, __nv_bfloat16 x) const {
        return __float2bfloat16(__bfloat162float(dy) * cosf(__bfloat162float(x)));
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_sin_backward_f32, float,
    baracuda::elementwise::SinBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_sin_backward_f16, __half,
    baracuda::elementwise::SinBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_sin_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::SinBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_sin_backward_f64, double,
    baracuda::elementwise::SinBackwardFunctor<double>)
