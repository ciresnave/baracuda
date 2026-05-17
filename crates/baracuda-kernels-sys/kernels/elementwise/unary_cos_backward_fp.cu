// baracuda-kernels Phase 3 unary backward: cos backward.
//
// Forward: `y = cos(x)`. Backward: `dx = -dy * sin(x)`. Saved-x.
// f32 uses `sinf`; f64 uses `sin`; f16 / bf16 use the f32-detour
// pattern, mirroring the forward Cos kernel.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct CosBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T x) const {
        return dy * x;  // placeholder — explicit specializations below are used.
    }
};

template <>
struct CosBackwardFunctor<float> {
    __device__ __forceinline__ float operator()(float dy, float x) const {
        return -dy * sinf(x);
    }
};

template <>
struct CosBackwardFunctor<double> {
    __device__ __forceinline__ double operator()(double dy, double x) const {
        return -dy * sin(x);
    }
};

template <>
struct CosBackwardFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half dy, __half x) const {
        return __float2half(-__half2float(dy) * sinf(__half2float(x)));
    }
};

template <>
struct CosBackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 dy, __nv_bfloat16 x) const {
        return __float2bfloat16(-__bfloat162float(dy) * sinf(__bfloat162float(x)));
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_cos_backward_f32, float,
    baracuda::elementwise::CosBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_cos_backward_f16, __half,
    baracuda::elementwise::CosBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_cos_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::CosBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_cos_backward_f64, double,
    baracuda::elementwise::CosBackwardFunctor<double>)
