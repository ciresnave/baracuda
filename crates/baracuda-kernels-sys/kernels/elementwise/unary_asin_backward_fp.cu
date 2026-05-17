// baracuda-kernels Phase 3 unary backward: asin backward.
//
// Forward: `y = asin(x)`. Backward: `dx = dy / sqrt(1 - x²)`. Saved-x.
// Domain: strictly `|x| < 1` (the derivative diverges at ±1). f32 uses
// `sqrtf`; f64 uses `sqrt`; f16 / bf16 use the f32-detour pattern.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct AsinBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T x) const {
        return dy * x;  // placeholder — explicit specializations below are used.
    }
};

template <>
struct AsinBackwardFunctor<float> {
    __device__ __forceinline__ float operator()(float dy, float x) const {
        return dy / sqrtf(1.0f - x * x);
    }
};

template <>
struct AsinBackwardFunctor<double> {
    __device__ __forceinline__ double operator()(double dy, double x) const {
        return dy / sqrt(1.0 - x * x);
    }
};

template <>
struct AsinBackwardFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half dy, __half x) const {
        float xf = __half2float(x);
        return __float2half(__half2float(dy) / sqrtf(1.0f - xf * xf));
    }
};

template <>
struct AsinBackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 dy, __nv_bfloat16 x) const {
        float xf = __bfloat162float(x);
        return __float2bfloat16(__bfloat162float(dy) / sqrtf(1.0f - xf * xf));
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_asin_backward_f32, float,
    baracuda::elementwise::AsinBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_asin_backward_f16, __half,
    baracuda::elementwise::AsinBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_asin_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::AsinBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_asin_backward_f64, double,
    baracuda::elementwise::AsinBackwardFunctor<double>)
