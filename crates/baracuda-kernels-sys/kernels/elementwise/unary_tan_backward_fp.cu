// baracuda-kernels Phase 3 unary backward: tan backward.
//
// Forward: `y = tan(x)`. Backward: `dx = dy * (1 + tan(x)^2)`. Saved-x.
// f32 uses `tanf`; f64 uses `tan`; f16 / bf16 use the f32-detour
// pattern. Callers must keep `x` away from π/2 + nπ (poles).

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct TanBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T x) const {
        return dy * x;  // placeholder — explicit specializations below are used.
    }
};

template <>
struct TanBackwardFunctor<float> {
    __device__ __forceinline__ float operator()(float dy, float x) const {
        float t = tanf(x);
        return dy * (1.0f + t * t);
    }
};

template <>
struct TanBackwardFunctor<double> {
    __device__ __forceinline__ double operator()(double dy, double x) const {
        double t = tan(x);
        return dy * (1.0 + t * t);
    }
};

template <>
struct TanBackwardFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half dy, __half x) const {
        float t = tanf(__half2float(x));
        return __float2half(__half2float(dy) * (1.0f + t * t));
    }
};

template <>
struct TanBackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 dy, __nv_bfloat16 x) const {
        float t = tanf(__bfloat162float(x));
        return __float2bfloat16(__bfloat162float(dy) * (1.0f + t * t));
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_tan_backward_f32, float,
    baracuda::elementwise::TanBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_tan_backward_f16, __half,
    baracuda::elementwise::TanBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_tan_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::TanBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_tan_backward_f64, double,
    baracuda::elementwise::TanBackwardFunctor<double>)
