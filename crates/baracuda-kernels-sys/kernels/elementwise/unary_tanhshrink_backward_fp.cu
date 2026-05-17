// baracuda-kernels Phase 3 unary backward: tanhshrink backward.
//
// Forward: `y = x - tanh(x)`. Backward: `dx = dy * tanh(x) * tanh(x)`.
// Saved-x. f32 uses `tanhf`; f64 uses `tanh`; f16 / bf16 use the
// f32-detour pattern, mirroring the Sin backward shape.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct TanhshrinkBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T x) const {
        return dy * x;  // placeholder — explicit specializations below are used.
    }
};

template <>
struct TanhshrinkBackwardFunctor<float> {
    __device__ __forceinline__ float operator()(float dy, float x) const {
        float t = tanhf(x);
        return dy * t * t;
    }
};

template <>
struct TanhshrinkBackwardFunctor<double> {
    __device__ __forceinline__ double operator()(double dy, double x) const {
        double t = tanh(x);
        return dy * t * t;
    }
};

template <>
struct TanhshrinkBackwardFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half dy, __half x) const {
        float t = tanhf(__half2float(x));
        return __float2half(__half2float(dy) * t * t);
    }
};

template <>
struct TanhshrinkBackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 dy, __nv_bfloat16 x) const {
        float t = tanhf(__bfloat162float(x));
        return __float2bfloat16(__bfloat162float(dy) * t * t);
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_tanhshrink_backward_f32, float,
    baracuda::elementwise::TanhshrinkBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_tanhshrink_backward_f16, __half,
    baracuda::elementwise::TanhshrinkBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_tanhshrink_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::TanhshrinkBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_tanhshrink_backward_f64, double,
    baracuda::elementwise::TanhshrinkBackwardFunctor<double>)
