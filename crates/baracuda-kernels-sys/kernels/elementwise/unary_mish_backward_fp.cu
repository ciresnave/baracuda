// baracuda-kernels Phase 3 unary backward fanout: mish backward.
//
// Forward: `y = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))`.
// Backward:
//   sp = softplus(x) = log1p(exp(x))
//   t  = tanh(sp)
//   s  = sigmoid(x) = 1 / (1 + exp(-x))
//   dx = dy * (t + x * s * (1 - t * t))
// Saved-x; smooth, multiple transcendentals (log1p / exp / tanh).
//
// 8×eps tolerance — three chained libdevice transcendentals (`expf`,
// `log1pf`, `tanhf`) plus the `(1 - t*t)` cancellation can each
// contribute ~1 ULP of drift vs Rust's libm, so 4×eps grazes the edge.
// f32 uses the `f`-suffixed libm; f64 uses the unsuffixed equivalents.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct MishBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T x) const {
        return dy * x;  // placeholder — explicit specializations below.
    }
};

template <>
struct MishBackwardFunctor<float> {
    __device__ __forceinline__ float operator()(float dy, float x) const {
        float sp = log1pf(expf(x));
        float t = tanhf(sp);
        float s = 1.0f / (1.0f + expf(-x));
        return dy * (t + x * s * (1.0f - t * t));
    }
};

template <>
struct MishBackwardFunctor<double> {
    __device__ __forceinline__ double operator()(double dy, double x) const {
        double sp = log1p(exp(x));
        double t = tanh(sp);
        double s = 1.0 / (1.0 + exp(-x));
        return dy * (t + x * s * (1.0 - t * t));
    }
};

template <>
struct MishBackwardFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half dy, __half x) const {
        float fx = __half2float(x);
        float fdy = __half2float(dy);
        float sp = log1pf(expf(fx));
        float t = tanhf(sp);
        float s = 1.0f / (1.0f + expf(-fx));
        return __float2half(fdy * (t + fx * s * (1.0f - t * t)));
    }
};

template <>
struct MishBackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 dy, __nv_bfloat16 x) const {
        float fx = __bfloat162float(x);
        float fdy = __bfloat162float(dy);
        float sp = log1pf(expf(fx));
        float t = tanhf(sp);
        float s = 1.0f / (1.0f + expf(-fx));
        return __float2bfloat16(fdy * (t + fx * s * (1.0f - t * t)));
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_mish_backward_f32, float,
    baracuda::elementwise::MishBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_mish_backward_f16, __half,
    baracuda::elementwise::MishBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_mish_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::MishBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_mish_backward_f64, double,
    baracuda::elementwise::MishBackwardFunctor<double>)
