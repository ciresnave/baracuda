// baracuda-kernels Phase 3 deferral: elementwise threshold backward.
//
// Forward: `y = (x > t) ? x : v`. Backward: `dx = (x > t) ? dy : 0`.
// Saved-x. Scalar params `t` and `v` are threaded through but only `t`
// is consulted (the replacement value `v` doesn't affect the gradient —
// `v` is a constant w.r.t. `x`). We accept both params on the ABI for
// shape parity with the forward; the second is ignored by the functor.
//
// PyTorch convention: at exactly `x == t` (the unmatched branch is taken
// because the comparison is strict `>`), the gradient is `0`.
//
// f16 / bf16 use the f32-detour for the compare; `dy` is returned
// bit-identically on the matched branch (no rounding) and zero
// otherwise.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct ThresholdBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T x, float t, float /*v*/) const {
        return (x > T(t)) ? dy : T(0);
    }
};

template <>
struct ThresholdBackwardFunctor<float> {
    __device__ __forceinline__ float operator()(float dy, float x, float t, float /*v*/) const {
        return (x > t) ? dy : 0.0f;
    }
};

template <>
struct ThresholdBackwardFunctor<double> {
    __device__ __forceinline__ double operator()(double dy, double x, float t, float /*v*/) const {
        return (x > (double)t) ? dy : 0.0;
    }
};

template <>
struct ThresholdBackwardFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half dy, __half x, float t, float /*v*/) const {
        float fx = __half2float(x);
        return (fx > t) ? dy : __float2half(0.0f);
    }
};

template <>
struct ThresholdBackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 dy, __nv_bfloat16 x, float t, float /*v*/) const {
        float fx = __bfloat162float(x);
        return (fx > t) ? dy : __float2bfloat16(0.0f);
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_PARAM_BACKWARD_INSTANTIATE(
    unary_threshold_backward_f32, float,
    baracuda::elementwise::ThresholdBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_PARAM_BACKWARD_INSTANTIATE(
    unary_threshold_backward_f16, __half,
    baracuda::elementwise::ThresholdBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_PARAM_BACKWARD_INSTANTIATE(
    unary_threshold_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::ThresholdBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_PARAM_BACKWARD_INSTANTIATE(
    unary_threshold_backward_f64, double,
    baracuda::elementwise::ThresholdBackwardFunctor<double>)
