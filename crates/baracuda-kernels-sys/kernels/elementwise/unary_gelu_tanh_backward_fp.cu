// baracuda-kernels Phase 3 unary backward fanout: gelu (tanh approximation) backward.
//
// Forward: `y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
//
// Backward — numerically stable reformulation. Let
//   c  = sqrt(2 / pi) ≈ 0.7978845608028654
//   u  = c * (x + 0.044715 * x^3)
//   u' = c * (1 + 3 * 0.044715 * x^2) = c * (1 + 0.134145 * x^2)
//   s  = sigmoid(2*u) = 1 / (1 + exp(-2*u))    [equivalently 0.5 * (1 + tanh(u))]
// then  1 - tanh(u)^2 = 4 * s * (1 - s)  (logistic identity), so
//   dx = dy * (s + 2 * x * s * (1 - s) * u')
//      = dy * s * (1 + 2 * x * (1 - s) * u')
//
// Why not the naive `0.5*(1+t) + 0.5*x*(1-t^2)*u'`: for `x → -∞` we have
// `t → -1`, so `(1+t)` and `(1-t^2) = (1-t)(1+t)` both suffer catastrophic
// cancellation (`(1+t)` is the small near-zero factor in both). The
// sigmoid form is well-conditioned everywhere — `s` is computed directly
// from `exp(-2u)` and never requires summing two near-cancelling terms.
//
// 8×eps tolerance — single `expf` (via the explicit sigmoid form) plus a
// well-conditioned multiplicative chain.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

namespace {
constexpr float  K_GELU_TANH_C_F  = 0.79788456080286535588f;  // sqrt(2 / pi)
constexpr float  K_GELU_TANH_A_F  = 0.044715f;
constexpr float  K_GELU_TANH_3A_F = 0.134145f;                // 3 * 0.044715
constexpr double K_GELU_TANH_C_D  = 0.79788456080286535588;
constexpr double K_GELU_TANH_A_D  = 0.044715;
constexpr double K_GELU_TANH_3A_D = 0.134145;
}

template <typename T>
struct GeluTanhBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T x) const {
        return dy * x;  // placeholder — explicit specializations below.
    }
};

template <>
struct GeluTanhBackwardFunctor<float> {
    __device__ __forceinline__ float operator()(float dy, float x) const {
        float xx = x * x;
        float u = K_GELU_TANH_C_F * (x + K_GELU_TANH_A_F * x * xx);
        float s = 1.0f / (1.0f + expf(-2.0f * u));
        float u_prime = K_GELU_TANH_C_F * (1.0f + K_GELU_TANH_3A_F * xx);
        return dy * s * (1.0f + 2.0f * x * (1.0f - s) * u_prime);
    }
};

template <>
struct GeluTanhBackwardFunctor<double> {
    __device__ __forceinline__ double operator()(double dy, double x) const {
        double xx = x * x;
        double u = K_GELU_TANH_C_D * (x + K_GELU_TANH_A_D * x * xx);
        double s = 1.0 / (1.0 + exp(-2.0 * u));
        double u_prime = K_GELU_TANH_C_D * (1.0 + K_GELU_TANH_3A_D * xx);
        return dy * s * (1.0 + 2.0 * x * (1.0 - s) * u_prime);
    }
};

template <>
struct GeluTanhBackwardFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half dy, __half x) const {
        float fx = __half2float(x);
        float fdy = __half2float(dy);
        float xx = fx * fx;
        float u = K_GELU_TANH_C_F * (fx + K_GELU_TANH_A_F * fx * xx);
        float s = 1.0f / (1.0f + expf(-2.0f * u));
        float u_prime = K_GELU_TANH_C_F * (1.0f + K_GELU_TANH_3A_F * xx);
        return __float2half(fdy * s * (1.0f + 2.0f * fx * (1.0f - s) * u_prime));
    }
};

template <>
struct GeluTanhBackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 dy, __nv_bfloat16 x) const {
        float fx = __bfloat162float(x);
        float fdy = __bfloat162float(dy);
        float xx = fx * fx;
        float u = K_GELU_TANH_C_F * (fx + K_GELU_TANH_A_F * fx * xx);
        float s = 1.0f / (1.0f + expf(-2.0f * u));
        float u_prime = K_GELU_TANH_C_F * (1.0f + K_GELU_TANH_3A_F * xx);
        return __float2bfloat16(fdy * s * (1.0f + 2.0f * fx * (1.0f - s) * u_prime));
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_gelu_tanh_backward_f32, float,
    baracuda::elementwise::GeluTanhBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_gelu_tanh_backward_f16, __half,
    baracuda::elementwise::GeluTanhBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_gelu_tanh_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::GeluTanhBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_gelu_tanh_backward_f64, double,
    baracuda::elementwise::GeluTanhBackwardFunctor<double>)
