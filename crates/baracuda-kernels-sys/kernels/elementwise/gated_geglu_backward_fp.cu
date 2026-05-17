// baracuda-kernels Phase 3 Category C′ — GeGLU backward (exact, erf-based).
//
// Forward: `y = a · gelu(b)`, gelu(b) = 0.5·b·(1 + erf(b/√2)).
// Backward (saved `x`):
//   da = dy · gelu(b)
//   db = dy · a · gelu'(b)
//     gelu'(b) = Φ(b) + b·φ(b)
//       Φ(b) = 0.5·(1 + erf(b/√2))         [standard-normal CDF]
//       φ(b) = (1/√(2π)) · exp(-b²/2)      [standard-normal PDF]
//
// Numerical stability: for `b < 0` we compute Φ(b) = 0.5·erfc(-b/√2)
// to avoid the catastrophic `1 + erf(b)` cancellation when `erf(b)` is
// near `-1`. Same trick as `unary_gelu_backward_fp.cu`.
//
// 8×eps tolerance (libdevice erff/erfcf/expf vs Rust libm libm ~1-2 ULP
// each, plus the multiplicative chain).

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

namespace {
constexpr float  K_GEGLU_INV_SQRT2_F   = 0.70710678118654752440f;
constexpr float  K_GEGLU_INV_SQRT2PI_F = 0.39894228040143267794f;
constexpr double K_GEGLU_INV_SQRT2_D   = 0.70710678118654752440;
constexpr double K_GEGLU_INV_SQRT2PI_D = 0.39894228040143267794;
}

template <typename T>
struct GegluBackwardFunctor {
    __device__ __forceinline__ void operator()(T dy, T a, T b, T& da_out, T& db_out) const {
        da_out = dy * b;
        db_out = dy * a;
    }
};

template <>
struct GegluBackwardFunctor<float> {
    __device__ __forceinline__ void operator()(float dy, float a, float b,
                                                float& da_out, float& db_out) const {
        float cdf = (b >= 0.0f)
            ? 0.5f * (1.0f + erff(b * K_GEGLU_INV_SQRT2_F))
            : 0.5f * erfcf(-b * K_GEGLU_INV_SQRT2_F);
        float pdf = K_GEGLU_INV_SQRT2PI_F * expf(-0.5f * b * b);
        float gelu_b = b * cdf;
        float gelu_prime_b = cdf + b * pdf;
        da_out = dy * gelu_b;
        db_out = dy * a * gelu_prime_b;
    }
};

template <>
struct GegluBackwardFunctor<double> {
    __device__ __forceinline__ void operator()(double dy, double a, double b,
                                                double& da_out, double& db_out) const {
        double cdf = (b >= 0.0)
            ? 0.5 * (1.0 + erf(b * K_GEGLU_INV_SQRT2_D))
            : 0.5 * erfc(-b * K_GEGLU_INV_SQRT2_D);
        double pdf = K_GEGLU_INV_SQRT2PI_D * exp(-0.5 * b * b);
        double gelu_b = b * cdf;
        double gelu_prime_b = cdf + b * pdf;
        da_out = dy * gelu_b;
        db_out = dy * a * gelu_prime_b;
    }
};

template <>
struct GegluBackwardFunctor<__half> {
    __device__ __forceinline__ void operator()(__half dy, __half a, __half b,
                                                __half& da_out, __half& db_out) const {
        float fdy = __half2float(dy);
        float fa  = __half2float(a);
        float fb  = __half2float(b);
        float cdf = (fb >= 0.0f)
            ? 0.5f * (1.0f + erff(fb * K_GEGLU_INV_SQRT2_F))
            : 0.5f * erfcf(-fb * K_GEGLU_INV_SQRT2_F);
        float pdf = K_GEGLU_INV_SQRT2PI_F * expf(-0.5f * fb * fb);
        float gelu_b = fb * cdf;
        float gelu_prime_b = cdf + fb * pdf;
        da_out = __float2half(fdy * gelu_b);
        db_out = __float2half(fdy * fa * gelu_prime_b);
    }
};

template <>
struct GegluBackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ void operator()(__nv_bfloat16 dy, __nv_bfloat16 a, __nv_bfloat16 b,
                                                __nv_bfloat16& da_out, __nv_bfloat16& db_out) const {
        float fdy = __bfloat162float(dy);
        float fa  = __bfloat162float(a);
        float fb  = __bfloat162float(b);
        float cdf = (fb >= 0.0f)
            ? 0.5f * (1.0f + erff(fb * K_GEGLU_INV_SQRT2_F))
            : 0.5f * erfcf(-fb * K_GEGLU_INV_SQRT2_F);
        float pdf = K_GEGLU_INV_SQRT2PI_F * expf(-0.5f * fb * fb);
        float gelu_b = fb * cdf;
        float gelu_prime_b = cdf + fb * pdf;
        da_out = __float2bfloat16(fdy * gelu_b);
        db_out = __float2bfloat16(fdy * fa * gelu_prime_b);
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_GATED_ACTIVATION_BACKWARD_INSTANTIATE(
    gated_geglu_backward_f32, float,
    baracuda::elementwise::GegluBackwardFunctor<float>)

BARACUDA_KERNELS_GATED_ACTIVATION_BACKWARD_INSTANTIATE(
    gated_geglu_backward_f16, __half,
    baracuda::elementwise::GegluBackwardFunctor<__half>)

BARACUDA_KERNELS_GATED_ACTIVATION_BACKWARD_INSTANTIATE(
    gated_geglu_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::GegluBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_GATED_ACTIVATION_BACKWARD_INSTANTIATE(
    gated_geglu_backward_f64, double,
    baracuda::elementwise::GegluBackwardFunctor<double>)
