// baracuda-kernels Phase 3 unary backward fanout: gelu backward (exact, erf-based).
//
// Forward: `y = 0.5 * x * (1 + erf(x / sqrt(2)))`. Backward:
//   dx = dy * (Φ(x) + x * φ(x))
//   where Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))   [standard normal CDF]
//         φ(x) = (1 / sqrt(2*pi)) * exp(-x*x / 2) [standard normal PDF]
// Saved-x; smooth, requires `erf` + `exp`.
//
// Numerical stability: for `x < 0` we compute `Φ(x) = 0.5 * erfc(-x /
// sqrt(2))` instead of `0.5 * (1 + erf(x / sqrt(2)))`. The two forms are
// mathematically identical, but `(1 + erf(x))` suffers catastrophic
// cancellation when `erf(x)` is near `-1` (i.e. `x` ≪ 0): a 1 ULP
// error in `erf` amplifies to dozens of ULPs in `(1 + erf)` because the
// sum is many orders of magnitude smaller than the summand. The `erfc`
// path computes the same small value directly with no cancellation.
//
// 8×eps tolerance — `erff` / `erfcf` / `expf` (libdevice) each diverge
// from Rust's libm by ~1-2 ULPs; the multiplicative chain plus the
// final `cdf + x*pdf` (which can still have ~10x cancellation in the
// `x ∈ [-1, 0]` band where `cdf ≈ -x*pdf`) brings worst-case relative
// error to ~6-8 ULPs.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

namespace {
constexpr float  K_INV_SQRT_2_F      = 0.70710678118654752440f;
constexpr float  K_INV_SQRT_2PI_F    = 0.39894228040143267794f;
constexpr double K_INV_SQRT_2_D      = 0.70710678118654752440;
constexpr double K_INV_SQRT_2PI_D    = 0.39894228040143267794;
}

template <typename T>
struct GeluBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T x) const {
        return dy * x;  // placeholder — explicit specializations below.
    }
};

template <>
struct GeluBackwardFunctor<float> {
    __device__ __forceinline__ float operator()(float dy, float x) const {
        float cdf = (x >= 0.0f)
            ? 0.5f * (1.0f + erff(x * K_INV_SQRT_2_F))
            : 0.5f * erfcf(-x * K_INV_SQRT_2_F);
        float pdf = K_INV_SQRT_2PI_F * expf(-0.5f * x * x);
        return dy * (cdf + x * pdf);
    }
};

template <>
struct GeluBackwardFunctor<double> {
    __device__ __forceinline__ double operator()(double dy, double x) const {
        double cdf = (x >= 0.0)
            ? 0.5 * (1.0 + erf(x * K_INV_SQRT_2_D))
            : 0.5 * erfc(-x * K_INV_SQRT_2_D);
        double pdf = K_INV_SQRT_2PI_D * exp(-0.5 * x * x);
        return dy * (cdf + x * pdf);
    }
};

template <>
struct GeluBackwardFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half dy, __half x) const {
        float fx = __half2float(x);
        float fdy = __half2float(dy);
        float cdf = (fx >= 0.0f)
            ? 0.5f * (1.0f + erff(fx * K_INV_SQRT_2_F))
            : 0.5f * erfcf(-fx * K_INV_SQRT_2_F);
        float pdf = K_INV_SQRT_2PI_F * expf(-0.5f * fx * fx);
        return __float2half(fdy * (cdf + fx * pdf));
    }
};

template <>
struct GeluBackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 dy, __nv_bfloat16 x) const {
        float fx = __bfloat162float(x);
        float fdy = __bfloat162float(dy);
        float cdf = (fx >= 0.0f)
            ? 0.5f * (1.0f + erff(fx * K_INV_SQRT_2_F))
            : 0.5f * erfcf(-fx * K_INV_SQRT_2_F);
        float pdf = K_INV_SQRT_2PI_F * expf(-0.5f * fx * fx);
        return __float2bfloat16(fdy * (cdf + fx * pdf));
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_gelu_backward_f32, float,
    baracuda::elementwise::GeluBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_gelu_backward_f16, __half,
    baracuda::elementwise::GeluBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_gelu_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::GeluBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_gelu_backward_f64, double,
    baracuda::elementwise::GeluBackwardFunctor<double>)
