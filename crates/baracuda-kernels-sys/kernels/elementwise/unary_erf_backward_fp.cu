// baracuda-kernels Phase 3 unary backward: erf backward.
//
// Forward: `y = erf(x) = (2/√π) ∫_0^x exp(-t²) dt`. Backward:
// `dx = dy * (2/√π) * exp(-x²)`. Saved-x, transcendental (one `exp`).
// f32 uses `expf`; f64 uses `exp`; f16 / bf16 use the f32-detour
// pattern, mirroring the forward Erf kernel.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

constexpr float  TWO_OVER_SQRT_PI_F = 1.1283791670955126f;
constexpr double TWO_OVER_SQRT_PI_D = 1.1283791670955126;

template <typename T>
struct ErfBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T x) const {
        return dy * x;  // placeholder — explicit specializations below are used.
    }
};

template <>
struct ErfBackwardFunctor<float> {
    __device__ __forceinline__ float operator()(float dy, float x) const {
        return dy * TWO_OVER_SQRT_PI_F * expf(-x * x);
    }
};

template <>
struct ErfBackwardFunctor<double> {
    __device__ __forceinline__ double operator()(double dy, double x) const {
        return dy * TWO_OVER_SQRT_PI_D * exp(-x * x);
    }
};

template <>
struct ErfBackwardFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half dy, __half x) const {
        float xf = __half2float(x);
        return __float2half(__half2float(dy) * TWO_OVER_SQRT_PI_F * expf(-xf * xf));
    }
};

template <>
struct ErfBackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 dy, __nv_bfloat16 x) const {
        float xf = __bfloat162float(x);
        return __float2bfloat16(__bfloat162float(dy) * TWO_OVER_SQRT_PI_F * expf(-xf * xf));
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_erf_backward_f32, float,
    baracuda::elementwise::ErfBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_erf_backward_f16, __half,
    baracuda::elementwise::ErfBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_erf_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::ErfBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_erf_backward_f64, double,
    baracuda::elementwise::ErfBackwardFunctor<double>)
