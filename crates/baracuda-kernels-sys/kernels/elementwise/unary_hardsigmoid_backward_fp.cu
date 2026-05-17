// baracuda-kernels Phase 3 unary backward fanout: hardsigmoid backward.
//
// Forward: `y = clamp(x / 6 + 0.5, 0, 1)`. Backward: PyTorch convention
//   dx = (-3 < x < 3) ? dy / 6 : 0
// Saved-x; piecewise. At the exact boundaries x == -3 and x == 3 the
// gradient is zero (subgradient undefined).
//
// Numerical note: the scalar `dy / 6.0` is used directly (not `dy *
// (1/6)`) because `1/6` is not representable exactly in any IEEE-754
// binary format, so the multiplicative form would NOT round-trip
// bit-exact against a `f32::div` reference. f16 / bf16 round once on
// store (the f32-detour pattern).

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct HardsigmoidBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T x) const {
        return dy * x;  // placeholder — explicit specializations below.
    }
};

template <>
struct HardsigmoidBackwardFunctor<float> {
    __device__ __forceinline__ float operator()(float dy, float x) const {
        return (x > -3.0f && x < 3.0f) ? (dy / 6.0f) : 0.0f;
    }
};

template <>
struct HardsigmoidBackwardFunctor<double> {
    __device__ __forceinline__ double operator()(double dy, double x) const {
        return (x > -3.0 && x < 3.0) ? (dy / 6.0) : 0.0;
    }
};

template <>
struct HardsigmoidBackwardFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half dy, __half x) const {
        float fx = __half2float(x);
        if (fx > -3.0f && fx < 3.0f) {
            return __float2half(__half2float(dy) / 6.0f);
        }
        return __float2half(0.0f);
    }
};

template <>
struct HardsigmoidBackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 dy, __nv_bfloat16 x) const {
        float fx = __bfloat162float(x);
        if (fx > -3.0f && fx < 3.0f) {
            return __float2bfloat16(__bfloat162float(dy) / 6.0f);
        }
        return __float2bfloat16(0.0f);
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_hardsigmoid_backward_f32, float,
    baracuda::elementwise::HardsigmoidBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_hardsigmoid_backward_f16, __half,
    baracuda::elementwise::HardsigmoidBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_hardsigmoid_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::HardsigmoidBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_hardsigmoid_backward_f64, double,
    baracuda::elementwise::HardsigmoidBackwardFunctor<double>)
