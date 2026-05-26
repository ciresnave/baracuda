// baracuda-kernels Phase 31 — elementwise Heaviside step.
//
// `y = (x > 0) ? 1 : 0`. NaN policy: `NaN > 0` is false → 0 (matches
// PyTorch's `heaviside(x, values=0)` for the > branch). For Fuel's
// activation-suite needs the equivalence is straight `step` not the
// generalized `heaviside(x, v0)` so we stay 0/1 only.
//
// All four FP dtypes route through f32 comparisons for the half-width
// dtypes (same convention as the rest of the unary family).

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct StepFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct StepFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const {
        return (x > 0.0f) ? 1.0f : 0.0f;
    }
};

template <>
struct StepFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const {
        return (x > 0.0) ? 1.0 : 0.0;
    }
};

template <>
struct StepFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        float f = __half2float(x);
        return __float2half((f > 0.0f) ? 1.0f : 0.0f);
    }
};

template <>
struct StepFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float f = __bfloat162float(x);
        return __float2bfloat16((f > 0.0f) ? 1.0f : 0.0f);
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_step_f32,
    float,
    baracuda::elementwise::StepFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_step_f32,
    float,
    baracuda::elementwise::StepFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_step_f16,
    __half,
    baracuda::elementwise::StepFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_step_f16,
    __half,
    baracuda::elementwise::StepFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_step_bf16,
    __nv_bfloat16,
    baracuda::elementwise::StepFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_step_bf16,
    __nv_bfloat16,
    baracuda::elementwise::StepFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_step_f64,
    double,
    baracuda::elementwise::StepFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_step_f64,
    double,
    baracuda::elementwise::StepFunctor<double>)
