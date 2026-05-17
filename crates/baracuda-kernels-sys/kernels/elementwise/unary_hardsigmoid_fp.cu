// baracuda-kernels Phase 3 unary fanout: elementwise Hardsigmoid for FP types.
//
// Implements `y = min(max((x + 3) / 6, 0), 1)`. Piecewise linear
// approximation of sigmoid. f32 / f64 use direct min/max intrinsics;
// f16 / bf16 use the f32 detour.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct HardsigmoidFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct HardsigmoidFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const {
        return fminf(fmaxf((x + 3.0f) * (1.0f / 6.0f), 0.0f), 1.0f);
    }
};

template <>
struct HardsigmoidFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const {
        return fmin(fmax((x + 3.0) * (1.0 / 6.0), 0.0), 1.0);
    }
};

template <>
struct HardsigmoidFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        float f = __half2float(x);
        float y = fminf(fmaxf((f + 3.0f) * (1.0f / 6.0f), 0.0f), 1.0f);
        return __float2half(y);
    }
};

template <>
struct HardsigmoidFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float f = __bfloat162float(x);
        float y = fminf(fmaxf((f + 3.0f) * (1.0f / 6.0f), 0.0f), 1.0f);
        return __float2bfloat16(y);
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_hardsigmoid_f32,
    float,
    baracuda::elementwise::HardsigmoidFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_hardsigmoid_f32,
    float,
    baracuda::elementwise::HardsigmoidFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_hardsigmoid_f16,
    __half,
    baracuda::elementwise::HardsigmoidFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_hardsigmoid_f16,
    __half,
    baracuda::elementwise::HardsigmoidFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_hardsigmoid_bf16,
    __nv_bfloat16,
    baracuda::elementwise::HardsigmoidFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_hardsigmoid_bf16,
    __nv_bfloat16,
    baracuda::elementwise::HardsigmoidFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_hardsigmoid_f64,
    double,
    baracuda::elementwise::HardsigmoidFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_hardsigmoid_f64,
    double,
    baracuda::elementwise::HardsigmoidFunctor<double>)
