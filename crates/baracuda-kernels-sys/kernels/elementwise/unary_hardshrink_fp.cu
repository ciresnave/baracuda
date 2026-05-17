// baracuda-kernels Phase 3 unary fanout: elementwise hardshrink for FP types.
//
// Implements `y = x if |x| > λ else 0`, with λ hardcoded to 0.5
// (PyTorch default `nn.Hardshrink(lambd=0.5)`). When the parameterized-
// unary plan ships, this kernel is re-emitted with λ as a runtime
// parameter. Pure piecewise — no transcendentals. f16 / bf16 use the
// f32-detour pattern (compare in f32, but `x` itself is preserved
// bit-exact on the kept branch).

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct HardshrinkFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct HardshrinkFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const {
        return (fabsf(x) > 0.5f) ? x : 0.0f;
    }
};

template <>
struct HardshrinkFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const {
        return (fabs(x) > 0.5) ? x : 0.0;
    }
};

template <>
struct HardshrinkFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        float f = __half2float(x);
        return (fabsf(f) > 0.5f) ? x : __float2half(0.0f);
    }
};

template <>
struct HardshrinkFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float f = __bfloat162float(x);
        return (fabsf(f) > 0.5f) ? x : __float2bfloat16(0.0f);
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_hardshrink_f32,
    float,
    baracuda::elementwise::HardshrinkFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_hardshrink_f32,
    float,
    baracuda::elementwise::HardshrinkFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_hardshrink_f16,
    __half,
    baracuda::elementwise::HardshrinkFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_hardshrink_f16,
    __half,
    baracuda::elementwise::HardshrinkFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_hardshrink_bf16,
    __nv_bfloat16,
    baracuda::elementwise::HardshrinkFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_hardshrink_bf16,
    __nv_bfloat16,
    baracuda::elementwise::HardshrinkFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_hardshrink_f64,
    double,
    baracuda::elementwise::HardshrinkFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_hardshrink_f64,
    double,
    baracuda::elementwise::HardshrinkFunctor<double>)
