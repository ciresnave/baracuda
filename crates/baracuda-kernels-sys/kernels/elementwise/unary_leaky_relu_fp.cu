// baracuda-kernels Phase 3 unary fanout: elementwise leaky-ReLU for FP types.
//
// Implements `y = x if x > 0 else α·x`, with α hardcoded to 0.01
// (PyTorch default `nn.LeakyReLU(negative_slope=0.01)`). When the
// parameterized-unary plan ships, this kernel is the template that gets
// re-emitted with α as a runtime parameter — same shape and dispatch.
// f16 / bf16 use the f32-detour pattern (compare and multiply in f32,
// round once on store).

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct LeakyReluFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct LeakyReluFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const {
        return (x > 0.0f) ? x : 0.01f * x;
    }
};

template <>
struct LeakyReluFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const {
        return (x > 0.0) ? x : 0.01 * x;
    }
};

template <>
struct LeakyReluFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        float f = __half2float(x);
        float y = (f > 0.0f) ? f : 0.01f * f;
        return __float2half(y);
    }
};

template <>
struct LeakyReluFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float f = __bfloat162float(x);
        float y = (f > 0.0f) ? f : 0.01f * f;
        return __float2bfloat16(y);
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_leaky_relu_f32,
    float,
    baracuda::elementwise::LeakyReluFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_leaky_relu_f32,
    float,
    baracuda::elementwise::LeakyReluFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_leaky_relu_f16,
    __half,
    baracuda::elementwise::LeakyReluFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_leaky_relu_f16,
    __half,
    baracuda::elementwise::LeakyReluFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_leaky_relu_bf16,
    __nv_bfloat16,
    baracuda::elementwise::LeakyReluFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_leaky_relu_bf16,
    __nv_bfloat16,
    baracuda::elementwise::LeakyReluFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_leaky_relu_f64,
    double,
    baracuda::elementwise::LeakyReluFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_leaky_relu_f64,
    double,
    baracuda::elementwise::LeakyReluFunctor<double>)
