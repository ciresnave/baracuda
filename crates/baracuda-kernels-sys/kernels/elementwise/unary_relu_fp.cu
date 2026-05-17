// baracuda-kernels Phase 3 unary fanout: elementwise ReLU for FP types.
//
// Implements `y = max(x, 0)` over contig + strided. f32 uses `fmaxf`;
// f64 uses `fmax`. f16 / bf16 use the f32-detour pattern.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct ReluFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct ReluFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const { return fmaxf(x, 0.0f); }
};

template <>
struct ReluFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const { return fmax(x, 0.0); }
};

template <>
struct ReluFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        return __float2half(fmaxf(__half2float(x), 0.0f));
    }
};

template <>
struct ReluFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        return __float2bfloat16(fmaxf(__bfloat162float(x), 0.0f));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_relu_f32,
    float,
    baracuda::elementwise::ReluFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_relu_f32,
    float,
    baracuda::elementwise::ReluFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_relu_f16,
    __half,
    baracuda::elementwise::ReluFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_relu_f16,
    __half,
    baracuda::elementwise::ReluFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_relu_bf16,
    __nv_bfloat16,
    baracuda::elementwise::ReluFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_relu_bf16,
    __nv_bfloat16,
    baracuda::elementwise::ReluFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_relu_f64,
    double,
    baracuda::elementwise::ReluFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_relu_f64,
    double,
    baracuda::elementwise::ReluFunctor<double>)
