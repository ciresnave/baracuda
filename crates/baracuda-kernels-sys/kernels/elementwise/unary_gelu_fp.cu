// baracuda-kernels Phase 3 unary fanout: elementwise GELU (exact, erf-based) for FP types.
//
// Implements `y = 0.5 * x * (1 + erf(x / sqrt(2)))`. PyTorch's default
// `nn.GELU()`. f32 uses `erff`; f64 uses `erf` (CUDA libdevice). f16 /
// bf16 use the f32 detour.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

// 1/sqrt(2). Used as a multiplicative factor for `x / sqrt(2)`.
__device__ __forceinline__ constexpr float gelu_inv_sqrt2_f() { return 0.7071067811865475f; }
__device__ __forceinline__ constexpr double gelu_inv_sqrt2_d() { return 0.7071067811865475; }

template <typename T>
struct GeluFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct GeluFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const {
        return 0.5f * x * (1.0f + erff(x * gelu_inv_sqrt2_f()));
    }
};

template <>
struct GeluFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const {
        return 0.5 * x * (1.0 + erf(x * gelu_inv_sqrt2_d()));
    }
};

template <>
struct GeluFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        float f = __half2float(x);
        float y = 0.5f * f * (1.0f + erff(f * gelu_inv_sqrt2_f()));
        return __float2half(y);
    }
};

template <>
struct GeluFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float f = __bfloat162float(x);
        float y = 0.5f * f * (1.0f + erff(f * gelu_inv_sqrt2_f()));
        return __float2bfloat16(y);
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_gelu_f32,
    float,
    baracuda::elementwise::GeluFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_gelu_f32,
    float,
    baracuda::elementwise::GeluFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_gelu_f16,
    __half,
    baracuda::elementwise::GeluFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_gelu_f16,
    __half,
    baracuda::elementwise::GeluFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_gelu_bf16,
    __nv_bfloat16,
    baracuda::elementwise::GeluFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_gelu_bf16,
    __nv_bfloat16,
    baracuda::elementwise::GeluFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_gelu_f64,
    double,
    baracuda::elementwise::GeluFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_gelu_f64,
    double,
    baracuda::elementwise::GeluFunctor<double>)
