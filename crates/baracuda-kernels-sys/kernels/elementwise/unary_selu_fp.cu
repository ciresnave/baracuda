// baracuda-kernels Phase 3 unary fanout: elementwise SELU for FP types.
//
// Implements `y = λ * (x > 0 ? x : α * (exp(x) - 1))` — the Scaled
// Exponential Linear Unit (Klambauer et al., 2017). f32 uses `expf`;
// f64 uses `exp` (CUDA libdevice). f16 / bf16 use the f32 detour
// pattern. The α / λ constants are baked at function scope.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

// SELU constants from the original paper. Kept as constexpr literals
// (not `__device__ const`) so they inline into the functor body.
__device__ __forceinline__ constexpr float selu_alpha_f()  { return 1.6732632423543772f; }
__device__ __forceinline__ constexpr float selu_lambda_f() { return 1.0507009873554805f; }
__device__ __forceinline__ constexpr double selu_alpha_d()  { return 1.6732632423543772; }
__device__ __forceinline__ constexpr double selu_lambda_d() { return 1.0507009873554805; }

template <typename T>
struct SeluFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct SeluFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const {
        float branch = (x > 0.0f) ? x : (selu_alpha_f() * (expf(x) - 1.0f));
        return selu_lambda_f() * branch;
    }
};

template <>
struct SeluFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const {
        double branch = (x > 0.0) ? x : (selu_alpha_d() * (exp(x) - 1.0));
        return selu_lambda_d() * branch;
    }
};

template <>
struct SeluFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        float f = __half2float(x);
        float branch = (f > 0.0f) ? f : (selu_alpha_f() * (expf(f) - 1.0f));
        return __float2half(selu_lambda_f() * branch);
    }
};

template <>
struct SeluFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float f = __bfloat162float(x);
        float branch = (f > 0.0f) ? f : (selu_alpha_f() * (expf(f) - 1.0f));
        return __float2bfloat16(selu_lambda_f() * branch);
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_selu_f32,
    float,
    baracuda::elementwise::SeluFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_selu_f32,
    float,
    baracuda::elementwise::SeluFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_selu_f16,
    __half,
    baracuda::elementwise::SeluFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_selu_f16,
    __half,
    baracuda::elementwise::SeluFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_selu_bf16,
    __nv_bfloat16,
    baracuda::elementwise::SeluFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_selu_bf16,
    __nv_bfloat16,
    baracuda::elementwise::SeluFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_selu_f64,
    double,
    baracuda::elementwise::SeluFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_selu_f64,
    double,
    baracuda::elementwise::SeluFunctor<double>)
