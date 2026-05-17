// baracuda-kernels Phase 3 unary fanout: elementwise GELU (tanh approximation) for FP types.
//
// Implements `y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
// PyTorch's `nn.GELU(approximate='tanh')`. f32 uses `tanhf`; f64 uses
// `tanh`. f16 / bf16 use the f32 detour.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

// sqrt(2 / pi).
__device__ __forceinline__ constexpr float gelu_tanh_kappa_f() { return 0.7978845608028654f; }
__device__ __forceinline__ constexpr double gelu_tanh_kappa_d() { return 0.7978845608028654; }
__device__ __forceinline__ constexpr float gelu_tanh_alpha_f() { return 0.044715f; }
__device__ __forceinline__ constexpr double gelu_tanh_alpha_d() { return 0.044715; }

template <typename T>
struct GeluTanhFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct GeluTanhFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const {
        float inner = gelu_tanh_kappa_f() * (x + gelu_tanh_alpha_f() * x * x * x);
        return 0.5f * x * (1.0f + tanhf(inner));
    }
};

template <>
struct GeluTanhFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const {
        double inner = gelu_tanh_kappa_d() * (x + gelu_tanh_alpha_d() * x * x * x);
        return 0.5 * x * (1.0 + tanh(inner));
    }
};

template <>
struct GeluTanhFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        float f = __half2float(x);
        float inner = gelu_tanh_kappa_f() * (f + gelu_tanh_alpha_f() * f * f * f);
        float y = 0.5f * f * (1.0f + tanhf(inner));
        return __float2half(y);
    }
};

template <>
struct GeluTanhFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float f = __bfloat162float(x);
        float inner = gelu_tanh_kappa_f() * (f + gelu_tanh_alpha_f() * f * f * f);
        float y = 0.5f * f * (1.0f + tanhf(inner));
        return __float2bfloat16(y);
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_gelu_tanh_f32,
    float,
    baracuda::elementwise::GeluTanhFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_gelu_tanh_f32,
    float,
    baracuda::elementwise::GeluTanhFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_gelu_tanh_f16,
    __half,
    baracuda::elementwise::GeluTanhFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_gelu_tanh_f16,
    __half,
    baracuda::elementwise::GeluTanhFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_gelu_tanh_bf16,
    __nv_bfloat16,
    baracuda::elementwise::GeluTanhFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_gelu_tanh_bf16,
    __nv_bfloat16,
    baracuda::elementwise::GeluTanhFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_gelu_tanh_f64,
    double,
    baracuda::elementwise::GeluTanhFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_gelu_tanh_f64,
    double,
    baracuda::elementwise::GeluTanhFunctor<double>)
