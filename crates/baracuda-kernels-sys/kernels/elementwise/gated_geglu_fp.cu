// baracuda-kernels Phase 3 Category C′ — GeGLU forward (exact, erf-based).
//
// Plan shape: split input `x` along `split_dim` into `(a, b)`; output
// `y = a · gelu(b)` with the exact erf-based GELU:
//   gelu(b) = 0.5 · b · (1 + erf(b / √2))
// Matches PyTorch's default `nn.GELU()` (not the tanh approximation).
//
// f32 uses `erff`; f64 uses `erf` (CUDA libdevice). f16 / bf16 detour
// through f32.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

namespace {
constexpr float  K_GEGLU_INV_SQRT2_F = 0.70710678118654752440f;
constexpr double K_GEGLU_INV_SQRT2_D = 0.70710678118654752440;
}

template <typename T>
struct GegluFunctor {
    __device__ __forceinline__ T operator()(T a, T b) const { return a * b; }
};

template <>
struct GegluFunctor<float> {
    __device__ __forceinline__ float operator()(float a, float b) const {
        float gelu_b = 0.5f * b * (1.0f + erff(b * K_GEGLU_INV_SQRT2_F));
        return a * gelu_b;
    }
};

template <>
struct GegluFunctor<double> {
    __device__ __forceinline__ double operator()(double a, double b) const {
        double gelu_b = 0.5 * b * (1.0 + erf(b * K_GEGLU_INV_SQRT2_D));
        return a * gelu_b;
    }
};

template <>
struct GegluFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half a, __half b) const {
        float fa = __half2float(a);
        float fb = __half2float(b);
        float gelu_b = 0.5f * fb * (1.0f + erff(fb * K_GEGLU_INV_SQRT2_F));
        return __float2half(fa * gelu_b);
    }
};

template <>
struct GegluFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 a, __nv_bfloat16 b) const {
        float fa = __bfloat162float(a);
        float fb = __bfloat162float(b);
        float gelu_b = 0.5f * fb * (1.0f + erff(fb * K_GEGLU_INV_SQRT2_F));
        return __float2bfloat16(fa * gelu_b);
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_GATED_ACTIVATION_INSTANTIATE(
    gated_geglu_f32, float,
    baracuda::elementwise::GegluFunctor<float>)

BARACUDA_KERNELS_GATED_ACTIVATION_INSTANTIATE(
    gated_geglu_f16, __half,
    baracuda::elementwise::GegluFunctor<__half>)

BARACUDA_KERNELS_GATED_ACTIVATION_INSTANTIATE(
    gated_geglu_bf16, __nv_bfloat16,
    baracuda::elementwise::GegluFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_GATED_ACTIVATION_INSTANTIATE(
    gated_geglu_f64, double,
    baracuda::elementwise::GegluFunctor<double>)
