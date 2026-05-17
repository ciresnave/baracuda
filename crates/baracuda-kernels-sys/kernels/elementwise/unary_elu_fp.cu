// baracuda-kernels Phase 3 unary fanout: elementwise ELU for FP types.
//
// Implements `y = x if x > 0 else α·(exp(x) - 1)`, with α hardcoded to
// 1.0 (PyTorch default `nn.ELU(alpha=1.0)`). When the parameterized-
// unary plan ships, this kernel is re-emitted with α as a runtime
// parameter. f32 uses `expf`; f64 uses `exp`. f16 / bf16 use the
// f32-detour with `expf` (same pattern as `unary_exp_fp.cu`).

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct EluFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct EluFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const {
        return (x > 0.0f) ? x : (expf(x) - 1.0f);
    }
};

template <>
struct EluFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const {
        return (x > 0.0) ? x : (exp(x) - 1.0);
    }
};

template <>
struct EluFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        float f = __half2float(x);
        float y = (f > 0.0f) ? f : (expf(f) - 1.0f);
        return __float2half(y);
    }
};

template <>
struct EluFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float f = __bfloat162float(x);
        float y = (f > 0.0f) ? f : (expf(f) - 1.0f);
        return __float2bfloat16(y);
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_elu_f32,
    float,
    baracuda::elementwise::EluFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_elu_f32,
    float,
    baracuda::elementwise::EluFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_elu_f16,
    __half,
    baracuda::elementwise::EluFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_elu_f16,
    __half,
    baracuda::elementwise::EluFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_elu_bf16,
    __nv_bfloat16,
    baracuda::elementwise::EluFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_elu_bf16,
    __nv_bfloat16,
    baracuda::elementwise::EluFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_elu_f64,
    double,
    baracuda::elementwise::EluFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_elu_f64,
    double,
    baracuda::elementwise::EluFunctor<double>)
