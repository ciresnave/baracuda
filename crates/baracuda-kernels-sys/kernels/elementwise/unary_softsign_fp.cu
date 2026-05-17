// baracuda-kernels Phase 3 unary fanout: elementwise softsign for FP types.
//
// Implements `y = x / (1 + |x|)` — a smooth, bounded-in-(-1, 1)
// activation. f32 uses `fabsf`; f64 uses `fabs` (CUDA libdevice).
// f16 / bf16 use the f32 detour pattern.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct SoftsignFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct SoftsignFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const {
        return x / (1.0f + fabsf(x));
    }
};

template <>
struct SoftsignFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const {
        return x / (1.0 + fabs(x));
    }
};

template <>
struct SoftsignFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        float f = __half2float(x);
        return __float2half(f / (1.0f + fabsf(f)));
    }
};

template <>
struct SoftsignFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float f = __bfloat162float(x);
        return __float2bfloat16(f / (1.0f + fabsf(f)));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_softsign_f32,
    float,
    baracuda::elementwise::SoftsignFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_softsign_f32,
    float,
    baracuda::elementwise::SoftsignFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_softsign_f16,
    __half,
    baracuda::elementwise::SoftsignFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_softsign_f16,
    __half,
    baracuda::elementwise::SoftsignFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_softsign_bf16,
    __nv_bfloat16,
    baracuda::elementwise::SoftsignFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_softsign_bf16,
    __nv_bfloat16,
    baracuda::elementwise::SoftsignFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_softsign_f64,
    double,
    baracuda::elementwise::SoftsignFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_softsign_f64,
    double,
    baracuda::elementwise::SoftsignFunctor<double>)
