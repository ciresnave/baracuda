// baracuda-kernels Phase 3 unary fanout: elementwise SiLU (Swish-1) for FP types.
//
// Implements `y = x * sigmoid(x)`. Uses the numerically stable two-branch
// sigmoid form to avoid overflow. f32 / f64 use direct intrinsic math;
// f16 / bf16 use the f32 detour.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct SiluFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct SiluFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const {
        float s;
        if (x >= 0.0f) {
            s = 1.0f / (1.0f + expf(-x));
        } else {
            float e = expf(x);
            s = e / (1.0f + e);
        }
        return x * s;
    }
};

template <>
struct SiluFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const {
        double s;
        if (x >= 0.0) {
            s = 1.0 / (1.0 + exp(-x));
        } else {
            double e = exp(x);
            s = e / (1.0 + e);
        }
        return x * s;
    }
};

template <>
struct SiluFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        float f = __half2float(x);
        float s;
        if (f >= 0.0f) {
            s = 1.0f / (1.0f + expf(-f));
        } else {
            float e = expf(f);
            s = e / (1.0f + e);
        }
        return __float2half(f * s);
    }
};

template <>
struct SiluFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float f = __bfloat162float(x);
        float s;
        if (f >= 0.0f) {
            s = 1.0f / (1.0f + expf(-f));
        } else {
            float e = expf(f);
            s = e / (1.0f + e);
        }
        return __float2bfloat16(f * s);
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_silu_f32,
    float,
    baracuda::elementwise::SiluFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_silu_f32,
    float,
    baracuda::elementwise::SiluFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_silu_f16,
    __half,
    baracuda::elementwise::SiluFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_silu_f16,
    __half,
    baracuda::elementwise::SiluFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_silu_bf16,
    __nv_bfloat16,
    baracuda::elementwise::SiluFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_silu_bf16,
    __nv_bfloat16,
    baracuda::elementwise::SiluFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_silu_f64,
    double,
    baracuda::elementwise::SiluFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_silu_f64,
    double,
    baracuda::elementwise::SiluFunctor<double>)
