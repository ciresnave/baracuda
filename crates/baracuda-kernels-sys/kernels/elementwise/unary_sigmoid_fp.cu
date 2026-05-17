// baracuda-kernels Phase 3 unary fanout: elementwise sigmoid for FP types.
//
// Implements `y = 1 / (1 + exp(-x))`. Uses the numerically stable
// two-branch form to avoid `exp(-x)` overflowing when x is large
// negative (and therefore `-x` is large positive):
//     x >= 0:  1 / (1 + exp(-x))
//     x <  0:  exp(x) / (1 + exp(x))
// f32 / f64 use direct intrinsic math; f16 / bf16 use the f32 detour.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct SigmoidFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct SigmoidFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const {
        if (x >= 0.0f) {
            return 1.0f / (1.0f + expf(-x));
        } else {
            float e = expf(x);
            return e / (1.0f + e);
        }
    }
};

template <>
struct SigmoidFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const {
        if (x >= 0.0) {
            return 1.0 / (1.0 + exp(-x));
        } else {
            double e = exp(x);
            return e / (1.0 + e);
        }
    }
};

template <>
struct SigmoidFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        float f = __half2float(x);
        float y;
        if (f >= 0.0f) {
            y = 1.0f / (1.0f + expf(-f));
        } else {
            float e = expf(f);
            y = e / (1.0f + e);
        }
        return __float2half(y);
    }
};

template <>
struct SigmoidFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float f = __bfloat162float(x);
        float y;
        if (f >= 0.0f) {
            y = 1.0f / (1.0f + expf(-f));
        } else {
            float e = expf(f);
            y = e / (1.0f + e);
        }
        return __float2bfloat16(y);
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_sigmoid_f32,
    float,
    baracuda::elementwise::SigmoidFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_sigmoid_f32,
    float,
    baracuda::elementwise::SigmoidFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_sigmoid_f16,
    __half,
    baracuda::elementwise::SigmoidFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_sigmoid_f16,
    __half,
    baracuda::elementwise::SigmoidFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_sigmoid_bf16,
    __nv_bfloat16,
    baracuda::elementwise::SigmoidFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_sigmoid_bf16,
    __nv_bfloat16,
    baracuda::elementwise::SigmoidFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_sigmoid_f64,
    double,
    baracuda::elementwise::SigmoidFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_sigmoid_f64,
    double,
    baracuda::elementwise::SigmoidFunctor<double>)
