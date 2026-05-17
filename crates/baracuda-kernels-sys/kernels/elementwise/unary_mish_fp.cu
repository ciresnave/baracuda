// baracuda-kernels Phase 3 unary fanout: elementwise Mish for FP types.
//
// Implements `y = x * tanh(softplus(x))` where `softplus(x) = log(1+exp(x))`.
// Uses the stable softplus form `x > 20 ? x : log1pf(expf(x))` then
// `tanhf`. f32 / f64 use direct intrinsic math; f16 / bf16 use the f32
// detour.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct MishFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct MishFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const {
        float sp = (x > 20.0f) ? x : log1pf(expf(x));
        return x * tanhf(sp);
    }
};

template <>
struct MishFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const {
        double sp = (x > 20.0) ? x : log1p(exp(x));
        return x * tanh(sp);
    }
};

template <>
struct MishFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        float f = __half2float(x);
        float sp = (f > 20.0f) ? f : log1pf(expf(f));
        float y = f * tanhf(sp);
        return __float2half(y);
    }
};

template <>
struct MishFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float f = __bfloat162float(x);
        float sp = (f > 20.0f) ? f : log1pf(expf(f));
        float y = f * tanhf(sp);
        return __float2bfloat16(y);
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_mish_f32,
    float,
    baracuda::elementwise::MishFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_mish_f32,
    float,
    baracuda::elementwise::MishFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_mish_f16,
    __half,
    baracuda::elementwise::MishFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_mish_f16,
    __half,
    baracuda::elementwise::MishFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_mish_bf16,
    __nv_bfloat16,
    baracuda::elementwise::MishFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_mish_bf16,
    __nv_bfloat16,
    baracuda::elementwise::MishFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_mish_f64,
    double,
    baracuda::elementwise::MishFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_mish_f64,
    double,
    baracuda::elementwise::MishFunctor<double>)
