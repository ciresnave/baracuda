// baracuda-kernels Phase 3 unary fanout: elementwise softplus for FP types.
//
// Implements `y = log(1 + exp(x))`. Uses the stable branch
//     x > 20:  x          (exp(x) so large that log(1 + exp(x)) == x to f32 precision)
//     else:    log1p(exp(x))
// to avoid `exp(x)` overflowing f32 when x is large. f64 uses a higher
// 700-cutoff. f16 / bf16 use the f32 detour with the same 20-cutoff.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct SoftplusFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct SoftplusFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const {
        return (x > 20.0f) ? x : log1pf(expf(x));
    }
};

template <>
struct SoftplusFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const {
        return (x > 20.0) ? x : log1p(exp(x));
    }
};

template <>
struct SoftplusFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        float f = __half2float(x);
        float y = (f > 20.0f) ? f : log1pf(expf(f));
        return __float2half(y);
    }
};

template <>
struct SoftplusFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float f = __bfloat162float(x);
        float y = (f > 20.0f) ? f : log1pf(expf(f));
        return __float2bfloat16(y);
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_softplus_f32,
    float,
    baracuda::elementwise::SoftplusFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_softplus_f32,
    float,
    baracuda::elementwise::SoftplusFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_softplus_f16,
    __half,
    baracuda::elementwise::SoftplusFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_softplus_f16,
    __half,
    baracuda::elementwise::SoftplusFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_softplus_bf16,
    __nv_bfloat16,
    baracuda::elementwise::SoftplusFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_softplus_bf16,
    __nv_bfloat16,
    baracuda::elementwise::SoftplusFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_softplus_f64,
    double,
    baracuda::elementwise::SoftplusFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_softplus_f64,
    double,
    baracuda::elementwise::SoftplusFunctor<double>)
