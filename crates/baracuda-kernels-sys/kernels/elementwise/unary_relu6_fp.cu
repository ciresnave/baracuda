// baracuda-kernels Phase 3 unary fanout: elementwise ReLU6 for FP types.
//
// Implements `y = min(max(x, 0), 6)` — the canonical MobileNet
// activation. f32 uses `fminf` / `fmaxf`; f64 uses `fmin` / `fmax`.
// f16 / bf16 use the f32 detour pattern.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct Relu6Functor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct Relu6Functor<float> {
    __device__ __forceinline__ float operator()(float x) const {
        return fminf(fmaxf(x, 0.0f), 6.0f);
    }
};

template <>
struct Relu6Functor<double> {
    __device__ __forceinline__ double operator()(double x) const {
        return fmin(fmax(x, 0.0), 6.0);
    }
};

template <>
struct Relu6Functor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        float f = __half2float(x);
        return __float2half(fminf(fmaxf(f, 0.0f), 6.0f));
    }
};

template <>
struct Relu6Functor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float f = __bfloat162float(x);
        return __float2bfloat16(fminf(fmaxf(f, 0.0f), 6.0f));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_relu6_f32,
    float,
    baracuda::elementwise::Relu6Functor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_relu6_f32,
    float,
    baracuda::elementwise::Relu6Functor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_relu6_f16,
    __half,
    baracuda::elementwise::Relu6Functor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_relu6_f16,
    __half,
    baracuda::elementwise::Relu6Functor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_relu6_bf16,
    __nv_bfloat16,
    baracuda::elementwise::Relu6Functor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_relu6_bf16,
    __nv_bfloat16,
    baracuda::elementwise::Relu6Functor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_relu6_f64,
    double,
    baracuda::elementwise::Relu6Functor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_relu6_f64,
    double,
    baracuda::elementwise::Relu6Functor<double>)
