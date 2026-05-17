// baracuda-kernels Phase 3 unary fanout: elementwise tanhshrink for FP types.
//
// Implements `y = x - tanh(x)` (the "tanh-shrink" residual activation).
// f32 uses `tanhf`; f64 uses `tanh` (CUDA libdevice). f16 / bf16 use
// the f32 detour pattern.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct TanhshrinkFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct TanhshrinkFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const {
        return x - tanhf(x);
    }
};

template <>
struct TanhshrinkFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const {
        return x - tanh(x);
    }
};

template <>
struct TanhshrinkFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        float f = __half2float(x);
        return __float2half(f - tanhf(f));
    }
};

template <>
struct TanhshrinkFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float f = __bfloat162float(x);
        return __float2bfloat16(f - tanhf(f));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_tanhshrink_f32,
    float,
    baracuda::elementwise::TanhshrinkFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_tanhshrink_f32,
    float,
    baracuda::elementwise::TanhshrinkFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_tanhshrink_f16,
    __half,
    baracuda::elementwise::TanhshrinkFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_tanhshrink_f16,
    __half,
    baracuda::elementwise::TanhshrinkFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_tanhshrink_bf16,
    __nv_bfloat16,
    baracuda::elementwise::TanhshrinkFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_tanhshrink_bf16,
    __nv_bfloat16,
    baracuda::elementwise::TanhshrinkFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_tanhshrink_f64,
    double,
    baracuda::elementwise::TanhshrinkFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_tanhshrink_f64,
    double,
    baracuda::elementwise::TanhshrinkFunctor<double>)
