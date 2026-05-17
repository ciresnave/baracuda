// baracuda-kernels Phase 3 unary fanout: elementwise Hardtanh for FP types.
//
// Implements `y = min(max(x, -1), +1)`. Piecewise linear clamp; PyTorch
// default bounds are -1 and +1 (`nn.Hardtanh()` with the default `(-1, 1)`).
// f32 / f64 use direct min/max intrinsics; f16 / bf16 use the f32 detour.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct HardtanhFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct HardtanhFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const {
        return fminf(fmaxf(x, -1.0f), 1.0f);
    }
};

template <>
struct HardtanhFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const {
        return fmin(fmax(x, -1.0), 1.0);
    }
};

template <>
struct HardtanhFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        float f = __half2float(x);
        return __float2half(fminf(fmaxf(f, -1.0f), 1.0f));
    }
};

template <>
struct HardtanhFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float f = __bfloat162float(x);
        return __float2bfloat16(fminf(fmaxf(f, -1.0f), 1.0f));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_hardtanh_f32,
    float,
    baracuda::elementwise::HardtanhFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_hardtanh_f32,
    float,
    baracuda::elementwise::HardtanhFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_hardtanh_f16,
    __half,
    baracuda::elementwise::HardtanhFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_hardtanh_f16,
    __half,
    baracuda::elementwise::HardtanhFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_hardtanh_bf16,
    __nv_bfloat16,
    baracuda::elementwise::HardtanhFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_hardtanh_bf16,
    __nv_bfloat16,
    baracuda::elementwise::HardtanhFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_hardtanh_f64,
    double,
    baracuda::elementwise::HardtanhFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_hardtanh_f64,
    double,
    baracuda::elementwise::HardtanhFunctor<double>)
