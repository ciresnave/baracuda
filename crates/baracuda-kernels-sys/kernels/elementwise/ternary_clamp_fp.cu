// baracuda-kernels Phase 3 ternary trailblazer: elementwise clamp for
// FP types.
//
// Implements `y = min(max(x, lo), hi)` over both contiguous tensors
// (fast path) and arbitrary strided / broadcast views. All four
// operands (x, lo, hi, y) share the same dtype `T`.
//
// All four FP dtypes are wired — f32 was the trailblazer; f16 / bf16 /
// f64 followed in the ternary-fanout session. The functor is
// per-dtype-specialized because `fmaxf` / `fmax` / the f16 / bf16
// detour pipelines differ. The remaining {Addcmul, Addcdiv} ops
// follow once the plan shape supports a scalar runtime parameter;
// Where is intentionally NOT in the same enum — it has a bool cond
// input and gets its own plan shape.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

// Clamp functor: `min(max(x, lo), hi)`. Per-dtype specialization
// because `fmaxf` / `fmax` / `__hmax` differ across the FP family.
template <typename T>
struct ClampFunctor {
    __device__ __forceinline__ T operator()(T x, T lo, T hi) const;
};

template <>
struct ClampFunctor<float> {
    __device__ __forceinline__ float operator()(float x, float lo, float hi) const {
        return fminf(fmaxf(x, lo), hi);
    }
};

template <>
struct ClampFunctor<double> {
    __device__ __forceinline__ double operator()(double x, double lo, double hi) const {
        return fmin(fmax(x, lo), hi);
    }
};

template <>
struct ClampFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x, __half lo, __half hi) const {
        return __float2half(fminf(fmaxf(__half2float(x), __half2float(lo)), __half2float(hi)));
    }
};

template <>
struct ClampFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16
    operator()(__nv_bfloat16 x, __nv_bfloat16 lo, __nv_bfloat16 hi) const {
        return __float2bfloat16(
            fminf(fmaxf(__bfloat162float(x), __bfloat162float(lo)), __bfloat162float(hi)));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_TERNARY_POINTWISE_INSTANTIATE(
    ternary_clamp_f32,
    float,
    baracuda::elementwise::ClampFunctor<float>)

BARACUDA_KERNELS_TERNARY_POINTWISE_INSTANTIATE_STRIDED(
    ternary_clamp_f32,
    float,
    baracuda::elementwise::ClampFunctor<float>)

BARACUDA_KERNELS_TERNARY_POINTWISE_INSTANTIATE(
    ternary_clamp_f16,
    __half,
    baracuda::elementwise::ClampFunctor<__half>)

BARACUDA_KERNELS_TERNARY_POINTWISE_INSTANTIATE_STRIDED(
    ternary_clamp_f16,
    __half,
    baracuda::elementwise::ClampFunctor<__half>)

BARACUDA_KERNELS_TERNARY_POINTWISE_INSTANTIATE(
    ternary_clamp_bf16,
    __nv_bfloat16,
    baracuda::elementwise::ClampFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_TERNARY_POINTWISE_INSTANTIATE_STRIDED(
    ternary_clamp_bf16,
    __nv_bfloat16,
    baracuda::elementwise::ClampFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_TERNARY_POINTWISE_INSTANTIATE(
    ternary_clamp_f64,
    double,
    baracuda::elementwise::ClampFunctor<double>)

BARACUDA_KERNELS_TERNARY_POINTWISE_INSTANTIATE_STRIDED(
    ternary_clamp_f64,
    double,
    baracuda::elementwise::ClampFunctor<double>)
