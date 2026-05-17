// baracuda-kernels Phase 3 unary fanout: elementwise absolute value for
// FP types.
//
// Implements `y = |x|` over both contiguous tensors (fast path) and
// arbitrary strided views. The kernel templates and INSTANTIATE macros
// live in `include/baracuda_elementwise.cuh`; this file supplies the
// `AbsFunctor<T>` and the per-dtype instantiations.
//
// f32 / f64 use the standard `fabsf` / `fabs` intrinsics. The half-
// precision specializations call CUDA's native `__habs` (cuda_fp16.h /
// cuda_bf16.h) — one PTX instruction on sm_53+ / sm_80+, cheaper than
// the f32-detour.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

// Generic fallback — emitted but unused; per-dtype specializations
// below cover every wired dtype.
template <typename T>
struct AbsFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct AbsFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const { return fabsf(x); }
};

template <>
struct AbsFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const { return fabs(x); }
};

template <>
struct AbsFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const { return __habs(x); }
};

template <>
struct AbsFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        return __habs(x);
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_abs_f32,
    float,
    baracuda::elementwise::AbsFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_abs_f32,
    float,
    baracuda::elementwise::AbsFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_abs_f16,
    __half,
    baracuda::elementwise::AbsFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_abs_f16,
    __half,
    baracuda::elementwise::AbsFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_abs_bf16,
    __nv_bfloat16,
    baracuda::elementwise::AbsFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_abs_bf16,
    __nv_bfloat16,
    baracuda::elementwise::AbsFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_abs_f64,
    double,
    baracuda::elementwise::AbsFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_abs_f64,
    double,
    baracuda::elementwise::AbsFunctor<double>)
