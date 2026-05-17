// baracuda-kernels Phase 3 unary fanout: elementwise expm1 for FP types.
//
// Implements `y = exp(x) - 1` over contig + strided. The dedicated
// `expm1f` / `expm1` intrinsics preserve precision near zero (vs the
// naive `expf(x) - 1.0f` form which suffers catastrophic cancellation
// for small `x`). f16 / bf16 use the f32-detour through `expm1f`.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct Expm1Functor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct Expm1Functor<float> {
    __device__ __forceinline__ float operator()(float x) const { return expm1f(x); }
};

template <>
struct Expm1Functor<double> {
    __device__ __forceinline__ double operator()(double x) const { return expm1(x); }
};

template <>
struct Expm1Functor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        return __float2half(expm1f(__half2float(x)));
    }
};

template <>
struct Expm1Functor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        return __float2bfloat16(expm1f(__bfloat162float(x)));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_expm1_f32,
    float,
    baracuda::elementwise::Expm1Functor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_expm1_f32,
    float,
    baracuda::elementwise::Expm1Functor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_expm1_f16,
    __half,
    baracuda::elementwise::Expm1Functor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_expm1_f16,
    __half,
    baracuda::elementwise::Expm1Functor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_expm1_bf16,
    __nv_bfloat16,
    baracuda::elementwise::Expm1Functor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_expm1_bf16,
    __nv_bfloat16,
    baracuda::elementwise::Expm1Functor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_expm1_f64,
    double,
    baracuda::elementwise::Expm1Functor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_expm1_f64,
    double,
    baracuda::elementwise::Expm1Functor<double>)
