// baracuda-kernels Phase 3 unary fanout: elementwise erfc for FP types.
//
// Implements `y = erfc(x)` (complementary error function, `1 - erf(x)`).
// f32 uses `erfcf`; f64 uses `erfc` (CUDA libdevice). f16 / bf16 use
// the f32 detour pattern.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct ErfcFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct ErfcFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const { return erfcf(x); }
};

template <>
struct ErfcFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const { return erfc(x); }
};

template <>
struct ErfcFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        return __float2half(erfcf(__half2float(x)));
    }
};

template <>
struct ErfcFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        return __float2bfloat16(erfcf(__bfloat162float(x)));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_erfc_f32,
    float,
    baracuda::elementwise::ErfcFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_erfc_f32,
    float,
    baracuda::elementwise::ErfcFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_erfc_f16,
    __half,
    baracuda::elementwise::ErfcFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_erfc_f16,
    __half,
    baracuda::elementwise::ErfcFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_erfc_bf16,
    __nv_bfloat16,
    baracuda::elementwise::ErfcFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_erfc_bf16,
    __nv_bfloat16,
    baracuda::elementwise::ErfcFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_erfc_f64,
    double,
    baracuda::elementwise::ErfcFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_erfc_f64,
    double,
    baracuda::elementwise::ErfcFunctor<double>)
