// baracuda-kernels Phase 3 unary fanout: elementwise sinh for FP types.
//
// Implements `y = sinh(x)` over contig + strided. f32 uses `sinhf`; f64
// uses `sinh`. f16 / bf16 use the f32-detour pattern.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct SinhFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct SinhFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const { return sinhf(x); }
};

template <>
struct SinhFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const { return sinh(x); }
};

template <>
struct SinhFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        return __float2half(sinhf(__half2float(x)));
    }
};

template <>
struct SinhFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        return __float2bfloat16(sinhf(__bfloat162float(x)));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_sinh_f32,
    float,
    baracuda::elementwise::SinhFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_sinh_f32,
    float,
    baracuda::elementwise::SinhFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_sinh_f16,
    __half,
    baracuda::elementwise::SinhFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_sinh_f16,
    __half,
    baracuda::elementwise::SinhFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_sinh_bf16,
    __nv_bfloat16,
    baracuda::elementwise::SinhFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_sinh_bf16,
    __nv_bfloat16,
    baracuda::elementwise::SinhFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_sinh_f64,
    double,
    baracuda::elementwise::SinhFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_sinh_f64,
    double,
    baracuda::elementwise::SinhFunctor<double>)
