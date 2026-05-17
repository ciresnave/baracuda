// baracuda-kernels Phase 3 unary fanout: elementwise tan for FP types.
//
// Implements `y = tan(x)` over contig + strided. f32 uses `tanf`; f64
// uses `tan`. f16 / bf16 use the f32-detour pattern. Note: `tan` has
// poles at `x = (k + 0.5) * π`; callers must keep inputs away from
// those values to stay in-representable for half precision.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct TanFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct TanFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const { return tanf(x); }
};

template <>
struct TanFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const { return tan(x); }
};

template <>
struct TanFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        return __float2half(tanf(__half2float(x)));
    }
};

template <>
struct TanFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        return __float2bfloat16(tanf(__bfloat162float(x)));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_tan_f32,
    float,
    baracuda::elementwise::TanFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_tan_f32,
    float,
    baracuda::elementwise::TanFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_tan_f16,
    __half,
    baracuda::elementwise::TanFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_tan_f16,
    __half,
    baracuda::elementwise::TanFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_tan_bf16,
    __nv_bfloat16,
    baracuda::elementwise::TanFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_tan_bf16,
    __nv_bfloat16,
    baracuda::elementwise::TanFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_tan_f64,
    double,
    baracuda::elementwise::TanFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_tan_f64,
    double,
    baracuda::elementwise::TanFunctor<double>)
