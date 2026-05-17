// baracuda-kernels Phase 3 binary fanout: elementwise fmax
// `y = fmax(a, b)` — IEEE 754 fmax (NaN-aware: returns the non-NaN
// operand when exactly one input is NaN; returns NaN only when both
// are NaN). Distinct from `BinaryKind::Maximum`, which propagates NaN.
//
// f32 → `fmaxf`, f64 → `fmax`, f16 / bf16 → f32-detour pattern matching
// the unary transcendental family. CUDA's libdevice `fmaxf` / `fmax`
// implement IEEE 754-2008 maxNum semantics on the f32-detour path.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct FmaxFunctor {
    __device__ __forceinline__ T operator()(T a, T b) const { return a; }
};

template <>
struct FmaxFunctor<float> {
    __device__ __forceinline__ float operator()(float a, float b) const {
        return fmaxf(a, b);
    }
};

template <>
struct FmaxFunctor<double> {
    __device__ __forceinline__ double operator()(double a, double b) const {
        return fmax(a, b);
    }
};

template <>
struct FmaxFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half a, __half b) const {
        return __float2half(fmaxf(__half2float(a), __half2float(b)));
    }
};

template <>
struct FmaxFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 a, __nv_bfloat16 b) const {
        return __float2bfloat16(fmaxf(__bfloat162float(a), __bfloat162float(b)));
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_fmax_f32, float, baracuda::elementwise::FmaxFunctor<float>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_fmax_f32, float, baracuda::elementwise::FmaxFunctor<float>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_fmax_f16, __half, baracuda::elementwise::FmaxFunctor<__half>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_fmax_f16, __half, baracuda::elementwise::FmaxFunctor<__half>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_fmax_bf16, __nv_bfloat16, baracuda::elementwise::FmaxFunctor<__nv_bfloat16>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_fmax_bf16, __nv_bfloat16, baracuda::elementwise::FmaxFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_fmax_f64, double, baracuda::elementwise::FmaxFunctor<double>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_fmax_f64, double, baracuda::elementwise::FmaxFunctor<double>)
