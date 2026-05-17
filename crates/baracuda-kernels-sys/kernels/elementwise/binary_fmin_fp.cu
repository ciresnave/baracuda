// baracuda-kernels Phase 3 binary fanout: elementwise fmin
// `y = fmin(a, b)` — IEEE 754 fmin (NaN-aware: returns the non-NaN
// operand when exactly one input is NaN; returns NaN only when both
// are NaN). Distinct from `BinaryKind::Minimum`, which propagates NaN.
//
// f32 → `fminf`, f64 → `fmin`, f16 / bf16 → f32-detour pattern matching
// the unary transcendental family. CUDA's libdevice `fminf` / `fmin`
// implement IEEE 754-2008 minNum semantics on the f32-detour path.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct FminFunctor {
    __device__ __forceinline__ T operator()(T a, T b) const { return a; }
};

template <>
struct FminFunctor<float> {
    __device__ __forceinline__ float operator()(float a, float b) const {
        return fminf(a, b);
    }
};

template <>
struct FminFunctor<double> {
    __device__ __forceinline__ double operator()(double a, double b) const {
        return fmin(a, b);
    }
};

template <>
struct FminFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half a, __half b) const {
        return __float2half(fminf(__half2float(a), __half2float(b)));
    }
};

template <>
struct FminFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 a, __nv_bfloat16 b) const {
        return __float2bfloat16(fminf(__bfloat162float(a), __bfloat162float(b)));
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_fmin_f32, float, baracuda::elementwise::FminFunctor<float>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_fmin_f32, float, baracuda::elementwise::FminFunctor<float>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_fmin_f16, __half, baracuda::elementwise::FminFunctor<__half>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_fmin_f16, __half, baracuda::elementwise::FminFunctor<__half>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_fmin_bf16, __nv_bfloat16, baracuda::elementwise::FminFunctor<__nv_bfloat16>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_fmin_bf16, __nv_bfloat16, baracuda::elementwise::FminFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_fmin_f64, double, baracuda::elementwise::FminFunctor<double>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_fmin_f64, double, baracuda::elementwise::FminFunctor<double>)
