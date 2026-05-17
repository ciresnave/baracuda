// baracuda-kernels Phase 3 binary fanout: elementwise copysign
// `y = copysign(a, b) = |a| · sign(b)`.
//
// f32 → `copysignf`, f64 → `copysign`, f16 / bf16 → f32-detour pattern
// matching the unary transcendental family. copysign is a pure
// sign-bit manipulation and is well-defined for every IEEE input
// including NaN; CUDA's `copysignf` / `copysign` preserve PyTorch
// semantics (sign of b is copied onto magnitude of a).

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct CopysignFunctor {
    __device__ __forceinline__ T operator()(T a, T b) const { return a; }
};

template <>
struct CopysignFunctor<float> {
    __device__ __forceinline__ float operator()(float a, float b) const {
        return copysignf(a, b);
    }
};

template <>
struct CopysignFunctor<double> {
    __device__ __forceinline__ double operator()(double a, double b) const {
        return copysign(a, b);
    }
};

template <>
struct CopysignFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half a, __half b) const {
        return __float2half(copysignf(__half2float(a), __half2float(b)));
    }
};

template <>
struct CopysignFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 a, __nv_bfloat16 b) const {
        return __float2bfloat16(copysignf(__bfloat162float(a), __bfloat162float(b)));
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_copysign_f32, float, baracuda::elementwise::CopysignFunctor<float>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_copysign_f32, float, baracuda::elementwise::CopysignFunctor<float>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_copysign_f16, __half, baracuda::elementwise::CopysignFunctor<__half>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_copysign_f16, __half, baracuda::elementwise::CopysignFunctor<__half>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_copysign_bf16, __nv_bfloat16, baracuda::elementwise::CopysignFunctor<__nv_bfloat16>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_copysign_bf16, __nv_bfloat16, baracuda::elementwise::CopysignFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_copysign_f64, double, baracuda::elementwise::CopysignFunctor<double>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_copysign_f64, double, baracuda::elementwise::CopysignFunctor<double>)
