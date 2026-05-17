// baracuda-kernels Phase 3 binary fanout: elementwise atan2 `y = atan2(a, b)`.
//
// f32 → `atan2f`, f64 → `atan2`, f16 / bf16 → f32-detour pattern matching
// the unary transcendental family. atan2 is well-defined everywhere
// except (a, b) == (0, 0); PyTorch / IEEE 754 return 0 in that case
// for `atan2f(0, 0)` on CUDA — we let the libdevice intrinsic produce
// whatever it does there.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct Atan2Functor {
    __device__ __forceinline__ T operator()(T a, T b) const { return a; }
};

template <>
struct Atan2Functor<float> {
    __device__ __forceinline__ float operator()(float a, float b) const {
        return atan2f(a, b);
    }
};

template <>
struct Atan2Functor<double> {
    __device__ __forceinline__ double operator()(double a, double b) const {
        return atan2(a, b);
    }
};

template <>
struct Atan2Functor<__half> {
    __device__ __forceinline__ __half operator()(__half a, __half b) const {
        return __float2half(atan2f(__half2float(a), __half2float(b)));
    }
};

template <>
struct Atan2Functor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 a, __nv_bfloat16 b) const {
        return __float2bfloat16(atan2f(__bfloat162float(a), __bfloat162float(b)));
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_atan2_f32, float, baracuda::elementwise::Atan2Functor<float>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_atan2_f32, float, baracuda::elementwise::Atan2Functor<float>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_atan2_f16, __half, baracuda::elementwise::Atan2Functor<__half>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_atan2_f16, __half, baracuda::elementwise::Atan2Functor<__half>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_atan2_bf16, __nv_bfloat16, baracuda::elementwise::Atan2Functor<__nv_bfloat16>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_atan2_bf16, __nv_bfloat16, baracuda::elementwise::Atan2Functor<__nv_bfloat16>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_atan2_f64, double, baracuda::elementwise::Atan2Functor<double>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_atan2_f64, double, baracuda::elementwise::Atan2Functor<double>)
