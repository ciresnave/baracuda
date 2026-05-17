// baracuda-kernels Phase 3 binary fanout: elementwise pow `y = a^b`.
//
// f32 → `powf`, f64 → `pow`, f16 / bf16 → f32-detour pattern matching
// the unary transcendental family (`exp`, `log`, ...). Caller is
// responsible for guarding against undefined regions (a < 0 with
// non-integer b → NaN, by IEEE 754 / C `pow` semantics — same as
// PyTorch's `torch.pow`).

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct PowFunctor {
    __device__ __forceinline__ T operator()(T a, T b) const { return a; }
};

template <>
struct PowFunctor<float> {
    __device__ __forceinline__ float operator()(float a, float b) const {
        return powf(a, b);
    }
};

template <>
struct PowFunctor<double> {
    __device__ __forceinline__ double operator()(double a, double b) const {
        return pow(a, b);
    }
};

template <>
struct PowFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half a, __half b) const {
        return __float2half(powf(__half2float(a), __half2float(b)));
    }
};

template <>
struct PowFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 a, __nv_bfloat16 b) const {
        return __float2bfloat16(powf(__bfloat162float(a), __bfloat162float(b)));
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_pow_f32, float, baracuda::elementwise::PowFunctor<float>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_pow_f32, float, baracuda::elementwise::PowFunctor<float>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_pow_f16, __half, baracuda::elementwise::PowFunctor<__half>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_pow_f16, __half, baracuda::elementwise::PowFunctor<__half>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_pow_bf16, __nv_bfloat16, baracuda::elementwise::PowFunctor<__nv_bfloat16>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_pow_bf16, __nv_bfloat16, baracuda::elementwise::PowFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_pow_f64, double, baracuda::elementwise::PowFunctor<double>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_pow_f64, double, baracuda::elementwise::PowFunctor<double>)
