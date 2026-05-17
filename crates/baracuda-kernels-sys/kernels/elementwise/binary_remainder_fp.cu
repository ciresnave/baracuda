// baracuda-kernels Phase 3 binary fanout: elementwise remainder
// `y = fmod(a, b)` — C-style remainder. The result's sign follows `a`.
// Matches the C99 `fmod` / `fmodf` semantic and PyTorch's
// `BinaryKind::Remainder` doc-comment ("sign matches a"). Distinct from
// `BinaryKind::Mod`, which is Python-style (sign of b).
//
// Note: PyTorch's user-facing `torch.remainder` is Python-style, while
// `torch.fmod` is C-style. This kernel implements the C-style
// `Remainder` discriminant as documented in `BinaryKind`; the
// Python-style behavior is exposed via `BinaryKind::Mod`.
//
// f32 → `fmodf`, f64 → `fmod`. f16 / bf16 → f32-detour pattern matching
// the unary transcendental family.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct RemainderFunctor {
    __device__ __forceinline__ T operator()(T a, T b) const { return a; }
};

template <>
struct RemainderFunctor<float> {
    __device__ __forceinline__ float operator()(float a, float b) const {
        return fmodf(a, b);
    }
};

template <>
struct RemainderFunctor<double> {
    __device__ __forceinline__ double operator()(double a, double b) const {
        return fmod(a, b);
    }
};

template <>
struct RemainderFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half a, __half b) const {
        return __float2half(fmodf(__half2float(a), __half2float(b)));
    }
};

template <>
struct RemainderFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 a, __nv_bfloat16 b) const {
        return __float2bfloat16(fmodf(__bfloat162float(a), __bfloat162float(b)));
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_remainder_f32, float, baracuda::elementwise::RemainderFunctor<float>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_remainder_f32, float, baracuda::elementwise::RemainderFunctor<float>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_remainder_f16, __half, baracuda::elementwise::RemainderFunctor<__half>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_remainder_f16, __half, baracuda::elementwise::RemainderFunctor<__half>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_remainder_bf16, __nv_bfloat16, baracuda::elementwise::RemainderFunctor<__nv_bfloat16>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_remainder_bf16, __nv_bfloat16, baracuda::elementwise::RemainderFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_remainder_f64, double, baracuda::elementwise::RemainderFunctor<double>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_remainder_f64, double, baracuda::elementwise::RemainderFunctor<double>)
