// baracuda-kernels Phase 3 binary fanout: elementwise mod
// `y = a - floor(a / b) * b` — Python-style modulo. The result's sign
// follows `b`. Matches Python's `%` operator and `torch.remainder`'s
// effective semantics (see also `BinaryKind::Mod` doc-comment: "sign
// matches b").
//
// Implementation: compute `r = fmod(a, b)` (C-style remainder, sign of
// `a`), then if `r != 0` and the signs of `r` and `b` differ, add `b`
// to flip the sign. This matches Python's modulo identity
// `a == (a // b) * b + (a % b)` with `a // b == floor(a / b)`.
//
// Distinct from `BinaryKind::Remainder`, which is C-style (sign of a).
//
// f32 → `fmodf` + sign-fix; f64 → `fmod` + sign-fix. f16 / bf16 →
// f32-detour pattern matching the unary transcendental family.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

__device__ __forceinline__ float python_mod_f32(float a, float b) {
    float r = fmodf(a, b);
    // Sign-mismatch fix: if remainder is nonzero and its sign differs
    // from b's, add b. The `(r < 0) != (b < 0)` predicate handles the
    // four-quadrant case without branching on signbit.
    if (r != 0.0f && ((r < 0.0f) != (b < 0.0f))) {
        r += b;
    }
    return r;
}

__device__ __forceinline__ double python_mod_f64(double a, double b) {
    double r = fmod(a, b);
    if (r != 0.0 && ((r < 0.0) != (b < 0.0))) {
        r += b;
    }
    return r;
}

template <typename T>
struct ModFunctor {
    __device__ __forceinline__ T operator()(T a, T b) const { return a; }
};

template <>
struct ModFunctor<float> {
    __device__ __forceinline__ float operator()(float a, float b) const {
        return python_mod_f32(a, b);
    }
};

template <>
struct ModFunctor<double> {
    __device__ __forceinline__ double operator()(double a, double b) const {
        return python_mod_f64(a, b);
    }
};

template <>
struct ModFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half a, __half b) const {
        return __float2half(python_mod_f32(__half2float(a), __half2float(b)));
    }
};

template <>
struct ModFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 a, __nv_bfloat16 b) const {
        return __float2bfloat16(python_mod_f32(__bfloat162float(a), __bfloat162float(b)));
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_mod_f32, float, baracuda::elementwise::ModFunctor<float>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_mod_f32, float, baracuda::elementwise::ModFunctor<float>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_mod_f16, __half, baracuda::elementwise::ModFunctor<__half>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_mod_f16, __half, baracuda::elementwise::ModFunctor<__half>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_mod_bf16, __nv_bfloat16, baracuda::elementwise::ModFunctor<__nv_bfloat16>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_mod_bf16, __nv_bfloat16, baracuda::elementwise::ModFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_mod_f64, double, baracuda::elementwise::ModFunctor<double>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_mod_f64, double, baracuda::elementwise::ModFunctor<double>)
