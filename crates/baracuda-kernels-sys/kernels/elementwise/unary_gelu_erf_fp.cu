// baracuda-kernels Phase 31 — elementwise GELU (exact, erf-based)
// exposed under the disambiguated `gelu_erf` name (Fuel ask).
//
// Implements `y = 0.5 * x * (1 + erf(x / sqrt(2)))`. PyTorch's default
// `nn.GELU()` math. Mathematically identical to the existing
// `unary_gelu` symbol — both ship under the same Phase 3 build — but
// Fuel needs an explicit `gelu_erf` symbol name so storage.rs can
// distinguish "exact erf GELU" from the tanh-approximation GELU
// (`unary_gelu_tanh_*`) without symbol-table heuristics.
//
// We intentionally duplicate the functor here rather than aliasing
// because (a) C++ external linkage names are simpler to reason about
// when the symbol points at its own translation unit, (b) future
// optimization work (e.g. an `erfcf` rewrite for the negative branch)
// might want to diverge between the two names, and (c) the cost is
// 4 extra _kernel symbols + a hundred bytes of fatbin.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

// 1/sqrt(2). Distinct from the `gelu_inv_sqrt2_*` constants in
// `unary_gelu_fp.cu` so this TU is self-contained — same value.
__device__ __forceinline__ constexpr float gelu_erf_inv_sqrt2_f() { return 0.7071067811865475f; }
__device__ __forceinline__ constexpr double gelu_erf_inv_sqrt2_d() { return 0.7071067811865475; }

template <typename T>
struct GeluErfFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct GeluErfFunctor<float> {
    __device__ __forceinline__ float operator()(float x) const {
        return 0.5f * x * (1.0f + erff(x * gelu_erf_inv_sqrt2_f()));
    }
};

template <>
struct GeluErfFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const {
        return 0.5 * x * (1.0 + erf(x * gelu_erf_inv_sqrt2_d()));
    }
};

template <>
struct GeluErfFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x) const {
        float f = __half2float(x);
        float y = 0.5f * f * (1.0f + erff(f * gelu_erf_inv_sqrt2_f()));
        return __float2half(y);
    }
};

template <>
struct GeluErfFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float f = __bfloat162float(x);
        float y = 0.5f * f * (1.0f + erff(f * gelu_erf_inv_sqrt2_f()));
        return __float2bfloat16(y);
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_gelu_erf_f32,
    float,
    baracuda::elementwise::GeluErfFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_gelu_erf_f32,
    float,
    baracuda::elementwise::GeluErfFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_gelu_erf_f16,
    __half,
    baracuda::elementwise::GeluErfFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_gelu_erf_f16,
    __half,
    baracuda::elementwise::GeluErfFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_gelu_erf_bf16,
    __nv_bfloat16,
    baracuda::elementwise::GeluErfFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_gelu_erf_bf16,
    __nv_bfloat16,
    baracuda::elementwise::GeluErfFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_gelu_erf_f64,
    double,
    baracuda::elementwise::GeluErfFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_gelu_erf_f64,
    double,
    baracuda::elementwise::GeluErfFunctor<double>)
