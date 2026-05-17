// baracuda-kernels Phase 3 fanout: binary elementwise mul for FP types.
//
// Implements `y = a * b` over contiguous tensors. The kernel template
// and INSTANTIATE macros live in `include/baracuda_elementwise.cuh`;
// this file supplies the MulFunctor and the per-dtype instantiations.
//
// Today only `f32` is wired (the trailblazer-pattern instantiation).
// `f16`, `bf16`, and `f64` instantiations follow as the Phase 3 fanout
// proceeds — they are mechanical re-instantiations of the same
// template + functor with the corresponding scalar type. The kernel
// itself contains no FP-specific magic — it's straight `a * b` — so the
// same functor reuses across the whole FP family.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

// Binary mul functor. Templated on T so the same definition serves
// f32 / f16 / bf16 / f64 once the fanout instantiations land.
template <typename T>
struct MulFunctor {
    __device__ __forceinline__ T operator()(T a, T b) const { return a * b; }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_mul_f32,
    float,
    baracuda::elementwise::MulFunctor<float>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_mul_f32,
    float,
    baracuda::elementwise::MulFunctor<float>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_mul_f16,
    __half,
    baracuda::elementwise::MulFunctor<__half>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_mul_f16,
    __half,
    baracuda::elementwise::MulFunctor<__half>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_mul_bf16,
    __nv_bfloat16,
    baracuda::elementwise::MulFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_mul_bf16,
    __nv_bfloat16,
    baracuda::elementwise::MulFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_mul_f64,
    double,
    baracuda::elementwise::MulFunctor<double>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_mul_f64,
    double,
    baracuda::elementwise::MulFunctor<double>)
