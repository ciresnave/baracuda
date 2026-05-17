// baracuda-kernels Phase 3 unary fanout: elementwise square for FP types.
//
// Implements `y = x * x`. The functor is fully generic — `operator*` is
// defined on all four FP dtypes (`float`, `double`, `__half`,
// `__nv_bfloat16`), so a single un-specialized template body covers
// every wired dtype without f32-detour. Hardware emits a single FMUL /
// HMUL instruction per element.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct SquareFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x * x; }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_square_f32,
    float,
    baracuda::elementwise::SquareFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_square_f32,
    float,
    baracuda::elementwise::SquareFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_square_f16,
    __half,
    baracuda::elementwise::SquareFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_square_f16,
    __half,
    baracuda::elementwise::SquareFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_square_bf16,
    __nv_bfloat16,
    baracuda::elementwise::SquareFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_square_bf16,
    __nv_bfloat16,
    baracuda::elementwise::SquareFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_square_f64,
    double,
    baracuda::elementwise::SquareFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_square_f64,
    double,
    baracuda::elementwise::SquareFunctor<double>)
