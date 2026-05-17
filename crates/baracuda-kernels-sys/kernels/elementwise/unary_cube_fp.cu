// baracuda-kernels Phase 3 unary fanout: elementwise cube for FP types.
//
// Implements `y = x * x * x`. Like Square, the functor is fully generic
// — `operator*` is defined on all four FP dtypes (`float`, `double`,
// `__half`, `__nv_bfloat16`), so a single un-specialized template body
// covers every wired dtype without f32-detour. The kernel evaluates as
// `(x * x) * x` left-to-right; on the GPU two FMUL / HMUL instructions
// per element.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct CubeFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x * x * x; }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_cube_f32,
    float,
    baracuda::elementwise::CubeFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_cube_f32,
    float,
    baracuda::elementwise::CubeFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_cube_f16,
    __half,
    baracuda::elementwise::CubeFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_cube_f16,
    __half,
    baracuda::elementwise::CubeFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_cube_bf16,
    __nv_bfloat16,
    baracuda::elementwise::CubeFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_cube_bf16,
    __nv_bfloat16,
    baracuda::elementwise::CubeFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_cube_f64,
    double,
    baracuda::elementwise::CubeFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_cube_f64,
    double,
    baracuda::elementwise::CubeFunctor<double>)
