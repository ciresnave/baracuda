// baracuda-kernels Phase 3 binary-comparison fanout: elementwise
// strict-greater-than for FP types.
//
// Implements `y = (a > b) ? 1 : 0` over both contiguous tensors (fast
// path) and arbitrary strided views (broadcast / transposed / sliced).
// Output type is `uint8_t` (0 = false, 1 = true) — PyTorch / NumPy bool
// storage convention.
//
// IEEE semantics: any comparison involving NaN returns false → 0. The
// functor is templated on T; `cuda_fp16.h` / `cuda_bf16.h` overload
// `operator>` for `__half` / `__nv_bfloat16` on sm_80+.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct GtFunctor {
    __device__ __forceinline__ uint8_t operator()(T a, T b) const {
        return (a > b) ? 1u : 0u;
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_BINARY_CMP_POINTWISE_INSTANTIATE(
    binary_cmp_gt_f32,
    float,
    baracuda::elementwise::GtFunctor<float>)

BARACUDA_KERNELS_BINARY_CMP_POINTWISE_INSTANTIATE_STRIDED(
    binary_cmp_gt_f32,
    float,
    baracuda::elementwise::GtFunctor<float>)

BARACUDA_KERNELS_BINARY_CMP_POINTWISE_INSTANTIATE(
    binary_cmp_gt_f16,
    __half,
    baracuda::elementwise::GtFunctor<__half>)

BARACUDA_KERNELS_BINARY_CMP_POINTWISE_INSTANTIATE_STRIDED(
    binary_cmp_gt_f16,
    __half,
    baracuda::elementwise::GtFunctor<__half>)

BARACUDA_KERNELS_BINARY_CMP_POINTWISE_INSTANTIATE(
    binary_cmp_gt_bf16,
    __nv_bfloat16,
    baracuda::elementwise::GtFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_BINARY_CMP_POINTWISE_INSTANTIATE_STRIDED(
    binary_cmp_gt_bf16,
    __nv_bfloat16,
    baracuda::elementwise::GtFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_BINARY_CMP_POINTWISE_INSTANTIATE(
    binary_cmp_gt_f64,
    double,
    baracuda::elementwise::GtFunctor<double>)

BARACUDA_KERNELS_BINARY_CMP_POINTWISE_INSTANTIATE_STRIDED(
    binary_cmp_gt_f64,
    double,
    baracuda::elementwise::GtFunctor<double>)
