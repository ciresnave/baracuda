// baracuda-kernels Phase 3 binary-comparison fanout: elementwise
// strict-less-than for FP types.
//
// Implements `y = (a < b) ? 1 : 0` over both contiguous tensors (fast
// path) and arbitrary strided views (broadcast / transposed / sliced).
// Output type is `uint8_t` (0 = false, 1 = true) — PyTorch / NumPy bool
// storage convention.
//
// IEEE semantics: any comparison involving NaN returns false → 0. The
// functor is templated on T; `cuda_fp16.h` / `cuda_bf16.h` overload
// `operator<` for `__half` / `__nv_bfloat16` on sm_80+.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct LtFunctor {
    __device__ __forceinline__ uint8_t operator()(T a, T b) const {
        return (a < b) ? 1u : 0u;
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_BINARY_CMP_POINTWISE_INSTANTIATE(
    binary_cmp_lt_f32,
    float,
    baracuda::elementwise::LtFunctor<float>)

BARACUDA_KERNELS_BINARY_CMP_POINTWISE_INSTANTIATE_STRIDED(
    binary_cmp_lt_f32,
    float,
    baracuda::elementwise::LtFunctor<float>)

BARACUDA_KERNELS_BINARY_CMP_POINTWISE_INSTANTIATE(
    binary_cmp_lt_f16,
    __half,
    baracuda::elementwise::LtFunctor<__half>)

BARACUDA_KERNELS_BINARY_CMP_POINTWISE_INSTANTIATE_STRIDED(
    binary_cmp_lt_f16,
    __half,
    baracuda::elementwise::LtFunctor<__half>)

BARACUDA_KERNELS_BINARY_CMP_POINTWISE_INSTANTIATE(
    binary_cmp_lt_bf16,
    __nv_bfloat16,
    baracuda::elementwise::LtFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_BINARY_CMP_POINTWISE_INSTANTIATE_STRIDED(
    binary_cmp_lt_bf16,
    __nv_bfloat16,
    baracuda::elementwise::LtFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_BINARY_CMP_POINTWISE_INSTANTIATE(
    binary_cmp_lt_f64,
    double,
    baracuda::elementwise::LtFunctor<double>)

BARACUDA_KERNELS_BINARY_CMP_POINTWISE_INSTANTIATE_STRIDED(
    binary_cmp_lt_f64,
    double,
    baracuda::elementwise::LtFunctor<double>)
