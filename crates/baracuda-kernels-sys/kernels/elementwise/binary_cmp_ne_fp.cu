// baracuda-kernels Phase 3 binary-comparison fanout: elementwise
// inequality for FP types.
//
// Implements `y = (a != b) ? 1 : 0` over both contiguous tensors (fast
// path) and arbitrary strided views (broadcast / transposed / sliced).
// Output type is `uint8_t` (0 = false, 1 = true) — PyTorch / NumPy bool
// storage convention.
//
// IEEE semantics: `NaN != anything` (including NaN) is true → 1. The
// functor is templated on T so the same definition serves every FP
// dtype; the `cuda_fp16.h` and `cuda_bf16.h` headers (included via
// `baracuda_elementwise.cuh`) overload `operator!=` for `__half` and
// `__nv_bfloat16` on sm_80+.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct NeFunctor {
    __device__ __forceinline__ uint8_t operator()(T a, T b) const {
        return (a != b) ? 1u : 0u;
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_BINARY_CMP_POINTWISE_INSTANTIATE(
    binary_cmp_ne_f32,
    float,
    baracuda::elementwise::NeFunctor<float>)

BARACUDA_KERNELS_BINARY_CMP_POINTWISE_INSTANTIATE_STRIDED(
    binary_cmp_ne_f32,
    float,
    baracuda::elementwise::NeFunctor<float>)

BARACUDA_KERNELS_BINARY_CMP_POINTWISE_INSTANTIATE(
    binary_cmp_ne_f16,
    __half,
    baracuda::elementwise::NeFunctor<__half>)

BARACUDA_KERNELS_BINARY_CMP_POINTWISE_INSTANTIATE_STRIDED(
    binary_cmp_ne_f16,
    __half,
    baracuda::elementwise::NeFunctor<__half>)

BARACUDA_KERNELS_BINARY_CMP_POINTWISE_INSTANTIATE(
    binary_cmp_ne_bf16,
    __nv_bfloat16,
    baracuda::elementwise::NeFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_BINARY_CMP_POINTWISE_INSTANTIATE_STRIDED(
    binary_cmp_ne_bf16,
    __nv_bfloat16,
    baracuda::elementwise::NeFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_BINARY_CMP_POINTWISE_INSTANTIATE(
    binary_cmp_ne_f64,
    double,
    baracuda::elementwise::NeFunctor<double>)

BARACUDA_KERNELS_BINARY_CMP_POINTWISE_INSTANTIATE_STRIDED(
    binary_cmp_ne_f64,
    double,
    baracuda::elementwise::NeFunctor<double>)
