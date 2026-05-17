// baracuda-kernels Phase 3 binary-comparison trailblazer: elementwise
// equality for FP types.
//
// Implements `y = (a == b) ? 1 : 0` over both contiguous tensors (fast
// path) and arbitrary strided views (broadcast / transposed / sliced).
// Output type is `uint8_t` (0 = false, 1 = true) — PyTorch / NumPy bool
// storage convention.
//
// All 4 FP dtypes (f32, f16, bf16, f64) are wired. The {Ne, Gt, Ge,
// Lt, Le} ops live in sibling files (`binary_cmp_ne_fp.cu` etc.). The
// functor is templated on T so the same definition serves every FP
// dtype; `cuda_fp16.h` / `cuda_bf16.h` (included via
// `baracuda_elementwise.cuh`) overload `operator==` for `__half` and
// `__nv_bfloat16` on sm_80+.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

// Equality functor. Returns 1 for equal, 0 otherwise. NaN equality
// follows IEEE semantics (NaN != NaN, so eq(NaN, NaN) = 0); this
// matches PyTorch's `torch.eq` and `==` operator on floating-point.
template <typename T>
struct EqFunctor {
    __device__ __forceinline__ uint8_t operator()(T a, T b) const {
        return (a == b) ? 1u : 0u;
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_BINARY_CMP_POINTWISE_INSTANTIATE(
    binary_cmp_eq_f32,
    float,
    baracuda::elementwise::EqFunctor<float>)

BARACUDA_KERNELS_BINARY_CMP_POINTWISE_INSTANTIATE_STRIDED(
    binary_cmp_eq_f32,
    float,
    baracuda::elementwise::EqFunctor<float>)

BARACUDA_KERNELS_BINARY_CMP_POINTWISE_INSTANTIATE(
    binary_cmp_eq_f16,
    __half,
    baracuda::elementwise::EqFunctor<__half>)

BARACUDA_KERNELS_BINARY_CMP_POINTWISE_INSTANTIATE_STRIDED(
    binary_cmp_eq_f16,
    __half,
    baracuda::elementwise::EqFunctor<__half>)

BARACUDA_KERNELS_BINARY_CMP_POINTWISE_INSTANTIATE(
    binary_cmp_eq_bf16,
    __nv_bfloat16,
    baracuda::elementwise::EqFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_BINARY_CMP_POINTWISE_INSTANTIATE_STRIDED(
    binary_cmp_eq_bf16,
    __nv_bfloat16,
    baracuda::elementwise::EqFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_BINARY_CMP_POINTWISE_INSTANTIATE(
    binary_cmp_eq_f64,
    double,
    baracuda::elementwise::EqFunctor<double>)

BARACUDA_KERNELS_BINARY_CMP_POINTWISE_INSTANTIATE_STRIDED(
    binary_cmp_eq_f64,
    double,
    baracuda::elementwise::EqFunctor<double>)
