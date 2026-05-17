// baracuda-kernels Phase 3.3 — binary bitwise `and` over the integer
// family. `y = a & b` on i32 / i64 contiguous tensors.
//
// Straight integer bitwise AND — no f32-detour, no rounding, no
// overflow concerns. The same functor template specializes to both
// signed-integer widths via the INSTANTIATE pair below; the kernel
// template in `baracuda_elementwise.cuh` is dtype-agnostic and reuses
// unchanged.
//
// Contig-only for the integer family today — broadcast / strided paths
// for int/bool elementwise ops are deferred to a follow-up milestone
// (the caller materializes if needed).

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct BitwiseAndFunctor {
    __device__ __forceinline__ T operator()(T a, T b) const { return a & b; }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_bitwise_and_i32, int32_t,
    baracuda::elementwise::BitwiseAndFunctor<int32_t>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_bitwise_and_i64, int64_t,
    baracuda::elementwise::BitwiseAndFunctor<int64_t>)
