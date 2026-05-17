// baracuda-kernels Phase 3.3 — binary bitwise `or` over the integer
// family. `y = a | b` on i32 / i64 contiguous tensors.
//
// Mirror of `binary_bitwise_and_int.cu` with the `|` operator.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct BitwiseOrFunctor {
    __device__ __forceinline__ T operator()(T a, T b) const { return a | b; }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_bitwise_or_i32, int32_t,
    baracuda::elementwise::BitwiseOrFunctor<int32_t>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_bitwise_or_i64, int64_t,
    baracuda::elementwise::BitwiseOrFunctor<int64_t>)
