// baracuda-kernels Phase 3.3 — binary bitwise left-shift over the
// integer family. `y = a << b` on i32 / i64 contiguous tensors.
//
// Out-of-range shift amounts (`b < 0` or `b >= 8 * sizeof(T)`) are
// undefined behavior in C / C++ on signed types. PyTorch documents the
// result as undefined / hardware-dependent in that range too, so we
// inherit the host architecture's behavior rather than masking. Callers
// who need defined behavior should clamp `b` themselves before launch.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct BitwiseLeftShiftFunctor {
    __device__ __forceinline__ T operator()(T a, T b) const { return a << b; }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_bitwise_left_shift_i32, int32_t,
    baracuda::elementwise::BitwiseLeftShiftFunctor<int32_t>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_bitwise_left_shift_i64, int64_t,
    baracuda::elementwise::BitwiseLeftShiftFunctor<int64_t>)
