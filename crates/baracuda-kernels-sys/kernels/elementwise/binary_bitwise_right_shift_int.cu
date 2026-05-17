// baracuda-kernels Phase 3.3 — binary bitwise right-shift over the
// integer family. `y = a >> b` on i32 / i64 contiguous tensors.
//
// **Arithmetic** right shift on signed integers — the sign bit is
// replicated into the vacated high bits, matching PyTorch's contract.
// C++ pre-C++20 leaves signed `>>` implementation-defined, but every
// CUDA-supported compiler (NVCC, MSVC, GCC, Clang) implements signed
// `>>` as arithmetic shift on the host and the device, and CUDA's PTX
// `shr.s32` / `shr.s64` instructions are arithmetic by construction.
// We therefore rely on the compiler's signed-`>>` lowering rather than
// hand-rolling the `(x < 0) ? ~(~x >> n) : (x >> n)` workaround — the
// resulting PTX is bit-identical to the manual form on every CUDA arch
// we target. C++20 standardizes arithmetic shift for signed types,
// closing this gap formally.
//
// Out-of-range shift amounts (`b < 0` or `b >= 8 * sizeof(T)`) inherit
// the host architecture's behavior (same caller contract as
// left-shift).

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct BitwiseRightShiftFunctor {
    __device__ __forceinline__ T operator()(T a, T b) const { return a >> b; }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_bitwise_right_shift_i32, int32_t,
    baracuda::elementwise::BitwiseRightShiftFunctor<int32_t>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_bitwise_right_shift_i64, int64_t,
    baracuda::elementwise::BitwiseRightShiftFunctor<int64_t>)
