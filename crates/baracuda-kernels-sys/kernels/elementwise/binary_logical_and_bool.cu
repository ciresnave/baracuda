// baracuda-kernels Phase 3.3 — binary logical `and` over the boolean
// family. `y = (a != 0) && (b != 0) ? 1 : 0` on Bool contiguous
// tensors.
//
// Bool storage on the GPU side is `uint8_t` (1 byte; PyTorch / NumPy
// convention: 0 = false, any non-zero = true). The functor normalizes
// each input to 0 or 1 before applying the logical op, so the output
// is always strictly 0 or 1 — even when the inputs are unnormalized
// (e.g. tensors built from a `reinterpret_cast` view of byte-storage).
// This is the safe choice for downstream consumers that expect
// canonical bool tensors.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

struct LogicalAndFunctor {
    __device__ __forceinline__ uint8_t operator()(uint8_t a, uint8_t b) const {
        return (a != 0 && b != 0) ? uint8_t{1} : uint8_t{0};
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_logical_and_bool, uint8_t,
    baracuda::elementwise::LogicalAndFunctor)
