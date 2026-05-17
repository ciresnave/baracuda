// baracuda-kernels Phase 3.3 — binary logical `xor` over the boolean
// family. `y = (a != 0) != (b != 0) ? 1 : 0` on Bool contiguous
// tensors.
//
// Mirror of `binary_logical_and_bool.cu` with the boolean-XOR (i.e.
// boolean inequality) operator. See that file for the input-
// normalization rationale.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

struct LogicalXorFunctor {
    __device__ __forceinline__ uint8_t operator()(uint8_t a, uint8_t b) const {
        return ((a != 0) != (b != 0)) ? uint8_t{1} : uint8_t{0};
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_logical_xor_bool, uint8_t,
    baracuda::elementwise::LogicalXorFunctor)
