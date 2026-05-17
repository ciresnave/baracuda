// baracuda-kernels Phase 3 backward: elementwise add backward.
//
// Forward: `y = a + b`. Backward: `(da, db) = (dy, dy)`. No saved
// tensors needed — gradient flows equally to both inputs.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

// Add backward functor — no saved tensors required.
template <typename T>
struct AddBackwardFunctor {
    __device__ __forceinline__ void operator()(T dy, T& da, T& db) const {
        da = dy;
        db = dy;
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_BINARY_BACKWARD_NOSAVE_INSTANTIATE(
    binary_add_backward_f32, float,
    baracuda::elementwise::AddBackwardFunctor<float>)

BARACUDA_KERNELS_BINARY_BACKWARD_NOSAVE_INSTANTIATE(
    binary_add_backward_f16, __half,
    baracuda::elementwise::AddBackwardFunctor<__half>)

BARACUDA_KERNELS_BINARY_BACKWARD_NOSAVE_INSTANTIATE(
    binary_add_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::AddBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_BINARY_BACKWARD_NOSAVE_INSTANTIATE(
    binary_add_backward_f64, double,
    baracuda::elementwise::AddBackwardFunctor<double>)
