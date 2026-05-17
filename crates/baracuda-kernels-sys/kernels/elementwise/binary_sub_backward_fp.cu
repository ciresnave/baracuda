// baracuda-kernels Phase 3 backward fanout: elementwise sub backward.
//
// Forward: `y = a - b`. Backward: `(da, db) = (dy, -dy)`. No saved
// tensors needed — gradient flows positively to `a` and negatively
// to `b`. We negate via `T(0) - dy` rather than the unary `-` operator
// because `__half` / `__nv_bfloat16` do not always provide unary minus
// across CUDA toolkits, whereas the binary subtraction operator and
// the `T(0)` constructor are uniformly available.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct SubBackwardFunctor {
    __device__ __forceinline__ void operator()(T dy, T& da, T& db) const {
        da = dy;
        db = T(0) - dy;
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_BINARY_BACKWARD_NOSAVE_INSTANTIATE(
    binary_sub_backward_f32, float,
    baracuda::elementwise::SubBackwardFunctor<float>)

BARACUDA_KERNELS_BINARY_BACKWARD_NOSAVE_INSTANTIATE(
    binary_sub_backward_f16, __half,
    baracuda::elementwise::SubBackwardFunctor<__half>)

BARACUDA_KERNELS_BINARY_BACKWARD_NOSAVE_INSTANTIATE(
    binary_sub_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::SubBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_BINARY_BACKWARD_NOSAVE_INSTANTIATE(
    binary_sub_backward_f64, double,
    baracuda::elementwise::SubBackwardFunctor<double>)
