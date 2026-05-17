// baracuda-kernels Phase 3 backward fanout: elementwise mul backward.
//
// Forward: `y = a * b`. Backward: `(da, db) = (dy * b, dy * a)`.
// Needs the saved forward inputs `a` and `b` from the caller. Both
// products use the same precision as the forward kernel (direct `T * T`)
// so f16 / bf16 results pick up the same single-rounding behaviour PyTorch
// expects from autograd at low precision.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct MulBackwardFunctor {
    __device__ __forceinline__ void operator()(T dy, T a, T b, T& da, T& db) const {
        da = dy * b;
        db = dy * a;
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_BINARY_BACKWARD_SAVES_INSTANTIATE(
    binary_mul_backward_f32, float,
    baracuda::elementwise::MulBackwardFunctor<float>)

BARACUDA_KERNELS_BINARY_BACKWARD_SAVES_INSTANTIATE(
    binary_mul_backward_f16, __half,
    baracuda::elementwise::MulBackwardFunctor<__half>)

BARACUDA_KERNELS_BINARY_BACKWARD_SAVES_INSTANTIATE(
    binary_mul_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::MulBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_BINARY_BACKWARD_SAVES_INSTANTIATE(
    binary_mul_backward_f64, double,
    baracuda::elementwise::MulBackwardFunctor<double>)
