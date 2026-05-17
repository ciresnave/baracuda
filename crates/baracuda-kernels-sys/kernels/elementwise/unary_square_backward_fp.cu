// baracuda-kernels Phase 3 unary backward fanout: square backward.
//
// Forward: `y = x²`. Backward: `dx = dy * 2 * x` — saved-x. Pure
// arithmetic; generic-on-T covers all four FP dtypes.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct SquareBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T x) const {
        return dy * T(2) * x;
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_square_backward_f32, float,
    baracuda::elementwise::SquareBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_square_backward_f16, __half,
    baracuda::elementwise::SquareBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_square_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::SquareBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_square_backward_f64, double,
    baracuda::elementwise::SquareBackwardFunctor<double>)
