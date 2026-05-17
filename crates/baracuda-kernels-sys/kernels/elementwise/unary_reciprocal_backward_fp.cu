// baracuda-kernels Phase 3 unary backward fanout: reciprocal backward.
//
// Forward: `y = 1 / x`. Backward: `dx = -dy / (x * x)`. Saved-x. No
// transcendental; one mul + one div + one negate covers all FP dtypes
// via a generic-on-T functor. The negation is written as `T(0) - ...`
// so it compiles uniformly for `__half` / `__nv_bfloat16`. Domain is
// strictly `x != 0`.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct ReciprocalBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T x) const {
        return T(0) - dy / (x * x);
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_reciprocal_backward_f32, float,
    baracuda::elementwise::ReciprocalBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_reciprocal_backward_f16, __half,
    baracuda::elementwise::ReciprocalBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_reciprocal_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::ReciprocalBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_reciprocal_backward_f64, double,
    baracuda::elementwise::ReciprocalBackwardFunctor<double>)
