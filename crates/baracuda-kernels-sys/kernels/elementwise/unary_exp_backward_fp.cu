// baracuda-kernels Phase 3 unary backward: exp backward.
//
// Forward: `y = exp(x)`. Backward: `dx = dy * y` — saved-y. Single
// multiplication; generic-on-T functor works for all FP dtypes
// (no transcendental specialization needed).

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct ExpBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T y) const {
        return dy * y;
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_exp_backward_f32, float,
    baracuda::elementwise::ExpBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_exp_backward_f16, __half,
    baracuda::elementwise::ExpBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_exp_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::ExpBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_exp_backward_f64, double,
    baracuda::elementwise::ExpBackwardFunctor<double>)
