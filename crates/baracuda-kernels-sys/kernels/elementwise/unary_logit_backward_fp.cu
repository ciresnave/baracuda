// baracuda-kernels Phase 3 unary backward fanout: logit backward.
//
// Forward: `y = log(x / (1 - x))`. Backward: `dx = dy / (x * (1 - x))`.
// Saved-x. No transcendental in the BW formula; one sub + one mul + one
// div covers all FP dtypes via a generic-on-T functor. Domain is
// strictly `0 < x < 1`.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct LogitBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T x) const {
        return dy / (x * (T(1) - x));
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_logit_backward_f32, float,
    baracuda::elementwise::LogitBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_logit_backward_f16, __half,
    baracuda::elementwise::LogitBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_logit_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::LogitBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_logit_backward_f64, double,
    baracuda::elementwise::LogitBackwardFunctor<double>)
