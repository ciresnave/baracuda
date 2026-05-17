// baracuda-kernels Phase 3 unary backward fanout: log backward.
//
// Forward: `y = ln(x)`. Backward: `dx = dy / x`. Saved-x.
// No transcendental in the BW formula; a single division covers all
// FP dtypes via a generic-on-T functor. Callers must ensure `x[i] > 0`
// (matching the forward Log domain).

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct LogBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T x) const {
        return dy / x;
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_log_backward_f32, float,
    baracuda::elementwise::LogBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_log_backward_f16, __half,
    baracuda::elementwise::LogBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_log_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::LogBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_log_backward_f64, double,
    baracuda::elementwise::LogBackwardFunctor<double>)
