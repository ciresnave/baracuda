// baracuda-kernels Phase 3 unary backward fanout: atanh backward.
//
// Forward: `y = atanh(x)`. Backward: `dx = dy / (1 - x²)`. Saved-x.
// No transcendental in the BW formula; one mul + one sub + one div
// covers all FP dtypes via a generic-on-T functor. Callers must keep
// `|x| < 1` (matches the forward Atanh domain).

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct AtanhBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T x) const {
        return dy / (T(1) - x * x);
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_atanh_backward_f32, float,
    baracuda::elementwise::AtanhBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_atanh_backward_f16, __half,
    baracuda::elementwise::AtanhBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_atanh_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::AtanhBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_atanh_backward_f64, double,
    baracuda::elementwise::AtanhBackwardFunctor<double>)
