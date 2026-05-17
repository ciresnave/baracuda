// baracuda-kernels Phase 3 unary backward fanout: atan backward.
//
// Forward: `y = atan(x)`. Backward: `dx = dy / (1 + x²)`. Saved-x.
// No transcendental in the BW formula; one add + one mul + one div
// covers all FP dtypes via a generic-on-T functor.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct AtanBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T x) const {
        return dy / (T(1) + x * x);
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_atan_backward_f32, float,
    baracuda::elementwise::AtanBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_atan_backward_f16, __half,
    baracuda::elementwise::AtanBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_atan_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::AtanBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_atan_backward_f64, double,
    baracuda::elementwise::AtanBackwardFunctor<double>)
