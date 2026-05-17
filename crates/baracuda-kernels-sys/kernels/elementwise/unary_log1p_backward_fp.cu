// baracuda-kernels Phase 3 unary backward fanout: log1p backward.
//
// Forward: `y = ln(1 + x)`. Backward: `dx = dy / (1 + x)`. Saved-x.
// No transcendental; single add + single div.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct Log1pBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T x) const {
        return dy / (T(1) + x);
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_log1p_backward_f32, float,
    baracuda::elementwise::Log1pBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_log1p_backward_f16, __half,
    baracuda::elementwise::Log1pBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_log1p_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::Log1pBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_log1p_backward_f64, double,
    baracuda::elementwise::Log1pBackwardFunctor<double>)
