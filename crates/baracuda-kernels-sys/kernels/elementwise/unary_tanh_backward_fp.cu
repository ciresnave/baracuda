// baracuda-kernels Phase 3 unary backward fanout: tanh backward.
//
// Forward: `y = tanh(x)`. Backward: `dx = dy * (1 - y²)` — saved-y.
// One mul + one sub + one mul; the compiler may fuse `1 - y*y` into
// an IEEE FMA. No per-dtype specialization needed.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct TanhBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T y) const {
        return dy * (T(1) - y * y);
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_tanh_backward_f32, float,
    baracuda::elementwise::TanhBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_tanh_backward_f16, __half,
    baracuda::elementwise::TanhBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_tanh_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::TanhBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_tanh_backward_f64, double,
    baracuda::elementwise::TanhBackwardFunctor<double>)
