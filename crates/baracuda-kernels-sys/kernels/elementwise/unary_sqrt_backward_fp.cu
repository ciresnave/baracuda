// baracuda-kernels Phase 3 unary backward fanout: sqrt backward.
//
// Forward: `y = sqrt(x)`. Backward: `dx = dy / (2y)` — saved-y. One
// mul (`T(2) * y`) then one division. Generic-on-T.
//
// Callers must ensure `y[i] != 0` for every cell (the forward Sqrt
// of zero would have produced zero, and dividing by it would NaN).

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct SqrtBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T y) const {
        return dy / (T(2) * y);
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_sqrt_backward_f32, float,
    baracuda::elementwise::SqrtBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_sqrt_backward_f16, __half,
    baracuda::elementwise::SqrtBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_sqrt_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::SqrtBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_sqrt_backward_f64, double,
    baracuda::elementwise::SqrtBackwardFunctor<double>)
