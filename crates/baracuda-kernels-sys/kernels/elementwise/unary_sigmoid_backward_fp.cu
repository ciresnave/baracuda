// baracuda-kernels Phase 3 unary backward fanout: sigmoid backward.
//
// Forward: `y = sigmoid(x) = 1 / (1 + exp(-x))`. Backward:
// `dx = dy * y * (1 - y)` — saved-y. Two muls + one sub.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct SigmoidBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T y) const {
        return dy * y * (T(1) - y);
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_sigmoid_backward_f32, float,
    baracuda::elementwise::SigmoidBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_sigmoid_backward_f16, __half,
    baracuda::elementwise::SigmoidBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_sigmoid_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::SigmoidBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_sigmoid_backward_f64, double,
    baracuda::elementwise::SigmoidBackwardFunctor<double>)
