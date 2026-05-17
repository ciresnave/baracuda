// baracuda-kernels Phase 3 unary backward fanout: exp2 backward.
//
// Forward: `y = 2^x`. Backward: `dx = dy * y * ln(2)` — saved-y. The
// constant `ln(2)` is hardcoded; `T(LN2)` truncates to the dtype on
// use (full precision for f64; f32-rounded for f32/f16/bf16).

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

constexpr double LN2 = 0.6931471805599453;

template <typename T>
struct Exp2BackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T y) const {
        return dy * y * T(LN2);
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_exp2_backward_f32, float,
    baracuda::elementwise::Exp2BackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_exp2_backward_f16, __half,
    baracuda::elementwise::Exp2BackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_exp2_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::Exp2BackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_exp2_backward_f64, double,
    baracuda::elementwise::Exp2BackwardFunctor<double>)
