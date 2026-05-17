// baracuda-kernels Phase 3 unary backward fanout: log2 backward.
//
// Forward: `y = log_2(x)`. Backward: `dx = dy / (x * ln(2))`. Saved-x.
// `ln(2)` is a compile-time constant; the kernel becomes a single
// (mul, div) for any dtype with no transcendental call.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

// ln(2), accurate to double precision. Truncated to the dtype on use.
constexpr double LN2 = 0.6931471805599453;

template <typename T>
struct Log2BackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T x) const {
        return dy / (x * T(LN2));
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_log2_backward_f32, float,
    baracuda::elementwise::Log2BackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_log2_backward_f16, __half,
    baracuda::elementwise::Log2BackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_log2_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::Log2BackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_log2_backward_f64, double,
    baracuda::elementwise::Log2BackwardFunctor<double>)
