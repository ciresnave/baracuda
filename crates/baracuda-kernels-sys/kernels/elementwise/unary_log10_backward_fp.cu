// baracuda-kernels Phase 3 unary backward fanout: log10 backward.
//
// Forward: `y = log_10(x)`. Backward: `dx = dy / (x * ln(10))`. Saved-x.
// `ln(10)` is a compile-time constant; the kernel becomes a single
// (mul, div) for any dtype with no transcendental call.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

// ln(10), accurate to double precision. Truncated to the dtype on use.
constexpr double LN10 = 2.302585092994046;

template <typename T>
struct Log10BackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T x) const {
        return dy / (x * T(LN10));
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_log10_backward_f32, float,
    baracuda::elementwise::Log10BackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_log10_backward_f16, __half,
    baracuda::elementwise::Log10BackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_log10_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::Log10BackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_log10_backward_f64, double,
    baracuda::elementwise::Log10BackwardFunctor<double>)
