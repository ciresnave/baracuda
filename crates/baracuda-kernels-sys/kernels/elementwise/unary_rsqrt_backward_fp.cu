// baracuda-kernels Phase 3 unary backward fanout: rsqrt backward.
//
// Forward: `y = 1 / sqrt(x)`. Backward: `dx = -0.5 * dy * y³`.
// Derivation: y = x^(-1/2), so dy/dx = -1/2 · x^(-3/2) = -1/2 · y³.
// Saved-y, no transcendental. Use `T(0) - x` for negation so the
// expression compiles uniformly across {float, double, __half,
// __nv_bfloat16} without depending on unary minus.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct RsqrtBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T y) const {
        T y3 = y * y * y;
        T half_dy_y3 = T(0.5) * dy * y3;
        return T(0) - half_dy_y3;
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_rsqrt_backward_f32, float,
    baracuda::elementwise::RsqrtBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_rsqrt_backward_f16, __half,
    baracuda::elementwise::RsqrtBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_rsqrt_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::RsqrtBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_rsqrt_backward_f64, double,
    baracuda::elementwise::RsqrtBackwardFunctor<double>)
