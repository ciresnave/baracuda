// baracuda-kernels Phase 3 unary backward fanout: leaky-ReLU backward.
//
// Forward: `y = x if x > 0 else α·x`. Backward: `dx = dy if x > 0 else
// dy·α`. Saved-x. α is hardcoded to 0.01 (PyTorch default) — when the
// parameterized-unary plan ships, this kernel is re-emitted with α as a
// runtime parameter. PyTorch convention: at exactly `x == 0` the
// gradient is `dy·α` (the negative branch is the closed half-plane).
//
// f16 / bf16 use the f32-detour pattern (compare and multiply in f32,
// round once on store). f32 / f64 use native ops.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct LeakyReluBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T x) const {
        return (x > T(0)) ? dy : dy * T(0.01);
    }
};

template <>
struct LeakyReluBackwardFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half dy, __half x) const {
        float fx = __half2float(x);
        float fdy = __half2float(dy);
        float dx = (fx > 0.0f) ? fdy : fdy * 0.01f;
        return __float2half(dx);
    }
};

template <>
struct LeakyReluBackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 dy, __nv_bfloat16 x) const {
        float fx = __bfloat162float(x);
        float fdy = __bfloat162float(dy);
        float dx = (fx > 0.0f) ? fdy : fdy * 0.01f;
        return __float2bfloat16(dx);
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_leaky_relu_backward_f32, float,
    baracuda::elementwise::LeakyReluBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_leaky_relu_backward_f16, __half,
    baracuda::elementwise::LeakyReluBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_leaky_relu_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::LeakyReluBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_leaky_relu_backward_f64, double,
    baracuda::elementwise::LeakyReluBackwardFunctor<double>)
