// baracuda-kernels Phase 3 unary backward fanout: relu backward.
//
// Forward: `y = relu(x) = max(x, 0)`. Backward: `dx = (x > 0) ? dy : 0`.
// Saved-x; first piecewise activation BW. PyTorch convention: gradient
// is zero at exactly `x == 0` (the subgradient is undefined; PyTorch
// picks 0).
//
// f16 / bf16 use the f32-detour pattern (compare and select in f32, round
// once on store), mirroring the activation forward family. f32 / f64 use
// native ops — bit-exact against `(x > 0) ? dy : 0` on the host.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct ReluBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T x) const {
        return (x > T(0)) ? dy : T(0);
    }
};

template <>
struct ReluBackwardFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half dy, __half x) const {
        float fx = __half2float(x);
        return (fx > 0.0f) ? dy : __float2half(0.0f);
    }
};

template <>
struct ReluBackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 dy, __nv_bfloat16 x) const {
        float fx = __bfloat162float(x);
        return (fx > 0.0f) ? dy : __float2bfloat16(0.0f);
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_relu_backward_f32, float,
    baracuda::elementwise::ReluBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_relu_backward_f16, __half,
    baracuda::elementwise::ReluBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_relu_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::ReluBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_relu_backward_f64, double,
    baracuda::elementwise::ReluBackwardFunctor<double>)
