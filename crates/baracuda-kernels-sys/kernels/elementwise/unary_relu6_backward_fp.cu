// baracuda-kernels Phase 3 unary backward fanout: relu6 backward.
//
// Forward: `y = clamp(x, 0, 6)`. Backward: `dx = (0 < x < 6) ? dy : 0`.
// Saved-x; piecewise activation BW. PyTorch convention picks zero at the
// exact boundary points x == 0 and x == 6 (subgradient is undefined).
//
// f16 / bf16 use the f32-detour pattern (compare in f32, select bits);
// the result is bit-exact against the host reference for every dtype.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct Relu6BackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T x) const {
        return (x > T(0) && x < T(6)) ? dy : T(0);
    }
};

template <>
struct Relu6BackwardFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half dy, __half x) const {
        float fx = __half2float(x);
        return (fx > 0.0f && fx < 6.0f) ? dy : __float2half(0.0f);
    }
};

template <>
struct Relu6BackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 dy, __nv_bfloat16 x) const {
        float fx = __bfloat162float(x);
        return (fx > 0.0f && fx < 6.0f) ? dy : __float2bfloat16(0.0f);
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_relu6_backward_f32, float,
    baracuda::elementwise::Relu6BackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_relu6_backward_f16, __half,
    baracuda::elementwise::Relu6BackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_relu6_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::Relu6BackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_relu6_backward_f64, double,
    baracuda::elementwise::Relu6BackwardFunctor<double>)
