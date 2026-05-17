// baracuda-kernels Phase 3 unary backward fanout: softplus backward.
//
// Forward: `y = softplus(x) = log(1 + exp(x))`. Backward:
//   dx = dy * sigmoid(x) = dy / (1 + exp(-x))
// Saved-x; smooth, requires one `exp`. The `1/(1+exp(-x))` form is
// numerically stable on both branches: large positive `x` → dy * 1, and
// large negative `x` → dy * 0, without intermediate overflow.
//
// f32 uses `expf`; f64 uses `exp`; f16 / bf16 detour through f32. Same
// 4×eps tolerance as the other transcendental BWs.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct SoftplusBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T x) const {
        return dy * x;  // placeholder — explicit specializations below.
    }
};

template <>
struct SoftplusBackwardFunctor<float> {
    __device__ __forceinline__ float operator()(float dy, float x) const {
        return dy / (1.0f + expf(-x));
    }
};

template <>
struct SoftplusBackwardFunctor<double> {
    __device__ __forceinline__ double operator()(double dy, double x) const {
        return dy / (1.0 + exp(-x));
    }
};

template <>
struct SoftplusBackwardFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half dy, __half x) const {
        float fx = __half2float(x);
        float fdy = __half2float(dy);
        return __float2half(fdy / (1.0f + expf(-fx)));
    }
};

template <>
struct SoftplusBackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 dy, __nv_bfloat16 x) const {
        float fx = __bfloat162float(x);
        float fdy = __bfloat162float(dy);
        return __float2bfloat16(fdy / (1.0f + expf(-fx)));
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_softplus_backward_f32, float,
    baracuda::elementwise::SoftplusBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_softplus_backward_f16, __half,
    baracuda::elementwise::SoftplusBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_softplus_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::SoftplusBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_softplus_backward_f64, double,
    baracuda::elementwise::SoftplusBackwardFunctor<double>)
