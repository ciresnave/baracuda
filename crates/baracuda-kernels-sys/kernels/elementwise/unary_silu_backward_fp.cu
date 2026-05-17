// baracuda-kernels Phase 3 unary backward fanout: silu (swish) backward.
//
// Forward: `y = silu(x) = x * sigmoid(x)`. Backward:
//   dx = dy * sigmoid(x) * (1 + x * (1 - sigmoid(x)))
// Saved-x; smooth, one `exp`. f32 uses `expf`; f64 uses `exp`; f16 /
// bf16 detour through f32. 4×eps tolerance (the multiplicative chain
// `s * (1 + x * (1 - s))` has bounded magnitudes near the origin).

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct SiluBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T x) const {
        return dy * x;  // placeholder — explicit specializations below.
    }
};

template <>
struct SiluBackwardFunctor<float> {
    __device__ __forceinline__ float operator()(float dy, float x) const {
        float s = 1.0f / (1.0f + expf(-x));
        return dy * s * (1.0f + x * (1.0f - s));
    }
};

template <>
struct SiluBackwardFunctor<double> {
    __device__ __forceinline__ double operator()(double dy, double x) const {
        double s = 1.0 / (1.0 + exp(-x));
        return dy * s * (1.0 + x * (1.0 - s));
    }
};

template <>
struct SiluBackwardFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half dy, __half x) const {
        float fx = __half2float(x);
        float fdy = __half2float(dy);
        float s = 1.0f / (1.0f + expf(-fx));
        return __float2half(fdy * s * (1.0f + fx * (1.0f - s)));
    }
};

template <>
struct SiluBackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 dy, __nv_bfloat16 x) const {
        float fx = __bfloat162float(x);
        float fdy = __bfloat162float(dy);
        float s = 1.0f / (1.0f + expf(-fx));
        return __float2bfloat16(fdy * s * (1.0f + fx * (1.0f - s)));
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_silu_backward_f32, float,
    baracuda::elementwise::SiluBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_silu_backward_f16, __half,
    baracuda::elementwise::SiluBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_silu_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::SiluBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_silu_backward_f64, double,
    baracuda::elementwise::SiluBackwardFunctor<double>)
