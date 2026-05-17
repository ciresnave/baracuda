// baracuda-kernels Phase 3 unary backward fanout: ELU backward.
//
// Forward: `y = x if x > 0 else α·(exp(x) - 1)`. Backward:
// `dx = dy if x > 0 else dy·α·exp(x)`. Saved-x. α is hardcoded to 1.0
// (PyTorch default) — when the parameterized-unary plan ships, this
// kernel is re-emitted with α as a runtime parameter.
//
// f32 uses `expf`; f64 uses `exp`. f16 / bf16 use the f32-detour with
// `expf` (compare and multiply in f32, round once on store).

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct EluBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T x) const { return dy; }
};

template <>
struct EluBackwardFunctor<float> {
    __device__ __forceinline__ float operator()(float dy, float x) const {
        return (x > 0.0f) ? dy : dy * expf(x);
    }
};

template <>
struct EluBackwardFunctor<double> {
    __device__ __forceinline__ double operator()(double dy, double x) const {
        return (x > 0.0) ? dy : dy * exp(x);
    }
};

template <>
struct EluBackwardFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half dy, __half x) const {
        float fx = __half2float(x);
        float fdy = __half2float(dy);
        float dx = (fx > 0.0f) ? fdy : fdy * expf(fx);
        return __float2half(dx);
    }
};

template <>
struct EluBackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 dy, __nv_bfloat16 x) const {
        float fx = __bfloat162float(x);
        float fdy = __bfloat162float(dy);
        float dx = (fx > 0.0f) ? fdy : fdy * expf(fx);
        return __float2bfloat16(dx);
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_elu_backward_f32, float,
    baracuda::elementwise::EluBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_elu_backward_f16, __half,
    baracuda::elementwise::EluBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_elu_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::EluBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_elu_backward_f64, double,
    baracuda::elementwise::EluBackwardFunctor<double>)
