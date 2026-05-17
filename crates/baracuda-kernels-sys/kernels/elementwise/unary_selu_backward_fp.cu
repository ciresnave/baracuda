// baracuda-kernels Phase 3 unary backward fanout: selu backward.
//
// Forward: `y = scale * (x > 0 ? x : alpha * (exp(x) - 1))`.
// Backward (piecewise):
//   x  > 0 → dy * scale
//   x <= 0 → dy * scale * alpha * exp(x)
// Saved-x. Constants (per the SELU paper):
//   scale = 1.0507009873554804
//   alpha = 1.6732632423543772
// f32 / f16 / bf16 use `expf`; f64 uses `exp`. 4×eps tolerance — the
// positive branch is a single scalar multiply (no transcendental); the
// negative branch has one `exp` plus two multiplies.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

namespace {
constexpr float  K_SELU_SCALE_F = 1.05070098735548049342f;
constexpr float  K_SELU_ALPHA_F = 1.67326324235437728481f;
constexpr double K_SELU_SCALE_D = 1.05070098735548049342;
constexpr double K_SELU_ALPHA_D = 1.67326324235437728481;
}

template <typename T>
struct SeluBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T x) const {
        return dy * x;  // placeholder — explicit specializations below.
    }
};

template <>
struct SeluBackwardFunctor<float> {
    __device__ __forceinline__ float operator()(float dy, float x) const {
        if (x > 0.0f) return dy * K_SELU_SCALE_F;
        return dy * K_SELU_SCALE_F * K_SELU_ALPHA_F * expf(x);
    }
};

template <>
struct SeluBackwardFunctor<double> {
    __device__ __forceinline__ double operator()(double dy, double x) const {
        if (x > 0.0) return dy * K_SELU_SCALE_D;
        return dy * K_SELU_SCALE_D * K_SELU_ALPHA_D * exp(x);
    }
};

template <>
struct SeluBackwardFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half dy, __half x) const {
        float fx = __half2float(x);
        float fdy = __half2float(dy);
        if (fx > 0.0f) return __float2half(fdy * K_SELU_SCALE_F);
        return __float2half(fdy * K_SELU_SCALE_F * K_SELU_ALPHA_F * expf(fx));
    }
};

template <>
struct SeluBackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 dy, __nv_bfloat16 x) const {
        float fx = __bfloat162float(x);
        float fdy = __bfloat162float(dy);
        if (fx > 0.0f) return __float2bfloat16(fdy * K_SELU_SCALE_F);
        return __float2bfloat16(fdy * K_SELU_SCALE_F * K_SELU_ALPHA_F * expf(fx));
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_selu_backward_f32, float,
    baracuda::elementwise::SeluBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_selu_backward_f16, __half,
    baracuda::elementwise::SeluBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_selu_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::SeluBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_selu_backward_f64, double,
    baracuda::elementwise::SeluBackwardFunctor<double>)
