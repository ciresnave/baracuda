// baracuda-kernels Phase 3 unary backward: acosh backward.
//
// Forward: `y = acosh(x)`. Backward: `dx = dy / sqrt(x² - 1)`. Saved-x.
// Domain: strictly `x > 1` (the derivative diverges at 1). f32 uses
// `sqrtf`; f64 uses `sqrt`; f16 / bf16 use the f32-detour pattern.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct AcoshBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T x) const {
        return dy * x;  // placeholder — explicit specializations below are used.
    }
};

template <>
struct AcoshBackwardFunctor<float> {
    __device__ __forceinline__ float operator()(float dy, float x) const {
        return dy / sqrtf(x * x - 1.0f);
    }
};

template <>
struct AcoshBackwardFunctor<double> {
    __device__ __forceinline__ double operator()(double dy, double x) const {
        return dy / sqrt(x * x - 1.0);
    }
};

template <>
struct AcoshBackwardFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half dy, __half x) const {
        float xf = __half2float(x);
        return __float2half(__half2float(dy) / sqrtf(xf * xf - 1.0f));
    }
};

template <>
struct AcoshBackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 dy, __nv_bfloat16 x) const {
        float xf = __bfloat162float(x);
        return __float2bfloat16(__bfloat162float(dy) / sqrtf(xf * xf - 1.0f));
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_acosh_backward_f32, float,
    baracuda::elementwise::AcoshBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_acosh_backward_f16, __half,
    baracuda::elementwise::AcoshBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_acosh_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::AcoshBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_acosh_backward_f64, double,
    baracuda::elementwise::AcoshBackwardFunctor<double>)
