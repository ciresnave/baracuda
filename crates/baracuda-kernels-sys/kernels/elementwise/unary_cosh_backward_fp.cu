// baracuda-kernels Phase 3 unary backward: cosh backward.
//
// Forward: `y = cosh(x)`. Backward: `dx = dy * sinh(x)`. Saved-x.
// f32 uses `sinhf`; f64 uses `sinh`; f16 / bf16 use the f32-detour
// pattern.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct CoshBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T x) const {
        return dy * x;  // placeholder — explicit specializations below are used.
    }
};

template <>
struct CoshBackwardFunctor<float> {
    __device__ __forceinline__ float operator()(float dy, float x) const {
        return dy * sinhf(x);
    }
};

template <>
struct CoshBackwardFunctor<double> {
    __device__ __forceinline__ double operator()(double dy, double x) const {
        return dy * sinh(x);
    }
};

template <>
struct CoshBackwardFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half dy, __half x) const {
        return __float2half(__half2float(dy) * sinhf(__half2float(x)));
    }
};

template <>
struct CoshBackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 dy, __nv_bfloat16 x) const {
        return __float2bfloat16(__bfloat162float(dy) * sinhf(__bfloat162float(x)));
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_cosh_backward_f32, float,
    baracuda::elementwise::CoshBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_cosh_backward_f16, __half,
    baracuda::elementwise::CoshBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_cosh_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::CoshBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_cosh_backward_f64, double,
    baracuda::elementwise::CoshBackwardFunctor<double>)
