// baracuda-kernels Phase 3 unary backward: sinh backward.
//
// Forward: `y = sinh(x)`. Backward: `dx = dy * cosh(x)`. Saved-x.
// f32 uses `coshf`; f64 uses `cosh`; f16 / bf16 use the f32-detour
// pattern.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct SinhBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T x) const {
        return dy * x;  // placeholder — explicit specializations below are used.
    }
};

template <>
struct SinhBackwardFunctor<float> {
    __device__ __forceinline__ float operator()(float dy, float x) const {
        return dy * coshf(x);
    }
};

template <>
struct SinhBackwardFunctor<double> {
    __device__ __forceinline__ double operator()(double dy, double x) const {
        return dy * cosh(x);
    }
};

template <>
struct SinhBackwardFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half dy, __half x) const {
        return __float2half(__half2float(dy) * coshf(__half2float(x)));
    }
};

template <>
struct SinhBackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 dy, __nv_bfloat16 x) const {
        return __float2bfloat16(__bfloat162float(dy) * coshf(__bfloat162float(x)));
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_sinh_backward_f32, float,
    baracuda::elementwise::SinhBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_sinh_backward_f16, __half,
    baracuda::elementwise::SinhBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_sinh_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::SinhBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_sinh_backward_f64, double,
    baracuda::elementwise::SinhBackwardFunctor<double>)
