// baracuda-kernels Phase 3 unary backward fanout: hardswish backward.
//
// Forward: `y = x * relu6(x + 3) / 6`. Backward (three regions):
//   x <= -3 → 0
//   x >=  3 → dy
//   else    → dy * (2x + 3) / 6
// Saved-x; piecewise + scalar arithmetic in the middle region. f16 /
// bf16 use the f32-detour pattern (compute `(2x + 3) / 6` in f32, round
// once on store). f64 is bit-exact against `(2*x + 3) / 6` on the host.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct HardswishBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T x) const {
        return dy * x;  // placeholder — explicit specializations below.
    }
};

template <>
struct HardswishBackwardFunctor<float> {
    __device__ __forceinline__ float operator()(float dy, float x) const {
        if (x <= -3.0f) return 0.0f;
        if (x >= 3.0f) return dy;
        return dy * (2.0f * x + 3.0f) / 6.0f;
    }
};

template <>
struct HardswishBackwardFunctor<double> {
    __device__ __forceinline__ double operator()(double dy, double x) const {
        if (x <= -3.0) return 0.0;
        if (x >= 3.0) return dy;
        return dy * (2.0 * x + 3.0) / 6.0;
    }
};

template <>
struct HardswishBackwardFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half dy, __half x) const {
        float fx = __half2float(x);
        if (fx <= -3.0f) return __float2half(0.0f);
        if (fx >= 3.0f) return dy;
        float fdy = __half2float(dy);
        return __float2half(fdy * (2.0f * fx + 3.0f) / 6.0f);
    }
};

template <>
struct HardswishBackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 dy, __nv_bfloat16 x) const {
        float fx = __bfloat162float(x);
        if (fx <= -3.0f) return __float2bfloat16(0.0f);
        if (fx >= 3.0f) return dy;
        float fdy = __bfloat162float(dy);
        return __float2bfloat16(fdy * (2.0f * fx + 3.0f) / 6.0f);
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_hardswish_backward_f32, float,
    baracuda::elementwise::HardswishBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_hardswish_backward_f16, __half,
    baracuda::elementwise::HardswishBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_hardswish_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::HardswishBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_hardswish_backward_f64, double,
    baracuda::elementwise::HardswishBackwardFunctor<double>)
