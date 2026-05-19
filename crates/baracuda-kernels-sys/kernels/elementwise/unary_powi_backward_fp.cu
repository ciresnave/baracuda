// baracuda-kernels Phase 12.1 — PowI backward.
//
// Forward:  `y  = x^n`.
// Backward: `dx = n * x^(n-1) * dy`.
//
// Special cases (computed branchlessly via the standard rule):
//   `n == 0` → `dx = 0` (gradient of a constant).
//   `n == 1` → `dx = dy` (gradient of identity).
//   `n == 2` → `dx = 2 * x * dy`  (just falls out of the formula).
//
// The integer exponent `n` is threaded as `p0` (cast to `int` at kernel
// entry); `p1` is unused (ABI parity with the rest of the unary_param
// family).
//
// Same f16/bf16 → f32 precision convention as the forward: power-by-
// squaring runs in f32 for half-precision inputs, then the product
// `n * x^(n-1) * dy` is also computed in f32, with a single cast back
// to T at the end.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename Acc>
__device__ __forceinline__ Acc powi_kernel_bw(Acc x, int n) {
    if (n == 0) return Acc(1);
    int absn = n < 0 ? -n : n;
    Acc base = (n < 0) ? (Acc(1) / x) : x;
    Acc result = Acc(1);
    while (absn > 0) {
        if (absn & 1) result = result * base;
        base = base * base;
        absn >>= 1;
    }
    return result;
}

template <typename T>
struct PowIBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T x, float p0, float /*p1*/) const {
        // Default specialization unused — explicit specs below.
        int n = static_cast<int>(p0);
        if (n == 0) return T(0);
        if (n == 1) return dy;
        return T(n) * powi_kernel_bw<T>(x, n - 1) * dy;
    }
};

template <>
struct PowIBackwardFunctor<float> {
    __device__ __forceinline__ float operator()(float dy, float x, float p0, float /*p1*/) const {
        int n = static_cast<int>(p0);
        if (n == 0) return 0.0f;
        if (n == 1) return dy;
        return static_cast<float>(n) * powi_kernel_bw<float>(x, n - 1) * dy;
    }
};

template <>
struct PowIBackwardFunctor<double> {
    __device__ __forceinline__ double operator()(double dy, double x, float p0, float /*p1*/) const {
        int n = static_cast<int>(p0);
        if (n == 0) return 0.0;
        if (n == 1) return dy;
        return static_cast<double>(n) * powi_kernel_bw<double>(x, n - 1) * dy;
    }
};

template <>
struct PowIBackwardFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half dy, __half x, float p0, float /*p1*/) const {
        int n = static_cast<int>(p0);
        if (n == 0) return __float2half(0.0f);
        if (n == 1) return dy;
        float fx = __half2float(x);
        float fdy = __half2float(dy);
        float g = static_cast<float>(n) * powi_kernel_bw<float>(fx, n - 1) * fdy;
        return __float2half(g);
    }
};

template <>
struct PowIBackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 dy, __nv_bfloat16 x, float p0, float /*p1*/) const {
        int n = static_cast<int>(p0);
        if (n == 0) return __float2bfloat16(0.0f);
        if (n == 1) return dy;
        float fx = __bfloat162float(x);
        float fdy = __bfloat162float(dy);
        float g = static_cast<float>(n) * powi_kernel_bw<float>(fx, n - 1) * fdy;
        return __float2bfloat16(g);
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_PARAM_BACKWARD_INSTANTIATE(
    unary_powi_backward_f32, float,
    baracuda::elementwise::PowIBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_PARAM_BACKWARD_INSTANTIATE(
    unary_powi_backward_f16, __half,
    baracuda::elementwise::PowIBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_PARAM_BACKWARD_INSTANTIATE(
    unary_powi_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::PowIBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_PARAM_BACKWARD_INSTANTIATE(
    unary_powi_backward_f64, double,
    baracuda::elementwise::PowIBackwardFunctor<double>)
