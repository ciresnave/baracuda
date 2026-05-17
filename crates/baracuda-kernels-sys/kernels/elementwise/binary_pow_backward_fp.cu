// baracuda-kernels Phase 3 backward fanout: elementwise pow backward.
//
// Forward: `y = a^b`. Backward needs both saved inputs:
//   da = dy * b * a^(b-1)
//   db = dy * a^b * ln(a)
//
// For `a > 0` the formulas are well-defined. For `a == 0`: a^(b-1) is
// either 0 (b > 1), 1 (b == 1), or +inf (b < 1) — and `ln(0) = -inf`
// makes `db` non-finite. PyTorch convention is "let the floating-point
// arithmetic produce inf/nan as it does." We compute both gradient
// directions in f32 for f16/bf16 inputs (matching the FW f32-detour);
// f32 and f64 use the matching libdevice intrinsics directly.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct PowBackwardFunctor {
    __device__ __forceinline__ void operator()(T dy, T a, T b, T& da, T& db) const {
        // Fallback (never reached): the dtype specializations cover all
        // wired SKUs.
        da = dy;
        db = dy;
    }
};

template <>
struct PowBackwardFunctor<float> {
    __device__ __forceinline__ void operator()(
        float dy, float a, float b, float& da, float& db) const
    {
        float pow_b   = powf(a, b);
        float pow_bm1 = powf(a, b - 1.0f);
        da = dy * b * pow_bm1;
        db = dy * pow_b * logf(a);
    }
};

template <>
struct PowBackwardFunctor<double> {
    __device__ __forceinline__ void operator()(
        double dy, double a, double b, double& da, double& db) const
    {
        double pow_b   = pow(a, b);
        double pow_bm1 = pow(a, b - 1.0);
        da = dy * b * pow_bm1;
        db = dy * pow_b * log(a);
    }
};

template <>
struct PowBackwardFunctor<__half> {
    __device__ __forceinline__ void operator()(
        __half dy, __half a, __half b, __half& da, __half& db) const
    {
        float dyf = __half2float(dy);
        float af  = __half2float(a);
        float bf  = __half2float(b);
        float pow_b   = powf(af, bf);
        float pow_bm1 = powf(af, bf - 1.0f);
        da = __float2half(dyf * bf * pow_bm1);
        db = __float2half(dyf * pow_b * logf(af));
    }
};

template <>
struct PowBackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ void operator()(
        __nv_bfloat16 dy, __nv_bfloat16 a, __nv_bfloat16 b,
        __nv_bfloat16& da, __nv_bfloat16& db) const
    {
        float dyf = __bfloat162float(dy);
        float af  = __bfloat162float(a);
        float bf  = __bfloat162float(b);
        float pow_b   = powf(af, bf);
        float pow_bm1 = powf(af, bf - 1.0f);
        da = __float2bfloat16(dyf * bf * pow_bm1);
        db = __float2bfloat16(dyf * pow_b * logf(af));
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_BINARY_BACKWARD_SAVES_INSTANTIATE(
    binary_pow_backward_f32, float,
    baracuda::elementwise::PowBackwardFunctor<float>)

BARACUDA_KERNELS_BINARY_BACKWARD_SAVES_INSTANTIATE(
    binary_pow_backward_f16, __half,
    baracuda::elementwise::PowBackwardFunctor<__half>)

BARACUDA_KERNELS_BINARY_BACKWARD_SAVES_INSTANTIATE(
    binary_pow_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::PowBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_BINARY_BACKWARD_SAVES_INSTANTIATE(
    binary_pow_backward_f64, double,
    baracuda::elementwise::PowBackwardFunctor<double>)
