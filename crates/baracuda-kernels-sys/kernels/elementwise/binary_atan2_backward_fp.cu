// baracuda-kernels Phase 3 backward fanout: elementwise atan2 backward.
//
// Forward: `y = atan2(a, b)`. Backward needs both saved inputs:
//   denom = a² + b²
//   da =  dy * b / denom
//   db = -dy * a / denom
//
// `denom` is zero only when both `a == 0` and `b == 0` — caller is
// responsible for avoiding that point (test inputs ensure at least one
// of a/b is nonzero). f16/bf16 backward computes in f32 (matching the
// FW f32-detour); f32 / f64 use native arithmetic directly.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct Atan2BackwardFunctor {
    __device__ __forceinline__ void operator()(T dy, T a, T b, T& da, T& db) const {
        da = dy;
        db = dy;
    }
};

template <>
struct Atan2BackwardFunctor<float> {
    __device__ __forceinline__ void operator()(
        float dy, float a, float b, float& da, float& db) const
    {
        float denom = a * a + b * b;
        float inv   = 1.0f / denom;
        da =  dy * b * inv;
        db = -dy * a * inv;
    }
};

template <>
struct Atan2BackwardFunctor<double> {
    __device__ __forceinline__ void operator()(
        double dy, double a, double b, double& da, double& db) const
    {
        double denom = a * a + b * b;
        double inv   = 1.0 / denom;
        da =  dy * b * inv;
        db = -dy * a * inv;
    }
};

template <>
struct Atan2BackwardFunctor<__half> {
    __device__ __forceinline__ void operator()(
        __half dy, __half a, __half b, __half& da, __half& db) const
    {
        float dyf = __half2float(dy);
        float af  = __half2float(a);
        float bf  = __half2float(b);
        float denom = af * af + bf * bf;
        float inv   = 1.0f / denom;
        da = __float2half( dyf * bf * inv);
        db = __float2half(-dyf * af * inv);
    }
};

template <>
struct Atan2BackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ void operator()(
        __nv_bfloat16 dy, __nv_bfloat16 a, __nv_bfloat16 b,
        __nv_bfloat16& da, __nv_bfloat16& db) const
    {
        float dyf = __bfloat162float(dy);
        float af  = __bfloat162float(a);
        float bf  = __bfloat162float(b);
        float denom = af * af + bf * bf;
        float inv   = 1.0f / denom;
        da = __float2bfloat16( dyf * bf * inv);
        db = __float2bfloat16(-dyf * af * inv);
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_BINARY_BACKWARD_SAVES_INSTANTIATE(
    binary_atan2_backward_f32, float,
    baracuda::elementwise::Atan2BackwardFunctor<float>)

BARACUDA_KERNELS_BINARY_BACKWARD_SAVES_INSTANTIATE(
    binary_atan2_backward_f16, __half,
    baracuda::elementwise::Atan2BackwardFunctor<__half>)

BARACUDA_KERNELS_BINARY_BACKWARD_SAVES_INSTANTIATE(
    binary_atan2_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::Atan2BackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_BINARY_BACKWARD_SAVES_INSTANTIATE(
    binary_atan2_backward_f64, double,
    baracuda::elementwise::Atan2BackwardFunctor<double>)
