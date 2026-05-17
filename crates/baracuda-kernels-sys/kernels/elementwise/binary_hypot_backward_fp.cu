// baracuda-kernels Phase 3 backward fanout: elementwise hypot backward.
//
// Forward: `y = sqrt(a² + b²)`. Backward:
//   da = dy * a / y
//   db = dy * b / y
//
// Mathematically the saved `y` could be reused, but the existing
// `BinaryBackwardArgs` only carries saved `a` and `b` — no slot for
// `y`. To avoid surgery on the host-side arg shape, we reconstruct
// `y = sqrt(a² + b²)` inline from the saved inputs. This costs one
// extra sqrt per cell vs reading a saved `y`; for the f32/f64 paths
// libdevice `sqrtf` / `sqrt` is ~1 SFU op and the overall BW is
// memory-bound, so the cost is negligible. f16/bf16 backward computes
// in f32 (matching FW f32-detour).
//
// `y == 0` only when both `a == 0` and `b == 0` — caller is responsible
// for avoiding that point (test inputs ensure at least one is nonzero).

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct HypotBackwardFunctor {
    __device__ __forceinline__ void operator()(T dy, T a, T b, T& da, T& db) const {
        da = dy;
        db = dy;
    }
};

template <>
struct HypotBackwardFunctor<float> {
    __device__ __forceinline__ void operator()(
        float dy, float a, float b, float& da, float& db) const
    {
        float y   = sqrtf(a * a + b * b);
        float inv = 1.0f / y;
        da = dy * a * inv;
        db = dy * b * inv;
    }
};

template <>
struct HypotBackwardFunctor<double> {
    __device__ __forceinline__ void operator()(
        double dy, double a, double b, double& da, double& db) const
    {
        double y   = sqrt(a * a + b * b);
        double inv = 1.0 / y;
        da = dy * a * inv;
        db = dy * b * inv;
    }
};

template <>
struct HypotBackwardFunctor<__half> {
    __device__ __forceinline__ void operator()(
        __half dy, __half a, __half b, __half& da, __half& db) const
    {
        float dyf = __half2float(dy);
        float af  = __half2float(a);
        float bf  = __half2float(b);
        float y   = sqrtf(af * af + bf * bf);
        float inv = 1.0f / y;
        da = __float2half(dyf * af * inv);
        db = __float2half(dyf * bf * inv);
    }
};

template <>
struct HypotBackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ void operator()(
        __nv_bfloat16 dy, __nv_bfloat16 a, __nv_bfloat16 b,
        __nv_bfloat16& da, __nv_bfloat16& db) const
    {
        float dyf = __bfloat162float(dy);
        float af  = __bfloat162float(a);
        float bf  = __bfloat162float(b);
        float y   = sqrtf(af * af + bf * bf);
        float inv = 1.0f / y;
        da = __float2bfloat16(dyf * af * inv);
        db = __float2bfloat16(dyf * bf * inv);
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_BINARY_BACKWARD_SAVES_INSTANTIATE(
    binary_hypot_backward_f32, float,
    baracuda::elementwise::HypotBackwardFunctor<float>)

BARACUDA_KERNELS_BINARY_BACKWARD_SAVES_INSTANTIATE(
    binary_hypot_backward_f16, __half,
    baracuda::elementwise::HypotBackwardFunctor<__half>)

BARACUDA_KERNELS_BINARY_BACKWARD_SAVES_INSTANTIATE(
    binary_hypot_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::HypotBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_BINARY_BACKWARD_SAVES_INSTANTIATE(
    binary_hypot_backward_f64, double,
    baracuda::elementwise::HypotBackwardFunctor<double>)
