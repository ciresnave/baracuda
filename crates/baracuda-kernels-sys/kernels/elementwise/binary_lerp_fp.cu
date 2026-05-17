// baracuda-kernels Phase 3 deferral: elementwise lerp for FP types.
//
// Implements `y = a + weight·(b - a)` where `weight` is a runtime scalar
// parameter threaded through via the new `BINARY_PARAM_INSTANTIATE`
// macro. Matches PyTorch's `torch.lerp(a, b, weight)` semantics with a
// scalar (broadcast) weight. Algebraically equivalent to
// `(1 - weight)·a + weight·b` — we use the `a + w·(b - a)` form (one
// fewer rounding) to match PyTorch's native implementation.
//
// f16 / bf16 do the arithmetic in f32 (the param itself is f32) and
// round once on the store. f32 / f64 use native arithmetic (f64 widens
// the f32 weight losslessly).

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct LerpFunctor {
    __device__ __forceinline__ T operator()(T a, T b, float w) const {
        // Generic fallback — explicit specs below.
        return a + T(w) * (b - a);
    }
};

template <>
struct LerpFunctor<float> {
    __device__ __forceinline__ float operator()(float a, float b, float w) const {
        return a + w * (b - a);
    }
};

template <>
struct LerpFunctor<double> {
    __device__ __forceinline__ double operator()(double a, double b, float w) const {
        return a + (double)w * (b - a);
    }
};

template <>
struct LerpFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half a, __half b, float w) const {
        float fa = __half2float(a);
        float fb = __half2float(b);
        float y = fa + w * (fb - fa);
        return __float2half(y);
    }
};

template <>
struct LerpFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 a, __nv_bfloat16 b, float w) const {
        float fa = __bfloat162float(a);
        float fb = __bfloat162float(b);
        float y = fa + w * (fb - fa);
        return __float2bfloat16(y);
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_BINARY_PARAM_INSTANTIATE(
    binary_lerp_f32,
    float,
    baracuda::elementwise::LerpFunctor<float>)

BARACUDA_KERNELS_BINARY_PARAM_INSTANTIATE(
    binary_lerp_f16,
    __half,
    baracuda::elementwise::LerpFunctor<__half>)

BARACUDA_KERNELS_BINARY_PARAM_INSTANTIATE(
    binary_lerp_bf16,
    __nv_bfloat16,
    baracuda::elementwise::LerpFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_BINARY_PARAM_INSTANTIATE(
    binary_lerp_f64,
    double,
    baracuda::elementwise::LerpFunctor<double>)
