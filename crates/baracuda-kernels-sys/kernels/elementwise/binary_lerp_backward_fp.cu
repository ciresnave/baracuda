// baracuda-kernels Phase 3 deferral: elementwise lerp backward.
//
// Forward: `y = a + weight·(b - a) = (1 - weight)·a + weight·b`. Backward:
// `da = (1 - weight)·dy`, `db = weight·dy`. No saved tensors needed — the
// gradient is a pure linear scaling of `dy` by constants derived from
// the scalar `weight`.
//
// We use the `BINARY_PARAM_BACKWARD_INSTANTIATE` macro (no-save variant)
// since the functor only needs `dy` and the weight.
//
// f16 / bf16 perform the scaling in f32 (the weight is f32) and round
// once on each store. f32 / f64 use native arithmetic.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct LerpBackwardFunctor {
    __device__ __forceinline__ void operator()(T dy, float w, T& da, T& db) const {
        // Generic fallback — explicit specs below.
        da = T((1.0f - w)) * dy;
        db = T(w) * dy;
    }
};

template <>
struct LerpBackwardFunctor<float> {
    __device__ __forceinline__ void operator()(float dy, float w, float& da, float& db) const {
        da = (1.0f - w) * dy;
        db = w * dy;
    }
};

template <>
struct LerpBackwardFunctor<double> {
    __device__ __forceinline__ void operator()(double dy, float w, double& da, double& db) const {
        double dw = (double)w;
        da = (1.0 - dw) * dy;
        db = dw * dy;
    }
};

template <>
struct LerpBackwardFunctor<__half> {
    __device__ __forceinline__ void operator()(__half dy, float w, __half& da, __half& db) const {
        float fdy = __half2float(dy);
        da = __float2half((1.0f - w) * fdy);
        db = __float2half(w * fdy);
    }
};

template <>
struct LerpBackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ void operator()(__nv_bfloat16 dy, float w, __nv_bfloat16& da, __nv_bfloat16& db) const {
        float fdy = __bfloat162float(dy);
        da = __float2bfloat16((1.0f - w) * fdy);
        db = __float2bfloat16(w * fdy);
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_BINARY_PARAM_BACKWARD_INSTANTIATE(
    binary_lerp_backward_f32, float,
    baracuda::elementwise::LerpBackwardFunctor<float>)

BARACUDA_KERNELS_BINARY_PARAM_BACKWARD_INSTANTIATE(
    binary_lerp_backward_f16, __half,
    baracuda::elementwise::LerpBackwardFunctor<__half>)

BARACUDA_KERNELS_BINARY_PARAM_BACKWARD_INSTANTIATE(
    binary_lerp_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::LerpBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_BINARY_PARAM_BACKWARD_INSTANTIATE(
    binary_lerp_backward_f64, double,
    baracuda::elementwise::LerpBackwardFunctor<double>)
