// baracuda-kernels Phase 3 unary backward fanout: hardshrink backward.
//
// Forward: `y = x if |x| > λ else 0`. Backward: `dx = dy if |x| > λ else
// 0`. Saved-x. λ is hardcoded to 0.5 (PyTorch default) — when the
// parameterized-unary plan ships, this kernel is re-emitted with λ as a
// runtime parameter. PyTorch convention: at the boundaries `|x| == λ`
// the gradient is `0` (the kept region is the open exterior).
//
// f16 / bf16 use the f32-detour pattern (compare in f32; the `dy` bits
// are preserved on the kept branch, zero on the shrunk branch — same
// shape as relu BW).

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct HardshrinkBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T x) const {
        T ax = (x < T(0)) ? -x : x;
        return (ax > T(0.5)) ? dy : T(0);
    }
};

template <>
struct HardshrinkBackwardFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half dy, __half x) const {
        float fx = __half2float(x);
        return (fabsf(fx) > 0.5f) ? dy : __float2half(0.0f);
    }
};

template <>
struct HardshrinkBackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 dy, __nv_bfloat16 x) const {
        float fx = __bfloat162float(x);
        return (fabsf(fx) > 0.5f) ? dy : __float2bfloat16(0.0f);
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_hardshrink_backward_f32, float,
    baracuda::elementwise::HardshrinkBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_hardshrink_backward_f16, __half,
    baracuda::elementwise::HardshrinkBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_hardshrink_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::HardshrinkBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_hardshrink_backward_f64, double,
    baracuda::elementwise::HardshrinkBackwardFunctor<double>)
