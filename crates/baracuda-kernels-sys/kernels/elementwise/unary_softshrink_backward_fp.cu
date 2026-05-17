// baracuda-kernels Phase 3 unary backward fanout: softshrink backward.
//
// Forward: `y = x - λ if x > λ; x + λ if x < -λ; else 0`. Backward:
// `dx = dy if |x| > λ else 0` — identical mask to hardshrink BW, only
// the FW differs. Saved-x. λ is hardcoded to 0.5 (PyTorch default) —
// when the parameterized-unary plan ships, this kernel is re-emitted
// with λ as a runtime parameter. PyTorch convention: at the boundaries
// `|x| == λ` the gradient is `0`.
//
// f16 / bf16 use the f32-detour pattern (compare in f32; `dy` bits
// preserved on the kept branch, zero on the shrunk branch).

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct SoftshrinkBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T x) const {
        T ax = (x < T(0)) ? -x : x;
        return (ax > T(0.5)) ? dy : T(0);
    }
};

template <>
struct SoftshrinkBackwardFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half dy, __half x) const {
        float fx = __half2float(x);
        return (fabsf(fx) > 0.5f) ? dy : __float2half(0.0f);
    }
};

template <>
struct SoftshrinkBackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 dy, __nv_bfloat16 x) const {
        float fx = __bfloat162float(x);
        return (fabsf(fx) > 0.5f) ? dy : __float2bfloat16(0.0f);
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_softshrink_backward_f32, float,
    baracuda::elementwise::SoftshrinkBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_softshrink_backward_f16, __half,
    baracuda::elementwise::SoftshrinkBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_softshrink_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::SoftshrinkBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_softshrink_backward_f64, double,
    baracuda::elementwise::SoftshrinkBackwardFunctor<double>)
