// baracuda-kernels Phase 3 unary backward fanout: hardtanh backward.
//
// Forward: `y = clamp(x, -1, 1)`. Backward: `dx = (-1 < x < 1) ? dy : 0`.
// Saved-x; piecewise activation BW. PyTorch picks zero at the exact
// boundary points x == -1 and x == 1 (subgradient is undefined there).
//
// f16 / bf16 use the f32-detour pattern (compare in f32, select bits;
// the dy bits are preserved on the inside branch, zero on the outside),
// so the result is bit-exact against the host reference for every dtype.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct HardtanhBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T x) const {
        return (x > T(-1) && x < T(1)) ? dy : T(0);
    }
};

template <>
struct HardtanhBackwardFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half dy, __half x) const {
        float fx = __half2float(x);
        return (fx > -1.0f && fx < 1.0f) ? dy : __float2half(0.0f);
    }
};

template <>
struct HardtanhBackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 dy, __nv_bfloat16 x) const {
        float fx = __bfloat162float(x);
        return (fx > -1.0f && fx < 1.0f) ? dy : __float2bfloat16(0.0f);
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_hardtanh_backward_f32, float,
    baracuda::elementwise::HardtanhBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_hardtanh_backward_f16, __half,
    baracuda::elementwise::HardtanhBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_hardtanh_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::HardtanhBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_hardtanh_backward_f64, double,
    baracuda::elementwise::HardtanhBackwardFunctor<double>)
