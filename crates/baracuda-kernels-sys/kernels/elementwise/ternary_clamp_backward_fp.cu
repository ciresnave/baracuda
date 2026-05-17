// baracuda-kernels Phase 3 backward fanout (Milestone F):
// elementwise CLAMP backward.
//
// Forward: `y = min(max(a, b), c)` — clamp `a` to `[b, c]` (b = lo,
// c = hi). Backward (subgradient — see boundary note below):
//   da = dy if b <= a <= c else 0
//   db = dy if a <  b      else 0
//   dc = dy if a >  c      else 0
//
// Boundary subgradient (a == b or a == c): the gradient is multi-
// valued; we route `dy` to `da` on the tied boundary and 0 to `db` /
// `dc`. This matches PyTorch's `torch.clamp` autograd convention
// (`b <= a <= c` is closed at both ends; `a < b` and `a > c` are open).
//
// The three masks are mutually exclusive and exhaustive for finite
// inputs satisfying `b <= c` — only one of {da, db, dc} ends up as
// `dy`, the other two as 0. (If a caller passes `b > c` the forward
// clamp degenerates and the masks here don't sum to `dy` everywhere;
// that's a caller bug consistent with PyTorch behavior.)
//
// NaN: if any of a / b / c is NaN, all `<` / `<=` / `>` comparisons
// return false and all three grads come out 0. This matches PyTorch
// (NaN inputs zero out the gradient — no propagation through clamp's
// piecewise-linear boundary). For uniformity with the rest of the
// backward family, f16 / bf16 detour through f32 for the comparisons.
//
// No math is performed beyond the comparison + select, so f16 / bf16
// results are bit-exact `dy` or zero — no rounding cost.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct ClampBackwardFunctor {
    __device__ __forceinline__ void operator()(
        T dy, T a, T b, T c, T& da, T& db, T& dc) const;
};

template <>
struct ClampBackwardFunctor<float> {
    __device__ __forceinline__ void operator()(
        float dy, float a, float b, float c,
        float& da, float& db, float& dc) const
    {
        da = (b <= a && a <= c) ? dy : 0.0f;
        db = (a <  b)            ? dy : 0.0f;
        dc = (a >  c)            ? dy : 0.0f;
    }
};

template <>
struct ClampBackwardFunctor<double> {
    __device__ __forceinline__ void operator()(
        double dy, double a, double b, double c,
        double& da, double& db, double& dc) const
    {
        da = (b <= a && a <= c) ? dy : 0.0;
        db = (a <  b)            ? dy : 0.0;
        dc = (a >  c)            ? dy : 0.0;
    }
};

template <>
struct ClampBackwardFunctor<__half> {
    __device__ __forceinline__ void operator()(
        __half dy, __half a, __half b, __half c,
        __half& da, __half& db, __half& dc) const
    {
        float af = __half2float(a);
        float bf = __half2float(b);
        float cf = __half2float(c);
        __half zero = __float2half(0.0f);
        da = (bf <= af && af <= cf) ? dy : zero;
        db = (af <  bf)              ? dy : zero;
        dc = (af >  cf)              ? dy : zero;
    }
};

template <>
struct ClampBackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ void operator()(
        __nv_bfloat16 dy, __nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c,
        __nv_bfloat16& da, __nv_bfloat16& db, __nv_bfloat16& dc) const
    {
        float af = __bfloat162float(a);
        float bf = __bfloat162float(b);
        float cf = __bfloat162float(c);
        __nv_bfloat16 zero = __float2bfloat16(0.0f);
        da = (bf <= af && af <= cf) ? dy : zero;
        db = (af <  bf)              ? dy : zero;
        dc = (af >  cf)              ? dy : zero;
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_TERNARY_BACKWARD_INSTANTIATE(
    ternary_clamp_backward_f32, float,
    baracuda::elementwise::ClampBackwardFunctor<float>)

BARACUDA_KERNELS_TERNARY_BACKWARD_INSTANTIATE(
    ternary_clamp_backward_f16, __half,
    baracuda::elementwise::ClampBackwardFunctor<__half>)

BARACUDA_KERNELS_TERNARY_BACKWARD_INSTANTIATE(
    ternary_clamp_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::ClampBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_TERNARY_BACKWARD_INSTANTIATE(
    ternary_clamp_backward_f64, double,
    baracuda::elementwise::ClampBackwardFunctor<double>)
