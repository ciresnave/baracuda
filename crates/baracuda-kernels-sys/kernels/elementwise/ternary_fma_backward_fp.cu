// baracuda-kernels Phase 3 backward fanout (Milestone F):
// elementwise FMA backward.
//
// Forward: `y = a*b + c` (two-step rounding — see `ternary_fma_fp.cu`
// for the no-fmad-fusion rationale). Backward:
//   da = dy * b
//   db = dy * a
//   dc = dy
//
// Saves: a and b are referenced by the gradient; c is unused but read
// by the kernel for ABI uniformity (the caller always passes all three).
//
// f16 / bf16 use the f32 detour for the two multiplies — `dy * a` and
// `dy * b` go through `float` to keep the rounding behavior aligned
// with PyTorch's autograd convention for low-precision dtypes.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct FmaBackwardFunctor {
    __device__ __forceinline__ void operator()(
        T dy, T a, T b, T /*c*/, T& da, T& db, T& dc) const
    {
        da = dy * b;
        db = dy * a;
        dc = dy;
    }
};

template <>
struct FmaBackwardFunctor<__half> {
    __device__ __forceinline__ void operator()(
        __half dy, __half a, __half b, __half /*c*/,
        __half& da, __half& db, __half& dc) const
    {
        float dyf = __half2float(dy);
        float af  = __half2float(a);
        float bf  = __half2float(b);
        da = __float2half(dyf * bf);
        db = __float2half(dyf * af);
        dc = dy;
    }
};

template <>
struct FmaBackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ void operator()(
        __nv_bfloat16 dy, __nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 /*c*/,
        __nv_bfloat16& da, __nv_bfloat16& db, __nv_bfloat16& dc) const
    {
        float dyf = __bfloat162float(dy);
        float af  = __bfloat162float(a);
        float bf  = __bfloat162float(b);
        da = __float2bfloat16(dyf * bf);
        db = __float2bfloat16(dyf * af);
        dc = dy;
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_TERNARY_BACKWARD_INSTANTIATE(
    ternary_fma_backward_f32, float,
    baracuda::elementwise::FmaBackwardFunctor<float>)

BARACUDA_KERNELS_TERNARY_BACKWARD_INSTANTIATE(
    ternary_fma_backward_f16, __half,
    baracuda::elementwise::FmaBackwardFunctor<__half>)

BARACUDA_KERNELS_TERNARY_BACKWARD_INSTANTIATE(
    ternary_fma_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::FmaBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_TERNARY_BACKWARD_INSTANTIATE(
    ternary_fma_backward_f64, double,
    baracuda::elementwise::FmaBackwardFunctor<double>)
