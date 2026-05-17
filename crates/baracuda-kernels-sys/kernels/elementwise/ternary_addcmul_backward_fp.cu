// baracuda-kernels Phase 3 backward fanout (Milestone F):
// elementwise ADDCMUL backward.
//
// Forward: `y = a + scale * b * c` (see `ternary_addcmul_fp.cu` —
// plain mul+add, no fma fusion, with `__fmul_rn` / `__fadd_rn` on
// f32 / f64). Backward:
//   da = dy
//   db = dy * scale * c
//   dc = dy * scale * b
//
// Saves: b and c are referenced by the gradient; a is unused but
// read by the kernel for ABI uniformity (the caller always passes
// all three).
//
// f32 / f64 use the matching round-to-nearest intrinsics
// (`__fmul_rn` / `__dmul_rn`) for parity with the forward rounding
// chain — keeps autograd numerically symmetric with the FW for the
// optimizer-loop case (Adam / RMSProp call addcmul both forward in
// the moment update and backward in the gradient accumulation).
//
// f16 / bf16 detour through f32 with the same intrinsics.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct AddcmulBackwardFunctor {
    __device__ __forceinline__ void operator()(
        T dy, T /*a*/, T b, T c, float scale, T& da, T& db, T& dc) const;
};

template <>
struct AddcmulBackwardFunctor<float> {
    __device__ __forceinline__ void operator()(
        float dy, float /*a*/, float b, float c, float scale,
        float& da, float& db, float& dc) const
    {
        da = dy;
        // db = dy * scale * c  (two unfused rounding steps)
        float t1 = __fmul_rn(scale, c);
        db = __fmul_rn(dy, t1);
        // dc = dy * scale * b
        float t2 = __fmul_rn(scale, b);
        dc = __fmul_rn(dy, t2);
    }
};

template <>
struct AddcmulBackwardFunctor<double> {
    __device__ __forceinline__ void operator()(
        double dy, double /*a*/, double b, double c, float scale,
        double& da, double& db, double& dc) const
    {
        da = dy;
        double sd = static_cast<double>(scale);
        double t1 = __dmul_rn(sd, c);
        db = __dmul_rn(dy, t1);
        double t2 = __dmul_rn(sd, b);
        dc = __dmul_rn(dy, t2);
    }
};

template <>
struct AddcmulBackwardFunctor<__half> {
    __device__ __forceinline__ void operator()(
        __half dy, __half /*a*/, __half b, __half c, float scale,
        __half& da, __half& db, __half& dc) const
    {
        float dyf = __half2float(dy);
        float bf  = __half2float(b);
        float cf  = __half2float(c);
        da = dy;
        float t1 = __fmul_rn(scale, cf);
        db = __float2half(__fmul_rn(dyf, t1));
        float t2 = __fmul_rn(scale, bf);
        dc = __float2half(__fmul_rn(dyf, t2));
    }
};

template <>
struct AddcmulBackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ void operator()(
        __nv_bfloat16 dy, __nv_bfloat16 /*a*/, __nv_bfloat16 b, __nv_bfloat16 c,
        float scale,
        __nv_bfloat16& da, __nv_bfloat16& db, __nv_bfloat16& dc) const
    {
        float dyf = __bfloat162float(dy);
        float bf  = __bfloat162float(b);
        float cf  = __bfloat162float(c);
        da = dy;
        float t1 = __fmul_rn(scale, cf);
        db = __float2bfloat16(__fmul_rn(dyf, t1));
        float t2 = __fmul_rn(scale, bf);
        dc = __float2bfloat16(__fmul_rn(dyf, t2));
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_TERNARY_BACKWARD_SCALED_INSTANTIATE(
    ternary_addcmul_backward_f32, float,
    baracuda::elementwise::AddcmulBackwardFunctor<float>)

BARACUDA_KERNELS_TERNARY_BACKWARD_SCALED_INSTANTIATE(
    ternary_addcmul_backward_f16, __half,
    baracuda::elementwise::AddcmulBackwardFunctor<__half>)

BARACUDA_KERNELS_TERNARY_BACKWARD_SCALED_INSTANTIATE(
    ternary_addcmul_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::AddcmulBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_TERNARY_BACKWARD_SCALED_INSTANTIATE(
    ternary_addcmul_backward_f64, double,
    baracuda::elementwise::AddcmulBackwardFunctor<double>)
