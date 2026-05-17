// baracuda-kernels Phase 3 backward fanout (Milestone F):
// elementwise ADDCDIV backward.
//
// Forward: `y = a + scale * (b / c)`. Backward:
//   da = dy
//   db = dy * scale / c
//   dc = -dy * scale * b / (c * c)
//
// Saves: b and c are referenced by the gradient; a is unused but
// read by the kernel for ABI uniformity (the caller always passes
// all three).
//
// Divide-by-zero: produces ±inf in f32 / f64; the kernel doesn't
// special-case it. Callers must avoid zero `c` if they don't want
// inf propagation. The f16 / bf16 detour propagates inf through the
// round-to-half conversion as expected.
//
// f32 / f64 use unfused round-to-nearest intrinsics for parity with
// the FW (see `ternary_addcdiv_fp.cu` for the rationale). f16 / bf16
// detour through f32 with the same intrinsics.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct AddcdivBackwardFunctor {
    __device__ __forceinline__ void operator()(
        T dy, T /*a*/, T b, T c, float scale, T& da, T& db, T& dc) const;
};

template <>
struct AddcdivBackwardFunctor<float> {
    __device__ __forceinline__ void operator()(
        float dy, float /*a*/, float b, float c, float scale,
        float& da, float& db, float& dc) const
    {
        da = dy;
        // db = dy * scale / c — order: scale/c, then * dy.
        float inv_c = __fdiv_rn(scale, c);
        db = __fmul_rn(dy, inv_c);
        // dc = -dy * scale * b / (c*c) = -(db * b) / c
        // We compute via the canonical form: -(dy * scale * b) / (c*c).
        float t1 = __fmul_rn(scale, b);
        float t2 = __fmul_rn(dy, t1);
        float c2 = __fmul_rn(c, c);
        dc = __fdiv_rn(-t2, c2);
    }
};

template <>
struct AddcdivBackwardFunctor<double> {
    __device__ __forceinline__ void operator()(
        double dy, double /*a*/, double b, double c, float scale,
        double& da, double& db, double& dc) const
    {
        da = dy;
        double sd = static_cast<double>(scale);
        double inv_c = __ddiv_rn(sd, c);
        db = __dmul_rn(dy, inv_c);
        double t1 = __dmul_rn(sd, b);
        double t2 = __dmul_rn(dy, t1);
        double c2 = __dmul_rn(c, c);
        dc = __ddiv_rn(-t2, c2);
    }
};

template <>
struct AddcdivBackwardFunctor<__half> {
    __device__ __forceinline__ void operator()(
        __half dy, __half /*a*/, __half b, __half c, float scale,
        __half& da, __half& db, __half& dc) const
    {
        float dyf = __half2float(dy);
        float bf  = __half2float(b);
        float cf  = __half2float(c);
        da = dy;
        float inv_c = __fdiv_rn(scale, cf);
        db = __float2half(__fmul_rn(dyf, inv_c));
        float t1 = __fmul_rn(scale, bf);
        float t2 = __fmul_rn(dyf, t1);
        float c2 = __fmul_rn(cf, cf);
        dc = __float2half(__fdiv_rn(-t2, c2));
    }
};

template <>
struct AddcdivBackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ void operator()(
        __nv_bfloat16 dy, __nv_bfloat16 /*a*/, __nv_bfloat16 b, __nv_bfloat16 c,
        float scale,
        __nv_bfloat16& da, __nv_bfloat16& db, __nv_bfloat16& dc) const
    {
        float dyf = __bfloat162float(dy);
        float bf  = __bfloat162float(b);
        float cf  = __bfloat162float(c);
        da = dy;
        float inv_c = __fdiv_rn(scale, cf);
        db = __float2bfloat16(__fmul_rn(dyf, inv_c));
        float t1 = __fmul_rn(scale, bf);
        float t2 = __fmul_rn(dyf, t1);
        float c2 = __fmul_rn(cf, cf);
        dc = __float2bfloat16(__fdiv_rn(-t2, c2));
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_TERNARY_BACKWARD_SCALED_INSTANTIATE(
    ternary_addcdiv_backward_f32, float,
    baracuda::elementwise::AddcdivBackwardFunctor<float>)

BARACUDA_KERNELS_TERNARY_BACKWARD_SCALED_INSTANTIATE(
    ternary_addcdiv_backward_f16, __half,
    baracuda::elementwise::AddcdivBackwardFunctor<__half>)

BARACUDA_KERNELS_TERNARY_BACKWARD_SCALED_INSTANTIATE(
    ternary_addcdiv_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::AddcdivBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_TERNARY_BACKWARD_SCALED_INSTANTIATE(
    ternary_addcdiv_backward_f64, double,
    baracuda::elementwise::AddcdivBackwardFunctor<double>)
