// baracuda-kernels Phase 3 ternary fma: elementwise `y = a * b + c`
// for FP types.
//
// Implements `y = a * b + c` over both contiguous tensors (fast path)
// and arbitrary strided / broadcast views. All four operands
// (a, b, c, y) share the same dtype `T`. This matches PyTorch's
// `torch.addcmul(c, a, b)` (with the implicit `value=1` multiplier),
// NOT the IEEE single-rounding `fma` intrinsic.
//
// Why two separate rounding steps instead of `fmaf` / `fma` / `__hfma`:
// the host reference computes the value as two separate rounding steps
// (multiply, then add). The IEEE single-rounding fused fma would
// disagree with the host at the last bit on f32 / f64. Two-step
// rounding gives bit-exact compare on f32 / f64.
//
// Wrinkle: nvcc fuses `a * b + c` into a single IEEE fma by default
// (the "fmad" / contracted-fma optimization) when compiling f32 / f64
// arithmetic with optimization on. To prevent that, the f32 / f64
// specializations split the expression via the `__fmul_rn` / `__fadd_rn`
// / `__dmul_rn` / `__dadd_rn` round-to-nearest intrinsics, which the
// compiler must NOT fuse. f16 / bf16 use the generic operator path
// (no fmad fusion is applied to those types — each op already detours
// through f32 internally).
//
// f16 / bf16 share one generic functor body — `operator*` and
// `operator+` are defined for `__half` and `__nv_bfloat16` on sm_80+
// (cuda_fp16.h / cuda_bf16.h), so no per-dtype specialization is
// needed.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

// Fma functor: `y = a * b + c` with two separate rounding steps —
// NOT the IEEE single-rounding fma intrinsic. See file-level doc-
// header for the rationale (bit-exact compare with host reference)
// and the per-dtype specialization for f32 / f64 (prevents nvcc
// fmad fusion).
template <typename T>
struct FmaFunctor {
    __device__ __forceinline__ T operator()(T a, T b, T c) const {
        return a * b + c;
    }
};

template <>
struct FmaFunctor<float> {
    __device__ __forceinline__ float operator()(float a, float b, float c) const {
        // Round-to-nearest mul / add intrinsics — explicit two-step
        // rounding that nvcc must NOT fuse into a single fma.f32.
        return __fadd_rn(__fmul_rn(a, b), c);
    }
};

template <>
struct FmaFunctor<double> {
    __device__ __forceinline__ double operator()(double a, double b, double c) const {
        // Same as the f32 specialization but f64 — `__dmul_rn` /
        // `__dadd_rn` enforce no-fusion.
        return __dadd_rn(__dmul_rn(a, b), c);
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_TERNARY_POINTWISE_INSTANTIATE(
    ternary_fma_f32,
    float,
    baracuda::elementwise::FmaFunctor<float>)

BARACUDA_KERNELS_TERNARY_POINTWISE_INSTANTIATE_STRIDED(
    ternary_fma_f32,
    float,
    baracuda::elementwise::FmaFunctor<float>)

BARACUDA_KERNELS_TERNARY_POINTWISE_INSTANTIATE(
    ternary_fma_f16,
    __half,
    baracuda::elementwise::FmaFunctor<__half>)

BARACUDA_KERNELS_TERNARY_POINTWISE_INSTANTIATE_STRIDED(
    ternary_fma_f16,
    __half,
    baracuda::elementwise::FmaFunctor<__half>)

BARACUDA_KERNELS_TERNARY_POINTWISE_INSTANTIATE(
    ternary_fma_bf16,
    __nv_bfloat16,
    baracuda::elementwise::FmaFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_TERNARY_POINTWISE_INSTANTIATE_STRIDED(
    ternary_fma_bf16,
    __nv_bfloat16,
    baracuda::elementwise::FmaFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_TERNARY_POINTWISE_INSTANTIATE(
    ternary_fma_f64,
    double,
    baracuda::elementwise::FmaFunctor<double>)

BARACUDA_KERNELS_TERNARY_POINTWISE_INSTANTIATE_STRIDED(
    ternary_fma_f64,
    double,
    baracuda::elementwise::FmaFunctor<double>)
