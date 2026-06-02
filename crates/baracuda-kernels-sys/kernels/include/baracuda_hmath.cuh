// baracuda_hmath.cuh — uniform half-precision (f16 / bf16) math facade.
// Phase 67f.
//
// CUDA's half-precision math API is fragmented across architectures and
// across the two half formats (`__half` / `__nv_bfloat16`) plus their
// packed 32-bit siblings (`__half2` / `__nv_bfloat162`):
//
//   - Native scalar intrinsics (`__hadd`, `__hmul`, `__hsub`, `__hfma`)
//     exist for f16 from sm_53 (Pascal) and for bf16 from sm_80 (Ampere).
//   - Packed intrinsics (`__hadd2`, `__hmul2`, `__hfma2`) follow the same
//     arch story but pay off ~2x on bandwidth-bound elementwise kernels.
//   - Transcendentals (`hexp`, `hlog`, ...) have native half forms only
//     on recent arches and with reduced precision — not worth the risk.
//
// This facade gives every kernel one set of names that works everywhere:
//
//   - `hadd` / `hmul` / `hsub` / `hfma` — overloaded for f16 + bf16, use
//     the native instruction when `__CUDA_ARCH__` supports it, otherwise
//     promote to f32, compute, and narrow back.
//   - `hexp` / `hlog` / `htanh` / `hsigmoid` / `hsqrt` — always computed
//     in f32 (the stable path; see note below) and narrowed back.
//   - `hadd2` / `hmul2` / `hfma2` — packed variants, native where
//     available, component-wise f32 fallback otherwise.
//
// Why f32 for transcendentals: the half-native transcendental intrinsics
// (`hexp`, `hlog`, `hsin`, ...) were added piecemeal across CUDA versions
// and arches and carry larger ULP error than `expf(__half2float(x))`.
// Promoting to f32, using the well-tested f32 libdevice routine, and
// narrowing back is both more portable and more accurate. The narrowing
// rounding step dominates the half-precision error budget anyway, so the
// extra f32 work is essentially free in accuracy terms.
//
// All functions are `__device__ __forceinline__`. This is a header-only,
// zero-side-effect-on-inclusion helper: including it does not pull in any
// kernel or trigger a rebuild of TUs that don't use it.
//
// Self-contained: the f32 fallbacks use the universally-available
// `__half2float` / `__float2half` (and bf16 equivalents) conversion
// intrinsics directly. When `baracuda_dtype_promote.cuh` lands, these can
// be refactored to share its `load_as_f32` / `store_from_f32`.

#ifndef BARACUDA_HMATH_CUH
#define BARACUDA_HMATH_CUH

#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace baracuda {
namespace hmath {

// =============================================================================
// Scalar arithmetic — uniform names across f16 / bf16.
//
// f16 native arithmetic is available from sm_53; bf16 native arithmetic
// from sm_80. Below those thresholds (or in host-side `__CUDA_ARCH__`
// undefined parsing) we promote to f32.
// =============================================================================

// ---- f16 -------------------------------------------------------------------

__device__ __forceinline__ __half hadd(__half a, __half b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    return __hadd(a, b);
#else
    return __float2half(__half2float(a) + __half2float(b));
#endif
}

__device__ __forceinline__ __half hsub(__half a, __half b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    return __hsub(a, b);
#else
    return __float2half(__half2float(a) - __half2float(b));
#endif
}

__device__ __forceinline__ __half hmul(__half a, __half b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    return __hmul(a, b);
#else
    return __float2half(__half2float(a) * __half2float(b));
#endif
}

// Fused-multiply-add: a*b + c. Native `__hfma` keeps the intermediate at
// full precision; the f32 fallback does likewise (single f32 fmaf).
__device__ __forceinline__ __half hfma(__half a, __half b, __half c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    return __hfma(a, b, c);
#else
    return __float2half(fmaf(__half2float(a), __half2float(b), __half2float(c)));
#endif
}

// ---- bf16 ------------------------------------------------------------------

__device__ __forceinline__ __nv_bfloat16 hadd(__nv_bfloat16 a, __nv_bfloat16 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __hadd(a, b);
#else
    return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));
#endif
}

__device__ __forceinline__ __nv_bfloat16 hsub(__nv_bfloat16 a, __nv_bfloat16 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __hsub(a, b);
#else
    return __float2bfloat16(__bfloat162float(a) - __bfloat162float(b));
#endif
}

__device__ __forceinline__ __nv_bfloat16 hmul(__nv_bfloat16 a, __nv_bfloat16 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __hmul(a, b);
#else
    return __float2bfloat16(__bfloat162float(a) * __bfloat162float(b));
#endif
}

__device__ __forceinline__ __nv_bfloat16 hfma(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __hfma(a, b, c);
#else
    return __float2bfloat16(fmaf(__bfloat162float(a), __bfloat162float(b), __bfloat162float(c)));
#endif
}

// =============================================================================
// Transcendental / nonlinear — always computed in f32 then narrowed back.
// This is the stable, portable path (see the file-top note). Use these for
// activation/normalizer math where a native half intrinsic would only save
// a conversion or two while risking precision + arch-support surprises.
// =============================================================================

// ---- f16 -------------------------------------------------------------------

__device__ __forceinline__ __half hexp(__half x) {
    return __float2half(expf(__half2float(x)));
}

__device__ __forceinline__ __half hlog(__half x) {
    return __float2half(logf(__half2float(x)));
}

__device__ __forceinline__ __half htanh(__half x) {
    return __float2half(tanhf(__half2float(x)));
}

__device__ __forceinline__ __half hsigmoid(__half x) {
    // Numerically-stable logistic on the f32 value: 1 / (1 + exp(-x)).
    return __float2half(1.0f / (1.0f + expf(-__half2float(x))));
}

__device__ __forceinline__ __half hsqrt(__half x) {
    return __float2half(sqrtf(__half2float(x)));
}

// ---- bf16 ------------------------------------------------------------------

__device__ __forceinline__ __nv_bfloat16 hexp(__nv_bfloat16 x) {
    return __float2bfloat16(expf(__bfloat162float(x)));
}

__device__ __forceinline__ __nv_bfloat16 hlog(__nv_bfloat16 x) {
    return __float2bfloat16(logf(__bfloat162float(x)));
}

__device__ __forceinline__ __nv_bfloat16 htanh(__nv_bfloat16 x) {
    return __float2bfloat16(tanhf(__bfloat162float(x)));
}

__device__ __forceinline__ __nv_bfloat16 hsigmoid(__nv_bfloat16 x) {
    return __float2bfloat16(1.0f / (1.0f + expf(-__bfloat162float(x))));
}

__device__ __forceinline__ __nv_bfloat16 hsqrt(__nv_bfloat16 x) {
    return __float2bfloat16(sqrtf(__bfloat162float(x)));
}

// =============================================================================
// Packed (vectorized) arithmetic — two half values per 32-bit register.
//
// The native `*2` intrinsics map to a single SIMD instruction and roughly
// double throughput on bandwidth-bound elementwise kernels. Below the arch
// threshold we unpack to the two components, compute in f32, and repack.
// =============================================================================

// ---- f16x2 -----------------------------------------------------------------

__device__ __forceinline__ __half2 hadd2(__half2 a, __half2 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    return __hadd2(a, b);
#else
    float2 fa = __half22float2(a);
    float2 fb = __half22float2(b);
    return __floats2half2_rn(fa.x + fb.x, fa.y + fb.y);
#endif
}

__device__ __forceinline__ __half2 hmul2(__half2 a, __half2 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    return __hmul2(a, b);
#else
    float2 fa = __half22float2(a);
    float2 fb = __half22float2(b);
    return __floats2half2_rn(fa.x * fb.x, fa.y * fb.y);
#endif
}

__device__ __forceinline__ __half2 hfma2(__half2 a, __half2 b, __half2 c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    return __hfma2(a, b, c);
#else
    float2 fa = __half22float2(a);
    float2 fb = __half22float2(b);
    float2 fc = __half22float2(c);
    return __floats2half2_rn(fmaf(fa.x, fb.x, fc.x), fmaf(fa.y, fb.y, fc.y));
#endif
}

// ---- bf16x2 ----------------------------------------------------------------

__device__ __forceinline__ __nv_bfloat162 hadd2(__nv_bfloat162 a, __nv_bfloat162 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __hadd2(a, b);
#else
    float2 fa = __bfloat1622float2(a);
    float2 fb = __bfloat1622float2(b);
    return __floats2bfloat162_rn(fa.x + fb.x, fa.y + fb.y);
#endif
}

__device__ __forceinline__ __nv_bfloat162 hmul2(__nv_bfloat162 a, __nv_bfloat162 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __hmul2(a, b);
#else
    float2 fa = __bfloat1622float2(a);
    float2 fb = __bfloat1622float2(b);
    return __floats2bfloat162_rn(fa.x * fb.x, fa.y * fb.y);
#endif
}

__device__ __forceinline__ __nv_bfloat162 hfma2(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __hfma2(a, b, c);
#else
    float2 fa = __bfloat1622float2(a);
    float2 fb = __bfloat1622float2(b);
    float2 fc = __bfloat1622float2(c);
    return __floats2bfloat162_rn(fmaf(fa.x, fb.x, fc.x), fmaf(fa.y, fb.y, fc.y));
#endif
}

}  // namespace hmath
}  // namespace baracuda

#endif  // BARACUDA_HMATH_CUH
