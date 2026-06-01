// SPDX-FileCopyrightText: 2026 Eric Evans and the baracuda contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// baracuda_dtype_promote.cuh — shared dtype promotion helpers for the
// "load a narrow storage dtype into a wide compute accumulator, then
// narrow back at write time" kernel pattern.
//
// Many baracuda kernels (normalization, attention, RoPE, reductions,
// losses, …) cannot compute directly in their storage dtype: f16 / bf16
// arithmetic is numerically catastrophic for variance / softmax /
// inverse-square-root, and integer reductions overflow their storage
// width long before the canonical modular answer is reached. The
// universal fix is to promote on load (T → f32 / f64 / i64), do the
// compute in the wide type, and narrow on store (wide → T). That
// `load`/`store` pair was duplicated verbatim across many `.cuh` files
// (`load_as_acc` in baracuda_norm.cuh; `load_as_f32` in
// baracuda_attention.cuh / baracuda_sdpa.cuh / …). This header is the
// single canonical home so future kernels share one implementation.
//
// Three promotion lanes are provided:
//   * f32 lane — `load_as_f32` / `store_from_f32`  (the common case:
//     half/bf16 → f32 compute → half/bf16). This is the direct
//     successor of norm.cuh's `load_as_acc` / `store_from_acc`.
//   * f64 lane — `load_as_f64` / `store_from_f64`  (when a kernel
//     accumulates in double for extra precision).
//   * i64 lane — `load_as_i64` / `store_from_i64`  (integer reductions:
//     sign-extend / zero-extend on load, two's-complement modular
//     narrow on store — matches baracuda_reduce_int.cuh's WidePolicy
//     contract: same low bits as the unwrapped infinite-precision
//     result).
//
// Pure device-side, header-only, zero side effects on inclusion. No
// host-callable functions. Out of scope: sub-byte dtypes (S4/U4/Bin/
// FP8 — bespoke pack/unpack) and SIMD-vectorized loads (see
// baracuda_vec_load.cuh).

#ifndef BARACUDA_DTYPE_PROMOTE_CUH
#define BARACUDA_DTYPE_PROMOTE_CUH

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace baracuda {

// =============================================================================
// f32 lane — promote storage T to f32 for compute, narrow f32 back to T.
// =============================================================================
//
// The generic template covers `float` (identity), `double` (narrowing
// cast — precision loss is expected, the caller opted into an f32
// accumulator), and any type with an implicit conversion to/from float
// (e.g. `cutlass::half_t` / `cutlass::bfloat16_t`, which route through
// their own conversion operators without needing a special case here).
// `__half` / `__nv_bfloat16` get explicit specializations because their
// fast intrinsic conversions are preferable to the implicit path.

template <typename T>
__device__ __forceinline__ float load_as_f32(T x) { return static_cast<float>(x); }

template <>
__device__ __forceinline__ float load_as_f32<__half>(__half x) { return __half2float(x); }

template <>
__device__ __forceinline__ float load_as_f32<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <typename T>
__device__ __forceinline__ T store_from_f32(float v) { return static_cast<T>(v); }

template <>
__device__ __forceinline__ __half store_from_f32<__half>(float v) {
    return __float2half(v);
}

template <>
__device__ __forceinline__ __nv_bfloat16 store_from_f32<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

// =============================================================================
// f64 lane — promote storage T to f64 for compute, narrow f64 back to T.
// =============================================================================
//
// Used by kernels that accumulate in double (f32 inputs needing extra
// headroom, or f64 storage). half / bf16 round-trip through f32
// intrinsics — there is no direct double↔half device intrinsic, and the
// extra f32 hop is lossless relative to the half mantissa anyway.

template <typename T>
__device__ __forceinline__ double load_as_f64(T x) { return static_cast<double>(x); }

template <>
__device__ __forceinline__ double load_as_f64<__half>(__half x) {
    return static_cast<double>(__half2float(x));
}

template <>
__device__ __forceinline__ double load_as_f64<__nv_bfloat16>(__nv_bfloat16 x) {
    return static_cast<double>(__bfloat162float(x));
}

template <typename T>
__device__ __forceinline__ T store_from_f64(double v) { return static_cast<T>(v); }

template <>
__device__ __forceinline__ __half store_from_f64<__half>(double v) {
    return __float2half(static_cast<float>(v));
}

template <>
__device__ __forceinline__ __nv_bfloat16 store_from_f64<__nv_bfloat16>(double v) {
    return __float2bfloat16(static_cast<float>(v));
}

// =============================================================================
// i64 lane — promote integer storage T to a 64-bit accumulator, narrow
// back to T at store time.
// =============================================================================
//
// Load semantics: signed types sign-extend, unsigned types zero-extend
// (the natural `static_cast` widening). For `uint64_t` the value may
// exceed `INT64_MAX`; the bit pattern is preserved (well-defined
// two's-complement reinterpretation in C++20 and in device code) and a
// matching narrow on store recovers it exactly.
//
// Store semantics: `static_cast<T>(v)` performs two's-complement
// modular narrowing (wrap on overflow). This matches the documented
// contract of baracuda_reduce_int.cuh's `WidePolicy::narrow` and Fuel's
// same-dtype int reduce/affine reference — the narrowed result has the
// same low bits as the unwrapped infinite-precision answer.
//
// The generic template handles every integer width uniformly
// (i8/u8/i16/u16/i32/u32/i64/u64); no per-type specialization is needed
// because the cast already encodes the correct sign/zero-extension and
// modular-narrow behavior. Intended for integer `T` only — do not
// instantiate with a floating-point type (use the f32 / f64 lanes).

template <typename T>
__device__ __forceinline__ int64_t load_as_i64(T x) { return static_cast<int64_t>(x); }

template <typename T>
__device__ __forceinline__ T store_from_i64(int64_t v) { return static_cast<T>(v); }

}  // namespace baracuda

#endif  // BARACUDA_DTYPE_PROMOTE_CUH
