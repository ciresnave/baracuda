// SPDX-FileCopyrightText: 2026 Eric Evans and the baracuda contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// baracuda_cast_subbyte.cuh
//
// Phase 13.3 — cast paths for sub-byte / non-arithmetic dtypes that the
// classic `baracuda_cast.cuh` doesn't cover.
//
// Covers three families:
//
//   * **Bool ↔ T** — Bool storage is one byte per element with the
//     PyTorch / NumPy truthiness convention (0 = false, any non-zero =
//     true). T → Bool returns 0 or 1 strictly; Bool → T returns 0.0 or
//     1.0 (normalizing whatever non-zero byte the caller stored).
//
//   * **Fp8E4M3 / Fp8E5M2 ↔ {f32, f16, bf16}** — round-trip through f32
//     via NVIDIA's `__nv_cvt_*_to_fp8` / `__nv_cvt_fp8_to_halfraw`
//     intrinsics from `<cuda_fp8.h>`. Saturating semantics on the
//     `f32 → fp8` direction (matches the existing baracuda Fp8 GEMM
//     epilogue's `SATFINITE` convention).
//
//   * **S4 / U4 ↔ {i32, i64, f32}** — packed-pair nibble storage (low
//     nibble = even index, high nibble = odd index — same convention as
//     baracuda_dtype.cuh's `unpack_s4_byte` and the host-side
//     `S4::pack` / `::unpack`). Two directions per pair:
//       - **Unpack** (S4/U4 → wide T): one thread per 2 output elements,
//         reads one byte from the packed source. Sign-extends for S4,
//         zero-extends for U4.
//       - **Pack** (wide T → S4/U4): one thread per 2 input elements,
//         writes one byte to the packed dest. Saturates to [-8, 7] for
//         S4 or [0, 15] for U4 before nibble-masking — matches the
//         `sat_cast_f32_to_s4` / `_to_u4` helpers in baracuda_dtype.cuh.
//
// Numel convention: for the sub-byte 4-bit families, `numel` is the
// element count (NOT the byte count). The caller must pass an even
// numel — odd numels are rejected with status code 2 (`InvalidProblem`).
// The kernels read / write `numel / 2` bytes of packed storage.
//
// Status codes mirror the rest of the kernel family (see crate-level
// doc): 0 = success, 2 = invalid problem, 5 = launch failure.

#ifndef BARACUDA_CAST_SUBBYTE_CUH
#define BARACUDA_CAST_SUBBYTE_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "baracuda_dtype.cuh"

namespace baracuda { namespace cast_subbyte {

// ----------------------------------------------------------------------------
// Bool ↔ T conversions
// ----------------------------------------------------------------------------
//
// Bool storage is one byte. Truthiness convention: 0 = false, any
// non-zero byte = true. Both directions normalize: Bool → T always
// produces 0.0 or 1.0 (not 0 / 255 even when the caller passed 255);
// T → Bool always produces 0 or 1 (not whatever bit pattern the source
// happens to have).

template <typename TOut>
__device__ __forceinline__ TOut bool_to_value(uint8_t b) {
    // Any non-zero byte becomes 1; zero stays 0. Branchless via the
    // implicit bool conversion of `b != 0`.
    return static_cast<TOut>(b != 0 ? 1 : 0);
}

// Specializations for __half / __nv_bfloat16 (need float detour).
template <> __device__ __forceinline__ __half bool_to_value<__half>(uint8_t b) {
    return __float2half(b != 0 ? 1.0f : 0.0f);
}
template <> __device__ __forceinline__ __nv_bfloat16 bool_to_value<__nv_bfloat16>(uint8_t b) {
    return __float2bfloat16(b != 0 ? 1.0f : 0.0f);
}

// T → Bool: source-truthy test for arithmetic / half types.
template <typename TIn>
__device__ __forceinline__ uint8_t value_to_bool(TIn x) {
    return x != static_cast<TIn>(0) ? 1u : 0u;
}

// __half / __nv_bfloat16 don't have a generic `!=` for `0` — go through
// the explicit conversion intrinsics. We compare bits to 0 to handle
// both +0 and -0 as "false" (matches PyTorch's bool-cast semantics: any
// non-zero finite OR NaN OR ±inf is truthy; only ±0 is falsy).
template <> __device__ __forceinline__ uint8_t value_to_bool<__half>(__half x) {
    float f = __half2float(x);
    return (f != 0.0f) ? 1u : 0u;
}
template <> __device__ __forceinline__ uint8_t value_to_bool<__nv_bfloat16>(__nv_bfloat16 x) {
    float f = __bfloat162float(x);
    return (f != 0.0f) ? 1u : 0u;
}

template <typename TOut>
__global__ void bool_to_t_kernel(
    const uint8_t* __restrict__ x,
    TOut* __restrict__ y,
    int64_t numel)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        y[i] = bool_to_value<TOut>(x[i]);
    }
}

template <typename TIn>
__global__ void t_to_bool_kernel(
    const TIn* __restrict__ x,
    uint8_t* __restrict__ y,
    int64_t numel)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        y[i] = value_to_bool<TIn>(x[i]);
    }
}

template <typename TOut>
__host__ inline int32_t launch_bool_to_t(
    const uint8_t* x, TOut* y, int64_t numel, cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    bool_to_t_kernel<TOut><<<blocks, kBlock, 0, stream>>>(x, y, numel);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename TIn>
__host__ inline int32_t launch_t_to_bool(
    const TIn* x, uint8_t* y, int64_t numel, cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    t_to_bool_kernel<TIn><<<blocks, kBlock, 0, stream>>>(x, y, numel);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// ----------------------------------------------------------------------------
// Fp8 ↔ {f32, f16, bf16} conversions
// ----------------------------------------------------------------------------
//
// All Fp8 conversions route through f32 via NVIDIA's
// `__nv_cvt_fp8_to_halfraw` (returns `__nv_fp16_storage_t` which is
// bit-compatible with `__half_raw`, then we use `__half2float`) and
// `__nv_cvt_float_to_fp8(x, __NV_SATFINITE, <format>)` for the reverse.
// This matches the existing baracuda Fp8 GEMM epilogue's saturating
// semantics (clamp |x| to the format's max-finite — 448 for E4M3,
// 57344 for E5M2 — rather than producing infinities).

__device__ __forceinline__ float e4m3_to_f32(uint8_t x) {
    // __nv_cvt_fp8_to_halfraw returns a `__half_raw` (struct with a
    // single `unsigned short x` field). Wrap it as `__half` and convert.
    __half_raw raw = __nv_cvt_fp8_to_halfraw(
        static_cast<__nv_fp8_storage_t>(x), __NV_E4M3);
    return __half2float(__half(raw));
}

__device__ __forceinline__ float e5m2_to_f32(uint8_t x) {
    __half_raw raw = __nv_cvt_fp8_to_halfraw(
        static_cast<__nv_fp8_storage_t>(x), __NV_E5M2);
    return __half2float(__half(raw));
}

__device__ __forceinline__ uint8_t f32_to_e4m3(float x) {
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ uint8_t f32_to_e5m2(float x) {
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E5M2));
}

// Generic kernel: one Fp8 (storage = uint8_t) byte in, one wide-T out.
// `FP8_TO_F32_FN` is a __device__ function `float(uint8_t)`.
template <typename TOut, float (*FP8_TO_F32_FN)(uint8_t)>
__global__ void fp8_to_t_kernel(
    const uint8_t* __restrict__ x,
    TOut* __restrict__ y,
    int64_t numel)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        float f = FP8_TO_F32_FN(x[i]);
        if constexpr (sizeof(TOut) == sizeof(__half) && !std::is_same<TOut, __nv_bfloat16>::value) {
            y[i] = __float2half(f);
        } else {
            y[i] = static_cast<TOut>(f);
        }
    }
}

// Specialized launchers — explicit overload set keeps the kernel
// template simple (avoids the std::is_same constexpr branch which the
// compiler dislikes when TOut == __nv_bfloat16).
template <typename TOut>
__device__ __forceinline__ TOut f32_to_wide(float f);

template <> __device__ __forceinline__ float f32_to_wide<float>(float f) { return f; }
template <> __device__ __forceinline__ __half f32_to_wide<__half>(float f) { return __float2half(f); }
template <> __device__ __forceinline__ __nv_bfloat16 f32_to_wide<__nv_bfloat16>(float f) { return __float2bfloat16(f); }

template <typename TOut, float (*FP8_TO_F32_FN)(uint8_t)>
__global__ void fp8_to_t_kernel_v2(
    const uint8_t* __restrict__ x,
    TOut* __restrict__ y,
    int64_t numel)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        y[i] = f32_to_wide<TOut>(FP8_TO_F32_FN(x[i]));
    }
}

template <typename TIn>
__device__ __forceinline__ float wide_to_f32(TIn x);

template <> __device__ __forceinline__ float wide_to_f32<float>(float x) { return x; }
template <> __device__ __forceinline__ float wide_to_f32<__half>(__half x) { return __half2float(x); }
template <> __device__ __forceinline__ float wide_to_f32<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename TIn, uint8_t (*F32_TO_FP8_FN)(float)>
__global__ void t_to_fp8_kernel(
    const TIn* __restrict__ x,
    uint8_t* __restrict__ y,
    int64_t numel)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        y[i] = F32_TO_FP8_FN(wide_to_f32<TIn>(x[i]));
    }
}

template <typename TOut, float (*FP8_TO_F32_FN)(uint8_t)>
__host__ inline int32_t launch_fp8_to_t(
    const uint8_t* x, TOut* y, int64_t numel, cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    fp8_to_t_kernel_v2<TOut, FP8_TO_F32_FN><<<blocks, kBlock, 0, stream>>>(x, y, numel);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename TIn, uint8_t (*F32_TO_FP8_FN)(float)>
__host__ inline int32_t launch_t_to_fp8(
    const TIn* x, uint8_t* y, int64_t numel, cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    t_to_fp8_kernel<TIn, F32_TO_FP8_FN><<<blocks, kBlock, 0, stream>>>(x, y, numel);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// ----------------------------------------------------------------------------
// S4 / U4 ↔ {i32, i64, f32} conversions
// ----------------------------------------------------------------------------
//
// Sub-byte packed nibble dtypes. `numel` is the ELEMENT count (must be
// even). The packed buffer holds `numel / 2` bytes; one byte stores two
// elements: low nibble = even index, high nibble = odd index.

// Per-byte sign-extending nibble unpack: low nibble + high nibble both
// in [-8, +7]. Same convention as `baracuda_dtype.cuh::unpack_s4_byte`.
template <typename TOut>
__device__ __forceinline__ TOut s4_nibble_to_t(uint8_t nibble) {
    // Sign-extend the low 4 bits.
    int32_t v = (int32_t)((int8_t)(nibble << 4) >> 4);
    return static_cast<TOut>(v);
}

// f32 / half specializations need explicit float conversion.
template <> __device__ __forceinline__ float s4_nibble_to_t<float>(uint8_t nibble) {
    int32_t v = (int32_t)((int8_t)(nibble << 4) >> 4);
    return static_cast<float>(v);
}

template <typename TOut>
__device__ __forceinline__ TOut u4_nibble_to_t(uint8_t nibble) {
    int32_t v = (int32_t)(nibble & 0x0F);
    return static_cast<TOut>(v);
}

template <> __device__ __forceinline__ float u4_nibble_to_t<float>(uint8_t nibble) {
    int32_t v = (int32_t)(nibble & 0x0F);
    return static_cast<float>(v);
}

// UNPACK direction: one thread per output PAIR (each thread reads 1
// source byte, writes 2 dest elements).
template <typename TOut, TOut (*NIBBLE_TO_T_FN)(uint8_t)>
__global__ void int4_unpack_kernel(
    const uint8_t* __restrict__ x_bytes,
    TOut* __restrict__ y,
    int64_t numel_elements)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    int64_t pair_count = numel_elements / 2;
    for (int64_t i = tid; i < pair_count; i += step) {
        uint8_t byte = x_bytes[i];
        y[2 * i]     = NIBBLE_TO_T_FN(byte & 0x0F);
        y[2 * i + 1] = NIBBLE_TO_T_FN((byte >> 4) & 0x0F);
    }
}

template <typename TOut, TOut (*NIBBLE_TO_T_FN)(uint8_t)>
__host__ inline int32_t launch_int4_unpack(
    const uint8_t* x_bytes, TOut* y, int64_t numel_elements, cudaStream_t stream)
{
    if (numel_elements % 2 != 0) return 2;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t pair_count = numel_elements / 2;
    int64_t blocks_i64 = (pair_count + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    int4_unpack_kernel<TOut, NIBBLE_TO_T_FN><<<blocks, kBlock, 0, stream>>>(x_bytes, y, numel_elements);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// PACK direction: one thread per input PAIR (each thread reads 2 source
// elements, writes 1 dest byte). Saturates to the nibble range.

template <typename TIn>
__device__ __forceinline__ uint8_t t_to_s4_nibble(TIn x) {
    // Round-to-nearest (matches `sat_cast_f32_to_s4` for floats) plus
    // saturate to [-8, +7]. For integer TIn we already have integer
    // semantics so `__float2int_rn` is a no-op via static_cast<float>.
    int32_t v;
    if constexpr (std::is_floating_point<TIn>::value || std::is_same<TIn, float>::value) {
        v = __float2int_rn(static_cast<float>(x));
    } else {
        v = static_cast<int32_t>(x);
    }
    if (v < -8) v = -8;
    if (v > 7) v = 7;
    return static_cast<uint8_t>(v & 0x0F);
}

template <typename TIn>
__device__ __forceinline__ uint8_t t_to_u4_nibble(TIn x) {
    int32_t v;
    if constexpr (std::is_floating_point<TIn>::value || std::is_same<TIn, float>::value) {
        v = __float2int_rn(static_cast<float>(x));
    } else {
        v = static_cast<int32_t>(x);
    }
    if (v < 0) v = 0;
    if (v > 15) v = 15;
    return static_cast<uint8_t>(v & 0x0F);
}

template <typename TIn, uint8_t (*T_TO_NIBBLE_FN)(TIn)>
__global__ void int4_pack_kernel(
    const TIn* __restrict__ x,
    uint8_t* __restrict__ y_bytes,
    int64_t numel_elements)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    int64_t pair_count = numel_elements / 2;
    for (int64_t i = tid; i < pair_count; i += step) {
        uint8_t lo = T_TO_NIBBLE_FN(x[2 * i]);
        uint8_t hi = T_TO_NIBBLE_FN(x[2 * i + 1]);
        y_bytes[i] = (lo & 0x0F) | ((hi & 0x0F) << 4);
    }
}

template <typename TIn, uint8_t (*T_TO_NIBBLE_FN)(TIn)>
__host__ inline int32_t launch_int4_pack(
    const TIn* x, uint8_t* y_bytes, int64_t numel_elements, cudaStream_t stream)
{
    if (numel_elements % 2 != 0) return 2;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t pair_count = numel_elements / 2;
    int64_t blocks_i64 = (pair_count + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    int4_pack_kernel<TIn, T_TO_NIBBLE_FN><<<blocks, kBlock, 0, stream>>>(x, y_bytes, numel_elements);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

} } // namespace baracuda::cast_subbyte

// ----------------------------------------------------------------------------
// FFI emit macros — pair-specific extern "C" launchers.
// ----------------------------------------------------------------------------

// Bool → T : src byte interpreted as 0/1 truthiness, dest is wide T.
// NAME : symbol body — e.g. `cast_bool_f32`.
// TOUT : destination scalar type.
#define BARACUDA_KERNELS_CAST_BOOL_TO_T_INSTANTIATE(NAME, TOUT)                                    \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t numel, const void* x, void* y,                                                     \
        void* /*workspace*/, size_t /*workspace_bytes*/, void* stream_ptr)                         \
    {                                                                                              \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                  \
        if (x == nullptr || y == nullptr) return 2;                                                \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                               \
        return baracuda::cast_subbyte::launch_bool_to_t<TOUT>(                                     \
            static_cast<const uint8_t*>(x), static_cast<TOUT*>(y), numel, stream);                 \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                    \
        int64_t numel, const void* /*x*/, const void* /*y*/)                                       \
    { if (numel < 0) return 2; return 0; }

// T → Bool: dest is 0/1 truthiness.
#define BARACUDA_KERNELS_CAST_T_TO_BOOL_INSTANTIATE(NAME, TIN)                                     \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t numel, const void* x, void* y,                                                     \
        void* /*workspace*/, size_t /*workspace_bytes*/, void* stream_ptr)                         \
    {                                                                                              \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                  \
        if (x == nullptr || y == nullptr) return 2;                                                \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                               \
        return baracuda::cast_subbyte::launch_t_to_bool<TIN>(                                      \
            static_cast<const TIN*>(x), static_cast<uint8_t*>(y), numel, stream);                  \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                    \
        int64_t numel, const void* /*x*/, const void* /*y*/)                                       \
    { if (numel < 0) return 2; return 0; }

// Fp8 → T conversion. FP8_FN_NAME = `e4m3_to_f32` or `e5m2_to_f32`.
#define BARACUDA_KERNELS_CAST_FP8_TO_T_INSTANTIATE(NAME, TOUT, FP8_FN_NAME)                        \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t numel, const void* x, void* y,                                                     \
        void* /*workspace*/, size_t /*workspace_bytes*/, void* stream_ptr)                         \
    {                                                                                              \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                  \
        if (x == nullptr || y == nullptr) return 2;                                                \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                               \
        return baracuda::cast_subbyte::launch_fp8_to_t<                                            \
            TOUT, baracuda::cast_subbyte::FP8_FN_NAME>(                                            \
            static_cast<const uint8_t*>(x), static_cast<TOUT*>(y), numel, stream);                 \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                    \
        int64_t numel, const void* /*x*/, const void* /*y*/)                                       \
    { if (numel < 0) return 2; return 0; }

// T → Fp8 conversion. FP8_FN_NAME = `f32_to_e4m3` or `f32_to_e5m2`.
#define BARACUDA_KERNELS_CAST_T_TO_FP8_INSTANTIATE(NAME, TIN, FP8_FN_NAME)                         \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t numel, const void* x, void* y,                                                     \
        void* /*workspace*/, size_t /*workspace_bytes*/, void* stream_ptr)                         \
    {                                                                                              \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                  \
        if (x == nullptr || y == nullptr) return 2;                                                \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                               \
        return baracuda::cast_subbyte::launch_t_to_fp8<                                            \
            TIN, baracuda::cast_subbyte::FP8_FN_NAME>(                                             \
            static_cast<const TIN*>(x), static_cast<uint8_t*>(y), numel, stream);                  \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                    \
        int64_t numel, const void* /*x*/, const void* /*y*/)                                       \
    { if (numel < 0) return 2; return 0; }

// S4/U4 unpack. NIBBLE_FN_NAME = `s4_nibble_to_t<TOUT>` or `u4_nibble_to_t<TOUT>`.
#define BARACUDA_KERNELS_CAST_INT4_UNPACK_INSTANTIATE(NAME, TOUT, NIBBLE_FN)                       \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t numel, const void* x, void* y,                                                     \
        void* /*workspace*/, size_t /*workspace_bytes*/, void* stream_ptr)                         \
    {                                                                                              \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                  \
        if (x == nullptr || y == nullptr) return 2;                                                \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                               \
        return baracuda::cast_subbyte::launch_int4_unpack<TOUT, NIBBLE_FN>(                        \
            static_cast<const uint8_t*>(x), static_cast<TOUT*>(y), numel, stream);                 \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                    \
        int64_t numel, const void* /*x*/, const void* /*y*/)                                       \
    {                                                                                              \
        if (numel < 0) return 2;                                                                   \
        if (numel % 2 != 0) return 2;                                                              \
        return 0;                                                                                  \
    }

// S4/U4 pack. NIBBLE_FN_NAME = `t_to_s4_nibble<TIN>` or `t_to_u4_nibble<TIN>`.
#define BARACUDA_KERNELS_CAST_INT4_PACK_INSTANTIATE(NAME, TIN, NIBBLE_FN)                          \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t numel, const void* x, void* y,                                                     \
        void* /*workspace*/, size_t /*workspace_bytes*/, void* stream_ptr)                         \
    {                                                                                              \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                  \
        if (x == nullptr || y == nullptr) return 2;                                                \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                               \
        return baracuda::cast_subbyte::launch_int4_pack<TIN, NIBBLE_FN>(                           \
            static_cast<const TIN*>(x), static_cast<uint8_t*>(y), numel, stream);                  \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                    \
        int64_t numel, const void* /*x*/, const void* /*y*/)                                       \
    {                                                                                              \
        if (numel < 0) return 2;                                                                   \
        if (numel % 2 != 0) return 2;                                                              \
        return 0;                                                                                  \
    }

#endif // BARACUDA_CAST_SUBBYTE_CUH
