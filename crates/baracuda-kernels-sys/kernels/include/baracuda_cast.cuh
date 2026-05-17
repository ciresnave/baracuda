// SPDX-FileCopyrightText: 2024 Eric Holscher and the candle / fuel-cuda-kernels contributors
// SPDX-FileCopyrightText: 2026 Eric Evans and the baracuda contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// baracuda_cast.cuh
//
// Templated kernel + INSTANTIATE macros for elementwise dtype casts
// (`y = (TOut) x`). Adapted from `fuel-cuda-kernels/src/cast.cu` (which
// itself descends from the huggingface/candle CUDA kernel set, dual-
// licensed MIT/Apache-2.0). The bespoke adaptations vs. Fuel:
//
//   * Contig-only fast path. baracuda's plan layer always materializes
//     strided views before the launch — no need for the `info` pointer +
//     `get_strided_index` branch Fuel ships.
//   * `extern "C" int32_t baracuda_kernels_cast_<sin>_<sout>_run(...)`
//     status-code ABI matching the rest of the family (0 success,
//     2 invalid problem, 5 launch failure).
//   * f16 / bf16 conversions route through `float` via explicit
//     overload selection on a `cast_value` helper, instead of the
//     `cast_through<TIn, TOut, float>` template Fuel uses (which forces
//     the caller to know whether to detour at site-of-use).
//
// Status codes mirror the GEMM family (see crate-level doc).

#ifndef BARACUDA_CAST_CUH
#define BARACUDA_CAST_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace baracuda { namespace cast {

// ----------------------------------------------------------------------------
// `cast_value<TIn, TOut>` — unified per-element conversion entry point.
//
// Default: plain `static_cast<TOut>(x)`. Handles every (TIn, TOut) pair
// where both endpoints are arithmetic ABI types (float / double / signed
// + unsigned integers).
//
// Half-precision endpoints get function overloads (NOT template
// specializations) on a `cast_value_impl` 2-arg dispatcher, picked by
// argument-dependent overload resolution. This sidesteps the
// partial-specialization-ambiguity problem you hit if you specialize
// both on `(__half, TOut)` and `(TIn, __half)`: a single overload set
// can't be ambiguous between two specs because each pair has at most
// one overload that matches both ends.
// ----------------------------------------------------------------------------

// Primary template — both ends are non-half arithmetic types.
template <typename TIn, typename TOut>
__device__ __forceinline__ TOut cast_value(TIn x) {
    return static_cast<TOut>(x);
}

// __half source -> any arithmetic destination. Detour through f32.
template <> __device__ __forceinline__ float           cast_value<__half, float>(__half x)           { return __half2float(x); }
template <> __device__ __forceinline__ double          cast_value<__half, double>(__half x)          { return static_cast<double>(__half2float(x)); }
template <> __device__ __forceinline__ int8_t          cast_value<__half, int8_t>(__half x)          { return static_cast<int8_t>(__half2float(x)); }
template <> __device__ __forceinline__ int16_t         cast_value<__half, int16_t>(__half x)         { return static_cast<int16_t>(__half2float(x)); }
template <> __device__ __forceinline__ int32_t         cast_value<__half, int32_t>(__half x)         { return static_cast<int32_t>(__half2float(x)); }
template <> __device__ __forceinline__ int64_t         cast_value<__half, int64_t>(__half x)         { return static_cast<int64_t>(__half2float(x)); }
template <> __device__ __forceinline__ uint8_t         cast_value<__half, uint8_t>(__half x)         { return static_cast<uint8_t>(__half2float(x)); }
template <> __device__ __forceinline__ uint32_t        cast_value<__half, uint32_t>(__half x)        { return static_cast<uint32_t>(__half2float(x)); }
template <> __device__ __forceinline__ __half          cast_value<__half, __half>(__half x)          { return x; }
template <> __device__ __forceinline__ __nv_bfloat16   cast_value<__half, __nv_bfloat16>(__half x)   { return __float2bfloat16(__half2float(x)); }

// __nv_bfloat16 source -> any arithmetic destination.
template <> __device__ __forceinline__ float           cast_value<__nv_bfloat16, float>(__nv_bfloat16 x)           { return __bfloat162float(x); }
template <> __device__ __forceinline__ double          cast_value<__nv_bfloat16, double>(__nv_bfloat16 x)          { return static_cast<double>(__bfloat162float(x)); }
template <> __device__ __forceinline__ int8_t          cast_value<__nv_bfloat16, int8_t>(__nv_bfloat16 x)          { return static_cast<int8_t>(__bfloat162float(x)); }
template <> __device__ __forceinline__ int16_t         cast_value<__nv_bfloat16, int16_t>(__nv_bfloat16 x)         { return static_cast<int16_t>(__bfloat162float(x)); }
template <> __device__ __forceinline__ int32_t         cast_value<__nv_bfloat16, int32_t>(__nv_bfloat16 x)         { return static_cast<int32_t>(__bfloat162float(x)); }
template <> __device__ __forceinline__ int64_t         cast_value<__nv_bfloat16, int64_t>(__nv_bfloat16 x)         { return static_cast<int64_t>(__bfloat162float(x)); }
template <> __device__ __forceinline__ uint8_t         cast_value<__nv_bfloat16, uint8_t>(__nv_bfloat16 x)         { return static_cast<uint8_t>(__bfloat162float(x)); }
template <> __device__ __forceinline__ uint32_t        cast_value<__nv_bfloat16, uint32_t>(__nv_bfloat16 x)        { return static_cast<uint32_t>(__bfloat162float(x)); }
template <> __device__ __forceinline__ __half          cast_value<__nv_bfloat16, __half>(__nv_bfloat16 x)          { return __float2half(__bfloat162float(x)); }
template <> __device__ __forceinline__ __nv_bfloat16   cast_value<__nv_bfloat16, __nv_bfloat16>(__nv_bfloat16 x)   { return x; }

// Arithmetic source -> __half (any TIn except __half itself, which is
// handled above).
template <> __device__ __forceinline__ __half cast_value<float, __half>(float x)        { return __float2half(x); }
template <> __device__ __forceinline__ __half cast_value<double, __half>(double x)      { return __float2half(static_cast<float>(x)); }
template <> __device__ __forceinline__ __half cast_value<int8_t, __half>(int8_t x)      { return __float2half(static_cast<float>(x)); }
template <> __device__ __forceinline__ __half cast_value<int16_t, __half>(int16_t x)    { return __float2half(static_cast<float>(x)); }
template <> __device__ __forceinline__ __half cast_value<int32_t, __half>(int32_t x)    { return __float2half(static_cast<float>(x)); }
template <> __device__ __forceinline__ __half cast_value<int64_t, __half>(int64_t x)    { return __float2half(static_cast<float>(x)); }
template <> __device__ __forceinline__ __half cast_value<uint8_t, __half>(uint8_t x)    { return __float2half(static_cast<float>(x)); }
template <> __device__ __forceinline__ __half cast_value<uint32_t, __half>(uint32_t x)  { return __float2half(static_cast<float>(x)); }

// Arithmetic source -> __nv_bfloat16.
template <> __device__ __forceinline__ __nv_bfloat16 cast_value<float, __nv_bfloat16>(float x)        { return __float2bfloat16(x); }
template <> __device__ __forceinline__ __nv_bfloat16 cast_value<double, __nv_bfloat16>(double x)      { return __float2bfloat16(static_cast<float>(x)); }
template <> __device__ __forceinline__ __nv_bfloat16 cast_value<int8_t, __nv_bfloat16>(int8_t x)      { return __float2bfloat16(static_cast<float>(x)); }
template <> __device__ __forceinline__ __nv_bfloat16 cast_value<int16_t, __nv_bfloat16>(int16_t x)    { return __float2bfloat16(static_cast<float>(x)); }
template <> __device__ __forceinline__ __nv_bfloat16 cast_value<int32_t, __nv_bfloat16>(int32_t x)    { return __float2bfloat16(static_cast<float>(x)); }
template <> __device__ __forceinline__ __nv_bfloat16 cast_value<int64_t, __nv_bfloat16>(int64_t x)    { return __float2bfloat16(static_cast<float>(x)); }
template <> __device__ __forceinline__ __nv_bfloat16 cast_value<uint8_t, __nv_bfloat16>(uint8_t x)    { return __float2bfloat16(static_cast<float>(x)); }
template <> __device__ __forceinline__ __nv_bfloat16 cast_value<uint32_t, __nv_bfloat16>(uint32_t x)  { return __float2bfloat16(static_cast<float>(x)); }

// Contig cast kernel. One thread per element, grid-cap loop to handle
// arbitrarily large numel.
template <typename TIn, typename TOut>
__global__ void cast_contig_kernel(
    const TIn* __restrict__ x,
    TOut* __restrict__ y,
    int64_t numel)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        y[i] = cast_value<TIn, TOut>(x[i]);
    }
}

template <typename TIn, typename TOut>
__host__ inline int32_t launch_cast_contig(
    const TIn* x, TOut* y,
    int64_t numel,
    cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    cast_contig_kernel<TIn, TOut><<<blocks, kBlock, 0, stream>>>(x, y, numel);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

} } // namespace baracuda::cast

// Emit one `cast_<sin>_<sout>` launcher pair (`_run` + `_can_implement`).
//
// NAME : symbol body — e.g. `cast_f32_i32`.
// TIN  : source scalar type.
// TOUT : destination scalar type.
#define BARACUDA_KERNELS_CAST_INSTANTIATE(NAME, TIN, TOUT)                                         \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t numel,                                                                              \
        const void* x, void* y,                                                                    \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                           \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                  \
        if (x == nullptr || y == nullptr) return 2;                                                \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                               \
        return baracuda::cast::launch_cast_contig<TIN, TOUT>(                                      \
            static_cast<const TIN*>(x), static_cast<TOUT*>(y), numel, stream);                     \
    }                                                                                               \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                    \
        int64_t numel,                                                                              \
        const void* /*x*/, const void* /*y*/)                                                      \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        return 0;                                                                                   \
    }

#endif // BARACUDA_CAST_CUH
