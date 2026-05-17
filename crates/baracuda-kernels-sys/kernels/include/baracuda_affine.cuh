// SPDX-FileCopyrightText: 2024 Eric Holscher and the candle / fuel-cuda-kernels contributors
// SPDX-FileCopyrightText: 2026 Eric Evans and the baracuda contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// baracuda_affine.cuh
//
// Templated kernel + INSTANTIATE macros for the fused affine op
// (`y[i] = a * x[i] + b`). Adapted from `fuel-cuda-kernels/src/affine.cu`
// (which itself descends from huggingface/candle, dual-licensed
// MIT/Apache-2.0). The bespoke adaptations vs. Fuel:
//
//   * Contig-only fast path. Fuel ships a strided variant via the
//     `info` pointer; baracuda's plan layer materializes strided
//     before launch.
//   * `extern "C" int32_t baracuda_kernels_affine_<dtype>_run(numel,
//     x, y, a, b, ws, ws_bytes, stream)` status-code ABI matching
//     the elementwise family. `x == nullptr` is rejected (Fuel
//     interprets it as "in-place: read y instead"; baracuda's plan
//     layer can express in-place by aliasing `y` to `x`).
//   * f16 / bf16 compute through f32 to match the precision-guarantee
//     contract the rest of the elementwise family follows (f32
//     accumulator, single rounding at store).
//   * f16 / bf16 take `a` / `b` as `float` over the FFI — the kernel
//     truncates back to half on store. Matches the Pad-constant
//     family's value-by-float convention.
//
// Status codes mirror the GEMM family (see crate-level doc).

#ifndef BARACUDA_AFFINE_CUH
#define BARACUDA_AFFINE_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace baracuda { namespace affine {

// Generic contig affine kernel — both compute and storage at type `T`.
template <typename T>
__global__ void affine_contig_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    T a, T b,
    int64_t numel)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        y[i] = a * x[i] + b;
    }
}

// Half-precision contig affine kernel — compute at f32, store at __half.
__global__ inline void affine_contig_kernel_f16(
    const __half* __restrict__ x,
    __half* __restrict__ y,
    float a, float b,
    int64_t numel)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        float xi = __half2float(x[i]);
        y[i] = __float2half(a * xi + b);
    }
}

// bf16 contig affine kernel — compute at f32, store at __nv_bfloat16.
__global__ inline void affine_contig_kernel_bf16(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    float a, float b,
    int64_t numel)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        float xi = __bfloat162float(x[i]);
        y[i] = __float2bfloat16(a * xi + b);
    }
}

template <typename T>
__host__ inline int32_t launch_affine_contig(
    const T* x, T* y, T a, T b,
    int64_t numel,
    cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    affine_contig_kernel<T><<<blocks, kBlock, 0, stream>>>(x, y, a, b, numel);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

__host__ inline int32_t launch_affine_contig_f16(
    const __half* x, __half* y, float a, float b,
    int64_t numel,
    cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    affine_contig_kernel_f16<<<blocks, kBlock, 0, stream>>>(x, y, a, b, numel);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

__host__ inline int32_t launch_affine_contig_bf16(
    const __nv_bfloat16* x, __nv_bfloat16* y, float a, float b,
    int64_t numel,
    cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    affine_contig_kernel_bf16<<<blocks, kBlock, 0, stream>>>(x, y, a, b, numel);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

} } // namespace baracuda::affine

// Emit one same-dtype-compute `affine_<dtype>` launcher pair (e.g.
// f32 / f64 / integer dtypes). `a` and `b` arrive in scalar type `T`.
//
// NAME : symbol body — e.g. `affine_f32`.
// T    : scalar element type.
#define BARACUDA_KERNELS_AFFINE_INSTANTIATE(NAME, T)                                               \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t numel,                                                                              \
        const void* x, void* y,                                                                    \
        T a, T b,                                                                                   \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                           \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                  \
        if (x == nullptr || y == nullptr) return 2;                                                \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                               \
        return baracuda::affine::launch_affine_contig<T>(                                          \
            static_cast<const T*>(x), static_cast<T*>(y), a, b, numel, stream);                    \
    }                                                                                               \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                    \
        int64_t numel,                                                                              \
        const void* /*x*/, const void* /*y*/)                                                      \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        return 0;                                                                                   \
    }

// Emit one f16-storage / f32-compute `affine_f16` launcher pair.
// `a` and `b` arrive as `float`.
#define BARACUDA_KERNELS_AFFINE_INSTANTIATE_F16(NAME)                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t numel,                                                                              \
        const void* x, void* y,                                                                    \
        float a, float b,                                                                          \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                           \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                  \
        if (x == nullptr || y == nullptr) return 2;                                                \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                               \
        return baracuda::affine::launch_affine_contig_f16(                                         \
            static_cast<const __half*>(x), static_cast<__half*>(y), a, b, numel, stream);          \
    }                                                                                               \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                    \
        int64_t numel,                                                                              \
        const void* /*x*/, const void* /*y*/)                                                      \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        return 0;                                                                                   \
    }

// Emit one bf16-storage / f32-compute `affine_bf16` launcher pair.
#define BARACUDA_KERNELS_AFFINE_INSTANTIATE_BF16(NAME)                                             \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t numel,                                                                              \
        const void* x, void* y,                                                                    \
        float a, float b,                                                                          \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                           \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                  \
        if (x == nullptr || y == nullptr) return 2;                                                \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                               \
        return baracuda::affine::launch_affine_contig_bf16(                                        \
            static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(y),                  \
            a, b, numel, stream);                                                                   \
    }                                                                                               \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                    \
        int64_t numel,                                                                              \
        const void* /*x*/, const void* /*y*/)                                                      \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        return 0;                                                                                   \
    }

#endif // BARACUDA_AFFINE_CUH
