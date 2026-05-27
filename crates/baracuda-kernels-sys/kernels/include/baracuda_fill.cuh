// SPDX-FileCopyrightText: 2024 Eric Holscher and the candle / fuel-cuda-kernels contributors
// SPDX-FileCopyrightText: 2026 Eric Evans and the baracuda contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// baracuda_fill.cuh
//
// Templated kernel + INSTANTIATE macros for elementwise fill (`y[i] =
// value` for all `i`). Adapted from `fuel-cuda-kernels/src/fill.cu`
// (which itself descends from huggingface/candle, dual-licensed
// MIT/Apache-2.0). The bespoke adaptations vs. Fuel:
//
//   * Contig-only fast path. Fuel ships both a `fill_<dtype>(ptr,
//     value, numel)` flavor and a `const_set_<dtype>` strided flavor
//     with the `info` pointer; we keep only the contig flavor since
//     baracuda's plan layer materializes strided views upstream.
//   * `extern "C" int32_t baracuda_kernels_fill_<dtype>_run(numel, y,
//     value, ws, ws_bytes, stream)` status-code ABI.
//   * f16 / bf16 take the value by raw u16 bit pattern, reinterpreted
//     on the device side, so we don't depend on the __half / __nv_bfloat16
//     Windows-x64 small-struct ABI corner case (the same trick the Pad
//     kernel family uses).
//
// Status codes mirror the GEMM family (see crate-level doc).

#ifndef BARACUDA_FILL_CUH
#define BARACUDA_FILL_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace baracuda { namespace fill {

// Contig fill kernel. One thread per element, grid-cap loop for
// unbounded numel. `value` is passed by value (small POD).
template <typename T>
__global__ void fill_contig_kernel(
    T* __restrict__ y,
    T value,
    int64_t numel)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        y[i] = value;
    }
}

template <typename T>
__host__ inline int32_t launch_fill_contig(
    T* y, T value,
    int64_t numel,
    cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    fill_contig_kernel<T><<<blocks, kBlock, 0, stream>>>(y, value, numel);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Strided fill — Phase 36 (Fuel ask Gap 4).
// =============================================================================
//
// `y[lin(coord)] = value` where `lin(coord) = Σ coord[axis] *
// stride_y[axis]`. The output coords iterate row-major over the
// virtual shape `[shape[0], ..., shape[rank-1]]`; `numel = Π shape[d]`.
// Strides are signed `int64_t` (negative-stride / broadcast-stride
// supported). Rank-polymorphic up to `MAX_RANK = 8`.
//
// Mirrors the affine.cuh / softmax.cuh strided contract: shape /
// strides live on the HOST and are copied into a small POD struct
// passed by value to the kernel.

inline constexpr int MAX_RANK = 8;
struct DimsI32 { int32_t v[MAX_RANK]; };
struct DimsI64 { int64_t v[MAX_RANK]; };

template <typename T>
__global__ void fill_strided_kernel(
    T* __restrict__ y,
    T value,
    int64_t numel,
    int32_t rank,
    DimsI32 shape,
    DimsI64 stride_y)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_y = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t c = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_y += c * stride_y.v[d];
        }
        y[off_y] = value;
    }
}

template <typename T>
__host__ inline int32_t launch_fill_strided(
    T* y, T value,
    int64_t numel,
    int32_t rank,
    const int32_t* shape_host,
    const int64_t* stride_y_host,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    DimsI32 shape{};
    DimsI64 sy{};
    for (int d = 0; d < rank; ++d) {
        shape.v[d] = shape_host[d];
        sy.v[d]    = stride_y_host[d];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    fill_strided_kernel<T><<<blocks, kBlock, 0, stream>>>(
        y, value, numel, rank, shape, sy);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

} } // namespace baracuda::fill

// Emit one `fill_<dtype>` launcher pair (`_run` + `_can_implement`) for
// trivially-copyable scalar types passed by value.
//
// NAME : symbol body — e.g. `fill_f32`.
// T    : scalar element type.
#define BARACUDA_KERNELS_FILL_INSTANTIATE(NAME, T)                                                 \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t numel,                                                                              \
        void* y,                                                                                    \
        T value,                                                                                    \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                           \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                  \
        if (y == nullptr) return 2;                                                                \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                               \
        return baracuda::fill::launch_fill_contig<T>(                                              \
            static_cast<T*>(y), value, numel, stream);                                             \
    }                                                                                               \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                    \
        int64_t numel,                                                                              \
        const void* /*y*/)                                                                          \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        return 0;                                                                                   \
    }

// Emit one `fill_<dtype>` launcher pair for f16 / bf16 — `value` is
// transported as a raw u16 bit pattern and reinterpreted on the device
// side via memcpy (`__half` / `__nv_bfloat16` are `#[repr(transparent)]`
// around `unsigned short`, so a bit-cast is the canonical way to land
// them across the FFI boundary).
#define BARACUDA_KERNELS_FILL_INSTANTIATE_HALF(NAME, T)                                            \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t numel,                                                                              \
        void* y,                                                                                    \
        uint16_t value_bits,                                                                       \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                           \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                  \
        if (y == nullptr) return 2;                                                                \
        T value;                                                                                    \
        static_assert(sizeof(T) == sizeof(uint16_t), "half-precision type must be 16 bits");      \
        memcpy(&value, &value_bits, sizeof(T));                                                    \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                               \
        return baracuda::fill::launch_fill_contig<T>(                                              \
            static_cast<T*>(y), value, numel, stream);                                             \
    }                                                                                               \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                    \
        int64_t numel,                                                                              \
        const void* /*y*/)                                                                          \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        return 0;                                                                                   \
    }

// Strided fill INSTANTIATEs — Phase 36 (Fuel ask Gap 4). Same ABI as
// the contig variant plus `rank`, `shape`, `stride_y` (host-side
// arrays). Numel is the product of `shape[0..rank]`; rank up to 8.
#define BARACUDA_KERNELS_FILL_STRIDED_INSTANTIATE(NAME, T)                                         \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t numel,                                                                              \
        int32_t rank,                                                                               \
        const int32_t* shape,                                                                       \
        const int64_t* stride_y,                                                                    \
        T value,                                                                                    \
        void* y,                                                                                    \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                           \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (rank < 0 || rank > baracuda::fill::MAX_RANK) return 2;                                  \
        if (rank > 0 && (shape == nullptr || stride_y == nullptr)) return 2;                       \
        if (numel == 0) return 0;                                                                  \
        if (y == nullptr) return 2;                                                                \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                               \
        return baracuda::fill::launch_fill_strided<T>(                                             \
            static_cast<T*>(y), value, numel, rank, shape, stride_y, stream);                      \
    }                                                                                               \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                    \
        int64_t numel, int32_t rank)                                                               \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (rank < 0 || rank > baracuda::fill::MAX_RANK) return 2;                                  \
        return 0;                                                                                   \
    }

// Half-precision strided fill — same trick as the contig HALF variant
// (transport `value` as a raw u16 bit pattern, reinterpret on device).
#define BARACUDA_KERNELS_FILL_STRIDED_INSTANTIATE_HALF(NAME, T)                                    \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t numel,                                                                              \
        int32_t rank,                                                                               \
        const int32_t* shape,                                                                       \
        const int64_t* stride_y,                                                                    \
        uint16_t value_bits,                                                                       \
        void* y,                                                                                    \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                           \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (rank < 0 || rank > baracuda::fill::MAX_RANK) return 2;                                  \
        if (rank > 0 && (shape == nullptr || stride_y == nullptr)) return 2;                       \
        if (numel == 0) return 0;                                                                  \
        if (y == nullptr) return 2;                                                                \
        T value;                                                                                    \
        static_assert(sizeof(T) == sizeof(uint16_t), "half-precision type must be 16 bits");      \
        memcpy(&value, &value_bits, sizeof(T));                                                    \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                               \
        return baracuda::fill::launch_fill_strided<T>(                                             \
            static_cast<T*>(y), value, numel, rank, shape, stride_y, stream);                      \
    }                                                                                               \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                    \
        int64_t numel, int32_t rank)                                                               \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (rank < 0 || rank > baracuda::fill::MAX_RANK) return 2;                                  \
        return 0;                                                                                   \
    }

#endif // BARACUDA_FILL_CUH
