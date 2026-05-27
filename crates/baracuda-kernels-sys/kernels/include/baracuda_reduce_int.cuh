// SPDX-FileCopyrightText: 2026 Eric Evans and the baracuda contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// baracuda_reduce_int.cuh
//
// Phase 37 Gap 1b — integer-dtype Sum and Prod single-axis reductions
// with a widened internal accumulator (i64 for signed, u64 for unsigned)
// and store-time narrowing back to the input dtype. Wrap-on-overflow
// is the documented contract (matches Fuel's CPU reference, which uses
// same-dtype output).
//
// Why widen at all?  Without widening, partial sums of an int8/u8/i16
// axis would wrap MANY times during accumulation, often producing a
// reduced value that differs from the canonical "sum modulo 2^N"
// answer that Fuel's CPU side expects. By accumulating in the wider
// type and narrowing only at the store site, we guarantee the same
// final low bits as the unwrapped infinite-precision result.
//
// The kernel mirrors the shape of `reduce_axis_kernel` from
// `baracuda_elementwise.cuh`:
//   * One thread per output cell.
//   * Walks the reduce axis with `reduce_stride_x`.
//   * Output shape is input shape with the reduced axis collapsed to 1.

#ifndef BARACUDA_REDUCE_INT_CUH
#define BARACUDA_REDUCE_INT_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "baracuda_elementwise.cuh"   // for DimsI32 / DimsI64 / MAX_RANK

namespace baracuda { namespace reduce_int {

// Per-dtype widened-accumulator policy. Each policy provides:
//   * AccT             — accumulator type (i64 / u64).
//   * sum_identity()   — 0.
//   * prod_identity()  — 1.
//   * load(p)          — read one T into AccT (signed sign-extends, unsigned zero-extends).
//   * narrow(v)        — narrow AccT back to T at store time (wrap on overflow).
template <typename T> struct WidePolicy;

template <> struct WidePolicy<uint8_t> {
    using AccT = uint64_t;
    static __device__ __forceinline__ AccT sum_identity()  { return 0; }
    static __device__ __forceinline__ AccT prod_identity() { return 1; }
    static __device__ __forceinline__ AccT load(uint8_t v)  { return static_cast<AccT>(v); }
    static __device__ __forceinline__ uint8_t narrow(AccT v) { return static_cast<uint8_t>(v); }
};

template <> struct WidePolicy<int8_t> {
    using AccT = int64_t;
    static __device__ __forceinline__ AccT sum_identity()  { return 0; }
    static __device__ __forceinline__ AccT prod_identity() { return 1; }
    static __device__ __forceinline__ AccT load(int8_t v)   { return static_cast<AccT>(v); }
    static __device__ __forceinline__ int8_t narrow(AccT v) { return static_cast<int8_t>(v); }
};

template <> struct WidePolicy<uint32_t> {
    using AccT = uint64_t;
    static __device__ __forceinline__ AccT sum_identity()  { return 0; }
    static __device__ __forceinline__ AccT prod_identity() { return 1; }
    static __device__ __forceinline__ AccT load(uint32_t v) { return static_cast<AccT>(v); }
    static __device__ __forceinline__ uint32_t narrow(AccT v) { return static_cast<uint32_t>(v); }
};

template <> struct WidePolicy<int16_t> {
    using AccT = int64_t;
    static __device__ __forceinline__ AccT sum_identity()  { return 0; }
    static __device__ __forceinline__ AccT prod_identity() { return 1; }
    static __device__ __forceinline__ AccT load(int16_t v)  { return static_cast<AccT>(v); }
    static __device__ __forceinline__ int16_t narrow(AccT v) { return static_cast<int16_t>(v); }
};

template <> struct WidePolicy<int32_t> {
    using AccT = int64_t;
    static __device__ __forceinline__ AccT sum_identity()  { return 0; }
    static __device__ __forceinline__ AccT prod_identity() { return 1; }
    static __device__ __forceinline__ AccT load(int32_t v)  { return static_cast<AccT>(v); }
    static __device__ __forceinline__ int32_t narrow(AccT v) { return static_cast<int32_t>(v); }
};

// i64: accumulator is already i64. Wrap-on-overflow is the natural
// signed-arithmetic UB-in-host-C++ but well-defined modulo-2^64 in
// device code; both sides agree on bit-level result for the +/* ops
// we care about.
template <> struct WidePolicy<int64_t> {
    using AccT = int64_t;
    static __device__ __forceinline__ AccT sum_identity()  { return 0; }
    static __device__ __forceinline__ AccT prod_identity() { return 1; }
    static __device__ __forceinline__ AccT load(int64_t v)  { return v; }
    static __device__ __forceinline__ int64_t narrow(AccT v) { return v; }
};

enum class IntReduceOp { Sum = 0, Prod = 1 };

template <typename T, IntReduceOp OP>
__global__ void reduce_int_axis_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int64_t output_numel,
    int32_t rank,
    baracuda::elementwise::DimsI32 output_shape,
    baracuda::elementwise::DimsI64 stride_x,
    baracuda::elementwise::DimsI64 stride_y,
    int32_t reduce_axis,
    int32_t reduce_extent,
    int64_t reduce_stride_x)
{
    using Pol  = WidePolicy<T>;
    using AccT = typename Pol::AccT;

    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < output_numel; i += step) {
        int64_t linear = i;
        int64_t off_y = 0;
        int64_t off_x_base = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = output_shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_y += coord * stride_y.v[d];
            if (d != reduce_axis) {
                off_x_base += coord * stride_x.v[d];
            }
        }

        AccT acc = (OP == IntReduceOp::Sum) ? Pol::sum_identity() : Pol::prod_identity();
        for (int32_t k = 0; k < reduce_extent; ++k) {
            int64_t off_x = off_x_base + (int64_t)k * reduce_stride_x;
            AccT v = Pol::load(x[off_x]);
            if (OP == IntReduceOp::Sum) {
                acc = acc + v;
            } else {
                acc = acc * v;
            }
        }
        y[off_y] = Pol::narrow(acc);
    }
}

template <typename T, IntReduceOp OP>
__host__ inline int32_t launch_reduce_int_axis(
    const T* x, T* y,
    int64_t output_numel,
    int32_t rank,
    const int32_t* output_shape_host,
    const int64_t* stride_x_host,
    const int64_t* stride_y_host,
    int32_t reduce_axis,
    int32_t reduce_extent,
    int64_t reduce_stride_x,
    cudaStream_t stream)
{
    if (rank < 0 || rank > baracuda::elementwise::MAX_RANK) return 2;
    if (reduce_axis < 0 || reduce_axis >= rank) return 2;
    baracuda::elementwise::DimsI32 out_shape = {};
    baracuda::elementwise::DimsI64 sx = {}, sy = {};
    for (int i = 0; i < rank; ++i) {
        out_shape.v[i] = output_shape_host[i];
        sx.v[i]        = stride_x_host[i];
        sy.v[i]        = stride_y_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (output_numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    reduce_int_axis_kernel<T, OP><<<blocks, kBlock, 0, stream>>>(
        x, y, output_numel, rank, out_shape, sx, sy,
        reduce_axis, reduce_extent, reduce_stride_x);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

}} // namespace baracuda::reduce_int

// =============================================================================
// INSTANTIATE macros — emit one `extern "C"` launcher per (op, dtype).
// Same parameter shape as REDUCE_AXIS_INSTANTIATE.
// =============================================================================

#define BARACUDA_KERNELS_REDUCE_INT_SUM_INSTANTIATE(NAME, T)                                          \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                                 \
        int64_t output_numel,                                                                          \
        int32_t rank,                                                                                  \
        const int32_t* output_shape,                                                                  \
        const int64_t* stride_x,                                                                      \
        const int64_t* stride_y,                                                                      \
        int32_t reduce_axis,                                                                          \
        int32_t reduce_extent,                                                                        \
        int64_t reduce_stride_x,                                                                      \
        const void* x, void* y,                                                                       \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                              \
        void* stream_ptr)                                                                              \
    {                                                                                                  \
        if (output_numel < 0) return 2;                                                               \
        if (output_numel == 0) return 0;                                                              \
        if (x == nullptr || y == nullptr) return 2;                                                   \
        if (output_shape == nullptr || stride_x == nullptr || stride_y == nullptr) return 2;          \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                  \
        return baracuda::reduce_int::launch_reduce_int_axis<T, baracuda::reduce_int::IntReduceOp::Sum>(\
            static_cast<const T*>(x), static_cast<T*>(y), output_numel, rank,                         \
            output_shape, stride_x, stride_y,                                                         \
            reduce_axis, reduce_extent, reduce_stride_x, stream);                                     \
    }

#define BARACUDA_KERNELS_REDUCE_INT_PROD_INSTANTIATE(NAME, T)                                         \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                                 \
        int64_t output_numel,                                                                          \
        int32_t rank,                                                                                  \
        const int32_t* output_shape,                                                                  \
        const int64_t* stride_x,                                                                      \
        const int64_t* stride_y,                                                                      \
        int32_t reduce_axis,                                                                          \
        int32_t reduce_extent,                                                                        \
        int64_t reduce_stride_x,                                                                      \
        const void* x, void* y,                                                                       \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                              \
        void* stream_ptr)                                                                              \
    {                                                                                                  \
        if (output_numel < 0) return 2;                                                               \
        if (output_numel == 0) return 0;                                                              \
        if (x == nullptr || y == nullptr) return 2;                                                   \
        if (output_shape == nullptr || stride_x == nullptr || stride_y == nullptr) return 2;          \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                  \
        return baracuda::reduce_int::launch_reduce_int_axis<T, baracuda::reduce_int::IntReduceOp::Prod>(\
            static_cast<const T*>(x), static_cast<T*>(y), output_numel, rank,                         \
            output_shape, stride_x, stride_y,                                                         \
            reduce_axis, reduce_extent, reduce_stride_x, stream);                                     \
    }

#endif // BARACUDA_REDUCE_INT_CUH
