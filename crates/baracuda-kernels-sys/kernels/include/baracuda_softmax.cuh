// baracuda_softmax.cuh
//
// Templated kernels and INSTANTIATE macros for the softmax op family
// (Phase 5 Category H of the comprehensive plan).
//
// Length-preserving transform along a single axis. Output shape ==
// input shape (in contrast to reductions). Each kernel does two passes
// over the softmax axis per output cell — first to find the max
// (for numerical stability) and the sum-of-exp, then to compute the
// per-cell output. Cost: O(numel · extent), naive but correct.
// Future tuning can use shared-memory + warp-shuffle to amortize the
// per-row reduction across threads of a block.
//
// Status codes match the elementwise family:
//   0 success, 1 misaligned, 2 invalid problem, 3 unsupported,
//   4 workspace too small, 5 internal launch failure.

#ifndef BARACUDA_SOFTMAX_CUH
#define BARACUDA_SOFTMAX_CUH

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// CUB block primitives must be included at GLOBAL scope (they bring in
// `cuda::std::...` symbols which would otherwise nest into whatever
// namespace this header is included from). Used by the Phase 11.6
// block-cooperative sparsemax FW kernel below.
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_reduce.cuh>
#include <cuda/functional> // for cuda::maximum (replaces removed cub::Max)

#include "baracuda_elementwise.cuh" // for DimsI32 / DimsI64 / MAX_RANK

namespace baracuda { namespace softmax {

// =============================================================================
// Softmax FW kernel — `y[k] = exp(x[k] - max(x)) / Σ_j exp(x[j] - max(x))`
// =============================================================================
//
// Two-pass per-output-cell scheme: find the row's max + sum-of-exp on
// pass 1, store output on pass 2 (well — actually each thread handles
// one output cell at coord (..., k, ...), so the "pass 2" is just the
// thread's own output write; the two passes are the two for-loops over
// the softmax axis inside the thread).
//
// `T` is the scalar element type; the accumulator is `float` (f64 uses
// `double`) regardless of T to preserve precision through the exp / sum
// chain. The dtype specializations below pick the right accumulator.

template <typename T>
__device__ __forceinline__ float load_as_acc(T x) { return (float)x; }

template <>
__device__ __forceinline__ float load_as_acc<__half>(__half x) { return __half2float(x); }

template <>
__device__ __forceinline__ float load_as_acc<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <typename T>
__device__ __forceinline__ T store_from_acc(float v) { return (T)v; }

template <>
__device__ __forceinline__ __half store_from_acc<__half>(float v) {
    return __float2half(v);
}

template <>
__device__ __forceinline__ __nv_bfloat16 store_from_acc<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

// f64 path: keep the accumulator at double precision (otherwise f64
// softmax would silently downgrade through the f32 detour).
template <typename T>
__device__ __forceinline__ double load_as_acc_d(T x) { return (double)x; }

template <typename T>
__device__ __forceinline__ T store_from_acc_d(double v) { return (T)v; }

template <typename T>
__global__ void softmax_fp_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int64_t numel,
    int32_t rank,
    baracuda::elementwise::DimsI32 shape,
    baracuda::elementwise::DimsI64 stride_x,
    baracuda::elementwise::DimsI64 stride_y,
    int32_t softmax_axis,
    int32_t softmax_extent,
    int64_t softmax_stride_x,
    int64_t softmax_stride_y)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_y = 0;
        int64_t off_x_base = 0;
        int32_t k = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_y += coord * stride_y.v[d];
            if (d != softmax_axis) {
                off_x_base += coord * stride_x.v[d];
            } else {
                k = (int32_t)coord;
            }
        }
        // Pass 1a: find the row's max.
        float m = -INFINITY;
        for (int32_t j = 0; j < softmax_extent; ++j) {
            int64_t off = off_x_base + (int64_t)j * softmax_stride_x;
            float v = load_as_acc<T>(x[off]);
            if (v > m) m = v;
        }
        // Pass 1b: sum of exp.
        float s = 0.0f;
        for (int32_t j = 0; j < softmax_extent; ++j) {
            int64_t off = off_x_base + (int64_t)j * softmax_stride_x;
            s += expf(load_as_acc<T>(x[off]) - m);
        }
        // Pass 2: compute this cell's output.
        int64_t off_xk = off_x_base + (int64_t)k * softmax_stride_x;
        float yk = expf(load_as_acc<T>(x[off_xk]) - m) / s;
        y[off_y] = store_from_acc<T>(yk);
    }
}

template <>
__global__ void softmax_fp_kernel<double>(
    const double* __restrict__ x,
    double* __restrict__ y,
    int64_t numel,
    int32_t rank,
    baracuda::elementwise::DimsI32 shape,
    baracuda::elementwise::DimsI64 stride_x,
    baracuda::elementwise::DimsI64 stride_y,
    int32_t softmax_axis,
    int32_t softmax_extent,
    int64_t softmax_stride_x,
    int64_t softmax_stride_y)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_y = 0;
        int64_t off_x_base = 0;
        int32_t k = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_y += coord * stride_y.v[d];
            if (d != softmax_axis) {
                off_x_base += coord * stride_x.v[d];
            } else {
                k = (int32_t)coord;
            }
        }
        double m = -INFINITY;
        for (int32_t j = 0; j < softmax_extent; ++j) {
            int64_t off = off_x_base + (int64_t)j * softmax_stride_x;
            double v = x[off];
            if (v > m) m = v;
        }
        double s = 0.0;
        for (int32_t j = 0; j < softmax_extent; ++j) {
            int64_t off = off_x_base + (int64_t)j * softmax_stride_x;
            s += exp(x[off] - m);
        }
        int64_t off_xk = off_x_base + (int64_t)k * softmax_stride_x;
        y[off_y] = exp(x[off_xk] - m) / s;
    }
}

template <typename T>
__host__ inline int32_t launch_softmax_fp(
    const T* x, T* y,
    int64_t numel,
    int32_t rank,
    const int32_t* shape_host,
    const int64_t* stride_x_host,
    const int64_t* stride_y_host,
    int32_t softmax_axis,
    int32_t softmax_extent,
    int64_t softmax_stride_x,
    int64_t softmax_stride_y,
    cudaStream_t stream)
{
    using baracuda::elementwise::MAX_RANK;
    using baracuda::elementwise::DimsI32;
    using baracuda::elementwise::DimsI64;
    if (rank < 0 || rank > MAX_RANK) return 2;
    if (softmax_axis < 0 || softmax_axis >= rank) return 2;
    DimsI32 shape = {};
    DimsI64 sx = {}, sy = {};
    for (int i = 0; i < rank; ++i) {
        shape.v[i] = shape_host[i];
        sx.v[i]    = stride_x_host[i];
        sy.v[i]    = stride_y_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    softmax_fp_kernel<T><<<blocks, kBlock, 0, stream>>>(
        x, y, numel, rank, shape, sx, sy,
        softmax_axis, softmax_extent, softmax_stride_x, softmax_stride_y);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Softmax BW kernel — `dx[k] = y[k] * (dy[k] - Σ_j y[j] · dy[j])`
// =============================================================================
//
// Needs the saved forward output `y`. The per-row "dot product"
// `Σ_j y[j] · dy[j]` is recomputed inline by each thread (single
// pass over the softmax axis), then the per-cell BW formula is
// applied.

template <typename T>
__global__ void softmax_backward_fp_kernel(
    const T* __restrict__ dy,
    const T* __restrict__ y,
    T* __restrict__ dx,
    int64_t numel,
    int32_t rank,
    baracuda::elementwise::DimsI32 shape,
    baracuda::elementwise::DimsI64 stride_dy,
    baracuda::elementwise::DimsI64 stride_y,
    baracuda::elementwise::DimsI64 stride_dx,
    int32_t softmax_axis,
    int32_t softmax_extent,
    int64_t softmax_stride_dy,
    int64_t softmax_stride_y)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_dy = 0, off_y = 0, off_dx = 0;
        int64_t off_dy_base = 0, off_y_base = 0;
        int32_t k = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_dx += coord * stride_dx.v[d];
            if (d != softmax_axis) {
                off_dy_base += coord * stride_dy.v[d];
                off_y_base  += coord * stride_y.v[d];
            } else {
                k = (int32_t)coord;
            }
        }
        // Compute dot = Σ_j y[j] · dy[j] along the softmax axis.
        float dot = 0.0f;
        for (int32_t j = 0; j < softmax_extent; ++j) {
            int64_t off_yj  = off_y_base  + (int64_t)j * softmax_stride_y;
            int64_t off_dyj = off_dy_base + (int64_t)j * softmax_stride_dy;
            dot += load_as_acc<T>(y[off_yj]) * load_as_acc<T>(dy[off_dyj]);
        }
        // Per-cell BW.
        off_dy = off_dy_base + (int64_t)k * softmax_stride_dy;
        off_y  = off_y_base  + (int64_t)k * softmax_stride_y;
        float yk  = load_as_acc<T>(y[off_y]);
        float dyk = load_as_acc<T>(dy[off_dy]);
        dx[off_dx] = store_from_acc<T>(yk * (dyk - dot));
    }
}

template <>
__global__ void softmax_backward_fp_kernel<double>(
    const double* __restrict__ dy,
    const double* __restrict__ y,
    double* __restrict__ dx,
    int64_t numel,
    int32_t rank,
    baracuda::elementwise::DimsI32 shape,
    baracuda::elementwise::DimsI64 stride_dy,
    baracuda::elementwise::DimsI64 stride_y,
    baracuda::elementwise::DimsI64 stride_dx,
    int32_t softmax_axis,
    int32_t softmax_extent,
    int64_t softmax_stride_dy,
    int64_t softmax_stride_y)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_dy = 0, off_y = 0, off_dx = 0;
        int64_t off_dy_base = 0, off_y_base = 0;
        int32_t k = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_dx += coord * stride_dx.v[d];
            if (d != softmax_axis) {
                off_dy_base += coord * stride_dy.v[d];
                off_y_base  += coord * stride_y.v[d];
            } else {
                k = (int32_t)coord;
            }
        }
        double dot = 0.0;
        for (int32_t j = 0; j < softmax_extent; ++j) {
            int64_t off_yj  = off_y_base  + (int64_t)j * softmax_stride_y;
            int64_t off_dyj = off_dy_base + (int64_t)j * softmax_stride_dy;
            dot += y[off_yj] * dy[off_dyj];
        }
        off_dy = off_dy_base + (int64_t)k * softmax_stride_dy;
        off_y  = off_y_base  + (int64_t)k * softmax_stride_y;
        dx[off_dx] = y[off_y] * (dy[off_dy] - dot);
    }
}

template <typename T>
__host__ inline int32_t launch_softmax_backward_fp(
    const T* dy, const T* y, T* dx,
    int64_t numel,
    int32_t rank,
    const int32_t* shape_host,
    const int64_t* stride_dy_host,
    const int64_t* stride_y_host,
    const int64_t* stride_dx_host,
    int32_t softmax_axis,
    int32_t softmax_extent,
    int64_t softmax_stride_dy,
    int64_t softmax_stride_y,
    cudaStream_t stream)
{
    using baracuda::elementwise::MAX_RANK;
    using baracuda::elementwise::DimsI32;
    using baracuda::elementwise::DimsI64;
    if (rank < 0 || rank > MAX_RANK) return 2;
    if (softmax_axis < 0 || softmax_axis >= rank) return 2;
    DimsI32 shape = {};
    DimsI64 sdy = {}, sy = {}, sdx = {};
    for (int i = 0; i < rank; ++i) {
        shape.v[i] = shape_host[i];
        sdy.v[i]   = stride_dy_host[i];
        sy.v[i]    = stride_y_host[i];
        sdx.v[i]   = stride_dx_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    softmax_backward_fp_kernel<T><<<blocks, kBlock, 0, stream>>>(
        dy, y, dx, numel, rank, shape, sdy, sy, sdx,
        softmax_axis, softmax_extent, softmax_stride_dy, softmax_stride_y);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// LogSoftmax FW kernel — `y[k] = (x[k] - max(x)) - log(Σ_j exp(x[j] - max(x)))`
// =============================================================================
//
// Same two-pass structure as `softmax_fp_kernel` (find row max, then sum
// of exp), but the per-cell output formula is the log-domain version:
// `y[k] = (x[k] - max) - log(sum_exp)`. Output values are log-probs, so
// they live in `[-INF, 0]` and `exp(y[k])` recovers `softmax(x)[k]`. The
// f64 specialization mirrors softmax_fp_kernel<double>.

template <typename T>
__global__ void log_softmax_fp_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int64_t numel,
    int32_t rank,
    baracuda::elementwise::DimsI32 shape,
    baracuda::elementwise::DimsI64 stride_x,
    baracuda::elementwise::DimsI64 stride_y,
    int32_t softmax_axis,
    int32_t softmax_extent,
    int64_t softmax_stride_x,
    int64_t softmax_stride_y)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_y = 0;
        int64_t off_x_base = 0;
        int32_t k = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_y += coord * stride_y.v[d];
            if (d != softmax_axis) {
                off_x_base += coord * stride_x.v[d];
            } else {
                k = (int32_t)coord;
            }
        }
        // Pass 1a: find the row's max.
        float m = -INFINITY;
        for (int32_t j = 0; j < softmax_extent; ++j) {
            int64_t off = off_x_base + (int64_t)j * softmax_stride_x;
            float v = load_as_acc<T>(x[off]);
            if (v > m) m = v;
        }
        // Pass 1b: sum of exp.
        float s = 0.0f;
        for (int32_t j = 0; j < softmax_extent; ++j) {
            int64_t off = off_x_base + (int64_t)j * softmax_stride_x;
            s += expf(load_as_acc<T>(x[off]) - m);
        }
        // Pass 2: `y[k] = (x[k] - m) - log(s)`.
        int64_t off_xk = off_x_base + (int64_t)k * softmax_stride_x;
        float yk = (load_as_acc<T>(x[off_xk]) - m) - logf(s);
        y[off_y] = store_from_acc<T>(yk);
    }
}

template <>
__global__ void log_softmax_fp_kernel<double>(
    const double* __restrict__ x,
    double* __restrict__ y,
    int64_t numel,
    int32_t rank,
    baracuda::elementwise::DimsI32 shape,
    baracuda::elementwise::DimsI64 stride_x,
    baracuda::elementwise::DimsI64 stride_y,
    int32_t softmax_axis,
    int32_t softmax_extent,
    int64_t softmax_stride_x,
    int64_t softmax_stride_y)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_y = 0;
        int64_t off_x_base = 0;
        int32_t k = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_y += coord * stride_y.v[d];
            if (d != softmax_axis) {
                off_x_base += coord * stride_x.v[d];
            } else {
                k = (int32_t)coord;
            }
        }
        double m = -INFINITY;
        for (int32_t j = 0; j < softmax_extent; ++j) {
            int64_t off = off_x_base + (int64_t)j * softmax_stride_x;
            double v = x[off];
            if (v > m) m = v;
        }
        double s = 0.0;
        for (int32_t j = 0; j < softmax_extent; ++j) {
            int64_t off = off_x_base + (int64_t)j * softmax_stride_x;
            s += exp(x[off] - m);
        }
        int64_t off_xk = off_x_base + (int64_t)k * softmax_stride_x;
        y[off_y] = (x[off_xk] - m) - log(s);
    }
}

template <typename T>
__host__ inline int32_t launch_log_softmax_fp(
    const T* x, T* y,
    int64_t numel,
    int32_t rank,
    const int32_t* shape_host,
    const int64_t* stride_x_host,
    const int64_t* stride_y_host,
    int32_t softmax_axis,
    int32_t softmax_extent,
    int64_t softmax_stride_x,
    int64_t softmax_stride_y,
    cudaStream_t stream)
{
    using baracuda::elementwise::MAX_RANK;
    using baracuda::elementwise::DimsI32;
    using baracuda::elementwise::DimsI64;
    if (rank < 0 || rank > MAX_RANK) return 2;
    if (softmax_axis < 0 || softmax_axis >= rank) return 2;
    DimsI32 shape = {};
    DimsI64 sx = {}, sy = {};
    for (int i = 0; i < rank; ++i) {
        shape.v[i] = shape_host[i];
        sx.v[i]    = stride_x_host[i];
        sy.v[i]    = stride_y_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    log_softmax_fp_kernel<T><<<blocks, kBlock, 0, stream>>>(
        x, y, numel, rank, shape, sx, sy,
        softmax_axis, softmax_extent, softmax_stride_x, softmax_stride_y);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// LogSoftmax BW kernel — `dx[k] = dy[k] - exp(y[k]) · Σ_j dy[j]`
// =============================================================================
//
// `y` is the SAVED forward output (log-softmax values, all in
// `[-INF, 0]`). Since `y = log(softmax(x))`, `exp(y)` recovers
// `softmax(x)`. Per-thread: walk the softmax axis once to compute the
// row-sum `dy_sum = Σ_j dy[j]`, then apply the per-cell formula.

template <typename T>
__global__ void log_softmax_backward_fp_kernel(
    const T* __restrict__ dy,
    const T* __restrict__ y,
    T* __restrict__ dx,
    int64_t numel,
    int32_t rank,
    baracuda::elementwise::DimsI32 shape,
    baracuda::elementwise::DimsI64 stride_dy,
    baracuda::elementwise::DimsI64 stride_y,
    baracuda::elementwise::DimsI64 stride_dx,
    int32_t softmax_axis,
    int32_t softmax_extent,
    int64_t softmax_stride_dy,
    int64_t softmax_stride_y)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_dy = 0, off_y = 0, off_dx = 0;
        int64_t off_dy_base = 0, off_y_base = 0;
        int32_t k = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_dx += coord * stride_dx.v[d];
            if (d != softmax_axis) {
                off_dy_base += coord * stride_dy.v[d];
                off_y_base  += coord * stride_y.v[d];
            } else {
                k = (int32_t)coord;
            }
        }
        // Compute dy_sum = Σ_j dy[j] along the softmax axis.
        float dy_sum = 0.0f;
        for (int32_t j = 0; j < softmax_extent; ++j) {
            int64_t off_dyj = off_dy_base + (int64_t)j * softmax_stride_dy;
            dy_sum += load_as_acc<T>(dy[off_dyj]);
        }
        // Per-cell BW: `dx[k] = dy[k] - exp(y[k]) * dy_sum`.
        off_dy = off_dy_base + (int64_t)k * softmax_stride_dy;
        off_y  = off_y_base  + (int64_t)k * softmax_stride_y;
        float yk  = load_as_acc<T>(y[off_y]);
        float dyk = load_as_acc<T>(dy[off_dy]);
        dx[off_dx] = store_from_acc<T>(dyk - expf(yk) * dy_sum);
    }
}

template <>
__global__ void log_softmax_backward_fp_kernel<double>(
    const double* __restrict__ dy,
    const double* __restrict__ y,
    double* __restrict__ dx,
    int64_t numel,
    int32_t rank,
    baracuda::elementwise::DimsI32 shape,
    baracuda::elementwise::DimsI64 stride_dy,
    baracuda::elementwise::DimsI64 stride_y,
    baracuda::elementwise::DimsI64 stride_dx,
    int32_t softmax_axis,
    int32_t softmax_extent,
    int64_t softmax_stride_dy,
    int64_t softmax_stride_y)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_dy = 0, off_y = 0, off_dx = 0;
        int64_t off_dy_base = 0, off_y_base = 0;
        int32_t k = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_dx += coord * stride_dx.v[d];
            if (d != softmax_axis) {
                off_dy_base += coord * stride_dy.v[d];
                off_y_base  += coord * stride_y.v[d];
            } else {
                k = (int32_t)coord;
            }
        }
        double dy_sum = 0.0;
        for (int32_t j = 0; j < softmax_extent; ++j) {
            int64_t off_dyj = off_dy_base + (int64_t)j * softmax_stride_dy;
            dy_sum += dy[off_dyj];
        }
        off_dy = off_dy_base + (int64_t)k * softmax_stride_dy;
        off_y  = off_y_base  + (int64_t)k * softmax_stride_y;
        dx[off_dx] = dy[off_dy] - exp(y[off_y]) * dy_sum;
    }
}

template <typename T>
__host__ inline int32_t launch_log_softmax_backward_fp(
    const T* dy, const T* y, T* dx,
    int64_t numel,
    int32_t rank,
    const int32_t* shape_host,
    const int64_t* stride_dy_host,
    const int64_t* stride_y_host,
    const int64_t* stride_dx_host,
    int32_t softmax_axis,
    int32_t softmax_extent,
    int64_t softmax_stride_dy,
    int64_t softmax_stride_y,
    cudaStream_t stream)
{
    using baracuda::elementwise::MAX_RANK;
    using baracuda::elementwise::DimsI32;
    using baracuda::elementwise::DimsI64;
    if (rank < 0 || rank > MAX_RANK) return 2;
    if (softmax_axis < 0 || softmax_axis >= rank) return 2;
    DimsI32 shape = {};
    DimsI64 sdy = {}, sy = {}, sdx = {};
    for (int i = 0; i < rank; ++i) {
        shape.v[i] = shape_host[i];
        sdy.v[i]   = stride_dy_host[i];
        sy.v[i]    = stride_y_host[i];
        sdx.v[i]   = stride_dx_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    log_softmax_backward_fp_kernel<T><<<blocks, kBlock, 0, stream>>>(
        dy, y, dx, numel, rank, shape, sdy, sy, sdx,
        softmax_axis, softmax_extent, softmax_stride_dy, softmax_stride_y);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// GumbelSoftmax FW kernel — `y = softmax((x + g) / τ)` where `g[i] = -log(-log(u[i]))`
// =============================================================================
//
// Reads a caller-supplied uniform-rand `float` buffer (`u` in [0, 1])
// generated by cuRAND. Adds Gumbel noise via the standard inverse-CDF
// transform `g = -log(-log(u))`, scales by `1/τ`, and applies the same
// two-pass softmax as `softmax_fp_kernel`. When `hard` is non-zero, the
// kernel additionally computes the row's argmax and emits a one-hot
// hardened output (the straight-through gradient lives in autograd —
// the saved `y_soft` is the post-noise softmax, so BW just calls the
// existing softmax_backward_fp_kernel).
//
// The uniform buffer is laid out flat (one f32 per output cell), indexed
// by the same coord walk as `x`. We add a tiny epsilon to `u` before
// the inner log to avoid `log(0) = -inf` poisoning the noise.

template <typename T>
__global__ void gumbel_softmax_fp_kernel(
    const T* __restrict__ x,
    const float* __restrict__ u_rand,
    T* __restrict__ y,
    int64_t numel,
    int32_t rank,
    baracuda::elementwise::DimsI32 shape,
    baracuda::elementwise::DimsI64 stride_x,
    baracuda::elementwise::DimsI64 stride_y,
    int32_t softmax_axis,
    int32_t softmax_extent,
    int64_t softmax_stride_x,
    int64_t softmax_stride_y,
    float inv_tau,
    int32_t hard)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_y = 0;
        int64_t off_x_base = 0;
        // The uniform buffer is in "y" order (flat, dense, same layout
        // as the output write). We need the base offset for the row's
        // start of `u`, which we compute by walking coord just like the
        // softmax does. The simplest approach: index `u` by the same
        // dense `i` as the output, and reconstruct the row base from
        // `i - k`. Since `u` is contiguous and `softmax_axis` may not
        // be the innermost dim, we rebuild the row's k=0 dense index
        // by replacing `coord[axis]` with 0 in the dense walk.
        int64_t off_u_base = 0;
        int32_t k = 0;
        // Walk coords. dense layout for `u_rand`: shape is `shape`, strides
        // are contiguous (last-axis-innermost — same as the input shape
        // in row-major).
        int64_t u_row_stride = 1;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_y += coord * stride_y.v[d];
            if (d != softmax_axis) {
                off_x_base += coord * stride_x.v[d];
                off_u_base += coord * u_row_stride;
            } else {
                k = (int32_t)coord;
            }
            u_row_stride *= (int64_t)((s == 0) ? 1 : s);
        }
        // u_row_stride for the softmax axis = product of dims to the right
        // of `axis`. Compute it explicitly so we can index `u` along
        // the softmax axis.
        int64_t u_axis_stride = 1;
        for (int d = rank - 1; d > softmax_axis; --d) {
            int32_t s = shape.v[d];
            u_axis_stride *= (int64_t)((s == 0) ? 1 : s);
        }
        // Pass 1a: find the row's max of (x + g) / τ.
        float m = -INFINITY;
        for (int32_t j = 0; j < softmax_extent; ++j) {
            int64_t off = off_x_base + (int64_t)j * softmax_stride_x;
            int64_t off_u = off_u_base + (int64_t)j * u_axis_stride;
            float xv = load_as_acc<T>(x[off]);
            float uv = u_rand[off_u];
            // Clamp uv into (eps, 1) so log(-log(uv)) is finite.
            const float eps = 1e-20f;
            if (uv < eps) uv = eps;
            if (uv > 1.0f - eps) uv = 1.0f - eps;
            float gv = -logf(-logf(uv));
            float v = (xv + gv) * inv_tau;
            if (v > m) m = v;
        }
        // Pass 1b: sum of exp.
        float s = 0.0f;
        for (int32_t j = 0; j < softmax_extent; ++j) {
            int64_t off = off_x_base + (int64_t)j * softmax_stride_x;
            int64_t off_u = off_u_base + (int64_t)j * u_axis_stride;
            float xv = load_as_acc<T>(x[off]);
            float uv = u_rand[off_u];
            const float eps = 1e-20f;
            if (uv < eps) uv = eps;
            if (uv > 1.0f - eps) uv = 1.0f - eps;
            float gv = -logf(-logf(uv));
            float v = (xv + gv) * inv_tau;
            s += expf(v - m);
        }
        // Pass 2: this cell's soft output.
        int64_t off_xk = off_x_base + (int64_t)k * softmax_stride_x;
        int64_t off_uk = off_u_base + (int64_t)k * u_axis_stride;
        float xv_k = load_as_acc<T>(x[off_xk]);
        float uv_k = u_rand[off_uk];
        const float eps = 1e-20f;
        if (uv_k < eps) uv_k = eps;
        if (uv_k > 1.0f - eps) uv_k = 1.0f - eps;
        float gv_k = -logf(-logf(uv_k));
        float vk = (xv_k + gv_k) * inv_tau;
        float yk_soft = expf(vk - m) / s;

        if (hard) {
            // Find argmax along the row (same noise applied per-cell).
            // We need a third pass to determine which `j` has the largest
            // post-noise logit. Could be folded into pass 1 by tracking
            // argmax there — fold here for clarity.
            int32_t argmax = 0;
            float best = -INFINITY;
            for (int32_t j = 0; j < softmax_extent; ++j) {
                int64_t off = off_x_base + (int64_t)j * softmax_stride_x;
                int64_t off_u = off_u_base + (int64_t)j * u_axis_stride;
                float xv = load_as_acc<T>(x[off]);
                float uv = u_rand[off_u];
                if (uv < eps) uv = eps;
                if (uv > 1.0f - eps) uv = 1.0f - eps;
                float gv = -logf(-logf(uv));
                float v = (xv + gv) * inv_tau;
                if (v > best) { best = v; argmax = j; }
            }
            // One-hot output at the argmax slot, zero elsewhere.
            float yk_hard = (k == argmax) ? 1.0f : 0.0f;
            y[off_y] = store_from_acc<T>(yk_hard);
        } else {
            y[off_y] = store_from_acc<T>(yk_soft);
        }
    }
}

// f64 specialization mirrors softmax_fp_kernel<double>, using double
// throughout. cuRAND uniform is still f32 (we cast on use).
template <>
__global__ void gumbel_softmax_fp_kernel<double>(
    const double* __restrict__ x,
    const float* __restrict__ u_rand,
    double* __restrict__ y,
    int64_t numel,
    int32_t rank,
    baracuda::elementwise::DimsI32 shape,
    baracuda::elementwise::DimsI64 stride_x,
    baracuda::elementwise::DimsI64 stride_y,
    int32_t softmax_axis,
    int32_t softmax_extent,
    int64_t softmax_stride_x,
    int64_t softmax_stride_y,
    float inv_tau,
    int32_t hard)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_y = 0;
        int64_t off_x_base = 0;
        int64_t off_u_base = 0;
        int32_t k = 0;
        int64_t u_row_stride = 1;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_y += coord * stride_y.v[d];
            if (d != softmax_axis) {
                off_x_base += coord * stride_x.v[d];
                off_u_base += coord * u_row_stride;
            } else {
                k = (int32_t)coord;
            }
            u_row_stride *= (int64_t)((s == 0) ? 1 : s);
        }
        int64_t u_axis_stride = 1;
        for (int d = rank - 1; d > softmax_axis; --d) {
            int32_t s = shape.v[d];
            u_axis_stride *= (int64_t)((s == 0) ? 1 : s);
        }
        double inv_tau_d = (double)inv_tau;
        double m = -INFINITY;
        for (int32_t j = 0; j < softmax_extent; ++j) {
            int64_t off = off_x_base + (int64_t)j * softmax_stride_x;
            int64_t off_u = off_u_base + (int64_t)j * u_axis_stride;
            double xv = x[off];
            double uv = (double)u_rand[off_u];
            const double eps = 1e-30;
            if (uv < eps) uv = eps;
            if (uv > 1.0 - eps) uv = 1.0 - eps;
            double gv = -log(-log(uv));
            double v = (xv + gv) * inv_tau_d;
            if (v > m) m = v;
        }
        double s = 0.0;
        for (int32_t j = 0; j < softmax_extent; ++j) {
            int64_t off = off_x_base + (int64_t)j * softmax_stride_x;
            int64_t off_u = off_u_base + (int64_t)j * u_axis_stride;
            double xv = x[off];
            double uv = (double)u_rand[off_u];
            const double eps = 1e-30;
            if (uv < eps) uv = eps;
            if (uv > 1.0 - eps) uv = 1.0 - eps;
            double gv = -log(-log(uv));
            double v = (xv + gv) * inv_tau_d;
            s += exp(v - m);
        }
        int64_t off_xk = off_x_base + (int64_t)k * softmax_stride_x;
        int64_t off_uk = off_u_base + (int64_t)k * u_axis_stride;
        double xv_k = x[off_xk];
        double uv_k = (double)u_rand[off_uk];
        const double eps = 1e-30;
        if (uv_k < eps) uv_k = eps;
        if (uv_k > 1.0 - eps) uv_k = 1.0 - eps;
        double gv_k = -log(-log(uv_k));
        double vk = (xv_k + gv_k) * inv_tau_d;
        double yk_soft = exp(vk - m) / s;
        if (hard) {
            int32_t argmax = 0;
            double best = -INFINITY;
            for (int32_t j = 0; j < softmax_extent; ++j) {
                int64_t off = off_x_base + (int64_t)j * softmax_stride_x;
                int64_t off_u = off_u_base + (int64_t)j * u_axis_stride;
                double xv = x[off];
                double uv = (double)u_rand[off_u];
                if (uv < eps) uv = eps;
                if (uv > 1.0 - eps) uv = 1.0 - eps;
                double gv = -log(-log(uv));
                double v = (xv + gv) * inv_tau_d;
                if (v > best) { best = v; argmax = j; }
            }
            y[off_y] = (k == argmax) ? 1.0 : 0.0;
        } else {
            y[off_y] = yk_soft;
        }
    }
}

template <typename T>
__host__ inline int32_t launch_gumbel_softmax_fp(
    const T* x,
    const float* u_rand,
    T* y,
    int64_t numel,
    int32_t rank,
    const int32_t* shape_host,
    const int64_t* stride_x_host,
    const int64_t* stride_y_host,
    int32_t softmax_axis,
    int32_t softmax_extent,
    int64_t softmax_stride_x,
    int64_t softmax_stride_y,
    float inv_tau,
    int32_t hard,
    cudaStream_t stream)
{
    using baracuda::elementwise::MAX_RANK;
    using baracuda::elementwise::DimsI32;
    using baracuda::elementwise::DimsI64;
    if (rank < 0 || rank > MAX_RANK) return 2;
    if (softmax_axis < 0 || softmax_axis >= rank) return 2;
    if (!(inv_tau > 0.0f) || !isfinite(inv_tau)) return 2;
    DimsI32 shape = {};
    DimsI64 sx = {}, sy = {};
    for (int i = 0; i < rank; ++i) {
        shape.v[i] = shape_host[i];
        sx.v[i]    = stride_x_host[i];
        sy.v[i]    = stride_y_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    gumbel_softmax_fp_kernel<T><<<blocks, kBlock, 0, stream>>>(
        x, u_rand, y, numel, rank, shape, sx, sy,
        softmax_axis, softmax_extent, softmax_stride_x, softmax_stride_y,
        inv_tau, hard);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Sparsemax FW kernel — `y = ProjSimplex(x)` via threshold τ
// =============================================================================
//
// Closed-form projection onto the probability simplex:
//   1. Sort row of K logits in descending order.
//   2. Find largest k such that `1 + (k+1)*x_sorted[k] > Σ_{i ≤ k} x_sorted[i]`.
//   3. `τ = (Σ_{i ≤ k_max} x_sorted[i] - 1) / (k_max + 1)`.
//   4. `y[i] = max(0, x[i] - τ)`.
//
// One thread per output cell. The sort runs in per-thread local memory
// (insertion sort, O(extent²)). Limit: `softmax_extent <= 64` —
// trailblazer constraint, documented at the safe layer.

// Phase 11.6 (Fuel #10): replace the per-thread serial sort with a
// block-cooperative `cub::BlockRadixSort` + `cub::BlockScan` + per-tile
// emit. One thread block per row; threads collectively sort the row
// into descending order, scan its prefix sum, find the threshold index
// K*, and then emit `max(0, x - τ)` for the tile of cells each thread
// owns. Lifts the practical row-extent cap from 64 to 1024.
//
// Two block-cooperative specializations are compiled:
//   - ITEMS_PER_THREAD = 1 → handles rows of size 1..=256.
//   - ITEMS_PER_THREAD = 4 → handles rows of size 257..=1024.
// Both use BLOCK_THREADS = 256. Unused tail slots are padded with
// `-FLT_MAX` (or `-DBL_MAX`) so they sink to the bottom of the
// descending sort and are ignored by the threshold scan.
//
// CUB ships with the CUDA toolkit; no extra dep is required and the
// headers resolve on nvcc's default include path. (Headers are
// `#include`d at global scope near the top of this file.)

#ifndef BARACUDA_SPARSEMAX_MAX_EXTENT
#define BARACUDA_SPARSEMAX_MAX_EXTENT 1024
#endif

#ifndef BARACUDA_SPARSEMAX_BLOCK_THREADS
#define BARACUDA_SPARSEMAX_BLOCK_THREADS 256
#endif

// `f64` needs a different store path: store_from_acc<double>(float)
// would lose precision. Provide an Acc-aware store helper.
template <typename T, typename Acc>
__device__ __forceinline__ T sparsemax_store(Acc v) {
    return store_from_acc<T>((float)v);
}
template <>
__device__ __forceinline__ double sparsemax_store<double, double>(double v) {
    return v;
}

// Block-cooperative sparsemax FW kernel. The accumulator type `Acc` is
// `float` for {f32, f16, bf16} and `double` for `f64`. The per-block
// tile size is `BLOCK_THREADS * ITEMS_PER_THREAD` and must be
// `>= softmax_extent` (the launcher dispatches the smallest spec that
// fits). One thread block per row.
template <typename T, typename Acc, int BLOCK_THREADS, int ITEMS_PER_THREAD, bool IsF64>
__global__ void sparsemax_block_kernel_v2(
    const T* __restrict__ x,
    T* __restrict__ y,
    int64_t num_rows,
    int32_t rank,
    baracuda::elementwise::DimsI32 shape,
    baracuda::elementwise::DimsI64 stride_x,
    baracuda::elementwise::DimsI64 stride_y,
    int32_t softmax_axis,
    int32_t softmax_extent,
    int64_t softmax_stride_x,
    int64_t softmax_stride_y,
    int64_t outer_size,
    int64_t inner_size)
{
    using BlockRadixSortT = cub::BlockRadixSort<Acc, BLOCK_THREADS, ITEMS_PER_THREAD>;
    using BlockScanT      = cub::BlockScan<Acc, BLOCK_THREADS>;
    using BlockReduceT    = cub::BlockReduce<int, BLOCK_THREADS>;

    __shared__ union {
        typename BlockRadixSortT::TempStorage sort;
        typename BlockScanT::TempStorage      scan;
        typename BlockReduceT::TempStorage    reduce;
    } temp_storage;
    constexpr int kTile = BLOCK_THREADS * ITEMS_PER_THREAD;
    __shared__ Acc cum_smem[kTile];
    __shared__ Acc tau_smem;

    int64_t row_id = (int64_t)blockIdx.x;
    if (row_id >= num_rows) return;

    int64_t off_x_base = 0;
    int64_t off_y_base = 0;
    {
        int64_t inner_id = (inner_size > 0) ? (row_id % inner_size) : 0;
        int64_t outer_id = (inner_size > 0) ? (row_id / inner_size) : row_id;
        int64_t lin_inner = inner_id;
        for (int d = rank - 1; d > softmax_axis; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (lin_inner % (int64_t)s);
            if (s != 0) lin_inner /= (int64_t)s;
            off_x_base += coord * stride_x.v[d];
            off_y_base += coord * stride_y.v[d];
        }
        int64_t lin_outer = outer_id;
        for (int d = softmax_axis - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (lin_outer % (int64_t)s);
            if (s != 0) lin_outer /= (int64_t)s;
            off_x_base += coord * stride_x.v[d];
            off_y_base += coord * stride_y.v[d];
        }
    }

    // CUB's BlockRadixSort with descending sort uses the radix encoding
    // of the key type. The "-inf" sentinel must be the smallest value
    // representable in `Acc` so the padded slots sort to the bottom.
    const Acc kNegInf = IsF64
        ? (Acc)(-1.0e308) // close enough to -DBL_MAX for sparsemax (real inputs are O(1))
        : (Acc)(-3.3e38); // close enough to -FLT_MAX

    Acc items[ITEMS_PER_THREAD];
    #pragma unroll
    for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
        int idx = threadIdx.x + it * BLOCK_THREADS;
        if (idx < softmax_extent) {
            int64_t off = off_x_base + (int64_t)idx * softmax_stride_x;
            if constexpr (IsF64) {
                items[it] = (Acc)x[off];
            } else {
                items[it] = (Acc)load_as_acc<T>(x[off]);
            }
        } else {
            items[it] = kNegInf;
        }
    }
    __syncthreads();

    BlockRadixSortT(temp_storage.sort).SortDescending(items);
    __syncthreads();

    Acc cum_items[ITEMS_PER_THREAD];
    BlockScanT(temp_storage.scan).InclusiveSum(items, cum_items);
    __syncthreads();

    #pragma unroll
    for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
        int sorted_idx = threadIdx.x * ITEMS_PER_THREAD + it;
        if (sorted_idx < softmax_extent) {
            cum_smem[sorted_idx] = cum_items[it];
        }
    }
    __syncthreads();

    int my_k_max = 0;
    #pragma unroll
    for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
        int sorted_idx = threadIdx.x * ITEMS_PER_THREAD + it;
        if (sorted_idx < softmax_extent) {
            Acc sval = items[it];
            Acc cval = cum_items[it];
            if ((Acc)1 + (Acc)(sorted_idx + 1) * sval > cval) {
                int cand = sorted_idx + 1;
                if (cand > my_k_max) my_k_max = cand;
            }
        }
    }
    int k_max = BlockReduceT(temp_storage.reduce).Reduce(my_k_max, ::cuda::maximum<int>{});
    if (threadIdx.x == 0) {
        if (k_max > 0) {
            Acc sum_top = cum_smem[k_max - 1];
            tau_smem = (sum_top - (Acc)1) / (Acc)k_max;
        } else {
            tau_smem = (Acc)0;
        }
    }
    __syncthreads();
    Acc tau = tau_smem;

    #pragma unroll
    for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
        int idx = threadIdx.x + it * BLOCK_THREADS;
        if (idx < softmax_extent) {
            int64_t off_x = off_x_base + (int64_t)idx * softmax_stride_x;
            int64_t off_y = off_y_base + (int64_t)idx * softmax_stride_y;
            Acc xv;
            if constexpr (IsF64) {
                xv = (Acc)x[off_x];
            } else {
                xv = (Acc)load_as_acc<T>(x[off_x]);
            }
            Acc yv = xv - tau;
            if (yv < (Acc)0) yv = (Acc)0;
            y[off_y] = sparsemax_store<T, Acc>(yv);
        }
    }
}

template <typename T>
__host__ inline int32_t launch_sparsemax_fp(
    const T* x, T* y,
    int64_t numel,
    int32_t rank,
    const int32_t* shape_host,
    const int64_t* stride_x_host,
    const int64_t* stride_y_host,
    int32_t softmax_axis,
    int32_t softmax_extent,
    int64_t softmax_stride_x,
    int64_t softmax_stride_y,
    cudaStream_t stream)
{
    using baracuda::elementwise::MAX_RANK;
    using baracuda::elementwise::DimsI32;
    using baracuda::elementwise::DimsI64;
    if (rank < 0 || rank > MAX_RANK) return 2;
    if (softmax_axis < 0 || softmax_axis >= rank) return 2;
    if (softmax_extent < 0) return 2;
    if (softmax_extent > BARACUDA_SPARSEMAX_MAX_EXTENT) return 3;
    DimsI32 shape = {};
    DimsI64 sx = {}, sy = {};
    int64_t outer_size = 1;
    int64_t inner_size = 1;
    for (int i = 0; i < rank; ++i) {
        shape.v[i] = shape_host[i];
        sx.v[i]    = stride_x_host[i];
        sy.v[i]    = stride_y_host[i];
        if (i < softmax_axis) outer_size *= (int64_t)shape_host[i];
        if (i > softmax_axis) inner_size *= (int64_t)shape_host[i];
    }
    // Degenerate: extent of 0 along the axis ⇒ no rows.
    if (softmax_extent == 0) return 0;
    int64_t num_rows = outer_size * inner_size;
    if (num_rows <= 0) return 0;

    // Dispatch to the smallest tile-size specialization that fits.
    constexpr int kBlock = BARACUDA_SPARSEMAX_BLOCK_THREADS;
    using AccT = typename std::conditional<std::is_same<T, double>::value, double, float>::type;
    constexpr bool kIsF64 = std::is_same<T, double>::value;

    dim3 grid((unsigned)num_rows);
    dim3 block((unsigned)kBlock);
    if (softmax_extent <= kBlock) {
        sparsemax_block_kernel_v2<T, AccT, kBlock, 1, kIsF64><<<grid, block, 0, stream>>>(
            x, y, num_rows, rank, shape, sx, sy,
            softmax_axis, softmax_extent, softmax_stride_x, softmax_stride_y,
            outer_size, inner_size);
    } else if (softmax_extent <= kBlock * 4) {
        sparsemax_block_kernel_v2<T, AccT, kBlock, 4, kIsF64><<<grid, block, 0, stream>>>(
            x, y, num_rows, rank, shape, sx, sy,
            softmax_axis, softmax_extent, softmax_stride_x, softmax_stride_y,
            outer_size, inner_size);
    } else {
        return 3;
    }

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Sparsemax BW kernel — Jacobian-vector product
// =============================================================================
//
// For active positions (where `y > 0`):
//   `dx[i] = dy[i] - sum_dy_active / n_active`.
// For inactive positions, `dx[i] = 0`.
//
// Needs the saved forward output `y`. Each thread walks the softmax axis
// once to count actives + sum their dy, then writes its cell.

template <typename T>
__global__ void sparsemax_backward_fp_kernel(
    const T* __restrict__ dy,
    const T* __restrict__ y,
    T* __restrict__ dx,
    int64_t numel,
    int32_t rank,
    baracuda::elementwise::DimsI32 shape,
    baracuda::elementwise::DimsI64 stride_dy,
    baracuda::elementwise::DimsI64 stride_y,
    baracuda::elementwise::DimsI64 stride_dx,
    int32_t softmax_axis,
    int32_t softmax_extent,
    int64_t softmax_stride_dy,
    int64_t softmax_stride_y)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_dy = 0, off_y = 0, off_dx = 0;
        int64_t off_dy_base = 0, off_y_base = 0;
        int32_t k = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_dx += coord * stride_dx.v[d];
            if (d != softmax_axis) {
                off_dy_base += coord * stride_dy.v[d];
                off_y_base  += coord * stride_y.v[d];
            } else {
                k = (int32_t)coord;
            }
        }
        // Walk row, accumulating sum_dy_active and n_active.
        float sum_dy_active = 0.0f;
        int32_t n_active = 0;
        for (int32_t j = 0; j < softmax_extent; ++j) {
            int64_t off_yj  = off_y_base  + (int64_t)j * softmax_stride_y;
            int64_t off_dyj = off_dy_base + (int64_t)j * softmax_stride_dy;
            float yj = load_as_acc<T>(y[off_yj]);
            if (yj > 0.0f) {
                sum_dy_active += load_as_acc<T>(dy[off_dyj]);
                ++n_active;
            }
        }
        float avg = (n_active > 0) ? (sum_dy_active / (float)n_active) : 0.0f;
        off_dy = off_dy_base + (int64_t)k * softmax_stride_dy;
        off_y  = off_y_base  + (int64_t)k * softmax_stride_y;
        float yk  = load_as_acc<T>(y[off_y]);
        float dyk = load_as_acc<T>(dy[off_dy]);
        float dxk = (yk > 0.0f) ? (dyk - avg) : 0.0f;
        dx[off_dx] = store_from_acc<T>(dxk);
    }
}

template <>
__global__ void sparsemax_backward_fp_kernel<double>(
    const double* __restrict__ dy,
    const double* __restrict__ y,
    double* __restrict__ dx,
    int64_t numel,
    int32_t rank,
    baracuda::elementwise::DimsI32 shape,
    baracuda::elementwise::DimsI64 stride_dy,
    baracuda::elementwise::DimsI64 stride_y,
    baracuda::elementwise::DimsI64 stride_dx,
    int32_t softmax_axis,
    int32_t softmax_extent,
    int64_t softmax_stride_dy,
    int64_t softmax_stride_y)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t linear = i;
        int64_t off_dy = 0, off_y = 0, off_dx = 0;
        int64_t off_dy_base = 0, off_y_base = 0;
        int32_t k = 0;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t coord = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            off_dx += coord * stride_dx.v[d];
            if (d != softmax_axis) {
                off_dy_base += coord * stride_dy.v[d];
                off_y_base  += coord * stride_y.v[d];
            } else {
                k = (int32_t)coord;
            }
        }
        double sum_dy_active = 0.0;
        int32_t n_active = 0;
        for (int32_t j = 0; j < softmax_extent; ++j) {
            int64_t off_yj  = off_y_base  + (int64_t)j * softmax_stride_y;
            int64_t off_dyj = off_dy_base + (int64_t)j * softmax_stride_dy;
            double yj = y[off_yj];
            if (yj > 0.0) {
                sum_dy_active += dy[off_dyj];
                ++n_active;
            }
        }
        double avg = (n_active > 0) ? (sum_dy_active / (double)n_active) : 0.0;
        off_dy = off_dy_base + (int64_t)k * softmax_stride_dy;
        off_y  = off_y_base  + (int64_t)k * softmax_stride_y;
        double yk  = y[off_y];
        double dyk = dy[off_dy];
        dx[off_dx] = (yk > 0.0) ? (dyk - avg) : 0.0;
    }
}

template <typename T>
__host__ inline int32_t launch_sparsemax_backward_fp(
    const T* dy, const T* y, T* dx,
    int64_t numel,
    int32_t rank,
    const int32_t* shape_host,
    const int64_t* stride_dy_host,
    const int64_t* stride_y_host,
    const int64_t* stride_dx_host,
    int32_t softmax_axis,
    int32_t softmax_extent,
    int64_t softmax_stride_dy,
    int64_t softmax_stride_y,
    cudaStream_t stream)
{
    using baracuda::elementwise::MAX_RANK;
    using baracuda::elementwise::DimsI32;
    using baracuda::elementwise::DimsI64;
    if (rank < 0 || rank > MAX_RANK) return 2;
    if (softmax_axis < 0 || softmax_axis >= rank) return 2;
    DimsI32 shape = {};
    DimsI64 sdy = {}, sy = {}, sdx = {};
    for (int i = 0; i < rank; ++i) {
        shape.v[i] = shape_host[i];
        sdy.v[i]   = stride_dy_host[i];
        sy.v[i]    = stride_y_host[i];
        sdx.v[i]   = stride_dx_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    sparsemax_backward_fp_kernel<T><<<blocks, kBlock, 0, stream>>>(
        dy, y, dx, numel, rank, shape, sdy, sy, sdx,
        softmax_axis, softmax_extent, softmax_stride_dy, softmax_stride_y);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

} } // namespace baracuda::softmax

// Emit one softmax FW launcher.
#define BARACUDA_KERNELS_SOFTMAX_INSTANTIATE(NAME, T)                                             \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t numel,                                                                             \
        int32_t rank,                                                                              \
        const int32_t* shape,                                                                     \
        const int64_t* stride_x,                                                                  \
        const int64_t* stride_y,                                                                  \
        int32_t softmax_axis,                                                                     \
        int32_t softmax_extent,                                                                   \
        int64_t softmax_stride_x,                                                                 \
        int64_t softmax_stride_y,                                                                 \
        const void* x, void* y,                                                                   \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                 \
        if (x == nullptr || y == nullptr) return 2;                                               \
        if (shape == nullptr || stride_x == nullptr || stride_y == nullptr) return 2;             \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::softmax::launch_softmax_fp<T>(                                           \
            static_cast<const T*>(x), static_cast<T*>(y), numel, rank,                            \
            shape, stride_x, stride_y,                                                            \
            softmax_axis, softmax_extent, softmax_stride_x, softmax_stride_y,                     \
            stream);                                                                              \
    }

// Emit one softmax BW launcher. Needs saved forward output `y`.
#define BARACUDA_KERNELS_SOFTMAX_BACKWARD_INSTANTIATE(NAME, T)                                    \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t numel,                                                                             \
        int32_t rank,                                                                              \
        const int32_t* shape,                                                                     \
        const int64_t* stride_dy,                                                                 \
        const int64_t* stride_y,                                                                  \
        const int64_t* stride_dx,                                                                 \
        int32_t softmax_axis,                                                                     \
        int32_t softmax_extent,                                                                   \
        int64_t softmax_stride_dy,                                                                \
        int64_t softmax_stride_y,                                                                 \
        const void* dy, const void* y, void* dx,                                                  \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                 \
        if (dy == nullptr || y == nullptr || dx == nullptr) return 2;                             \
        if (shape == nullptr || stride_dy == nullptr || stride_y == nullptr ||                    \
            stride_dx == nullptr) return 2;                                                       \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::softmax::launch_softmax_backward_fp<T>(                                  \
            static_cast<const T*>(dy), static_cast<const T*>(y), static_cast<T*>(dx),             \
            numel, rank, shape, stride_dy, stride_y, stride_dx,                                   \
            softmax_axis, softmax_extent, softmax_stride_dy, softmax_stride_y,                    \
            stream);                                                                              \
    }

// Emit one log-softmax FW launcher. ABI identical to softmax FW.
#define BARACUDA_KERNELS_LOG_SOFTMAX_INSTANTIATE(NAME, T)                                         \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t numel,                                                                             \
        int32_t rank,                                                                              \
        const int32_t* shape,                                                                     \
        const int64_t* stride_x,                                                                  \
        const int64_t* stride_y,                                                                  \
        int32_t softmax_axis,                                                                     \
        int32_t softmax_extent,                                                                   \
        int64_t softmax_stride_x,                                                                 \
        int64_t softmax_stride_y,                                                                 \
        const void* x, void* y,                                                                   \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                 \
        if (x == nullptr || y == nullptr) return 2;                                               \
        if (shape == nullptr || stride_x == nullptr || stride_y == nullptr) return 2;             \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::softmax::launch_log_softmax_fp<T>(                                       \
            static_cast<const T*>(x), static_cast<T*>(y), numel, rank,                            \
            shape, stride_x, stride_y,                                                            \
            softmax_axis, softmax_extent, softmax_stride_x, softmax_stride_y,                     \
            stream);                                                                              \
    }

// Emit one log-softmax BW launcher. ABI identical to softmax BW.
#define BARACUDA_KERNELS_LOG_SOFTMAX_BACKWARD_INSTANTIATE(NAME, T)                                \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t numel,                                                                             \
        int32_t rank,                                                                              \
        const int32_t* shape,                                                                     \
        const int64_t* stride_dy,                                                                 \
        const int64_t* stride_y,                                                                  \
        const int64_t* stride_dx,                                                                 \
        int32_t softmax_axis,                                                                     \
        int32_t softmax_extent,                                                                   \
        int64_t softmax_stride_dy,                                                                \
        int64_t softmax_stride_y,                                                                 \
        const void* dy, const void* y, void* dx,                                                  \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                 \
        if (dy == nullptr || y == nullptr || dx == nullptr) return 2;                             \
        if (shape == nullptr || stride_dy == nullptr || stride_y == nullptr ||                    \
            stride_dx == nullptr) return 2;                                                       \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::softmax::launch_log_softmax_backward_fp<T>(                              \
            static_cast<const T*>(dy), static_cast<const T*>(y), static_cast<T*>(dx),             \
            numel, rank, shape, stride_dy, stride_y, stride_dx,                                   \
            softmax_axis, softmax_extent, softmax_stride_dy, softmax_stride_y,                    \
            stream);                                                                              \
    }

// Emit one gumbel-softmax FW launcher. Reads `x` (T) + `u_rand` (f32)
// + writes `y` (T). ABI is softmax FW + (u_rand, inv_tau, hard).
#define BARACUDA_KERNELS_GUMBEL_SOFTMAX_INSTANTIATE(NAME, T)                                       \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t numel,                                                                              \
        int32_t rank,                                                                               \
        const int32_t* shape,                                                                       \
        const int64_t* stride_x,                                                                    \
        const int64_t* stride_y,                                                                    \
        int32_t softmax_axis,                                                                       \
        int32_t softmax_extent,                                                                     \
        int64_t softmax_stride_x,                                                                   \
        int64_t softmax_stride_y,                                                                   \
        float inv_tau,                                                                              \
        int32_t hard,                                                                               \
        const void* x, const void* u_rand, void* y,                                                 \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                  \
        if (x == nullptr || u_rand == nullptr || y == nullptr) return 2;                            \
        if (shape == nullptr || stride_x == nullptr || stride_y == nullptr) return 2;              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        return baracuda::softmax::launch_gumbel_softmax_fp<T>(                                     \
            static_cast<const T*>(x), static_cast<const float*>(u_rand),                            \
            static_cast<T*>(y), numel, rank,                                                        \
            shape, stride_x, stride_y,                                                              \
            softmax_axis, softmax_extent, softmax_stride_x, softmax_stride_y,                       \
            inv_tau, hard, stream);                                                                 \
    }

// Emit one sparsemax FW launcher. ABI identical to softmax FW.
#define BARACUDA_KERNELS_SPARSEMAX_INSTANTIATE(NAME, T)                                            \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t numel,                                                                              \
        int32_t rank,                                                                               \
        const int32_t* shape,                                                                       \
        const int64_t* stride_x,                                                                    \
        const int64_t* stride_y,                                                                    \
        int32_t softmax_axis,                                                                       \
        int32_t softmax_extent,                                                                     \
        int64_t softmax_stride_x,                                                                   \
        int64_t softmax_stride_y,                                                                   \
        const void* x, void* y,                                                                     \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                  \
        if (x == nullptr || y == nullptr) return 2;                                                \
        if (shape == nullptr || stride_x == nullptr || stride_y == nullptr) return 2;              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        return baracuda::softmax::launch_sparsemax_fp<T>(                                           \
            static_cast<const T*>(x), static_cast<T*>(y), numel, rank,                              \
            shape, stride_x, stride_y,                                                              \
            softmax_axis, softmax_extent, softmax_stride_x, softmax_stride_y,                       \
            stream);                                                                                \
    }

// Emit one sparsemax BW launcher. ABI identical to softmax BW.
#define BARACUDA_KERNELS_SPARSEMAX_BACKWARD_INSTANTIATE(NAME, T)                                   \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t numel,                                                                              \
        int32_t rank,                                                                               \
        const int32_t* shape,                                                                       \
        const int64_t* stride_dy,                                                                   \
        const int64_t* stride_y,                                                                    \
        const int64_t* stride_dx,                                                                   \
        int32_t softmax_axis,                                                                       \
        int32_t softmax_extent,                                                                     \
        int64_t softmax_stride_dy,                                                                  \
        int64_t softmax_stride_y,                                                                   \
        const void* dy, const void* y, void* dx,                                                    \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                  \
        if (dy == nullptr || y == nullptr || dx == nullptr) return 2;                              \
        if (shape == nullptr || stride_dy == nullptr || stride_y == nullptr ||                     \
            stride_dx == nullptr) return 2;                                                        \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        return baracuda::softmax::launch_sparsemax_backward_fp<T>(                                 \
            static_cast<const T*>(dy), static_cast<const T*>(y), static_cast<T*>(dx),               \
            numel, rank, shape, stride_dy, stride_y, stride_dx,                                     \
            softmax_axis, softmax_extent, softmax_stride_dy, softmax_stride_y,                      \
            stream);                                                                                \
    }

#endif // BARACUDA_SOFTMAX_CUH
