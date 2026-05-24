// baracuda_adaptive_pool.cuh
//
// Phase 16.1 — bit-exact PyTorch adaptive average / max pooling.
//
// **Why this kernel exists.** Phase 11.8 shipped six adaptive-pool plans
// (1D/2D/3D × Avg/Max) routed through cuDNN with a uniform-window
// approximation:
//   kernel = ceil(in / out); stride = floor(in / out); pad = 0
// That matches PyTorch when `in % out == 0`. When the axis is
// non-divisible, PyTorch instead uses **non-uniform** per-output-cell
// windows so the entire input range is tiled exactly. The cuDNN
// approximation can diverge by ±1 input cell on the boundary cells.
//
// **PyTorch convention** (bit-exact, the contract this file implements):
//   for each output cell `i ∈ [0, out)`:
//     start_i = floor(i * in / out)
//     end_i   = ceil((i + 1) * in / out)
//   window covers input cells `[start_i, end_i)`. Window length is
//   variable per output cell (especially on boundaries). For ND pools
//   the formula is applied independently per spatial axis.
//
//   AvgPool:  y[i] = mean(x[start_i..end_i])  (denominator = window len)
//   MaxPool:  y[i] = max(x[start_i..end_i]); save argmax for BW
//
// **Layout.** The kernels are rank-agnostic — one launch handles
// 1D / 2D / 3D by carrying the spatial rank + per-axis in/out shape
// arrays (capped at 3 spatial axes). Outer dim = batch × channels,
// treated as one independent batch of (in_spatial → out_spatial) pools.
// Input / output are contiguous row-major (NCL / NCHW / NCDHW).
//
// **FW kernels.** One thread per output element. Thread decomposes its
// linear index into (nc, out_coord), computes the per-axis [start, end)
// window via integer arithmetic, walks the window accumulating
// mean / scanning max.
//   - AvgPool divides by the product of per-axis window lengths.
//   - MaxPool also writes an i64 argmax (linear offset within the
//     per-NC spatial slab) to `indices` for BW reuse.
//
// **BW kernels.** AvgPool BW scatters `dy[i] / win_size` to every input
// cell covered by output cell `i`. MaxPool BW scatters `dy[i]` to the
// saved argmax index. Both use `atomicAdd` (via `baracuda_atomic.cuh`)
// because input cells may belong to multiple output windows (AvgPool
// boundary overlap) or — rarely — multiple max-outputs may tie on the
// same argmax (MaxPool). Determinism is non-bit-stable across launches
// per the rest of baracuda's pool BW family.
//
// **Dtype coverage.** `{f16, bf16, f32, f64}`. Integer adaptive pool
// (rounding semantics, no Fuel ask) is out of scope.
//
// Status codes mirror the rest of baracuda-kernels-sys:
//   0 success
//   1 misaligned operand
//   2 invalid problem
//   3 unsupported
//   4 workspace too small
//   5 internal kernel error (launch failure)

#ifndef BARACUDA_ADAPTIVE_POOL_CUH
#define BARACUDA_ADAPTIVE_POOL_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "baracuda_atomic.cuh"

namespace baracuda { namespace adaptive_pool {

inline constexpr int MAX_SPATIAL_RANK = 3;

// PyTorch's bit-exact adaptive-pool window bounds for axis index `i`:
//   start_i = floor(i * in_sz / out_sz)
//   end_i   = ceil((i + 1) * in_sz / out_sz)
// Implemented with int64 intermediates to avoid 32-bit overflow on
// large extents. Caller must ensure `out_sz > 0`.
__host__ __device__ __forceinline__ int32_t adaptive_start(
    int32_t i, int32_t in_sz, int32_t out_sz)
{
    int64_t num = (int64_t)i * (int64_t)in_sz;
    return (int32_t)(num / (int64_t)out_sz);
}

__host__ __device__ __forceinline__ int32_t adaptive_end(
    int32_t i, int32_t in_sz, int32_t out_sz)
{
    int64_t num = (int64_t)(i + 1) * (int64_t)in_sz + (int64_t)(out_sz - 1);
    return (int32_t)(num / (int64_t)out_sz);
}

// Accumulator selection: f64 for f64 operands, f32 for everything
// (including half / bf16) — matches the rest of baracuda's reduction
// family (see `baracuda_norm.cuh`, `reduce_mean`, etc.).
template <typename T> struct accum_of { using type = float; };
template <> struct accum_of<double> { using type = double; };

template <typename T>
__host__ __device__ __forceinline__ typename accum_of<T>::type to_accum(T v) {
    return (typename accum_of<T>::type)v;
}
template <>
__host__ __device__ __forceinline__ float to_accum<__half>(__half v) {
    return __half2float(v);
}
template <>
__host__ __device__ __forceinline__ float to_accum<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename T>
__host__ __device__ __forceinline__ T from_accum(typename accum_of<T>::type v) {
    return (T)v;
}
template <>
__host__ __device__ __forceinline__ __half from_accum<__half>(float v) {
    return __float2half(v);
}
template <>
__host__ __device__ __forceinline__ __nv_bfloat16 from_accum<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

// "Type-correct zero" helpers (for memset-by-kernel where atomicAdd into
// dx requires a zeroed target).
template <typename T> __host__ __device__ __forceinline__ T zero_of() { return (T)0; }
template <> __host__ __device__ __forceinline__ __half zero_of<__half>() {
    return __float2half(0.0f);
}
template <> __host__ __device__ __forceinline__ __nv_bfloat16 zero_of<__nv_bfloat16>() {
    return __float2bfloat16(0.0f);
}

// =============================================================================
// AvgPool — forward.
// =============================================================================
//
// One thread per output element. `out_numel_per_nc = ∏ out_spatial`,
// `total_out_numel = NC * out_numel_per_nc`. Thread decomposes its tid
// into (nc, out_coord[]), computes per-axis window via the PyTorch
// integer formula, walks the window in input-axis-major order
// accumulating in float / double, writes mean.
template <typename T>
__global__ void avg_pool_fw_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int64_t total_out_numel,
    int32_t nc,
    int32_t spatial_rank,
    int32_t in_d, int32_t in_h, int32_t in_w,
    int32_t out_d, int32_t out_h, int32_t out_w)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    using Acc = typename accum_of<T>::type;

    int64_t in_per_nc  = (int64_t)in_d * (int64_t)in_h * (int64_t)in_w;
    int64_t out_per_nc = (int64_t)out_d * (int64_t)out_h * (int64_t)out_w;

    for (int64_t k = tid; k < total_out_numel; k += step) {
        int64_t nc_idx = k / out_per_nc;
        int64_t within = k - nc_idx * out_per_nc;
        // Decompose `within` into (od, oh, ow) per the spatial rank.
        int32_t ow_i = (int32_t)(within % (int64_t)out_w);
        int64_t rem  = within / (int64_t)out_w;
        int32_t oh_i = (int32_t)(rem % (int64_t)out_h);
        int32_t od_i = (int32_t)(rem / (int64_t)out_h);

        // Per-axis window bounds. For rank < 3 the missing axes are
        // configured at in=1, out=1, so start=0 / end=1 — a degenerate
        // window of length 1 that adds nothing to the product.
        int32_t sd = adaptive_start(od_i, in_d, out_d);
        int32_t ed = adaptive_end  (od_i, in_d, out_d);
        int32_t sh = adaptive_start(oh_i, in_h, out_h);
        int32_t eh = adaptive_end  (oh_i, in_h, out_h);
        int32_t sw = adaptive_start(ow_i, in_w, out_w);
        int32_t ew = adaptive_end  (ow_i, in_w, out_w);

        int32_t win_d = ed - sd;
        int32_t win_h = eh - sh;
        int32_t win_w = ew - sw;
        int64_t win_count = (int64_t)win_d * (int64_t)win_h * (int64_t)win_w;
        if (win_count <= 0) {
            y[k] = zero_of<T>();
            continue;
        }

        Acc accum = (Acc)0;
        const T* x_base = x + nc_idx * in_per_nc;
        for (int32_t dd = sd; dd < ed; ++dd) {
            for (int32_t hh = sh; hh < eh; ++hh) {
                const T* row = x_base + ((int64_t)dd * (int64_t)in_h + (int64_t)hh) * (int64_t)in_w;
                for (int32_t ww = sw; ww < ew; ++ww) {
                    accum += to_accum<T>(row[ww]);
                }
            }
        }
        Acc mean = accum / (Acc)win_count;
        y[k] = from_accum<T>(mean);
        // Silence unused-param warnings when spatial_rank < 3.
        (void)spatial_rank;
        (void)nc;
    }
}

template <typename T>
__host__ inline int32_t launch_avg_pool_fw(
    const void* x, void* y,
    int32_t nc,
    int32_t spatial_rank,
    int32_t in_d, int32_t in_h, int32_t in_w,
    int32_t out_d, int32_t out_h, int32_t out_w,
    cudaStream_t stream)
{
    if (nc <= 0) return 2;
    if (spatial_rank < 1 || spatial_rank > MAX_SPATIAL_RANK) return 2;
    if (in_d <= 0 || in_h <= 0 || in_w <= 0) return 2;
    if (out_d <= 0 || out_h <= 0 || out_w <= 0) return 2;
    if (x == nullptr || y == nullptr) return 2;
    int64_t out_per_nc = (int64_t)out_d * (int64_t)out_h * (int64_t)out_w;
    int64_t total_out_numel = (int64_t)nc * out_per_nc;
    if (total_out_numel == 0) return 0;

    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total_out_numel + kBlock - 1) / kBlock;
    int blocks = (int)(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    avg_pool_fw_kernel<T><<<blocks, kBlock, 0, stream>>>(
        (const T*)x, (T*)y, total_out_numel, nc, spatial_rank,
        in_d, in_h, in_w, out_d, out_h, out_w);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// AvgPool — backward.
// =============================================================================
//
// We need to scatter `dy[i] / win_size_i` into every input cell `j` in
// output cell `i`'s window. Boundary windows overlap (when in % out !=
// 0), so two outputs can share an input cell — atomic scatter is
// required. We follow the established pool-BW pattern (atomicAdd
// scatter), with `baracuda::atomic::add` routing half / bf16 through
// CAS for determinism on those dtypes.
//
// `dx` is assumed to be pre-zeroed by a prior kernel (see
// `dx_zero_kernel` below — launched immediately before this one in the
// `launch_avg_pool_bw` helper).
template <typename T>
__global__ void avg_pool_bw_scatter_kernel(
    const T* __restrict__ dy,
    T* __restrict__ dx,
    int64_t total_out_numel,
    int32_t nc,
    int32_t spatial_rank,
    int32_t in_d, int32_t in_h, int32_t in_w,
    int32_t out_d, int32_t out_h, int32_t out_w)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    using Acc = typename accum_of<T>::type;

    int64_t in_per_nc  = (int64_t)in_d * (int64_t)in_h * (int64_t)in_w;
    int64_t out_per_nc = (int64_t)out_d * (int64_t)out_h * (int64_t)out_w;

    for (int64_t k = tid; k < total_out_numel; k += step) {
        int64_t nc_idx = k / out_per_nc;
        int64_t within = k - nc_idx * out_per_nc;
        int32_t ow_i = (int32_t)(within % (int64_t)out_w);
        int64_t rem  = within / (int64_t)out_w;
        int32_t oh_i = (int32_t)(rem % (int64_t)out_h);
        int32_t od_i = (int32_t)(rem / (int64_t)out_h);

        int32_t sd = adaptive_start(od_i, in_d, out_d);
        int32_t ed = adaptive_end  (od_i, in_d, out_d);
        int32_t sh = adaptive_start(oh_i, in_h, out_h);
        int32_t eh = adaptive_end  (oh_i, in_h, out_h);
        int32_t sw = adaptive_start(ow_i, in_w, out_w);
        int32_t ew = adaptive_end  (ow_i, in_w, out_w);

        int32_t win_d = ed - sd;
        int32_t win_h = eh - sh;
        int32_t win_w = ew - sw;
        int64_t win_count = (int64_t)win_d * (int64_t)win_h * (int64_t)win_w;
        if (win_count <= 0) continue;

        Acc share = to_accum<T>(dy[k]) / (Acc)win_count;
        T share_t = from_accum<T>(share);
        T* dx_base = dx + nc_idx * in_per_nc;
        for (int32_t dd = sd; dd < ed; ++dd) {
            for (int32_t hh = sh; hh < eh; ++hh) {
                T* row = dx_base + ((int64_t)dd * (int64_t)in_h + (int64_t)hh) * (int64_t)in_w;
                for (int32_t ww = sw; ww < ew; ++ww) {
                    baracuda::atomic::add<T>(row + ww, share_t);
                }
            }
        }
        (void)spatial_rank;
        (void)nc;
    }
}

template <typename T>
__global__ void dx_zero_kernel(T* dx, int64_t numel) {
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t k = tid; k < numel; k += step) {
        dx[k] = zero_of<T>();
    }
}

template <typename T>
__host__ inline int32_t launch_avg_pool_bw(
    const void* dy, void* dx,
    int32_t nc,
    int32_t spatial_rank,
    int32_t in_d, int32_t in_h, int32_t in_w,
    int32_t out_d, int32_t out_h, int32_t out_w,
    cudaStream_t stream)
{
    if (nc <= 0) return 2;
    if (spatial_rank < 1 || spatial_rank > MAX_SPATIAL_RANK) return 2;
    if (in_d <= 0 || in_h <= 0 || in_w <= 0) return 2;
    if (out_d <= 0 || out_h <= 0 || out_w <= 0) return 2;
    if (dy == nullptr || dx == nullptr) return 2;
    int64_t in_per_nc  = (int64_t)in_d * (int64_t)in_h * (int64_t)in_w;
    int64_t out_per_nc = (int64_t)out_d * (int64_t)out_h * (int64_t)out_w;
    int64_t total_in  = (int64_t)nc * in_per_nc;
    int64_t total_out = (int64_t)nc * out_per_nc;

    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;

    if (total_in > 0) {
        int64_t blocks_i64 = (total_in + kBlock - 1) / kBlock;
        int blocks = (int)(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
        if (blocks <= 0) blocks = 1;
        dx_zero_kernel<T><<<blocks, kBlock, 0, stream>>>((T*)dx, total_in);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return 5;
    }

    if (total_out == 0) return 0;
    {
        int64_t blocks_i64 = (total_out + kBlock - 1) / kBlock;
        int blocks = (int)(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
        if (blocks <= 0) blocks = 1;
        avg_pool_bw_scatter_kernel<T><<<blocks, kBlock, 0, stream>>>(
            (const T*)dy, (T*)dx, total_out, nc, spatial_rank,
            in_d, in_h, in_w, out_d, out_h, out_w);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return 5;
    }
    return 0;
}

// =============================================================================
// MaxPool — forward.
// =============================================================================
//
// Same iteration shape as AvgPool FW; instead of accumulating a mean we
// scan for the max value. PyTorch's `nn.AdaptiveMaxPool*d` also returns
// an `indices` tensor as a second output for autograd reuse, but here
// we keep the Phase-11.8 args shape (FW emits `y` only). The matching
// MaxPool BW recomputes the argmax from the saved `x` so the FW path
// doesn't need to materialize an extra i64 tensor — fewer FFI symbols,
// no API break vs the cuDNN-driven plan, modest extra work in BW
// (~1× re-scan of the input).
//
// On ties, the FIRST input cell in scan order wins — matches PyTorch's
// adaptive_max_pool* convention.
template <typename T>
__global__ void max_pool_fw_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int64_t total_out_numel,
    int32_t nc,
    int32_t spatial_rank,
    int32_t in_d, int32_t in_h, int32_t in_w,
    int32_t out_d, int32_t out_h, int32_t out_w)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    using Acc = typename accum_of<T>::type;

    int64_t in_per_nc  = (int64_t)in_d * (int64_t)in_h * (int64_t)in_w;
    int64_t out_per_nc = (int64_t)out_d * (int64_t)out_h * (int64_t)out_w;

    for (int64_t k = tid; k < total_out_numel; k += step) {
        int64_t nc_idx = k / out_per_nc;
        int64_t within = k - nc_idx * out_per_nc;
        int32_t ow_i = (int32_t)(within % (int64_t)out_w);
        int64_t rem  = within / (int64_t)out_w;
        int32_t oh_i = (int32_t)(rem % (int64_t)out_h);
        int32_t od_i = (int32_t)(rem / (int64_t)out_h);

        int32_t sd = adaptive_start(od_i, in_d, out_d);
        int32_t ed = adaptive_end  (od_i, in_d, out_d);
        int32_t sh = adaptive_start(oh_i, in_h, out_h);
        int32_t eh = adaptive_end  (oh_i, in_h, out_h);
        int32_t sw = adaptive_start(ow_i, in_w, out_w);
        int32_t ew = adaptive_end  (ow_i, in_w, out_w);

        if (ed <= sd || eh <= sh || ew <= sw) {
            y[k] = zero_of<T>();
            continue;
        }

        const T* x_base = x + nc_idx * in_per_nc;
        Acc best = to_accum<T>(x_base[((int64_t)sd * (int64_t)in_h + (int64_t)sh) * (int64_t)in_w + sw]);
        for (int32_t dd = sd; dd < ed; ++dd) {
            for (int32_t hh = sh; hh < eh; ++hh) {
                int64_t row_off = ((int64_t)dd * (int64_t)in_h + (int64_t)hh) * (int64_t)in_w;
                const T* row = x_base + row_off;
                for (int32_t ww = sw; ww < ew; ++ww) {
                    Acc v = to_accum<T>(row[ww]);
                    bool is_first = (dd == sd && hh == sh && ww == sw);
                    if (!is_first && v > best) {
                        best = v;
                    }
                }
            }
        }
        y[k] = from_accum<T>(best);
        (void)spatial_rank;
        (void)nc;
    }
}

template <typename T>
__host__ inline int32_t launch_max_pool_fw(
    const void* x, void* y,
    int32_t nc,
    int32_t spatial_rank,
    int32_t in_d, int32_t in_h, int32_t in_w,
    int32_t out_d, int32_t out_h, int32_t out_w,
    cudaStream_t stream)
{
    if (nc <= 0) return 2;
    if (spatial_rank < 1 || spatial_rank > MAX_SPATIAL_RANK) return 2;
    if (in_d <= 0 || in_h <= 0 || in_w <= 0) return 2;
    if (out_d <= 0 || out_h <= 0 || out_w <= 0) return 2;
    if (x == nullptr || y == nullptr) return 2;
    int64_t out_per_nc = (int64_t)out_d * (int64_t)out_h * (int64_t)out_w;
    int64_t total_out_numel = (int64_t)nc * out_per_nc;
    if (total_out_numel == 0) return 0;

    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total_out_numel + kBlock - 1) / kBlock;
    int blocks = (int)(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    max_pool_fw_kernel<T><<<blocks, kBlock, 0, stream>>>(
        (const T*)x, (T*)y, total_out_numel, nc, spatial_rank,
        in_d, in_h, in_w, out_d, out_h, out_w);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// MaxPool — backward.
// =============================================================================
//
// One thread per output element. Thread re-scans the per-output-cell
// window in `x` to recover the argmax (first-in-scan-order tie-break,
// PyTorch convention), then atomically adds `dy[k]` into
// `dx[nc_idx * in_per_nc + argmax]`. `dx` is zeroed internally by the
// launcher before the scatter.
//
// Tie ambiguity: when multiple windows tie on the same argmax cell,
// the atomicAdd accumulates correctly (PyTorch behavior). When a
// single window has ties, the first-in-scan-order tie-break is used —
// matches both the FW kernel here and PyTorch's CPU/CUDA reference.
template <typename T>
__global__ void max_pool_bw_scatter_kernel(
    const T* __restrict__ x,
    const T* __restrict__ dy,
    T* __restrict__ dx,
    int64_t total_out_numel,
    int32_t nc,
    int32_t spatial_rank,
    int32_t in_d, int32_t in_h, int32_t in_w,
    int32_t out_d, int32_t out_h, int32_t out_w)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    using Acc = typename accum_of<T>::type;

    int64_t in_per_nc  = (int64_t)in_d * (int64_t)in_h * (int64_t)in_w;
    int64_t out_per_nc = (int64_t)out_d * (int64_t)out_h * (int64_t)out_w;

    for (int64_t k = tid; k < total_out_numel; k += step) {
        int64_t nc_idx = k / out_per_nc;
        int64_t within = k - nc_idx * out_per_nc;
        int32_t ow_i = (int32_t)(within % (int64_t)out_w);
        int64_t rem  = within / (int64_t)out_w;
        int32_t oh_i = (int32_t)(rem % (int64_t)out_h);
        int32_t od_i = (int32_t)(rem / (int64_t)out_h);

        int32_t sd = adaptive_start(od_i, in_d, out_d);
        int32_t ed = adaptive_end  (od_i, in_d, out_d);
        int32_t sh = adaptive_start(oh_i, in_h, out_h);
        int32_t eh = adaptive_end  (oh_i, in_h, out_h);
        int32_t sw = adaptive_start(ow_i, in_w, out_w);
        int32_t ew = adaptive_end  (ow_i, in_w, out_w);

        if (ed <= sd || eh <= sh || ew <= sw) continue;

        const T* x_base = x + nc_idx * in_per_nc;
        int64_t best_idx = ((int64_t)sd * (int64_t)in_h + (int64_t)sh) * (int64_t)in_w + sw;
        Acc best = to_accum<T>(x_base[best_idx]);
        for (int32_t dd = sd; dd < ed; ++dd) {
            for (int32_t hh = sh; hh < eh; ++hh) {
                int64_t row_off = ((int64_t)dd * (int64_t)in_h + (int64_t)hh) * (int64_t)in_w;
                const T* row = x_base + row_off;
                for (int32_t ww = sw; ww < ew; ++ww) {
                    Acc v = to_accum<T>(row[ww]);
                    int64_t idx = row_off + ww;
                    bool is_first = (dd == sd && hh == sh && ww == sw);
                    if (!is_first && v > best) {
                        best = v;
                        best_idx = idx;
                    }
                }
            }
        }
        T* dx_base = dx + nc_idx * in_per_nc;
        baracuda::atomic::add<T>(dx_base + best_idx, dy[k]);
        (void)spatial_rank;
        (void)nc;
    }
}

template <typename T>
__host__ inline int32_t launch_max_pool_bw(
    const void* x, const void* dy, void* dx,
    int32_t nc,
    int32_t spatial_rank,
    int32_t in_d, int32_t in_h, int32_t in_w,
    int32_t out_d, int32_t out_h, int32_t out_w,
    cudaStream_t stream)
{
    if (nc <= 0) return 2;
    if (spatial_rank < 1 || spatial_rank > MAX_SPATIAL_RANK) return 2;
    if (in_d <= 0 || in_h <= 0 || in_w <= 0) return 2;
    if (out_d <= 0 || out_h <= 0 || out_w <= 0) return 2;
    if (x == nullptr || dy == nullptr || dx == nullptr) return 2;
    int64_t in_per_nc  = (int64_t)in_d * (int64_t)in_h * (int64_t)in_w;
    int64_t out_per_nc = (int64_t)out_d * (int64_t)out_h * (int64_t)out_w;
    int64_t total_in  = (int64_t)nc * in_per_nc;
    int64_t total_out = (int64_t)nc * out_per_nc;

    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;

    if (total_in > 0) {
        int64_t blocks_i64 = (total_in + kBlock - 1) / kBlock;
        int blocks = (int)(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
        if (blocks <= 0) blocks = 1;
        dx_zero_kernel<T><<<blocks, kBlock, 0, stream>>>((T*)dx, total_in);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return 5;
    }

    if (total_out == 0) return 0;
    {
        int64_t blocks_i64 = (total_out + kBlock - 1) / kBlock;
        int blocks = (int)(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
        if (blocks <= 0) blocks = 1;
        max_pool_bw_scatter_kernel<T><<<blocks, kBlock, 0, stream>>>(
            (const T*)x, (const T*)dy, (T*)dx,
            total_out, nc, spatial_rank,
            in_d, in_h, in_w, out_d, out_h, out_w);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return 5;
    }
    return 0;
}

} } // namespace baracuda::adaptive_pool

// =============================================================================
// extern "C" `_run` entry points — per-dtype, rank-agnostic. The Rust
// side packs the 1D / 2D / 3D shape into the (in_d, in_h, in_w) /
// (out_d, out_h, out_w) i32 args with degenerate 1's filling unused
// leading axes (e.g. 1D → in_d=1, in_h=1, in_w=L).
//
// AvgPool FW.
#define BARACUDA_KERNELS_ADAPTIVE_AVG_POOL_FW_INSTANTIATE(NAME, T)                                 \
    extern "C" int32_t baracuda_kernels_adaptive_avg_pool_##NAME##_fw_run(                          \
        const void* x, void* y,                                                                     \
        int32_t nc, int32_t spatial_rank,                                                           \
        int32_t in_d, int32_t in_h, int32_t in_w,                                                   \
        int32_t out_d, int32_t out_h, int32_t out_w,                                                \
        void* stream_ptr)                                                                            \
    {                                                                                                \
        cudaStream_t stream = (cudaStream_t)stream_ptr;                                              \
        return baracuda::adaptive_pool::launch_avg_pool_fw<T>(                                       \
            x, y, nc, spatial_rank,                                                                  \
            in_d, in_h, in_w, out_d, out_h, out_w, stream);                                          \
    }

// AvgPool BW.
#define BARACUDA_KERNELS_ADAPTIVE_AVG_POOL_BW_INSTANTIATE(NAME, T)                                 \
    extern "C" int32_t baracuda_kernels_adaptive_avg_pool_##NAME##_bw_run(                          \
        const void* dy, void* dx,                                                                   \
        int32_t nc, int32_t spatial_rank,                                                           \
        int32_t in_d, int32_t in_h, int32_t in_w,                                                   \
        int32_t out_d, int32_t out_h, int32_t out_w,                                                \
        void* stream_ptr)                                                                            \
    {                                                                                                \
        cudaStream_t stream = (cudaStream_t)stream_ptr;                                              \
        return baracuda::adaptive_pool::launch_avg_pool_bw<T>(                                       \
            dy, dx, nc, spatial_rank,                                                                \
            in_d, in_h, in_w, out_d, out_h, out_w, stream);                                          \
    }

// MaxPool FW (writes y only — argmax is recomputed inside BW from
// saved x to keep the Phase 11.8 args shape intact).
#define BARACUDA_KERNELS_ADAPTIVE_MAX_POOL_FW_INSTANTIATE(NAME, T)                                 \
    extern "C" int32_t baracuda_kernels_adaptive_max_pool_##NAME##_fw_run(                          \
        const void* x, void* y,                                                                     \
        int32_t nc, int32_t spatial_rank,                                                           \
        int32_t in_d, int32_t in_h, int32_t in_w,                                                   \
        int32_t out_d, int32_t out_h, int32_t out_w,                                                \
        void* stream_ptr)                                                                            \
    {                                                                                                \
        cudaStream_t stream = (cudaStream_t)stream_ptr;                                              \
        return baracuda::adaptive_pool::launch_max_pool_fw<T>(                                       \
            x, y, nc, spatial_rank,                                                                  \
            in_d, in_h, in_w, out_d, out_h, out_w, stream);                                          \
    }

// MaxPool BW (consumes saved x; recomputes argmax internally).
#define BARACUDA_KERNELS_ADAPTIVE_MAX_POOL_BW_INSTANTIATE(NAME, T)                                 \
    extern "C" int32_t baracuda_kernels_adaptive_max_pool_##NAME##_bw_run(                          \
        const void* x, const void* dy, void* dx,                                                    \
        int32_t nc, int32_t spatial_rank,                                                           \
        int32_t in_d, int32_t in_h, int32_t in_w,                                                   \
        int32_t out_d, int32_t out_h, int32_t out_w,                                                \
        void* stream_ptr)                                                                            \
    {                                                                                                \
        cudaStream_t stream = (cudaStream_t)stream_ptr;                                              \
        return baracuda::adaptive_pool::launch_max_pool_bw<T>(                                       \
            x, dy, dx, nc, spatial_rank,                                                             \
            in_d, in_h, in_w, out_d, out_h, out_w, stream);                                          \
    }

#endif // BARACUDA_ADAPTIVE_POOL_CUH
