// baracuda_smem_scan.cuh — warp + block prefix scans for the
// SMEM-staged kernel pattern. Phase 67d.
//
// Companion to `baracuda_smem_reduce.cuh`. Where the reduce helpers
// collapse a row to a single statistic (sum / max / min), the scan
// helpers compute a *running* statistic: each thread's result is the
// op over all values at-or-before (inclusive) — or strictly before
// (exclusive) — its `threadIdx.x` position. These are the O(log N)
// Kogge-Stone / cross-warp building blocks that replace the bespoke
// O(N) per-row sequential loops in the scan kernel family.
//
// When to use:
//   - cumsum / cumprod / cummax / cummin / logcumsumexp when the scan
//     axis maps onto a single thread block (one block scans one row).
//   - online-softmax running-max / running-sum updates.
//   - prefix-sum-driven indexing (compaction, stream-compact offsets).
//
// When NOT to use (out of scope — use CUB directly):
//   - Device-wide scans across many blocks: `cub::DeviceScan`.
//   - Block scans with multiple ITEMS_PER_THREAD: `cub::BlockScan`
//     already shines there (see sparsemax_fp.cu). These helpers are
//     the one-value-per-thread fast path with an explicit SMEM scratch
//     the caller owns (so they compose with row-stager SMEM budgets).
//   - Segmented (per-row-within-a-2D-tile) scans — kernel-specific
//     dispatch that doesn't generalize into a helper.
//
// Cross-warp scratch:
//   Block-wide scans need `warp_buf[BARACUDA_MAX_WARPS]` (32 slots —
//   one per warp; max blockDim.x == 1024 ⇒ ≤32 warps). Reserve it in
//   dynamic SMEM or as a static `__shared__ float warp_buf[32]`. The
//   buffer matches the layout `block_reduce_*` uses, so a kernel that
//   both reduces and scans can share one scratch buffer.
//
// All block-wide functions are cooperative: every thread in the block
// MUST call, and the function does internal `__syncthreads()`. The
// result is returned per-thread (no extra broadcast needed by callers).
//
// Partial-warp note: the cross-warp aggregation reads the per-warp
// total from lane 31 of each warp (matching `block_reduce_*`'s
// lane-0-writes convention). Callers should use a `blockDim.x` that is
// a multiple of 32 (the standard for these block-cooperative kernels)
// so the last warp is full. A partial final warp would leave its
// per-warp total slot unwritten.

#ifndef BARACUDA_SMEM_SCAN_CUH
#define BARACUDA_SMEM_SCAN_CUH

#include <cstdint>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace baracuda { namespace scan {

// =============================================================================
// Warp scans — inclusive prefix within a single warp (32 lanes) via
// `__shfl_up_sync`. Kogge-Stone: log2(32) = 5 rounds. Each lane returns
// the op over lanes [0 .. its own lane_id] of the warp.
// =============================================================================

__device__ __forceinline__ float warp_scan_inclusive_sum_f32(float v) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float n = __shfl_up_sync(0xffffffff, v, offset, 32);
        if ((threadIdx.x & 31) >= offset) v += n;
    }
    return v;
}

__device__ __forceinline__ float warp_scan_inclusive_max_f32(float v) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float n = __shfl_up_sync(0xffffffff, v, offset, 32);
        if ((threadIdx.x & 31) >= offset) v = fmaxf(v, n);
    }
    return v;
}

__device__ __forceinline__ float warp_scan_inclusive_min_f32(float v) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float n = __shfl_up_sync(0xffffffff, v, offset, 32);
        if ((threadIdx.x & 31) >= offset) v = fminf(v, n);
    }
    return v;
}

__device__ __forceinline__ double warp_scan_inclusive_sum_f64(double v) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        double n = __shfl_up_sync(0xffffffff, v, offset, 32);
        if ((threadIdx.x & 31) >= offset) v += n;
    }
    return v;
}

// =============================================================================
// Block scans — combine per-warp inclusive scans with a cross-warp
// offset computed in `warp_buf`. Pattern (mirrors block_reduce_*):
//   1. each warp scans its own lanes (warp_scan_inclusive_*).
//   2. lane 31 writes the warp total to warp_buf[warp_id].
//   3. warp 0 EXCLUSIVE-scans the per-warp totals → each warp's offset
//      (op over all *prior* warps; first warp gets the identity).
//   4. every thread combines its in-warp prefix with its warp offset.
// =============================================================================

__device__ __forceinline__ float block_scan_inclusive_sum_f32(
    float value,
    float* __restrict__ warp_buf)
{
    int lane_id  = threadIdx.x & 31;
    int warp_id  = threadIdx.x >> 5;
    int num_warps = (blockDim.x + 31) >> 5;

    float warp_inc = warp_scan_inclusive_sum_f32(value);

    // Lane 31 holds the warp's inclusive total.
    if (lane_id == 31) warp_buf[warp_id] = warp_inc;
    __syncthreads();

    // Warp 0 turns the per-warp totals into exclusive prefix offsets.
    if (warp_id == 0) {
        float t = (lane_id < num_warps) ? warp_buf[lane_id] : 0.0f;
        float inc = warp_scan_inclusive_sum_f32(t);
        float ex  = __shfl_up_sync(0xffffffff, inc, 1, 32);
        if (lane_id == 0) ex = 0.0f;          // identity for the first warp
        warp_buf[lane_id] = ex;
    }
    __syncthreads();

    return warp_inc + warp_buf[warp_id];
}

__device__ __forceinline__ float block_scan_inclusive_max_f32(
    float value,
    float* __restrict__ warp_buf)
{
    int lane_id  = threadIdx.x & 31;
    int warp_id  = threadIdx.x >> 5;
    int num_warps = (blockDim.x + 31) >> 5;

    float warp_inc = warp_scan_inclusive_max_f32(value);

    if (lane_id == 31) warp_buf[warp_id] = warp_inc;
    __syncthreads();

    if (warp_id == 0) {
        float t = (lane_id < num_warps) ? warp_buf[lane_id] : -CUDART_INF_F;
        float inc = warp_scan_inclusive_max_f32(t);
        float ex  = __shfl_up_sync(0xffffffff, inc, 1, 32);
        if (lane_id == 0) ex = -CUDART_INF_F;   // identity for the first warp
        warp_buf[lane_id] = ex;
    }
    __syncthreads();

    return fmaxf(warp_inc, warp_buf[warp_id]);
}

__device__ __forceinline__ float block_scan_inclusive_min_f32(
    float value,
    float* __restrict__ warp_buf)
{
    int lane_id  = threadIdx.x & 31;
    int warp_id  = threadIdx.x >> 5;
    int num_warps = (blockDim.x + 31) >> 5;

    float warp_inc = warp_scan_inclusive_min_f32(value);

    if (lane_id == 31) warp_buf[warp_id] = warp_inc;
    __syncthreads();

    if (warp_id == 0) {
        float t = (lane_id < num_warps) ? warp_buf[lane_id] : CUDART_INF_F;
        float inc = warp_scan_inclusive_min_f32(t);
        float ex  = __shfl_up_sync(0xffffffff, inc, 1, 32);
        if (lane_id == 0) ex = CUDART_INF_F;     // identity for the first warp
        warp_buf[lane_id] = ex;
    }
    __syncthreads();

    return fminf(warp_inc, warp_buf[warp_id]);
}

// Exclusive sum: each thread sees the sum of all strictly-prior threads;
// the first thread (threadIdx.x == 0) gets the identity (0). Derived
// from the inclusive scan by subtracting the thread's own value — exact
// for the additive group (no extra reduction pass).
__device__ __forceinline__ float block_scan_exclusive_sum_f32(
    float value,
    float* __restrict__ warp_buf)
{
    return block_scan_inclusive_sum_f32(value, warp_buf) - value;
}

__device__ __forceinline__ double block_scan_inclusive_sum_f64(
    double value,
    double* __restrict__ warp_buf)
{
    int lane_id  = threadIdx.x & 31;
    int warp_id  = threadIdx.x >> 5;
    int num_warps = (blockDim.x + 31) >> 5;

    double warp_inc = warp_scan_inclusive_sum_f64(value);

    if (lane_id == 31) warp_buf[warp_id] = warp_inc;
    __syncthreads();

    if (warp_id == 0) {
        double t = (lane_id < num_warps) ? warp_buf[lane_id] : 0.0;
        double inc = warp_scan_inclusive_sum_f64(t);
        double ex  = __shfl_up_sync(0xffffffff, inc, 1, 32);
        if (lane_id == 0) ex = 0.0;            // identity for the first warp
        warp_buf[lane_id] = ex;
    }
    __syncthreads();

    return warp_inc + warp_buf[warp_id];
}

__device__ __forceinline__ double block_scan_exclusive_sum_f64(
    double value,
    double* __restrict__ warp_buf)
{
    return block_scan_inclusive_sum_f64(value, warp_buf) - value;
}

} }  // namespace baracuda::scan

#endif  // BARACUDA_SMEM_SCAN_CUH
