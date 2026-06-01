// baracuda_smem_reduce.cuh — warp + block reductions for the
// SMEM-staged kernel pattern. Phase 65.
//
// Companion to `baracuda_smem_row_stager.cuh`. After cooperatively
// staging a row in SMEM, normalizer kernels typically need a block-wide
// reduction (sum, max) to compute statistics like mean / variance /
// log-sum-exp. These helpers provide the warp-shuffle + cross-warp SMEM
// aggregation in one place so we don't reimplement the pattern across
// the 6 normalizer families.
//
// The `warp_reduce_*` helpers were extracted from
// `baracuda_moe.cuh` (Phase 8.5 vintage); the `block_reduce_*` helpers
// + `WarpAggregator` are Phase 65 additions.
//
// Block-wide reductions require some SMEM scratch (32 floats — one
// per warp in a max-1024-thread block). Callers must reserve that
// scratch in their dynamic SMEM allocation OR declare a static
// `__shared__ float warp_buf[32]` separately. Both patterns work.

#ifndef BARACUDA_SMEM_REDUCE_CUH
#define BARACUDA_SMEM_REDUCE_CUH

#include <cstdint>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace baracuda {

// =============================================================================
// Warp reductions — operate within a single warp (32 threads) using
// `__shfl_*_sync`. Lane 0 returns the reduction result for the warp.
// =============================================================================

__device__ __forceinline__ float warp_reduce_sum_f32(float x) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, 32);
    }
    return x;
}

__device__ __forceinline__ float warp_reduce_max_f32(float x) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, 32));
    }
    return x;
}

__device__ __forceinline__ float warp_reduce_min_f32(float x) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x = fminf(x, __shfl_xor_sync(0xffffffff, x, offset, 32));
    }
    return x;
}

// =============================================================================
// Block reductions — aggregate per-warp results via SMEM scratch.
// Caller provides a `__shared__ float warp_buf[BARACUDA_MAX_WARPS]` of
// size 32 (one slot per warp; a block can have up to 32 warps since
// max blockDim.x == 1024).
//
// All threads must call this; the function does internal __syncthreads
// to ensure correctness. The result is broadcast to every thread on
// return (so callers don't need a final `__shfl_sync` themselves).
// =============================================================================

constexpr int BARACUDA_MAX_WARPS = 32;

__device__ __forceinline__ float block_reduce_sum_f32(
    float x,
    float* __restrict__ warp_buf)
{
    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = (blockDim.x + 31) >> 5;

    // Step 1: warp-level reduction.
    float sum = warp_reduce_sum_f32(x);

    // Step 2: lane 0 of each warp writes to warp_buf.
    if (lane_id == 0) warp_buf[warp_id] = sum;
    __syncthreads();

    // Step 3: warp 0 reads back from warp_buf + reduces across warps.
    if (warp_id == 0) {
        float v = (lane_id < num_warps) ? warp_buf[lane_id] : 0.0f;
        v = warp_reduce_sum_f32(v);
        // Broadcast slot 0 of warp_buf so every thread can read.
        if (lane_id == 0) warp_buf[0] = v;
    }
    __syncthreads();

    return warp_buf[0];
}

__device__ __forceinline__ float block_reduce_max_f32(
    float x,
    float* __restrict__ warp_buf)
{
    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = (blockDim.x + 31) >> 5;

    float m = warp_reduce_max_f32(x);

    if (lane_id == 0) warp_buf[warp_id] = m;
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane_id < num_warps) ? warp_buf[lane_id] : -CUDART_INF_F;
        v = warp_reduce_max_f32(v);
        if (lane_id == 0) warp_buf[0] = v;
    }
    __syncthreads();

    return warp_buf[0];
}

__device__ __forceinline__ float block_reduce_min_f32(
    float x,
    float* __restrict__ warp_buf)
{
    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = (blockDim.x + 31) >> 5;

    float m = warp_reduce_min_f32(x);

    if (lane_id == 0) warp_buf[warp_id] = m;
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane_id < num_warps) ? warp_buf[lane_id] : CUDART_INF_F;
        v = warp_reduce_min_f32(v);
        if (lane_id == 0) warp_buf[0] = v;
    }
    __syncthreads();

    return warp_buf[0];
}

// =============================================================================
// Aggregator — multi-stat in one block-reduction pass. Used by LayerNorm,
// online-softmax, RMSNorm to compute (sum, sum_sq) or (max, sum_exp)
// from a single sweep over the row.
//
// Pattern:
//
//     float sum = 0.0f, sum_sq = 0.0f;
//     for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
//         float v = smem_row[i];
//         sum    += v;
//         sum_sq += v * v;
//     }
//     float total_sum    = block_reduce_sum_f32(sum,    warp_buf);
//     float total_sum_sq = block_reduce_sum_f32(sum_sq, warp_buf);
//     // mean = total_sum / n
//     // var  = total_sum_sq / n - mean * mean
//
// Each call to block_reduce_sum_f32 is one full block-wide reduction.
// For multi-stat patterns (mean + var, max + sum-exp, etc.) we call
// the block_reduce primitive once per statistic.
// =============================================================================

}  // namespace baracuda

#endif  // BARACUDA_SMEM_REDUCE_CUH
