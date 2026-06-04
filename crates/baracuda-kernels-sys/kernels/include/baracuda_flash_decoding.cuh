// SPDX-FileCopyrightText: 2026 Eric Evans and the baracuda contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// FlashDecoding — split-K parallel attention decode for seq_q = 1.
//
// The Phase 10 trailblazer `flash_sdpa_fw_kernel` is tuned for prefill
// (seq_q comparable to seq_k); its Br=64 q-tile shape leaves the
// seq_q=1 decode regime under-parallelized. FA2 hits the same wall —
// FA2's tile structure assumes seq_q ≥ block-rows.
//
// FlashDecoding (Dao 2023, "Flash-Decoding for Long-Context Inference")
// flips the parallelism axis: split K into S chunks of size CHUNK_K
// each, launch one block per (b, h, k_split), and combine per-split
// online-softmax partials in a small reduction kernel.
//
// Pipeline:
//
//   Kernel 1: split_kernel<T>
//     gridDim  = (S, H, B)
//     blockDim = 128 (4 warps; each warp owns 32 elements of head_dim)
//
//     Each block:
//       1. Load Q[b, h, 0, :D] into SMEM (cooperative, vectorized).
//       2. For each k in [k_split * CHUNK_K, (k_split+1) * CHUNK_K):
//            - Load K[b, h, k, :D] into SMEM.
//            - Score s_k = Q · K[k] * scale (warp-reduce over D).
//            - Online softmax update with running (m, l, o) accumulators.
//       3. Write partial (m, l, o) for this split to workspace.
//
//   Kernel 2: combine_kernel<T>
//     gridDim  = (1, H, B)
//     blockDim = 128
//
//     Each block reads the S partials for its (b, h) and merges them
//     via the standard online-softmax associative merge:
//       global_m   = max over splits of partial_m[s]
//       alpha_s    = exp(partial_m[s] - global_m)
//       global_l   = Σ_s alpha_s * partial_l[s]
//       global_o_d = Σ_s alpha_s * partial_o_d[s]
//     Final: y[b, h, 0, d] = global_o_d / global_l.
//
// Per-block partial storage in workspace:
//   partial_m: [B, H, S]    × f32     (4 B per split)
//   partial_l: [B, H, S]    × f32     (4 B per split)
//   partial_o: [B, H, S, D] × f32     (4 D B per split)
//
// For (B=1, H=32, S=64, D=128) workspace ≈ 1 MB. The same workspace is
// reused across launches (caller passes it in via the Workspace::Borrowed
// path — same contract as FA2).
//
// Tier-1 scope (Phase 73 follow-up):
//   - dtypes: f16, bf16
//   - head_dim ∈ [1, 128]
//   - GQA via stride: K/V supply `head_stride` separately from the head
//     index — the launcher handles broadcast-stride at the host level.
//   - seq_q = 1 (decode); arbitrary seq_k.
//   - is_causal: ignored (decode is always non-causal vs the full KV
//     history — caller is responsible for slicing the cache).
//
// Out of scope (deferred):
//   - f32 / f64 (decode is half-precision in practice).
//   - sliding window, ALiBi, soft-cap (caller masks beforehand).
//   - backward (decode is FW-only).
//   - tensor-core MMA in the Q·K dot product — first cut is warp-shuffle
//     reduce. Tensor-core retune is a follow-up phase once perf bench
//     numbers are in.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "baracuda_smem_reduce.cuh"

namespace baracuda { namespace flash_decoding {

constexpr int kMaxD = 128;
constexpr int kChunkK = 256;
constexpr int kThreadsPerBlock = 128;

// =============================================================================
// Type helpers — half/bf16 → f32 accumulator.
// =============================================================================

template <typename T> struct LoadAcc;
template <> struct LoadAcc<__half> {
    static __device__ __forceinline__ float load(__half x) { return __half2float(x); }
    static __device__ __forceinline__ __half store(float x) { return __float2half(x); }
};
template <> struct LoadAcc<__nv_bfloat16> {
    static __device__ __forceinline__ float load(__nv_bfloat16 x) { return __bfloat162float(x); }
    static __device__ __forceinline__ __nv_bfloat16 store(float x) { return __float2bfloat16(x); }
};

// =============================================================================
// Split kernel — one block per (b, h, k_split). Produces a partial
// (m, l, o[D]) for its K chunk.
// =============================================================================
//
// Strides are in element units, matching the rest of baracuda's strided
// FFI convention. The caller computes per-(b, h) base offsets via the
// host-side strides (so GQA broadcast is `stride_kv[1] = 0`).

template <typename T>
__global__ void flash_decoding_split_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    float* __restrict__ partial_o,
    int32_t batch, int32_t heads, int32_t k_len,
    int32_t head_dim,
    int32_t num_splits,
    int64_t q_b_stride, int64_t q_h_stride,
    int64_t k_b_stride, int64_t k_h_stride, int64_t k_seq_stride,
    int64_t v_b_stride, int64_t v_h_stride, int64_t v_seq_stride,
    float scale)
{
    const int s = blockIdx.x;   // split idx
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    if (s >= num_splits || h >= heads || b >= batch) return;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int k_start = s * kChunkK;
    const int k_end   = min(k_start + kChunkK, k_len);
    if (k_start >= k_end) {
        // No work in this split — write neutral partials so the combine
        // kernel doesn't propagate NaNs.
        if (tid == 0) {
            int64_t pidx = ((int64_t)b * heads + h) * num_splits + s;
            partial_m[pidx] = -INFINITY;
            partial_l[pidx] = 0.0f;
        }
        for (int d = tid; d < head_dim; d += nthreads) {
            int64_t poff = (((int64_t)b * heads + h) * num_splits + s) * (int64_t)head_dim + d;
            partial_o[poff] = 0.0f;
        }
        return;
    }

    // Q tile in SMEM (only D elements — the single Q row for this BH).
    __shared__ float sQ[kMaxD];
    // Per-block running (m, l, o[D]) accumulators in SMEM.
    __shared__ float sM;
    __shared__ float sL;
    __shared__ float sO[kMaxD];
    // K tile in SMEM (kChunkK × D, but we stream one K row at a time).
    __shared__ float sS[kChunkK];     // scores for this chunk
    __shared__ float warp_buf[32];    // for block_reduce_*

    // Load Q[b, h, 0, :D] once. q_b/q_h strides give the BH base; seq=0
    // so no seq-stride contribution.
    const T* q_bh = q + (int64_t)b * q_b_stride + (int64_t)h * q_h_stride;
    for (int d = tid; d < head_dim; d += nthreads) {
        sQ[d] = LoadAcc<T>::load(q_bh[d]);
    }
    if (tid == 0) { sM = -INFINITY; sL = 0.0f; }
    for (int d = tid; d < head_dim; d += nthreads) {
        sO[d] = 0.0f;
    }
    __syncthreads();

    const T* k_bh = k + (int64_t)b * k_b_stride + (int64_t)h * k_h_stride;
    const T* v_bh = v + (int64_t)b * v_b_stride + (int64_t)h * v_h_stride;

    // Stage A — compute the score for each k in this split, then row-
    // softmax over the chunk + accumulate into O.
    //
    // We compute the full chunk's scores into sS first (one thread per
    // score, looping over d in-warp), then do the chunk-wide softmax
    // merge, then the V accumulation. This is the FlashDecoding
    // "vector inner product" pattern.

    const int chunk_len = k_end - k_start;

    // 1. Scores: sS[ki] = sum_d sQ[d] * K[k_start+ki][d] * scale.
    //    One thread per (ki). The d-loop is serial per thread; this is
    //    fine because head_dim ≤ 128 and we have chunk_len ≤ 256 threads
    //    of work per block.
    for (int ki = tid; ki < chunk_len; ki += nthreads) {
        const T* k_row = k_bh + (int64_t)(k_start + ki) * k_seq_stride;
        float acc = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < head_dim; ++d) {
            acc += sQ[d] * LoadAcc<T>::load(k_row[d]);
        }
        sS[ki] = acc * scale;
    }
    // Mask the tail of the chunk if k_len doesn't fill kChunkK.
    for (int ki = tid + chunk_len; ki < kChunkK; ki += nthreads) {
        sS[ki] = -INFINITY;
    }
    __syncthreads();

    // 2. Chunk-local max over scores.
    float local_max = -INFINITY;
    for (int ki = tid; ki < chunk_len; ki += nthreads) {
        if (sS[ki] > local_max) local_max = sS[ki];
    }
    float chunk_max = block_reduce_max_f32(local_max, warp_buf);

    // 3. Chunk-local sum of exp(s - chunk_max).
    float local_sum = 0.0f;
    for (int ki = tid; ki < chunk_len; ki += nthreads) {
        float p = expf(sS[ki] - chunk_max);
        sS[ki] = p;       // overwrite with softmax weight
        local_sum += p;
    }
    float chunk_sum = block_reduce_sum_f32(local_sum, warp_buf);

    // 4. O = sum_ki sS[ki] * V[k_start + ki].
    //    One thread per (d). Each thread walks all chunk_len rows of V.
    for (int d = tid; d < head_dim; d += nthreads) {
        float acc = 0.0f;
        for (int ki = 0; ki < chunk_len; ++ki) {
            const T* v_row = v_bh + (int64_t)(k_start + ki) * v_seq_stride;
            acc += sS[ki] * LoadAcc<T>::load(v_row[d]);
        }
        sO[d] = acc;
    }
    __syncthreads();

    // 5. Write partials. m and l describe the chunk-local softmax;
    //    o is the chunk-local weighted V sum (un-normalized). The
    //    combine kernel handles the global merge.
    if (tid == 0) {
        int64_t pidx = ((int64_t)b * heads + h) * num_splits + s;
        partial_m[pidx] = chunk_max;
        partial_l[pidx] = chunk_sum;
    }
    for (int d = tid; d < head_dim; d += nthreads) {
        int64_t poff = (((int64_t)b * heads + h) * num_splits + s) * (int64_t)head_dim + d;
        partial_o[poff] = sO[d];
    }
}

// =============================================================================
// Combine kernel — one block per (b, h). Reads `num_splits` partial
// (m, l, o[D]) triples for its BH and emits the final y[b, h, 0, :D].
// =============================================================================

template <typename T>
__global__ void flash_decoding_combine_kernel(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ partial_o,
    T* __restrict__ y,
    int32_t batch, int32_t heads,
    int32_t head_dim,
    int32_t num_splits,
    int64_t y_b_stride, int64_t y_h_stride)
{
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    if (h >= heads || b >= batch) return;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    __shared__ float warp_buf[32];

    // Phase 1 — find the global max across splits.
    const int64_t ml_base = ((int64_t)b * heads + h) * num_splits;
    float local_max = -INFINITY;
    for (int s = tid; s < num_splits; s += nthreads) {
        float m = partial_m[ml_base + s];
        if (m > local_max) local_max = m;
    }
    float global_max = block_reduce_max_f32(local_max, warp_buf);

    // Phase 2 — global_l = Σ_s exp(partial_m[s] - global_max) * partial_l[s].
    float local_l = 0.0f;
    for (int s = tid; s < num_splits; s += nthreads) {
        float pm = partial_m[ml_base + s];
        float pl = partial_l[ml_base + s];
        float alpha = (pm == -INFINITY) ? 0.0f : expf(pm - global_max);
        local_l += alpha * pl;
    }
    float global_l = block_reduce_sum_f32(local_l, warp_buf);
    // Guard against degenerate (all-masked) input.
    float inv_l = (global_l > 0.0f) ? (1.0f / global_l) : 0.0f;

    // Phase 3 — per-d, accumulate weighted partial_o.
    const int64_t o_base = (((int64_t)b * heads + h)) * (int64_t)num_splits * (int64_t)head_dim;
    T* y_bh = y + (int64_t)b * y_b_stride + (int64_t)h * y_h_stride;
    for (int d = tid; d < head_dim; d += nthreads) {
        float acc = 0.0f;
        for (int s = 0; s < num_splits; ++s) {
            float pm = partial_m[ml_base + s];
            float alpha = (pm == -INFINITY) ? 0.0f : expf(pm - global_max);
            float po = partial_o[o_base + (int64_t)s * head_dim + d];
            acc += alpha * po;
        }
        y_bh[d] = LoadAcc<T>::store(acc * inv_l);
    }
}

// =============================================================================
// Host launcher — workspace contract + 2-kernel dispatch.
// =============================================================================
//
// Workspace bytes:
//   partial_m:           sizeof(float) * B * H * S
//   partial_l:           sizeof(float) * B * H * S
//   partial_o:           sizeof(float) * B * H * S * D
//   total = B * H * S * (2 + D) * sizeof(float)

__host__ inline int64_t flash_decoding_num_splits(int32_t k_len) {
    if (k_len <= 0) return 0;
    return (int64_t)((k_len + kChunkK - 1) / kChunkK);
}

__host__ inline size_t flash_decoding_workspace_bytes(
    int32_t batch, int32_t heads, int32_t k_len, int32_t head_dim)
{
    int64_t s = flash_decoding_num_splits(k_len);
    if (s == 0) return 0;
    return (size_t)batch * (size_t)heads * (size_t)s
         * (size_t)(2 + head_dim) * sizeof(float);
}

template <typename T>
__host__ inline int32_t launch_flash_decoding(
    const T* q, const T* k, const T* v, T* y,
    void* workspace, size_t workspace_bytes,
    int32_t batch, int32_t heads, int32_t k_len, int32_t head_dim,
    int64_t q_b_stride, int64_t q_h_stride,
    int64_t k_b_stride, int64_t k_h_stride, int64_t k_seq_stride,
    int64_t v_b_stride, int64_t v_h_stride, int64_t v_seq_stride,
    int64_t y_b_stride, int64_t y_h_stride,
    float scale,
    cudaStream_t stream)
{
    if (batch <= 0 || heads <= 0 || head_dim <= 0) return 2;
    if (head_dim > kMaxD) return 3;
    if (k_len <= 0) {
        // No KV → write zeros + bail.
        // Caller is expected to zero-init y; nothing to do here.
        return 0;
    }

    int32_t num_splits = (int32_t)flash_decoding_num_splits(k_len);
    size_t need = (size_t)batch * (size_t)heads * (size_t)num_splits
                * (size_t)(2 + head_dim) * sizeof(float);
    if (workspace_bytes < need) return 4;
    if (workspace == nullptr) return 4;

    unsigned char* wp = (unsigned char*)workspace;
    size_t per_ml = (size_t)batch * (size_t)heads * (size_t)num_splits * sizeof(float);
    float* partial_m = (float*)wp;        wp += per_ml;
    float* partial_l = (float*)wp;        wp += per_ml;
    float* partial_o = (float*)wp;

    dim3 grid_split((unsigned)num_splits, (unsigned)heads, (unsigned)batch);
    dim3 block(kThreadsPerBlock);
    flash_decoding_split_kernel<T><<<grid_split, block, 0, stream>>>(
        q, k, v, partial_m, partial_l, partial_o,
        batch, heads, k_len, head_dim, num_splits,
        q_b_stride, q_h_stride,
        k_b_stride, k_h_stride, k_seq_stride,
        v_b_stride, v_h_stride, v_seq_stride,
        scale);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + (int32_t)err;

    dim3 grid_comb(1, (unsigned)heads, (unsigned)batch);
    flash_decoding_combine_kernel<T><<<grid_comb, block, 0, stream>>>(
        partial_m, partial_l, partial_o, y,
        batch, heads, head_dim, num_splits,
        y_b_stride, y_h_stride);
    err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + (int32_t)err;
    return 0;
}

} } // namespace baracuda::flash_decoding

// =============================================================================
// FFI macro — one symbol pair per dtype.
// =============================================================================

#define BARACUDA_KERNELS_FLASH_DECODING_INSTANTIATE(NAME, T)                                       \
    extern "C" int32_t baracuda_kernels_ ## NAME ## _run(                                           \
        const void* q, const void* k, const void* v, void* y,                                       \
        void* workspace, size_t workspace_bytes,                                                    \
        int32_t batch, int32_t heads, int32_t k_len, int32_t head_dim,                              \
        int64_t q_b_stride, int64_t q_h_stride,                                                     \
        int64_t k_b_stride, int64_t k_h_stride, int64_t k_seq_stride,                               \
        int64_t v_b_stride, int64_t v_h_stride, int64_t v_seq_stride,                               \
        int64_t y_b_stride, int64_t y_h_stride,                                                     \
        float scale,                                                                                \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        return baracuda::flash_decoding::launch_flash_decoding<T>(                                  \
            (const T*)q, (const T*)k, (const T*)v, (T*)y,                                           \
            workspace, workspace_bytes,                                                             \
            batch, heads, k_len, head_dim,                                                          \
            q_b_stride, q_h_stride,                                                                 \
            k_b_stride, k_h_stride, k_seq_stride,                                                   \
            v_b_stride, v_h_stride, v_seq_stride,                                                   \
            y_b_stride, y_h_stride,                                                                 \
            scale, stream);                                                                         \
    }                                                                                               \
    extern "C" int32_t baracuda_kernels_ ## NAME ## _can_implement(                                 \
        int32_t batch, int32_t heads, int32_t k_len, int32_t head_dim)                              \
    {                                                                                               \
        if (batch <= 0 || heads <= 0 || head_dim <= 0) return 2;                                    \
        if (head_dim > baracuda::flash_decoding::kMaxD) return 3;                                   \
        if (k_len < 0) return 2;                                                                    \
        return 0;                                                                                   \
    }                                                                                               \
    extern "C" size_t baracuda_kernels_ ## NAME ## _workspace_bytes(                                \
        int32_t batch, int32_t heads, int32_t k_len, int32_t head_dim)                              \
    {                                                                                               \
        return baracuda::flash_decoding::flash_decoding_workspace_bytes(                            \
            batch, heads, k_len, head_dim);                                                         \
    }
