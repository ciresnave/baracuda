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
#include <mma.h>

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
    int32_t group_size,           // H_q / H_kv (1 for pure MHA)
    int64_t q_b_stride, int64_t q_h_stride,
    int64_t k_b_stride, int64_t k_h_stride, int64_t k_seq_stride,
    int64_t v_b_stride, int64_t v_h_stride, int64_t v_seq_stride,
    float scale)
{
    const int s = blockIdx.x;   // split idx
    const int h = blockIdx.y;   // Q-head index (in [0, H_q))
    const int b = blockIdx.z;
    if (s >= num_splits || h >= heads || b >= batch) return;
    // For GQA: every `group_size` Q heads share one K/V head.
    const int h_kv = h / group_size;

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

    // K/V indexed by the KV-head id (collapses group_size Q heads onto
    // the same K/V slice — the standard GQA broadcast).
    const T* k_bh = k + (int64_t)b * k_b_stride + (int64_t)h_kv * k_h_stride;
    const T* v_bh = v + (int64_t)b * v_b_stride + (int64_t)h_kv * v_h_stride;

    const int chunk_len = k_end - k_start;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int num_warps = nthreads >> 5;   // 4 for kThreadsPerBlock = 128

    // Pass 1 — scores. Warp-cooperative dot product: one warp owns one
    // K-row at a time, all 32 lanes cooperate along the D axis.
    //
    // Why this layout (and not 1 thread per K row, walking D serially):
    // the per-thread serial-D pattern has each warp's 32 threads load
    // K from 32 *different* rows at the same d step. That's 32 cache
    // lines fetched per d step per warp — high pressure, poor reuse.
    //
    // With warp-along-D, 32 lanes of one warp load contiguous D=32
    // halfs of the SAME row — fully coalesced. The warp processes
    // `chunk_len / num_warps` rows over the chunk; 4 warps × 64 rows
    // = 256 rows total for kChunkK = 256.
    //
    // head_dim must be ≥ 32 for this to be meaningful; for D=128 each
    // lane handles D/32 = 4 elements per row.
    if (head_dim >= 32) {
        for (int k_off = warp_id; k_off < chunk_len; k_off += num_warps) {
            const int k_abs = k_start + k_off;
            const T* k_row = k_bh + (int64_t)k_abs * k_seq_stride;
            float acc = 0.0f;
            // Each lane covers D/32 contiguous d-slots, interleaved by
            // warp stride. For D=128: lanes 0..31 own d 0..31, then
            // d 32..63, etc.
            for (int d = lane; d < head_dim; d += 32) {
                acc += sQ[d] * LoadAcc<T>::load(k_row[d]);
            }
            // Warp-reduce sum across the 32 lanes.
            #pragma unroll
            for (int delta = 16; delta > 0; delta >>= 1) {
                acc += __shfl_xor_sync(0xffffffff, acc, delta, 32);
            }
            if (lane == 0) {
                sS[k_off] = acc * scale;
            }
        }
    } else {
        // Tiny-D fallback — one thread per K-row, serial-D. Same shape
        // as the legacy path; we don't care about perf here because the
        // bandwidth math is dominated by the long-D shapes anyway.
        for (int ki = tid; ki < chunk_len; ki += nthreads) {
            const T* k_row = k_bh + (int64_t)(k_start + ki) * k_seq_stride;
            float acc = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                acc += sQ[d] * LoadAcc<T>::load(k_row[d]);
            }
            sS[ki] = acc * scale;
        }
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

    // Pass 2 — V accumulation. The "1 thread per d, walks all K-rows"
    // pattern (used in v1) is ALREADY coalesced across a warp because
    // 32 lanes share the same `ki` and load V[ki, lane..lane+31] which
    // is one contiguous row segment per cache line. Keep this layout.
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
// GQA-batched WMMA split kernel — Tier-2 (Phase 73 follow-up #3).
//
// One block per (k_split, h_kv, b). The block computes attention for
// ALL `group_size` Q heads in this KV group at once, batching them
// in the WMMA M-tile. For Llama-3-class GQA (group_size=4 or 8) this
// uses 25-50% of WMMA's M-tile capacity; for full MQA (group_size=16+
// when H_q=H_kv*group) it uses 100%.
//
// Why this beats the SIMT kernel for GQA:
//   - 1 block does the work of `group_size` SIMT blocks (4-8× fewer
//     kernel launch grids).
//   - K/V loaded ONCE per block (vs once per Q head in the SIMT path)
//     — eliminates redundant L2 traffic across Q heads in a group.
//   - QK^T and PV both run on tensor cores at fp16/bf16 → fp32 MMA.
//
// Constraints:
//   - group_size ∈ [1, kWmmaM] (M-tile width = 16).
//   - head_dim must be a multiple of kWmmaK (= 16).
//   - chunk_len rounded up to kWmmaN multiples for the N-tile loop.
//   - dtype: __half or __nv_bfloat16.
//   - blockDim.x = kThreadsPerBlock = 128 (4 warps).
//
// SMEM layout (per block):
//   sQ        [kWmmaM × kMaxD]                    half/bf16
//   sK_tile   [kWmmaN × kMaxD]   one K sub-tile   half/bf16
//   sV_tile   [kWmmaN × kMaxD]   one V sub-tile   half/bf16
//   sScores   [kWmmaM × kChunkK]                  float
//   sO        [kWmmaM × kMaxD]                    float
//   warp_buf  [32]                                float
//
// Total ≈ 16*128*2 + 16*128*2 + 16*128*2 + 16*256*4 + 16*128*4 + 128
//       = 4K + 4K + 4K + 16K + 8K + 0.5K ≈ 36 KB — fits in 48 KB.
// =============================================================================

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;

namespace tc {
using namespace nvcuda;

// Convert f32 → T for storing a Q row into the WMMA half-precision
// fragment buffer.
template <typename T>
struct ToHalf;
template <>
struct ToHalf<__half> {
    static __device__ __forceinline__ __half cvt(float x) { return __float2half(x); }
};
template <>
struct ToHalf<__nv_bfloat16> {
    static __device__ __forceinline__ __nv_bfloat16 cvt(float x) { return __float2bfloat16(x); }
};

}  // namespace tc

template <typename T>
__global__ void flash_decoding_split_kernel_tc(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    float* __restrict__ partial_o,
    int32_t batch, int32_t heads, int32_t k_len,
    int32_t head_dim,
    int32_t num_splits,
    int32_t num_kv_heads,
    int32_t group_size,
    int64_t q_b_stride, int64_t q_h_stride,
    int64_t k_b_stride, int64_t k_h_stride, int64_t k_seq_stride,
    int64_t v_b_stride, int64_t v_h_stride, int64_t v_seq_stride,
    float scale)
{
    using namespace nvcuda;

    const int s     = blockIdx.x;
    const int h_kv  = blockIdx.y;
    const int b     = blockIdx.z;
    if (s >= num_splits || h_kv >= num_kv_heads || b >= batch) return;

    const int tid       = threadIdx.x;
    const int warp_id   = tid >> 5;
    const int lane      = tid & 31;
    const int num_warps = blockDim.x >> 5;   // 4

    const int k_start = s * kChunkK;
    const int k_end   = min(k_start + kChunkK, k_len);
    const int chunk_len = k_end - k_start;

    // Q-head range that maps to this KV head: [h_kv*group_size,
    // (h_kv+1)*group_size).
    const int q_head_base = h_kv * group_size;

    // SMEM allocations. sQ/sK/sV use the kernel's element type T;
    // sScores and sO accumulate in float.
    __shared__ T     sQ[kWmmaM * kMaxD];        // padded to M=16
    __shared__ T     sK_tile[kWmmaN * kMaxD];
    __shared__ T     sV_tile[kWmmaN * kMaxD];
    __shared__ float sScores[kWmmaM * kChunkK]; // 16 × 256
    __shared__ float sO[kWmmaM * kMaxD];
    __shared__ float sMaxRow[kWmmaM];
    __shared__ float sSumRow[kWmmaM];

    // Empty-chunk → write neutral partials for every Q head in the group.
    if (k_start >= k_end) {
        for (int g = 0; g < group_size; ++g) {
            const int h_q = q_head_base + g;
            if (tid == 0) {
                int64_t pidx = ((int64_t)b * heads + h_q) * num_splits + s;
                partial_m[pidx] = -INFINITY;
                partial_l[pidx] = 0.0f;
            }
            for (int d = tid; d < head_dim; d += blockDim.x) {
                int64_t poff = (((int64_t)b * heads + h_q) * num_splits + s)
                              * (int64_t)head_dim + d;
                partial_o[poff] = 0.0f;
            }
        }
        return;
    }

    // Load Q for all `group_size` heads in this KV group. Pad unused
    // M-rows with zeros (they contribute zero scores → become -inf
    // after the row-mask + softmax).
    for (int m = 0; m < kWmmaM; ++m) {
        if (m < group_size) {
            const int h_q = q_head_base + m;
            const T* q_row = q + (int64_t)b * q_b_stride
                               + (int64_t)h_q * q_h_stride;
            for (int d = tid; d < head_dim; d += blockDim.x) {
                sQ[m * head_dim + d] = q_row[d];
            }
        } else {
            for (int d = tid; d < head_dim; d += blockDim.x) {
                sQ[m * head_dim + d] = tc::ToHalf<T>::cvt(0.0f);
            }
        }
    }

    // Initialize sO to zero (we'll accumulate across K sub-tiles).
    for (int i = tid; i < kWmmaM * head_dim; i += blockDim.x) {
        sO[i] = 0.0f;
    }
    // Initialize row stats — running online softmax over the chunk.
    if (tid < kWmmaM) {
        sMaxRow[tid] = -INFINITY;
        sSumRow[tid] = 0.0f;
    }
    __syncthreads();

    const T* k_bh = k + (int64_t)b * k_b_stride + (int64_t)h_kv * k_h_stride;
    const T* v_bh = v + (int64_t)b * v_b_stride + (int64_t)h_kv * v_h_stride;

    // ==========================================================================
    // Pass 1 — compute all chunk_len scores into sScores via WMMA mma.
    // Stream K through SMEM in N-tiles of width kWmmaN = 16.
    // ==========================================================================
    //
    // For each N-tile (16 K-rows):
    //   1. Coop-load sK_tile [16, head_dim].
    //   2. For each warp w: compute fragment of S[0..16, w*kWmmaN..(w+1)*kWmmaN].
    //      Wait — N-tile width is 16; with 4 warps each warp would handle
    //      4 N-cols. Tiny. Instead let each warp process a different N-tile.
    //
    // Strategy: 4 warps × 1 N-tile each per outer iteration → 4 N-tiles per
    // outer step → chunk_len / 64 outer steps for chunk_len = 256 → 4 steps.
    //
    // Each warp does the full QK^T mma for ONE [16, 16] N-tile, walking
    // (head_dim / kWmmaK) K-tiles in the reduction direction.

    for (int n_base = 0; n_base < chunk_len; n_base += num_warps * kWmmaN) {
        const int n_warp = n_base + warp_id * kWmmaN;
        const bool warp_active = (n_warp < chunk_len);

        // Coop-load 4 K sub-tiles (one per warp) cooperatively into
        // sK_tile[warp_id][16, head_dim]. We split sK_tile across warps
        // by reusing the same SMEM with offset. Actually: simpler —
        // allocate sK_tile as [num_warps × kWmmaN × head_dim] mentally
        // and load per warp. SMEM budget allows this if kWmmaN=16,
        // num_warps=4: 4*16*128*2 = 16 KB ✓.
        //
        // For simplicity in this Tier-1 cut, use the SAME sK_tile for
        // all warps (one N-tile at a time per warp). Outer loop iterates
        // num_warps × kWmmaN at a time but each warp processes one
        // N-tile from the shared sK_tile sequentially.
        //
        // Actually that adds num_warps × syncs. Cleaner: allocate
        // sK_tile slice per warp.

        for (int w_inner = 0; w_inner < num_warps; ++w_inner) {
            const int n_tile_start = n_base + w_inner * kWmmaN;
            if (n_tile_start >= chunk_len) break;

            // Coop-load this N-tile into sK_tile.
            for (int i = tid; i < kWmmaN * head_dim; i += blockDim.x) {
                int row = i / head_dim;
                int d   = i % head_dim;
                int k_abs = n_tile_start + row;
                if (k_abs < k_end) {
                    sK_tile[row * head_dim + d] =
                        k_bh[(int64_t)k_abs * k_seq_stride + d];
                } else {
                    sK_tile[row * head_dim + d] = tc::ToHalf<T>::cvt(0.0f);
                }
            }
            __syncthreads();

            // Each warp computes QK^T for ITS assigned N-tile (= the
            // currently-loaded sK_tile). All warps reduce over the K
            // (head_dim) direction.
            //
            // Wait — re-reading: we WANT each warp to process a
            // DIFFERENT N-tile in parallel, not the same one.
            // Simpler reorganization: 1 outer K-load is shared by all
            // warps, all warps compute the SAME N-tile of S, then we
            // shift. That wastes warps. Skip.
            //
            // Correct approach: warp `warp_id` owns N-tile
            // (n_base + warp_id * kWmmaN). Each warp loads its OWN
            // sK_tile slice — needs num_warps separate slices.
            //
            // OK rewriting below with the right SMEM layout. For
            // now: only warp `w_inner` does mma; others idle. Suboptimal
            // but correct.
            if (warp_id == w_inner && warp_active && n_tile_start == n_warp) {
                wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, T,
                               wmma::row_major> q_frag;
                wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, T,
                               wmma::col_major> k_frag;
                wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_frag;
                wmma::fill_fragment(c_frag, 0.0f);

                for (int kk = 0; kk < head_dim; kk += kWmmaK) {
                    // A = sQ[0..M, kk..kk+K] row-major, ld = head_dim
                    wmma::load_matrix_sync(q_frag, sQ + kk, head_dim);
                    // B = K^T[kk..kk+K, n_warp..n_warp+N]. K stored row-
                    // major as [kWmmaN, head_dim] = K[k_abs, d]. To get
                    // K^T as col-major, we read sK_tile row-major and
                    // use the col_major layout tag — wmma reads it as
                    // K[d, kk+...] effectively transposed. The ld for
                    // col-major B is head_dim (the stride between rows
                    // of K).
                    wmma::load_matrix_sync(k_frag, sK_tile + kk, head_dim);
                    wmma::mma_sync(c_frag, q_frag, k_frag, c_frag);
                }

                // Store C [16, 16] into sScores at the right N-offset.
                // We can write directly via store_matrix_sync to the
                // sScores buffer at column n_warp.
                wmma::store_matrix_sync(
                    &sScores[0 * kChunkK + n_warp],
                    c_frag, kChunkK, wmma::mem_row_major);
            }
            __syncthreads();
        }
    }

    // Apply scale + chunk-tail mask (sScores beyond chunk_len → -inf).
    for (int i = tid; i < kWmmaM * kChunkK; i += blockDim.x) {
        int m = i / kChunkK;
        int n = i % kChunkK;
        if (n < chunk_len && m < group_size) {
            sScores[i] *= scale;
        } else {
            sScores[i] = -INFINITY;
        }
    }
    __syncthreads();

    // Per-row softmax over [kWmmaM, chunk_len]. Each warp owns ONE row
    // (only group_size rows are meaningful; the rest produce -inf).
    //
    // Use block_reduce-style helpers per row. With 4 warps and 16 rows,
    // each warp handles 4 rows sequentially.
    for (int m_local = warp_id; m_local < kWmmaM; m_local += num_warps) {
        // Phase 1: row max via warp-shuffle reduce.
        float row_max = -INFINITY;
        for (int n = lane; n < chunk_len; n += 32) {
            float v = sScores[m_local * kChunkK + n];
            if (v > row_max) row_max = v;
        }
        #pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1) {
            float other = __shfl_xor_sync(0xffffffff, row_max, delta, 32);
            if (other > row_max) row_max = other;
        }
        // Phase 2: row sum of exp(s - row_max).
        float row_sum = 0.0f;
        for (int n = lane; n < chunk_len; n += 32) {
            float p = expf(sScores[m_local * kChunkK + n] - row_max);
            sScores[m_local * kChunkK + n] = p;
            row_sum += p;
        }
        #pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1) {
            row_sum += __shfl_xor_sync(0xffffffff, row_sum, delta, 32);
        }
        if (lane == 0) {
            sMaxRow[m_local] = row_max;
            sSumRow[m_local] = row_sum;
        }
    }
    __syncthreads();

    // ==========================================================================
    // Pass 2 — accumulate sO = P @ V via WMMA mma.
    // ==========================================================================
    //
    // P is in sScores [kWmmaM, chunk_len]. V is in global, streamed
    // through sV_tile in K-direction sub-tiles of kWmmaK = 16 K-rows.
    // For each K sub-tile: each warp accumulates 1 N-tile of output
    // (kWmmaN = 16 D-elements).
    //
    // For head_dim = 128: 128 / 16 = 8 N-tiles to cover. 4 warps × 2
    // N-tiles each per outer step → 1 outer iteration per K sub-tile.

    // We need sP in T (half/bf16) for the WMMA A fragment. Convert
    // sScores → a half-precision buffer. Reuse sK_tile as sP storage
    // for the conversion (sK_tile is at least kWmmaN × head_dim ≥
    // 16 × 128 = 2048 elements; we need kWmmaM × chunk_len = 16 × 256
    // = 4096. Won't fit).
    //
    // Allocate sP separately. SMEM was already at ~36 KB; add another
    // 16*256*2 = 8 KB → 44 KB. Tight but fits in 48 KB.

    __shared__ T sP[kWmmaM * kChunkK];

    for (int i = tid; i < kWmmaM * kChunkK; i += blockDim.x) {
        sP[i] = tc::ToHalf<T>::cvt(sScores[i]);
    }
    __syncthreads();

    // For each K sub-tile in the chunk, all warps cooperatively load
    // sV_tile, then each warp accumulates its N-tile of the output.
    const int n_tiles_per_d = head_dim / kWmmaN;     // 8 for D=128
    const int n_tiles_per_warp = (n_tiles_per_d + num_warps - 1) / num_warps;

    for (int k_sub = 0; k_sub < chunk_len; k_sub += kWmmaK) {
        const int rows_to_load = min(kWmmaK, chunk_len - k_sub);

        // Coop-load V sub-tile [kWmmaK, head_dim].
        for (int i = tid; i < kWmmaK * head_dim; i += blockDim.x) {
            int row = i / head_dim;
            int d   = i % head_dim;
            int k_abs = k_start + k_sub + row;
            if (row < rows_to_load && k_abs < k_end) {
                sV_tile[row * head_dim + d] =
                    v_bh[(int64_t)k_abs * v_seq_stride + d];
            } else {
                sV_tile[row * head_dim + d] = tc::ToHalf<T>::cvt(0.0f);
            }
        }
        __syncthreads();

        // Each warp processes its assigned N-tile(s).
        for (int n_idx = 0; n_idx < n_tiles_per_warp; ++n_idx) {
            const int n_tile = warp_id + n_idx * num_warps;
            if (n_tile >= n_tiles_per_d) break;
            const int d_base = n_tile * kWmmaN;

            wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, T,
                           wmma::row_major> p_frag;
            wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, T,
                           wmma::row_major> v_frag;
            wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> o_frag;

            // Load existing sO accumulator for this [M, n_tile] block.
            wmma::load_matrix_sync(
                o_frag, &sO[0 * head_dim + d_base], head_dim,
                wmma::mem_row_major);

            // P fragment: sP[0..M, k_sub..k_sub+K]. ld = kChunkK.
            wmma::load_matrix_sync(p_frag, sP + k_sub, kChunkK);
            // V fragment: sV_tile[0..K, d_base..d_base+N], row_major, ld = head_dim.
            wmma::load_matrix_sync(v_frag, sV_tile + d_base, head_dim);
            wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);

            // Store back to sO.
            wmma::store_matrix_sync(
                &sO[0 * head_dim + d_base], o_frag, head_dim,
                wmma::mem_row_major);
        }
        __syncthreads();
    }

    // ==========================================================================
    // Pass 3 — write partials. Each of the `group_size` Q heads gets
    // its own (m, l, o[D]) tuple in workspace, indexed by the Q-head
    // ID. Padded M-rows (m >= group_size) are not written.
    // ==========================================================================
    for (int g = 0; g < group_size; ++g) {
        const int h_q = q_head_base + g;
        if (tid == 0) {
            int64_t pidx = ((int64_t)b * heads + h_q) * num_splits + s;
            partial_m[pidx] = sMaxRow[g];
            partial_l[pidx] = sSumRow[g];
        }
        for (int d = tid; d < head_dim; d += blockDim.x) {
            int64_t poff = (((int64_t)b * heads + h_q) * num_splits + s)
                          * (int64_t)head_dim + d;
            partial_o[poff] = sO[g * head_dim + d];
        }
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

// TC (tensor-core / WMMA) dispatch — DISABLED in single-batch decode.
//
// The WMMA kernel `flash_decoding_split_kernel_tc` below is preserved
// for reference. Empirical benchmark (RTX 4070, 2026-06-04) showed it
// LOSES to the warp-cooperative SIMT kernel at every tested GQA shape:
//
//   Shape (Hq, Hkv, K)         TC       SIMT-GQA   Winner
//   llama3-8b   (32, 8, 4096)  229µs    78µs       SIMT 2.94×
//   llama3-8b   (32, 8, 8192)  444µs    137µs      SIMT 3.24×
//   llama3-70b  (64, 8, 4096)  231µs    132µs      SIMT 1.75×
//   llama3-70b  (64, 8, 8192)  448µs    252µs      SIMT 1.78×
//   qwen2-14b   (32, 4, 8192)  231µs    134µs      SIMT 1.73×
//
// Why TC loses (analyzed in commit message of the Phase 73-followup
// GQA work):
//
//   1. Decode is bandwidth/L2-bound at the tested shapes. Tensor cores
//      attack compute throughput — useless for a non-compute-bound
//      workload.
//
//   2. TC grid is (num_splits, H_kv, B); SIMT grid is (num_splits,
//      H_q, B). With B=1 and H_kv ≤ H_q the TC grid undersaturates
//      RTX 4070's 36 SMs (e.g. 32 blocks for H_kv=8, K=1024). The
//      SIMT grid has group_size× more blocks → fully saturates SMs
//      and amortizes per-block fixed costs better.
//
//   3. TC kernel adds fp32→fp16→fp32 round-trips (sScores fp32 →
//      sP fp16 fragment-load → mma → sO fp32) that the SIMT path
//      doesn't have.
//
//   4. The WMMA M-tile is 16 rows; at single-batch decode, only
//      `group_size` rows are meaningful (4-8 for Llama 3). The
//      remaining 8-12 M-rows are pure padding — tensor cores process
//      them but their output is thrown away. Throughput penalty
//      proportional to (16 - group_size) / 16.
//
// The TC kernel would compete (and likely win) at MULTI-BATCH decode
// (B ≥ 8) where the M-tile fills with B × group_size rows. That
// workload is owned by `BatchPagedDecodePlan` (Phase 46, FlashInfer
// vendored) — a different op family with explicit paged-KV layout.
//
// Keeping the WMMA kernel code in this file: it compiles, smoke tests
// pass, and the design serves as a worked example for future
// optimizers. If a multi-batch contig decode plan is added later, the
// kernel + dispatch can be re-enabled by changing the body of this
// function. Until then it's documented dead code.
__host__ inline bool flash_decoding_should_use_tc(
    int32_t /*group_size*/, int32_t /*head_dim*/,
    int32_t /*batch*/, int32_t /*num_kv_heads*/, int32_t /*num_splits*/)
{
    return false;
}

template <typename T>
__host__ inline int32_t launch_flash_decoding(
    const T* q, const T* k, const T* v, T* y,
    void* workspace, size_t workspace_bytes,
    int32_t batch, int32_t heads, int32_t num_kv_heads,
    int32_t k_len, int32_t head_dim,
    int64_t q_b_stride, int64_t q_h_stride,
    int64_t k_b_stride, int64_t k_h_stride, int64_t k_seq_stride,
    int64_t v_b_stride, int64_t v_h_stride, int64_t v_seq_stride,
    int64_t y_b_stride, int64_t y_h_stride,
    float scale,
    cudaStream_t stream)
{
    if (batch <= 0 || heads <= 0 || num_kv_heads <= 0 || head_dim <= 0) return 2;
    if (heads % num_kv_heads != 0) return 2;
    if (head_dim > kMaxD) return 3;
    if (k_len <= 0) {
        // No KV → write zeros + bail.
        // Caller is expected to zero-init y; nothing to do here.
        return 0;
    }

    const int32_t group_size = heads / num_kv_heads;

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

    dim3 block(kThreadsPerBlock);

    if (flash_decoding_should_use_tc(
            group_size, head_dim, batch, num_kv_heads, num_splits))
    {
        // TC path — one block per (split, h_kv, b). Each block batches
        // all group_size Q heads into the WMMA M-tile.
        dim3 grid_split((unsigned)num_splits, (unsigned)num_kv_heads, (unsigned)batch);
        flash_decoding_split_kernel_tc<T><<<grid_split, block, 0, stream>>>(
            q, k, v, partial_m, partial_l, partial_o,
            batch, heads, k_len, head_dim, num_splits,
            num_kv_heads, group_size,
            q_b_stride, q_h_stride,
            k_b_stride, k_h_stride, k_seq_stride,
            v_b_stride, v_h_stride, v_seq_stride,
            scale);
    } else {
        // SIMT path — one block per (split, h_q, b). Each block handles
        // a single Q head; GQA broadcast handled via integer division
        // h_q / group_size inside the kernel.
        dim3 grid_split((unsigned)num_splits, (unsigned)heads, (unsigned)batch);
        flash_decoding_split_kernel<T><<<grid_split, block, 0, stream>>>(
            q, k, v, partial_m, partial_l, partial_o,
            batch, heads, k_len, head_dim, num_splits, group_size,
            q_b_stride, q_h_stride,
            k_b_stride, k_h_stride, k_seq_stride,
            v_b_stride, v_h_stride, v_seq_stride,
            scale);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + (int32_t)err;

    // Combine kernel — same shape (per Q head) regardless of which split
    // kernel ran.
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
        int32_t batch, int32_t heads, int32_t num_kv_heads,                                         \
        int32_t k_len, int32_t head_dim,                                                            \
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
            batch, heads, num_kv_heads, k_len, head_dim,                                            \
            q_b_stride, q_h_stride,                                                                 \
            k_b_stride, k_h_stride, k_seq_stride,                                                   \
            v_b_stride, v_h_stride, v_seq_stride,                                                   \
            y_b_stride, y_h_stride,                                                                 \
            scale, stream);                                                                         \
    }                                                                                               \
    extern "C" int32_t baracuda_kernels_ ## NAME ## _can_implement(                                 \
        int32_t batch, int32_t heads, int32_t num_kv_heads,                                         \
        int32_t k_len, int32_t head_dim)                                                            \
    {                                                                                               \
        if (batch <= 0 || heads <= 0 || num_kv_heads <= 0 || head_dim <= 0) return 2;               \
        if (heads % num_kv_heads != 0) return 2;                                                    \
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
