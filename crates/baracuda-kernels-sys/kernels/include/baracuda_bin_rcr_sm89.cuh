// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Shared kernel body for binary (B1) GEMM, RCR layout, sm_80+.
//
// Identity-only — no α / β / bias / activation chain. The popcount-
// based programming model doesn't have a meaningful place for them
// (the output is the raw popcount-of-XOR sum, an int32 accumulator;
// callers can post-process externally if they want a thresholded b1
// result or a "binary dot product" K - 2 * popcount).
//
// Computes
//   D[i,j] = sum_k popcount( A[i, k_byte] XOR B[k_byte, j] )   (k_byte in 0..K/8)
//
// Layout `RCR`
//   A : row-major  [M, K bits], stride lda_bytes along K (= K/8 bytes / row)
//   B : col-major  [K, N bits], stride ldb_bytes along K (= K/8 bytes / col)
//   D : row-major  [M, N],      stride ldd_elements along N — **int32 output**
//
// `lda_bytes` / `ldb_bytes` are in bytes (= packed-bit storage slots,
// 8 bits / byte). `ldd_elements` is in **i32 element count** (not bytes)
// — D is a plain int32 matrix, no packing.
//
// `M`, `N`, `K` are in **element** count. The kernel requires:
//   - `K` divisible by 8 (packing is byte-aligned along K).
// No constraint on M or N (sub-tiles padded with zeros internally).
//
// Tile / thread layout — mirrors the int4 RCR template byte arithmetic
// (K_TILE_BYTES = 32 = K_MMA / 8 = 256 / 8):
//   M_TILE=64 elements, N_TILE=64 elements, K_TILE=256 elements (=32 bytes).
//   2 warps × 32 threads. smem_A[M_TILE][K_TILE_BYTES],
//   smem_B[N_TILE][K_TILE_BYTES].
//
// Per-thread MMA register layout for `mma.sync m16n8k256.b1` (PTX 7.0+):
//   A : 4 b32 regs/thread; each b32 = 4 K-adjacent bytes = 32 b1 elements.
//       Same byte arithmetic as m16n8k64.s4 and m16n8k32.s8 — only the
//       K-elements-per-byte differs.
//   B : 2 b32 regs/thread; each b32 = 4 K-adjacent bytes = 32 b1 elements.
//   D : 4 s32 regs/thread (raw popcount accumulator).

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "baracuda_bin_common.cuh"  // mma_m16n8k256_xor_popc

namespace baracuda {

namespace bin_rcr_sm89 {

constexpr int M_TILE = 64;        // elements
constexpr int N_TILE = 64;        // elements
constexpr int K_TILE = 256;       // elements (= K_MMA)
constexpr int K_TILE_BYTES = K_TILE / 8;  // 32 bytes per row of smem

constexpr int M_MMA = 16;         // elements
constexpr int N_MMA = 8;          // elements
constexpr int K_MMA = 256;        // elements (m16n8k256 → K is 256 b1 elements)

constexpr int NUM_WARPS = 2;
constexpr int NUM_THREADS = NUM_WARPS * 32;

constexpr int M_WARP = M_TILE / NUM_WARPS;   // 32
constexpr int N_WARP = N_TILE;               // 64
constexpr int M_ITER = M_WARP / M_MMA;       // 2
constexpr int N_ITER = N_WARP / N_MMA;       // 8

__global__ void gemm_bin_rcr_sm89_kernel(
    int M, int N, int K,
    const uint8_t * __restrict__ A, int64_t lda_bytes,
    const uint8_t * __restrict__ B, int64_t ldb_bytes,
    int32_t * __restrict__ D, int64_t ldd_elements
) {
    const int bm = blockIdx.y;
    const int bn = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;

    const int row0 = bm * M_TILE;
    const int col0 = bn * N_TILE;

    __shared__ uint8_t smem_A[M_TILE][K_TILE_BYTES];
    __shared__ uint8_t smem_B[N_TILE][K_TILE_BYTES];

    int32_t acc[M_ITER][N_ITER][4];
    #pragma unroll
    for (int mi = 0; mi < M_ITER; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < N_ITER; ++ni) {
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc[mi][ni][r] = 0;
        }
    }

    // K loop in element units; each tile is K_TILE elements = K_TILE_BYTES bytes.
    for (int kt = 0; kt < K; kt += K_TILE) {
        // gmem → smem: A tile. Each thread loads one row of K_TILE_BYTES bytes.
        {
            const int row = tid;
            if (row < M_TILE && (row0 + row) < M) {
                const uint8_t *src =
                    &A[(int64_t)(row0 + row) * lda_bytes + (int64_t)(kt / 8)];
                #pragma unroll
                for (int kk = 0; kk < K_TILE_BYTES; ++kk) {
                    int k_elem = kt + 8 * kk;
                    smem_A[row][kk] = (k_elem < K) ? src[kk] : (uint8_t)0;
                }
            } else if (row < M_TILE) {
                #pragma unroll
                for (int kk = 0; kk < K_TILE_BYTES; ++kk) {
                    smem_A[row][kk] = (uint8_t)0;
                }
            }
        }

        // gmem → smem: B tile (col-major K-contig in gmem). Stash each
        // column unchanged into a row of smem_B.
        {
            const int col = tid;
            if (col < N_TILE && (col0 + col) < N) {
                const uint8_t *src =
                    &B[(int64_t)(col0 + col) * ldb_bytes + (int64_t)(kt / 8)];
                #pragma unroll
                for (int kk = 0; kk < K_TILE_BYTES; ++kk) {
                    int k_elem = kt + 8 * kk;
                    smem_B[col][kk] = (k_elem < K) ? src[kk] : (uint8_t)0;
                }
            } else if (col < N_TILE) {
                #pragma unroll
                for (int kk = 0; kk < K_TILE_BYTES; ++kk) {
                    smem_B[col][kk] = (uint8_t)0;
                }
            }
        }

        __syncthreads();

        // MMA pass — byte-arithmetic mirrors m16n8k32.s8 / m16n8k64.s4;
        // only the K-elements-per-byte interpretation differs.
        #pragma unroll
        for (int mi = 0; mi < M_ITER; ++mi) {
            const int frag_m0 = warp_id * M_WARP + mi * M_MMA;

            const int ar = lane / 4;
            const int ac = (lane % 4) * 4;

            const uint32_t a0 =
                *reinterpret_cast<const uint32_t *>(&smem_A[frag_m0 + ar][ac]);
            const uint32_t a1 =
                *reinterpret_cast<const uint32_t *>(&smem_A[frag_m0 + ar + 8][ac]);
            const uint32_t a2 =
                *reinterpret_cast<const uint32_t *>(&smem_A[frag_m0 + ar][ac + 16]);
            const uint32_t a3 =
                *reinterpret_cast<const uint32_t *>(&smem_A[frag_m0 + ar + 8][ac + 16]);

            #pragma unroll
            for (int ni = 0; ni < N_ITER; ++ni) {
                const int bk0 = (lane % 4) * 4;
                const int bn0 = lane / 4;
                const int frag_n0 = ni * N_MMA;

                const uint32_t b0 =
                    *reinterpret_cast<const uint32_t *>(&smem_B[frag_n0 + bn0][bk0]);
                const uint32_t b1 =
                    *reinterpret_cast<const uint32_t *>(&smem_B[frag_n0 + bn0][bk0 + 16]);

                mma_m16n8k256_xor_popc(
                    a0, a1, a2, a3,
                    b0, b1,
                    acc[mi][ni][0], acc[mi][ni][1],
                    acc[mi][ni][2], acc[mi][ni][3]);
            }
        }

        __syncthreads();
    }

    // Epilogue: direct int32 store, no sat-cast, no packing.
    //
    // D-fragment register layout per thread (m16n8 s32 — same as
    // int4/int8 epilogue):
    //   d0 = D_frag[lane/4    ][(lane%4)*2    ]
    //   d1 = D_frag[lane/4    ][(lane%4)*2 + 1]
    //   d2 = D_frag[lane/4 + 8][(lane%4)*2    ]
    //   d3 = D_frag[lane/4 + 8][(lane%4)*2 + 1]
    #pragma unroll
    for (int mi = 0; mi < M_ITER; ++mi) {
        const int frag_m0 = warp_id * M_WARP + mi * M_MMA;
        #pragma unroll
        for (int ni = 0; ni < N_ITER; ++ni) {
            const int frag_n0 = ni * N_MMA;
            const int r = lane / 4;
            const int c = (lane % 4) * 2;

            const int row_top = row0 + frag_m0 + r;
            const int row_bot = row0 + frag_m0 + r + 8;
            const int col_l   = col0 + frag_n0 + c;
            const int col_r   = col0 + frag_n0 + c + 1;

            if (row_top < M && col_l < N)
                D[(int64_t)row_top * ldd_elements + col_l] = acc[mi][ni][0];
            if (row_top < M && col_r < N)
                D[(int64_t)row_top * ldd_elements + col_r] = acc[mi][ni][1];
            if (row_bot < M && col_l < N)
                D[(int64_t)row_bot * ldd_elements + col_l] = acc[mi][ni][2];
            if (row_bot < M && col_r < N)
                D[(int64_t)row_bot * ldd_elements + col_r] = acc[mi][ni][3];
        }
    }
}

} // namespace bin_rcr_sm89

// Host-side launcher.
inline int32_t launch_gemm_bin_rcr_sm89(
    int32_t m, int32_t n, int32_t k,
    const void *a, int64_t lda_bytes,
    const void *b, int64_t ldb_bytes,
    void *d, int64_t ldd_elements,
    void *stream
) {
    using namespace bin_rcr_sm89;
    if (m <= 0 || n <= 0 || k <= 0) return 2;
    if ((k & 7) != 0) return 3;   // K must be divisible by 8

    dim3 grid((unsigned)((n + N_TILE - 1) / N_TILE),
              (unsigned)((m + M_TILE - 1) / M_TILE),
              1u);
    dim3 block((unsigned)NUM_THREADS, 1u, 1u);

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    gemm_bin_rcr_sm89_kernel<<<grid, block, 0, s>>>(
        m, n, k,
        reinterpret_cast<const uint8_t *>(a), lda_bytes,
        reinterpret_cast<const uint8_t *>(b), ldb_bytes,
        reinterpret_cast<int32_t *>(d), ldd_elements
    );

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

} // namespace baracuda
