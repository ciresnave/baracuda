// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Shared kernel body for binary (B1) GEMM, RRR layout, sm_80+.
//
// Identity-only — no α / β / bias / activation chain.
//
// Computes
//   D[i,j] = sum_k popcount( A[i, k_byte] XOR B_col_packed_byte_k(j) )
// where the kernel re-packs `B[k, j]` from N-pair-packed gmem storage
// into K-pair-packed smem storage before the MMA — see the
// novel-mechanic note below.
//
// Layout `RRR`
//   A : row-major  [M, K bits], stride lda_bytes along K (= K/8 bytes / row,
//                                                         bit-packed along K).
//   B : row-major  [K bits, N], stride ldb_bytes along N (= N/8 bytes / row,
//                                                         bit-packed along N).
//   D : row-major  [M, N i32],  stride ldd_elements along N — **int32 output**.
//
// `lda_bytes` / `ldb_bytes` are in bytes (= packed-bit storage slots,
// 8 bits / byte). `ldd_elements` is in **i32 element count** (not
// bytes). `M`, `N`, `K` are in **element** count.
//
// **Kernel requirements:**
//   - `K` divisible by 8 (packing is byte-aligned along K).
//   - `N` divisible by 8 (packing is byte-aligned along N for B in
//     gmem). The kernel could in principle handle N not divisible by 8
//     by zero-padding the last B byte's high bits, but the smoke test
//     and plan layer enforce divisibility for simplicity.
//
// **Novel mechanic for bin RRR**: B in gmem is bit-packed along N (a
// byte holds 8 N-cols of the same K-row), but the kernel needs B in
// smem K-pair-packed within each output column (the layout the MMA
// fragment expects, matching the RCR case). The load therefore
// **gathers** 8 bits from 8 different gmem K-row bytes — one bit per
// gmem byte read — into one K-pair smem byte per output column:
//
//     for each output column j (= col0 + col_in_tile):
//       smem_B[col][kk_byte] = OR over b ∈ [0..8) of
//         (((gmem byte at K=2*kk_byte + ... wait, no — bin K_TILE_BYTES = 32,
//           so kk_byte indexes K_BYTES not pairs; each byte holds 8 K-bits)
//          bit (col_g & 7) of gmem byte at offset
//          ((kt + 8*kk_byte + b) * ldb_bytes + (col_g >> 3)))
//          shifted into bit position `b`.
//
// Cost: 8 gmem byte reads per smem byte vs 1 for RCR (and 2 nibbles
// per smem byte for int4 RRR). Bandwidth-heavy but simple and
// correct; the bin RRR use case is rare so optimization is
// future work.
//
// Once smem_B is built, the MMA fragment byte arithmetic is identical
// to the RCR header — both use K-pair-packed smem.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "baracuda_bin_common.cuh"  // mma_m16n8k256_xor_popc — reused

namespace baracuda {

namespace bin_rrr_sm89 {

constexpr int M_TILE = 64;
constexpr int N_TILE = 64;
constexpr int K_TILE = 256;
constexpr int K_TILE_BYTES = K_TILE / 8;  // 32 bytes per row of smem

constexpr int M_MMA = 16;
constexpr int N_MMA = 8;
constexpr int K_MMA = 256;

constexpr int NUM_WARPS = 2;
constexpr int NUM_THREADS = NUM_WARPS * 32;

constexpr int M_WARP = M_TILE / NUM_WARPS;   // 32
constexpr int N_WARP = N_TILE;               // 64
constexpr int M_ITER = M_WARP / M_MMA;       // 2
constexpr int N_ITER = N_WARP / N_MMA;       // 8

__global__ void gemm_bin_rrr_sm89_kernel(
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

    for (int kt = 0; kt < K; kt += K_TILE) {
        // gmem → smem: A tile. Identical to the RCR header — A is row-
        // major in both layouts and bit-packed along K.
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

        // gmem → smem: B tile (NOVEL — see header comment). For each
        // output column, gather 8 bits from 8 K-row bytes in gmem into
        // one K-pair smem byte.
        {
            const int col = tid;
            if (col < N_TILE && (col0 + col) < N) {
                const int col_g = col0 + col;
                const int col_byte = col_g >> 3;     // gmem byte index along N
                const int col_bit  = col_g & 7;      // bit position within byte
                #pragma unroll
                for (int kk = 0; kk < K_TILE_BYTES; ++kk) {
                    const int k_base = kt + 8 * kk;
                    uint8_t out = 0;
                    #pragma unroll
                    for (int b = 0; b < 8; ++b) {
                        const int k_elem = k_base + b;
                        if (k_elem < K) {
                            uint8_t g =
                                B[(int64_t)k_elem * ldb_bytes + col_byte];
                            out |= (uint8_t)(((g >> col_bit) & 1u) << b);
                        }
                    }
                    smem_B[col][kk] = out;
                }
            } else if (col < N_TILE) {
                #pragma unroll
                for (int kk = 0; kk < K_TILE_BYTES; ++kk) {
                    smem_B[col][kk] = (uint8_t)0;
                }
            }
        }

        __syncthreads();

        // MMA pass — identical to the RCR header (smem layout matches).
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

    // Epilogue — identical to the RCR header (D is plain int32 in both).
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

} // namespace bin_rrr_sm89

// Host-side launcher — mirrors `launch_gemm_bin_rcr_sm89`.
inline int32_t launch_gemm_bin_rrr_sm89(
    int32_t m, int32_t n, int32_t k,
    const void *a, int64_t lda_bytes,
    const void *b, int64_t ldb_bytes,
    void *d, int64_t ldd_elements,
    void *stream
) {
    using namespace bin_rrr_sm89;
    if (m <= 0 || n <= 0 || k <= 0) return 2;
    if ((k & 7) != 0) return 3;   // K must be divisible by 8
    if ((n & 7) != 0) return 3;   // N must be divisible by 8 (B is N-bit-packed in gmem)

    dim3 grid((unsigned)((n + N_TILE - 1) / N_TILE),
              (unsigned)((m + M_TILE - 1) / M_TILE),
              1u);
    dim3 block((unsigned)NUM_THREADS, 1u, 1u);

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    gemm_bin_rrr_sm89_kernel<<<grid, block, 0, s>>>(
        m, n, k,
        reinterpret_cast<const uint8_t *>(a), lda_bytes,
        reinterpret_cast<const uint8_t *>(b), ldb_bytes,
        reinterpret_cast<int32_t *>(d), ldd_elements
    );

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

} // namespace baracuda
