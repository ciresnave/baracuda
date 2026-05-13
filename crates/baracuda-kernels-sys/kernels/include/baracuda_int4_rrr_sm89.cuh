// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Shared templated kernel body for int4 RRR GEMM on sm_89.
//
// Parameters mirror `baracuda_int4_rcr_sm89.cuh`:
//   Enc       : `Int4Encoding::S4` or `::U4`.
//   BiasT     : `float` or `int32_t`. Ignored when `Act == None`.
//   Act       : Activation / bias selector.
//
// Layout `RRR`
//   A : row-major  [M, K], stride lda_bytes along K (K elements per row
//                                                   packed as K/2 bytes).
//   B : row-major  [K, N], stride ldb_bytes along N (N elements per row
//                                                   packed as N/2 bytes,
//                                                   pair-packed along N).
//   C : row-major  [M, N], stride ldc_bytes along N (packed along N).
//   D : row-major  [M, N], stride ldd_bytes along N (packed along N).
//
// Same byte-counted FFI as the RCR header.
//
// **Novel mechanic for int4 RRR**: B in gmem is pair-packed along N
// (a byte holds B[k, 2j] in the low nibble and B[k, 2j+1] in the high
// nibble), but the kernel's MMA fragment wants B in smem pair-packed
// along K within each output column (the same K-contig packing the RCR
// path produces directly from col-major gmem). The load therefore
// **gathers** two nibbles from two different K-row bytes in gmem and
// re-packs them into one K-pair byte in smem:
//
//     smem_B[col][kk_byte] = lo_nib(gmem at (K=2*kk_byte    , col))
//                          | hi_nib(gmem at (K=2*kk_byte + 1, col)) << 4
//
// where `lo_nib` / `hi_nib` of a gmem byte are chosen based on the
// output column's parity (`col_g & 1`: 0 → low nibble, 1 → high
// nibble). This costs 2 gmem byte reads per smem byte (vs 1 in the RCR
// path).
//
// Once smem_B is built, the MMA fragment byte arithmetic is identical
// to the RCR header — both use K-contig packed-pair smem layout.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "baracuda_dtype.cuh"
#include "baracuda_epilogue_int8.cuh"
#include "baracuda_int4_rcr_sm89.cuh"  // Int4Encoding, mma_m16n8k64_s32_int4,
                                       // sat_cast_int4_from_f32,
                                       // unpack_int4_byte_to_s32 — reused.

namespace baracuda {

namespace int4_rrr_sm89 {

constexpr int M_TILE = 64;
constexpr int N_TILE = 64;
constexpr int K_TILE = 64;
constexpr int K_TILE_BYTES = K_TILE / 2;

constexpr int M_MMA = 16;
constexpr int N_MMA = 8;
constexpr int K_MMA = 64;

constexpr int NUM_WARPS = 2;
constexpr int NUM_THREADS = NUM_WARPS * 32;

constexpr int M_WARP = M_TILE / NUM_WARPS;   // 32
constexpr int N_WARP = N_TILE;               // 64
constexpr int M_ITER = M_WARP / M_MMA;       // 2
constexpr int N_ITER = N_WARP / N_MMA;       // 8

template <Int4Encoding Enc, typename BiasT, Activation Act>
__global__ void gemm_int4_rrr_sm89_kernel(
    int M, int N, int K,
    const uint8_t * __restrict__ A, int64_t lda_bytes,
    const uint8_t * __restrict__ B, int64_t ldb_bytes,
    const uint8_t * __restrict__ C, int64_t ldc_bytes,
    uint8_t * __restrict__ D, int64_t ldd_bytes,
    const BiasT * __restrict__ bias,
    float alpha, float beta
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
        // major in both layouts and pair-packed along K.
        {
            const int row = tid;
            if (row < M_TILE && (row0 + row) < M) {
                const uint8_t *src =
                    &A[(int64_t)(row0 + row) * lda_bytes + (int64_t)(kt / 2)];
                #pragma unroll
                for (int kk = 0; kk < K_TILE_BYTES; ++kk) {
                    int k_elem = kt + 2 * kk;
                    smem_A[row][kk] = (k_elem < K) ? src[kk] : (uint8_t)0;
                }
            } else if (row < M_TILE) {
                #pragma unroll
                for (int kk = 0; kk < K_TILE_BYTES; ++kk) {
                    smem_A[row][kk] = (uint8_t)0;
                }
            }
        }

        // gmem → smem: B tile (NOVEL — see header comment). B is row-
        // major in gmem and pair-packed along N; we gather two nibbles
        // from two K-row bytes to assemble one K-pair smem byte per
        // output column.
        {
            const int col = tid;                       // column in tile
            if (col < N_TILE && (col0 + col) < N) {
                const int col_g = col0 + col;          // gmem column (elem)
                const int col_byte = col_g >> 1;       // gmem byte index along N
                const int col_nib = col_g & 1;         // 0 → low, 1 → high nibble
                #pragma unroll
                for (int kk = 0; kk < K_TILE_BYTES; ++kk) {
                    const int k_lo_elem = kt + 2 * kk;
                    const int k_hi_elem = kt + 2 * kk + 1;
                    uint8_t byte_lo = (k_lo_elem < K)
                        ? B[(int64_t)k_lo_elem * ldb_bytes + col_byte]
                        : (uint8_t)0;
                    uint8_t byte_hi = (k_hi_elem < K)
                        ? B[(int64_t)k_hi_elem * ldb_bytes + col_byte]
                        : (uint8_t)0;
                    uint8_t lo_nib = (col_nib == 0)
                        ? (byte_lo & 0x0F)
                        : ((byte_lo >> 4) & 0x0F);
                    uint8_t hi_nib = (col_nib == 0)
                        ? (byte_hi & 0x0F)
                        : ((byte_hi >> 4) & 0x0F);
                    smem_B[col][kk] = lo_nib | (hi_nib << 4);
                }
            } else if (col < N_TILE) {
                #pragma unroll
                for (int kk = 0; kk < K_TILE_BYTES; ++kk) {
                    smem_B[col][kk] = (uint8_t)0;
                }
            }
        }

        __syncthreads();

        // MMA pass — identical to the RCR header (smem layout is the
        // same; the only difference was upstream in the gmem→smem
        // gather above).
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

                mma_m16n8k64_s32_int4<Enc>(
                    a0, a1, a2, a3,
                    b0, b1,
                    acc[mi][ni][0], acc[mi][ni][1],
                    acc[mi][ni][2], acc[mi][ni][3]);
            }
        }

        __syncthreads();
    }

    // Epilogue — identical to the RCR header (D is row-major pair-
    // packed along N in both layouts).
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
            const int pair_byte_col = col_l / 2;

            float v0 = alpha * (float)acc[mi][ni][0];
            float v1 = alpha * (float)acc[mi][ni][1];
            float v2 = alpha * (float)acc[mi][ni][2];
            float v3 = alpha * (float)acc[mi][ni][3];

            if (beta != 0.0f && C != nullptr) {
                if (row_top < M && col_l < N) {
                    uint8_t byte = C[(int64_t)row_top * ldc_bytes + pair_byte_col];
                    int32_t cl, cr;
                    unpack_int4_byte_to_s32<Enc>(byte, cl, cr);
                    v0 += beta * (float)cl;
                    if (col_r < N) v1 += beta * (float)cr;
                }
                if (row_bot < M && col_l < N) {
                    uint8_t byte = C[(int64_t)row_bot * ldc_bytes + pair_byte_col];
                    int32_t cl, cr;
                    unpack_int4_byte_to_s32<Enc>(byte, cl, cr);
                    v2 += beta * (float)cl;
                    if (col_r < N) v3 += beta * (float)cr;
                }
            }

            if constexpr (has_bias<Act>()) {
                if (col_l < N) {
                    float bl = bias_to_f32<BiasT>(bias[col_l]);
                    v0 += bl;
                    v2 += bl;
                }
                if (col_r < N) {
                    float br = bias_to_f32<BiasT>(bias[col_r]);
                    v1 += br;
                    v3 += br;
                }
            }

            v0 = apply_activation_f32<Act>(v0);
            v1 = apply_activation_f32<Act>(v1);
            v2 = apply_activation_f32<Act>(v2);
            v3 = apply_activation_f32<Act>(v3);

            if (row_top < M && col_l < N) {
                uint8_t lo = sat_cast_int4_from_f32<Enc>(v0);
                uint8_t hi = (col_r < N) ? sat_cast_int4_from_f32<Enc>(v1) : (uint8_t)0;
                D[(int64_t)row_top * ldd_bytes + pair_byte_col] = pack_int4_pair(lo, hi);
            }
            if (row_bot < M && col_l < N) {
                uint8_t lo = sat_cast_int4_from_f32<Enc>(v2);
                uint8_t hi = (col_r < N) ? sat_cast_int4_from_f32<Enc>(v3) : (uint8_t)0;
                D[(int64_t)row_bot * ldd_bytes + pair_byte_col] = pack_int4_pair(lo, hi);
            }
        }
    }
}

} // namespace int4_rrr_sm89

// Host-side launcher — mirrors `launch_gemm_int4_rcr_sm89`.
template <Int4Encoding Enc, typename BiasT, Activation Act>
int32_t launch_gemm_int4_rrr_sm89(
    int32_t m, int32_t n, int32_t k,
    const void *a, int64_t lda_bytes,
    const void *b, int64_t ldb_bytes,
    const void *c, int64_t ldc_bytes,
    void *d, int64_t ldd_bytes,
    const void *bias,
    float alpha, float beta,
    void *stream
) {
    using namespace int4_rrr_sm89;
    if (m <= 0 || n <= 0 || k <= 0) return 2;
    if ((k & 1) != 0) return 3;
    if ((n & 1) != 0) return 3;

    dim3 grid((unsigned)((n + N_TILE - 1) / N_TILE),
              (unsigned)((m + M_TILE - 1) / M_TILE),
              1u);
    dim3 block((unsigned)NUM_THREADS, 1u, 1u);

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    gemm_int4_rrr_sm89_kernel<Enc, BiasT, Act><<<grid, block, 0, s>>>(
        m, n, k,
        reinterpret_cast<const uint8_t *>(a), lda_bytes,
        reinterpret_cast<const uint8_t *>(b), ldb_bytes,
        reinterpret_cast<const uint8_t *>(c), ldc_bytes,
        reinterpret_cast<uint8_t *>(d), ldd_bytes,
        reinterpret_cast<const BiasT *>(bias),
        alpha, beta
    );

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

} // namespace baracuda
