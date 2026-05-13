// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Shared templated kernel body for FP8 RCR GEMM on sm_89.
//
// Parameters
//   Enc       : `Fp8Encoding::E4M3` or `Fp8Encoding::E5M2`.
//               Drives both the `mma.sync` operand tag (`.e4m3.e4m3.f32`
//               vs `.e5m2.e5m2.f32`) and the saturating-cast on store.
//   BiasT     : Bias element type, always `float` for the FP8 family
//               (no int32 bias convention for FP8). Ignored when
//               `Act == None`.
//   Act       : Activation / bias selector (see baracuda_epilogue_int8.cuh).
//
// Computes
//   D[i,j] = sat_cast_fp8<Enc>(activation(alpha * (A·B)[i,j]
//                                       + beta * dequant_fp8<Enc>(C[i,j])
//                                       + bias[j]))
//
// Layout `RCR`
//   A : row-major  [M, K], stride lda along K
//   B : col-major  [K, N], stride ldb along K   (each column K-contig)
//   C : row-major  [M, N], stride ldc along N   (optional; null + beta=0 to skip)
//   D : row-major  [M, N], stride ldd along N
//   bias : [N] (optional, gated by `has_bias<Act>()`)
//
// Tile / thread layout — identical to the int8 RRR template:
//   M_TILE=64, N_TILE=64, K_TILE=32, 2 warps × 32 threads.
//   smem_A[M_TILE][K_TILE], smem_B[N_TILE][K_TILE].
//   In RCR, `B` is col-major in gmem so its column data is K-contig;
//   we stash each column unchanged in a row of smem_B (no transpose
//   on load). The MMA fragment load then mirrors the int8 RRR template
//   verbatim — `&smem_B[frag_n0 + bn0][bk0]`.

#pragma once

#include <cstdint>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "baracuda_dtype.cuh"
#include "baracuda_epilogue_fp8.cuh"

namespace baracuda {

namespace fp8_rcr_sm89 {

constexpr int M_TILE = 64;
constexpr int N_TILE = 64;
constexpr int K_TILE = 32;

constexpr int M_MMA = 16;
constexpr int N_MMA = 8;
constexpr int K_MMA = 32;

constexpr int NUM_WARPS = 2;
constexpr int NUM_THREADS = NUM_WARPS * 32;

constexpr int M_WARP = M_TILE / NUM_WARPS;   // 32
constexpr int N_WARP = N_TILE;               // 64
constexpr int M_ITER = M_WARP / M_MMA;       // 2
constexpr int N_ITER = N_WARP / N_MMA;       // 8

// `mma.sync.aligned.m16n8k32.row.col.f32.{e4m3|e5m2}.{e4m3|e5m2}.f32`
// wrappers. Inputs are 4×b32 for A and 2×b32 for B (FP8 bytes packed
// 4-per-b32, same as int8 m16n8k32). Accumulator is 4×f32 per thread.
template <Fp8Encoding Enc>
__device__ __forceinline__ void mma_m16n8k32_f32_fp8(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float &d0, float &d1, float &d2, float &d3);

template <>
__device__ __forceinline__ void mma_m16n8k32_f32_fp8<Fp8Encoding::E4M3>(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float &d0, float &d1, float &d2, float &d3
) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1)
    );
}

template <>
__device__ __forceinline__ void mma_m16n8k32_f32_fp8<Fp8Encoding::E5M2>(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float &d0, float &d1, float &d2, float &d3
) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1)
    );
}

template <Fp8Encoding Enc, typename BiasT, Activation Act>
__global__ void gemm_fp8_rcr_sm89_kernel(
    int M, int N, int K,
    const uint8_t * __restrict__ A, int64_t lda,
    const uint8_t * __restrict__ B, int64_t ldb,
    const uint8_t * __restrict__ C, int64_t ldc,
    uint8_t * __restrict__ D, int64_t ldd,
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

    __shared__ uint8_t smem_A[M_TILE][K_TILE];
    __shared__ uint8_t smem_B[N_TILE][K_TILE];

    float acc[M_ITER][N_ITER][4];
    #pragma unroll
    for (int mi = 0; mi < M_ITER; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < N_ITER; ++ni) {
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc[mi][ni][r] = 0.0f;
        }
    }

    for (int kt = 0; kt < K; kt += K_TILE) {
        // gmem → smem: A tile. Each thread loads one row.
        {
            const int row = tid;
            if (row < M_TILE && (row0 + row) < M) {
                const uint8_t *src = &A[(int64_t)(row0 + row) * lda + kt];
                const int k_lim = (K - kt) < K_TILE ? (K - kt) : K_TILE;
                #pragma unroll
                for (int kk = 0; kk < K_TILE; ++kk) {
                    smem_A[row][kk] = (kk < k_lim) ? src[kk] : (uint8_t)0;
                }
            } else if (row < M_TILE) {
                #pragma unroll
                for (int kk = 0; kk < K_TILE; ++kk) {
                    smem_A[row][kk] = (uint8_t)0;
                }
            }
        }

        // gmem → smem: B tile (col-major in gmem; stash columns into
        // rows of smem_B unchanged — column data is K-contig in gmem).
        {
            const int col = tid;
            if (col < N_TILE && (col0 + col) < N) {
                const uint8_t *src =
                    &B[(int64_t)(col0 + col) * ldb + kt];
                const int k_lim = (K - kt) < K_TILE ? (K - kt) : K_TILE;
                #pragma unroll
                for (int kk = 0; kk < K_TILE; ++kk) {
                    smem_B[col][kk] = (kk < k_lim) ? src[kk] : (uint8_t)0;
                }
            } else if (col < N_TILE) {
                #pragma unroll
                for (int kk = 0; kk < K_TILE; ++kk) {
                    smem_B[col][kk] = (uint8_t)0;
                }
            }
        }

        __syncthreads();

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

                mma_m16n8k32_f32_fp8<Enc>(
                    a0, a1, a2, a3,
                    b0, b1,
                    acc[mi][ni][0], acc[mi][ni][1],
                    acc[mi][ni][2], acc[mi][ni][3]);
            }
        }

        __syncthreads();
    }

    // ----------------------------------------------------------------
    // Epilogue: alpha * acc + (beta * dequant(C)) + (bias) → activation →
    // sat-cast → store.
    //
    // D-fragment register layout per thread (m16n8 f32):
    //   d0 = D_frag[lane/4    ][(lane%4)*2    ]
    //   d1 = D_frag[lane/4    ][(lane%4)*2 + 1]
    //   d2 = D_frag[lane/4 + 8][(lane%4)*2    ]
    //   d3 = D_frag[lane/4 + 8][(lane%4)*2 + 1]
    // ----------------------------------------------------------------
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

            float v0 = alpha * acc[mi][ni][0];
            float v1 = alpha * acc[mi][ni][1];
            float v2 = alpha * acc[mi][ni][2];
            float v3 = alpha * acc[mi][ni][3];

            if (beta != 0.0f && C != nullptr) {
                if (row_top < M && col_l < N) {
                    v0 += beta * dequant_fp8_to_f32<Enc>(
                        C[(int64_t)row_top * ldc + col_l]);
                }
                if (row_top < M && col_r < N) {
                    v1 += beta * dequant_fp8_to_f32<Enc>(
                        C[(int64_t)row_top * ldc + col_r]);
                }
                if (row_bot < M && col_l < N) {
                    v2 += beta * dequant_fp8_to_f32<Enc>(
                        C[(int64_t)row_bot * ldc + col_l]);
                }
                if (row_bot < M && col_r < N) {
                    v3 += beta * dequant_fp8_to_f32<Enc>(
                        C[(int64_t)row_bot * ldc + col_r]);
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

            if (row_top < M && col_l < N) D[(int64_t)row_top * ldd + col_l] = sat_cast_fp8_from_f32<Enc>(v0);
            if (row_top < M && col_r < N) D[(int64_t)row_top * ldd + col_r] = sat_cast_fp8_from_f32<Enc>(v1);
            if (row_bot < M && col_l < N) D[(int64_t)row_bot * ldd + col_l] = sat_cast_fp8_from_f32<Enc>(v2);
            if (row_bot < M && col_r < N) D[(int64_t)row_bot * ldd + col_r] = sat_cast_fp8_from_f32<Enc>(v3);
        }
    }
}

} // namespace fp8_rcr_sm89

// Host-side launcher. Templated on the same parameters as the kernel.
template <Fp8Encoding Enc, typename BiasT, Activation Act>
int32_t launch_gemm_fp8_rcr_sm89(
    int32_t m, int32_t n, int32_t k,
    const void *a, int64_t lda,
    const void *b, int64_t ldb,
    const void *c, int64_t ldc,
    void *d, int64_t ldd,
    const void *bias,
    float alpha, float beta,
    void *stream
) {
    using namespace fp8_rcr_sm89;
    if (m <= 0 || n <= 0 || k <= 0) return 2;

    dim3 grid((unsigned)((n + N_TILE - 1) / N_TILE),
              (unsigned)((m + M_TILE - 1) / M_TILE),
              1u);
    dim3 block((unsigned)NUM_THREADS, 1u, 1u);

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    gemm_fp8_rcr_sm89_kernel<Enc, BiasT, Act><<<grid, block, 0, s>>>(
        m, n, k,
        reinterpret_cast<const uint8_t *>(a), lda,
        reinterpret_cast<const uint8_t *>(b), ldb,
        reinterpret_cast<const uint8_t *>(c), ldc,
        reinterpret_cast<uint8_t *>(d), ldd,
        reinterpret_cast<const BiasT *>(bias),
        alpha, beta
    );

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

} // namespace baracuda
