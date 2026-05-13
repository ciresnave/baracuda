// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Shared templated kernel body for int8 RRR GEMM on sm_80.
//
// Parameters
//   InT       : `int8_t` (signed) or `uint8_t` (unsigned) input element.
//   OutT      : `int8_t` (signed) or `uint8_t` (unsigned) output element.
//               Conventionally matches `InT` for the alpha.16 SKUs.
//   BiasT     : `float` or `int32_t`. Ignored when `Act == None`.
//   Act       : Activation / bias selector (see baracuda_epilogue_int8.cuh).
//   SatCast   : function pointer to the saturating f32→OutT cast.
//
// Computes
//   D[i,j] = SatCast(activation(alpha * (A·B)[i,j] + beta * C[i,j] + bias[j]))
//
// Layout
//   A : row-major  [M, K], stride lda along K
//   B : row-major  [K, N], stride ldb along N   (the RRR case)
//   C : row-major  [M, N], stride ldc along N   (optional; null + beta=0 to skip)
//   D : row-major  [M, N], stride ldd along N
//   bias : [N] (optional, gated by `has_bias<Act>()`)
//
// Threadblock / register layout
//   M_TILE=64  N_TILE=64  K_TILE=32 ; 2 warps × 32 threads.
//   Each warp owns 32×64 of the M×N output tile.
//   Per-warp MMA grid: M_ITER=2 (× 16-row mma), N_ITER=8 (× 8-col mma).
//   Single K iteration per K-tile (K_TILE == K_MMA == 32).
//   smem layout: smem_A[M_TILE][K_TILE], smem_B[N_TILE][K_TILE]
//     — both K-contig; B is stored on-the-fly-transposed so
//     mma.sync.row.col consumes it directly.
//
// Per-thread MMA register layout for m16n8k32 with s8/u8 multiplicands:
//   A : 4 b32 regs/thread, 4 K-adjacent bytes per b32
//   B : 2 b32 regs/thread, 4 K-adjacent bytes per b32
//   D : 4 s32 regs/thread, layout below in the epilogue section.
//
// PTX form differs only in the multiplicand encoding:
//   mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 …  (signed)
//   mma.sync.aligned.m16n8k32.row.col.satfinite.s32.u8.u8.s32 …  (unsigned)
//
// This header is `#include`d by the four `gemm_{s8,u8}_rrr_sm80{,_bias}.cu`
// files which instantiate the matching combination and expose the
// extern "C" launcher.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <type_traits>

#include "baracuda_dtype.cuh"
#include "baracuda_epilogue_int8.cuh"

namespace baracuda {

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

// mma.sync wrappers, parameterized on the input element type. The s8
// and u8 forms differ only in the `.s8` vs `.u8` operand tags.
template <typename InT>
__device__ __forceinline__ void mma_m16n8k32_satfinite(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    int32_t &d0, int32_t &d1, int32_t &d2, int32_t &d3);

template <>
__device__ __forceinline__ void mma_m16n8k32_satfinite<int8_t>(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    int32_t &d0, int32_t &d1, int32_t &d2, int32_t &d3
) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1)
    );
}

template <>
__device__ __forceinline__ void mma_m16n8k32_satfinite<uint8_t>(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    int32_t &d0, int32_t &d1, int32_t &d2, int32_t &d3
) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.u8.u8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1)
    );
}

// Saturating f32→OutT cast trampoline. Specialized for s8 / u8 outputs
// so the `__float2int_rn`+clamp lives in one place (baracuda_dtype.cuh).
template <typename OutT>
__device__ __forceinline__ OutT sat_cast_from_f32(float x);

template <>
__device__ __forceinline__ int8_t sat_cast_from_f32<int8_t>(float x) {
    return sat_cast_f32_to_s8(x);
}

template <>
__device__ __forceinline__ uint8_t sat_cast_from_f32<uint8_t>(float x) {
    return sat_cast_f32_to_u8(x);
}

// ============================================================================
// Templated GEMM kernel.
// ============================================================================
template <typename InT, typename OutT, typename BiasT, Activation Act>
__global__ void gemm_int8_rrr_sm80_kernel(
    int M, int N, int K,
    const InT * __restrict__ A, int64_t lda,
    const InT * __restrict__ B, int64_t ldb,
    const OutT * __restrict__ C, int64_t ldc,
    OutT * __restrict__ D, int64_t ldd,
    const BiasT * __restrict__ bias,
    float alpha, float beta
) {
    static_assert(std::is_same<InT, int8_t>::value || std::is_same<InT, uint8_t>::value,
                  "InT must be int8_t or uint8_t");

    const int bm = blockIdx.y;
    const int bn = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;

    const int row0 = bm * M_TILE;
    const int col0 = bn * N_TILE;

    __shared__ InT smem_A[M_TILE][K_TILE];
    __shared__ InT smem_B[N_TILE][K_TILE];

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
        // gmem → smem: A tile. Each thread loads one row.
        {
            const int row = tid;
            if (row < M_TILE && (row0 + row) < M) {
                const InT *src = &A[(int64_t)(row0 + row) * lda + kt];
                const int k_lim = (K - kt) < K_TILE ? (K - kt) : K_TILE;
                #pragma unroll
                for (int kk = 0; kk < K_TILE; ++kk) {
                    smem_A[row][kk] = (kk < k_lim) ? src[kk] : (InT)0;
                }
            } else if (row < M_TILE) {
                #pragma unroll
                for (int kk = 0; kk < K_TILE; ++kk) {
                    smem_A[row][kk] = (InT)0;
                }
            }
        }

        // gmem → smem: B tile (transposed). Each thread covers one N-column.
        {
            const int col = tid;
            if (col < N_TILE && (col0 + col) < N) {
                const int k_lim = (K - kt) < K_TILE ? (K - kt) : K_TILE;
                #pragma unroll
                for (int kk = 0; kk < K_TILE; ++kk) {
                    InT v = (InT)0;
                    if (kk < k_lim) {
                        v = B[(int64_t)(kt + kk) * ldb + (col0 + col)];
                    }
                    smem_B[col][kk] = v;
                }
            } else if (col < N_TILE) {
                #pragma unroll
                for (int kk = 0; kk < K_TILE; ++kk) {
                    smem_B[col][kk] = (InT)0;
                }
            }
        }

        __syncthreads();

        // ----------------------------------------------------------------
        // MMA pass. K_TILE == K_MMA, so one mma per (mi, ni) per K-tile.
        // ----------------------------------------------------------------
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

                mma_m16n8k32_satfinite<InT>(
                    a0, a1, a2, a3,
                    b0, b1,
                    acc[mi][ni][0], acc[mi][ni][1],
                    acc[mi][ni][2], acc[mi][ni][3]);
            }
        }

        __syncthreads();
    }

    // ----------------------------------------------------------------
    // Epilogue: alpha * acc + (beta * C) + (bias) → activation → sat-cast → store.
    //
    // D-fragment register layout per thread (m16n8 s32):
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

            float v0 = alpha * (float)acc[mi][ni][0];
            float v1 = alpha * (float)acc[mi][ni][1];
            float v2 = alpha * (float)acc[mi][ni][2];
            float v3 = alpha * (float)acc[mi][ni][3];

            if (beta != 0.0f && C != nullptr) {
                if (row_top < M && col_l < N) v0 += beta * (float)C[(int64_t)row_top * ldc + col_l];
                if (row_top < M && col_r < N) v1 += beta * (float)C[(int64_t)row_top * ldc + col_r];
                if (row_bot < M && col_l < N) v2 += beta * (float)C[(int64_t)row_bot * ldc + col_l];
                if (row_bot < M && col_r < N) v3 += beta * (float)C[(int64_t)row_bot * ldc + col_r];
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

            // Activation. For Act ∈ {None, Bias} this compiles to a pass-through.
            v0 = apply_activation_f32<Act>(v0);
            v1 = apply_activation_f32<Act>(v1);
            v2 = apply_activation_f32<Act>(v2);
            v3 = apply_activation_f32<Act>(v3);

            if (row_top < M && col_l < N) D[(int64_t)row_top * ldd + col_l] = sat_cast_from_f32<OutT>(v0);
            if (row_top < M && col_r < N) D[(int64_t)row_top * ldd + col_r] = sat_cast_from_f32<OutT>(v1);
            if (row_bot < M && col_l < N) D[(int64_t)row_bot * ldd + col_l] = sat_cast_from_f32<OutT>(v2);
            if (row_bot < M && col_r < N) D[(int64_t)row_bot * ldd + col_r] = sat_cast_from_f32<OutT>(v3);
        }
    }
}

// Host-side launcher. Templated on the same parameters as the kernel.
// `m, n, k` are int32; the kernel uses int64 strides internally.
template <typename InT, typename OutT, typename BiasT, Activation Act>
int32_t launch_gemm_int8_rrr_sm80(
    int32_t m, int32_t n, int32_t k,
    const void *a, int64_t lda,
    const void *b, int64_t ldb,
    const void *c, int64_t ldc,
    void *d, int64_t ldd,
    const void *bias,
    float alpha, float beta,
    void *stream
) {
    if (m <= 0 || n <= 0 || k <= 0) return 2;

    dim3 grid((unsigned)((n + N_TILE - 1) / N_TILE),
              (unsigned)((m + M_TILE - 1) / M_TILE),
              1u);
    dim3 block((unsigned)NUM_THREADS, 1u, 1u);

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    gemm_int8_rrr_sm80_kernel<InT, OutT, BiasT, Act><<<grid, block, 0, s>>>(
        m, n, k,
        reinterpret_cast<const InT *>(a), lda,
        reinterpret_cast<const InT *>(b), ldb,
        reinterpret_cast<const OutT *>(c), ldc,
        reinterpret_cast<OutT *>(d), ldd,
        reinterpret_cast<const BiasT *>(bias),
        alpha, beta
    );

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

} // namespace baracuda
