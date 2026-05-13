// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Shared templated kernel body for int4 RCR GEMM on sm_89.
//
// Parameters
//   Enc       : `Int4Encoding::S4` (signed) or `::U4` (unsigned).
//               Drives the `mma.sync` operand tag (`.s4.s4` vs `.u4.u4`)
//               and the saturating-cast on store (clamp to [-8, +7] or
//               [0, 15]).
//   BiasT     : Bias element type — `float` or `int32_t`. Ignored when
//               `Act == Activation::None` (Identity epilogue).
//   Act       : Activation / bias selector (see baracuda_epilogue_int8.cuh).
//
// Computes
//   D[i,j] = sat_cast_int4<Enc>(activation(alpha * (A·B)[i,j]
//                                       + beta * C[i,j]
//                                       + bias[j]))
//
// Layout `RCR`
//   A : row-major  [M, K], stride lda_bytes along K  (K elements / row =
//                                                     K/2 bytes / row)
//   B : col-major  [K, N], stride ldb_bytes along K  (K elements / col =
//                                                     K/2 bytes / col)
//   C : row-major  [M, N], stride ldc_bytes along N  (N elements / row =
//                                                     N/2 bytes / row;
//                                                     optional; null + beta=0
//                                                     to skip)
//   D : row-major  [M, N], stride ldd_bytes along N  (N elements / row =
//                                                     N/2 bytes / row)
//   bias : [N] (optional, gated by `has_bias<Act>()`)
//
// `lda_bytes` / `ldb_bytes` / `ldc_bytes` / `ldd_bytes` are leading
// dimensions in **bytes** (= storage-slot count), not in element count.
// Each storage slot is a packed-pair of two int4 elements (low nibble =
// even element, high nibble = odd element along the K axis for A/B and
// along the N axis for C/D).
//
// `M`, `N`, `K` are in **element** count. The kernel requires:
//   - `K` divisible by 2 (else the packing convention is ambiguous);
//   - `N` divisible by 2 (else the output packed pairs would straddle
//     unpacked elements; enforced by `can_implement`).
//   The kernel pads sub-tiles up to `K_TILE` and `N_TILE` with zeros
//   internally; non-multiples-of-tile shapes are supported.
//
// Tile / thread layout — mirrors the int8 RRR template byte-arithmetic:
//   M_TILE=64 elements, N_TILE=64 elements, K_TILE=64 elements (=32 bytes).
//   2 warps × 32 threads. smem_A[M_TILE][K_TILE_BYTES],
//   smem_B[N_TILE][K_TILE_BYTES] (both K-contig in bytes).
//   In RCR, `B` is col-major in gmem so its column data is K-contig;
//   columns are stashed unchanged into rows of smem_B (no transpose
//   on load).
//
// Per-thread MMA register layout for m16n8k64 with s4/u4 multiplicands
// (PTX 8.0+):
//   A : 4 b32 regs/thread; each b32 = 4 K-adjacent bytes = 8 int4
//       elements along K. Same byte-arithmetic as m16n8k32.s8.
//   B : 2 b32 regs/thread; each b32 = 4 K-adjacent bytes = 8 int4
//       elements along K.
//   D : 4 s32 regs/thread.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "baracuda_dtype.cuh"
#include "baracuda_epilogue_int8.cuh"

namespace baracuda {

// int4 encoding selector — drives both the MMA operand tag and the
// saturating-cast on store.
enum class Int4Encoding : int {
    S4 = 0,
    U4 = 1,
};

// `mma.sync.aligned.m16n8k64.row.col.satfinite.s32.{s4|u4}.{s4|u4}.s32`
// wrappers. A is 4×b32, B is 2×b32, D accum is 4×s32 (same per-thread
// register footprint as m16n8k32.s8 — the K dimension doubles in
// element count but the byte count per b32 stays at 4).
//
// Lives in `baracuda` scope (not the per-layout sub-namespace) so the
// RRR header can reuse the same MMA / sat-cast / unpack templates.
template <Int4Encoding Enc>
__device__ __forceinline__ void mma_m16n8k64_s32_int4(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    int32_t &d0, int32_t &d1, int32_t &d2, int32_t &d3);

template <>
__device__ __forceinline__ void mma_m16n8k64_s32_int4<Int4Encoding::S4>(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    int32_t &d0, int32_t &d1, int32_t &d2, int32_t &d3
) {
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.s4.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1)
    );
}

template <>
__device__ __forceinline__ void mma_m16n8k64_s32_int4<Int4Encoding::U4>(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    int32_t &d0, int32_t &d1, int32_t &d2, int32_t &d3
) {
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.satfinite.s32.u4.u4.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1)
    );
}

// Saturating f32 → int4 nibble cast trampoline. Specialized on `Enc` so
// the underlying `__float2int_rn` + clamp lives in baracuda_dtype.cuh.
template <Int4Encoding Enc>
__device__ __forceinline__ uint8_t sat_cast_int4_from_f32(float x);

template <>
__device__ __forceinline__ uint8_t sat_cast_int4_from_f32<Int4Encoding::S4>(float x) {
    return sat_cast_f32_to_s4(x);
}

template <>
__device__ __forceinline__ uint8_t sat_cast_int4_from_f32<Int4Encoding::U4>(float x) {
    return sat_cast_f32_to_u4(x);
}

// Unpack one byte of packed int4 storage into two s32 values, branching
// on the encoding (s4 = sign-extend, u4 = zero-extend).
template <Int4Encoding Enc>
__device__ __forceinline__ void unpack_int4_byte_to_s32(
    uint8_t byte, int32_t &lo, int32_t &hi);

template <>
__device__ __forceinline__ void unpack_int4_byte_to_s32<Int4Encoding::S4>(
    uint8_t byte, int32_t &lo, int32_t &hi) {
    unpack_s4_byte(byte, lo, hi);
}

template <>
__device__ __forceinline__ void unpack_int4_byte_to_s32<Int4Encoding::U4>(
    uint8_t byte, int32_t &lo, int32_t &hi) {
    unpack_u4_byte(byte, lo, hi);
}

namespace int4_rcr_sm89 {

constexpr int M_TILE = 64;        // elements
constexpr int N_TILE = 64;        // elements
constexpr int K_TILE = 64;        // elements
constexpr int K_TILE_BYTES = K_TILE / 2;  // 32 bytes per row of smem

constexpr int M_MMA = 16;         // elements
constexpr int N_MMA = 8;          // elements
constexpr int K_MMA = 64;         // elements (m16n8k64 → K is 64 in element count)

constexpr int NUM_WARPS = 2;
constexpr int NUM_THREADS = NUM_WARPS * 32;

constexpr int M_WARP = M_TILE / NUM_WARPS;   // 32
constexpr int N_WARP = N_TILE;               // 64
constexpr int M_ITER = M_WARP / M_MMA;       // 2
constexpr int N_ITER = N_WARP / N_MMA;       // 8

template <Int4Encoding Enc, typename BiasT, Activation Act>
__global__ void gemm_int4_rcr_sm89_kernel(
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

    const int row0 = bm * M_TILE;   // row index in elements
    const int col0 = bn * N_TILE;   // col index in elements

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

    // K loop: iterate in element units. Each tile is K_TILE elements
    // = K_TILE_BYTES bytes along K.
    for (int kt = 0; kt < K; kt += K_TILE) {
        // gmem → smem: A tile. Each thread loads one row's worth of
        // K_TILE_BYTES bytes (= K_TILE int4 elements along K).
        {
            const int row = tid;
            if (row < M_TILE && (row0 + row) < M) {
                const uint8_t *src =
                    &A[(int64_t)(row0 + row) * lda_bytes + (int64_t)(kt / 2)];
                // Per-byte K-element predicate: byte `kk` holds elements
                // [kt + 2*kk, kt + 2*kk + 1] from this row of A.
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

        // gmem → smem: B tile (col-major in gmem; K-contig within each
        // column). Stash each column unchanged into a row of smem_B.
        {
            const int col = tid;
            if (col < N_TILE && (col0 + col) < N) {
                const uint8_t *src =
                    &B[(int64_t)(col0 + col) * ldb_bytes + (int64_t)(kt / 2)];
                #pragma unroll
                for (int kk = 0; kk < K_TILE_BYTES; ++kk) {
                    int k_elem = kt + 2 * kk;
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

        // ----------------------------------------------------------------
        // MMA pass. K_TILE == K_MMA (in elements), so one mma per
        // (mi, ni) per K-tile. Fragment byte arithmetic is identical to
        // the int8 RRR / FP8 RCR templates: each b32 covers 4 bytes of
        // smem (just 8 int4 elements per b32 instead of 4 int8).
        // ----------------------------------------------------------------
        #pragma unroll
        for (int mi = 0; mi < M_ITER; ++mi) {
            const int frag_m0 = warp_id * M_WARP + mi * M_MMA;

            const int ar = lane / 4;        // row in 16-row A fragment
            const int ac = (lane % 4) * 4;  // byte offset along K (within 32-byte K-tile)

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

    // ----------------------------------------------------------------
    // Epilogue: alpha * acc + (beta * dequant(C)) + (bias) → activation →
    // sat-cast → packed-pair store.
    //
    // D-fragment register layout per thread (m16n8 s32):
    //   d0 = D_frag[lane/4    ][(lane%4)*2    ]   ← even col, low nibble
    //   d1 = D_frag[lane/4    ][(lane%4)*2 + 1]   ← odd  col, high nibble
    //   d2 = D_frag[lane/4 + 8][(lane%4)*2    ]
    //   d3 = D_frag[lane/4 + 8][(lane%4)*2 + 1]
    //
    // `col_l` is always even (frag_n0 = ni * 8 is even, (lane%4)*2 is
    // even), so `col_l` / `col_r` form a packed-pair output byte at
    // byte index `col_l / 2`.
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
            const int pair_byte_col = col_l / 2;   // shared byte index for (col_l, col_r)

            float v0 = alpha * (float)acc[mi][ni][0];
            float v1 = alpha * (float)acc[mi][ni][1];
            float v2 = alpha * (float)acc[mi][ni][2];
            float v3 = alpha * (float)acc[mi][ni][3];

            if (beta != 0.0f && C != nullptr) {
                // Read packed-pair byte at (row, pair_byte_col); unpack
                // low/high nibbles into the two adjacent column cells.
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

            // Pack adjacent column pairs into one byte and write. Each
            // thread owns the (col_l, col_r) pair for its row(s), so
            // there is no inter-thread byte contention.
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

} // namespace int4_rcr_sm89

// Host-side launcher. Templated on the same parameters as the kernel.
// `lda` / `ldb` / `ldc` / `ldd` at this boundary are in **bytes** —
// the safe-layer dispatcher in `baracuda-kernels` converts from
// element-counted leading dimensions (the MatrixRef convention) before
// invoking the FFI.
template <Int4Encoding Enc, typename BiasT, Activation Act>
int32_t launch_gemm_int4_rcr_sm89(
    int32_t m, int32_t n, int32_t k,
    const void *a, int64_t lda_bytes,
    const void *b, int64_t ldb_bytes,
    const void *c, int64_t ldc_bytes,
    void *d, int64_t ldd_bytes,
    const void *bias,
    float alpha, float beta,
    void *stream
) {
    using namespace int4_rcr_sm89;
    if (m <= 0 || n <= 0 || k <= 0) return 2;
    if ((k & 1) != 0) return 3;   // packing requires K even
    if ((n & 1) != 0) return 3;   // packed output requires N even

    dim3 grid((unsigned)((n + N_TILE - 1) / N_TILE),
              (unsigned)((m + M_TILE - 1) / M_TILE),
              1u);
    dim3 block((unsigned)NUM_THREADS, 1u, 1u);

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    gemm_int4_rcr_sm89_kernel<Enc, BiasT, Act><<<grid, block, 0, s>>>(
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
