// SPDX-License-Identifier: MIT OR Apache-2.0
//
// gemm_s8_rrr_sm80.cu — S8 GEMM with RowMajor × RowMajor × RowMajor layout
// and Identity epilogue, for sm_80 (Ampere; runs forward-compatibly on
// Ada / Hopper).
//
// Computes D = saturating_cast<s8>(alpha * (A @ B) + beta * C)
//   A: row-major  [M, K]      stride lda along K
//   B: row-major  [K, N]      stride ldb along N
//   C: row-major  [M, N]      stride ldc along N (optional; null + beta=0 to skip)
//   D: row-major  [M, N]      stride ldd along N
//
// Accumulator: int32 via mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32.
// Epilogue alpha/beta in f32, saturating cast back to s8 on store.
//
// Design (correctness-first; perf is Phase 10 tuning):
//   - Threadblock tile:  M_TILE=64, N_TILE=64, K_TILE=32
//   - 2 warps × 32 threads per block = 64 threads
//   - Each warp owns 32×64 of the M×N output tile
//   - Per-warp MMA grid:  M_ITER=2 (× 16-row mma), N_ITER=8 (× 8-col mma)
//   - Single K iteration per K-tile (K_TILE = K_MMA = 32)
//   - smem holds A as [M_TILE][K_TILE] (K-contig) and B-transposed as
//     [N_TILE][K_TILE] (K-contig per N column — the smem layout that lets
//     mma.sync.row.col consume the B operand directly).
//   - gmem→smem: simplest correct pattern (each thread serially scatters
//     one byte at a time for B). Coalesced cp.async is a later optimization.
//   - smem→reg: plain pointer dereference of 4-byte aligned smem (no
//     ldmatrix — direct uint32_t loads from the K-contig smem). Slower
//     than ldmatrix but easier to reason about correctness against the
//     mma.sync register-layout spec.
//
// Why RRR isn't CUTLASS-expressible on sm_80 — see
// ~/.claude/plans/baracuda-kernels-comprehensive.md §5. Short version:
// CUTLASS 4.2.0 lacks the 8-bit TensorOpMultiplicandCongruous warp
// iterator, and the partial-spec gap can't be papered over via vendoring
// alone (two attempts in baracuda-cutlass were reverted in
// commit 6a1a4dd).

#include <cstdint>
#include <cuda_runtime.h>

#include "baracuda_dtype.cuh"

namespace {

constexpr int M_TILE = 64;
constexpr int N_TILE = 64;
constexpr int K_TILE = 32;

constexpr int M_MMA = 16;
constexpr int N_MMA = 8;
constexpr int K_MMA = 32;

constexpr int NUM_WARPS = 2;
constexpr int NUM_THREADS = NUM_WARPS * 32;

constexpr int M_WARP = M_TILE / NUM_WARPS; // 32 — each warp covers 32 rows of M
constexpr int N_WARP = N_TILE;             // 64 — each warp covers all of N
constexpr int M_ITER = M_WARP / M_MMA;     // 2
constexpr int N_ITER = N_WARP / N_MMA;     // 8

// PTX wrapper for one mma.sync.m16n8k32 invocation with s8 inputs and
// saturating s32 accumulator.
//
// Register layout (per PTX ISA for m16n8k32 with .s8 multiplicands):
//   A: 4 × b32 registers per thread, packed 4 K-adjacent s8 per b32.
//      thread g (laneid) holds:
//        a0 = A[g/4][(g%4)*4 .. (g%4)*4 + 3]
//        a1 = A[g/4 + 8][(g%4)*4 .. (g%4)*4 + 3]
//        a2 = A[g/4][(g%4)*4 + 16 .. (g%4)*4 + 19]
//        a3 = A[g/4 + 8][(g%4)*4 + 16 .. (g%4)*4 + 19]
//   B: 2 × b32 registers per thread, packed 4 K-adjacent s8 per b32.
//      thread g holds:
//        b0 = B[(g%4)*4 .. (g%4)*4 + 3][g/4]
//        b1 = B[(g%4)*4 + 16 .. (g%4)*4 + 19][g/4]
//   D (= C): 4 × s32 registers per thread.
//      thread g holds:
//        d0 = D[g/4][(g%4)*2]
//        d1 = D[g/4][(g%4)*2 + 1]
//        d2 = D[g/4 + 8][(g%4)*2]
//        d3 = D[g/4 + 8][(g%4)*2 + 1]
__device__ __forceinline__ void mma_m16n8k32_s8s8s32_satfinite(
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

__global__ void gemm_s8_rrr_sm80_kernel(
    int M, int N, int K,
    const int8_t * __restrict__ A, int64_t lda,
    const int8_t * __restrict__ B, int64_t ldb,
    const int8_t * __restrict__ C, int64_t ldc,
    int8_t * __restrict__ D, int64_t ldd,
    float alpha, float beta
) {
    const int bm = blockIdx.y; // block-row index along M
    const int bn = blockIdx.x; // block-col index along N
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;

    // The block-tile origin in the global output.
    const int row0 = bm * M_TILE;
    const int col0 = bn * N_TILE;

    // Shared memory.
    // A:  [M_TILE rows][K_TILE cols]   K-contig (matches A row-major + K=inner).
    // Bt: [N_TILE rows][K_TILE cols]   K-contig per N-column; this is the
    //                                  smem-side transpose of the gmem B
    //                                  tile, achieved by writing
    //                                  Bt[n][k] = B[kt+k][bn*N_TILE+n] in
    //                                  the gmem→smem copy below.
    __shared__ int8_t smem_A[M_TILE][K_TILE];
    __shared__ int8_t smem_B[N_TILE][K_TILE];

    // Per-thread accumulators: M_ITER × N_ITER tiles, 4 s32 registers each.
    int32_t acc[M_ITER][N_ITER][4];
    #pragma unroll
    for (int mi = 0; mi < M_ITER; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < N_ITER; ++ni) {
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc[mi][ni][r] = 0;
        }
    }

    // ------------------------------------------------------------------
    // Main loop over K-tiles.
    // ------------------------------------------------------------------
    for (int kt = 0; kt < K; kt += K_TILE) {
        // gmem → smem: A tile.
        //
        // 64 threads × 32 bytes/thread = 2048 bytes = M_TILE * K_TILE ✓.
        // Thread t loads row t of the A tile (M_TILE rows = NUM_THREADS).
        // K_TILE = 32 bytes/row → 2 × int4 (8 × int32) loads per row.
        {
            const int row = tid; // 0..63 → M_TILE row
            if (row < M_TILE && (row0 + row) < M) {
                const int8_t *src = &A[(int64_t)(row0 + row) * lda + kt];
                // Slow but always-correct fallback if (kt + K_TILE) > K:
                // write zero into the smem cells past K. We do it in
                // bytes so misaligned tails don't trip a 4-byte read.
                int8_t *dst = &smem_A[row][0];
                const int k_lim = (K - kt) < K_TILE ? (K - kt) : K_TILE;
                #pragma unroll
                for (int kk = 0; kk < K_TILE; ++kk) {
                    dst[kk] = (kk < k_lim) ? src[kk] : (int8_t)0;
                }
            } else if (row < M_TILE) {
                #pragma unroll
                for (int kk = 0; kk < K_TILE; ++kk) {
                    smem_A[row][kk] = (int8_t)0;
                }
            }
        }

        // gmem → smem: B tile (transposed on the fly).
        //
        // Each thread covers one N-column of the smem-B tile (N_TILE = 64 = NUM_THREADS).
        // Reads K_TILE = 32 elements scattered along K from gmem (stride
        // ldb) and writes them contiguously along the K axis of smem_B.
        {
            const int col = tid; // 0..63 → N_TILE col
            if (col < N_TILE && (col0 + col) < N) {
                const int k_lim = (K - kt) < K_TILE ? (K - kt) : K_TILE;
                #pragma unroll
                for (int kk = 0; kk < K_TILE; ++kk) {
                    int8_t v = (int8_t)0;
                    if (kk < k_lim) {
                        v = B[(int64_t)(kt + kk) * ldb + (col0 + col)];
                    }
                    smem_B[col][kk] = v;
                }
            } else if (col < N_TILE) {
                #pragma unroll
                for (int kk = 0; kk < K_TILE; ++kk) {
                    smem_B[col][kk] = (int8_t)0;
                }
            }
        }

        __syncthreads();

        // ------------------------------------------------------------------
        // MMA pass: one K-iteration per K-tile (K_TILE == K_MMA == 32).
        //
        // Per warp:
        //   M_ITER × N_ITER = 2 × 8 = 16 mma.sync instructions.
        // Per threadblock: 2 warps × 16 = 32 mma instructions per K-tile.
        // ------------------------------------------------------------------
        #pragma unroll
        for (int mi = 0; mi < M_ITER; ++mi) {
            // A fragment for this (warp, mi): 16 rows × 32 cols.
            //   row base in smem_A = warp_id * M_WARP + mi * M_MMA
            //   col base = 0 (K_TILE == K_MMA, no offset)
            const int frag_m0 = warp_id * M_WARP + mi * M_MMA;

            const int ar = lane / 4;          // 0..7  → row within 8-row half
            const int ac = (lane % 4) * 4;    // 0,4,8,12 → byte-col within first 16 cols

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
                // B fragment for this ni: 32 rows × 8 cols (K × N tile).
                //   smem_B[n][k] is K-contig per N-column.
                //   The b32 layout per thread maps:
                //     b0 lane g = B_frag[(g%4)*4 .. (g%4)*4 + 3][g/4]
                //              = smem_B[ni*N_MMA + g/4][(g%4)*4 .. (g%4)*4 + 3]
                //     b1 lane g = B_frag[(g%4)*4 + 16 .. (g%4)*4 + 19][g/4]
                //              = smem_B[ni*N_MMA + g/4][(g%4)*4 + 16 .. (g%4)*4 + 19]
                const int bk0 = (lane % 4) * 4;
                const int bn0 = lane / 4;
                const int frag_n0 = ni * N_MMA;

                const uint32_t b0 =
                    *reinterpret_cast<const uint32_t *>(&smem_B[frag_n0 + bn0][bk0]);
                const uint32_t b1 =
                    *reinterpret_cast<const uint32_t *>(&smem_B[frag_n0 + bn0][bk0 + 16]);

                mma_m16n8k32_s8s8s32_satfinite(
                    a0, a1, a2, a3,
                    b0, b1,
                    acc[mi][ni][0], acc[mi][ni][1],
                    acc[mi][ni][2], acc[mi][ni][3]);
            }
        }

        __syncthreads();
    }

    // ------------------------------------------------------------------
    // Epilogue: alpha * acc + beta * C → saturating cast to s8, store to D.
    //
    // D fragment register layout (m16n8 s32):
    //   thread g holds 4 s32 values:
    //     d0 = D[g/4][(g%4)*2]
    //     d1 = D[g/4][(g%4)*2 + 1]
    //     d2 = D[g/4 + 8][(g%4)*2]
    //     d3 = D[g/4 + 8][(g%4)*2 + 1]
    // ------------------------------------------------------------------
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

            // alpha * acc
            float v0 = alpha * (float)acc[mi][ni][0];
            float v1 = alpha * (float)acc[mi][ni][1];
            float v2 = alpha * (float)acc[mi][ni][2];
            float v3 = alpha * (float)acc[mi][ni][3];

            // optional beta * C
            if (beta != 0.0f && C != nullptr) {
                if (row_top < M && col_l < N) {
                    v0 += beta * (float)C[(int64_t)row_top * ldc + col_l];
                }
                if (row_top < M && col_r < N) {
                    v1 += beta * (float)C[(int64_t)row_top * ldc + col_r];
                }
                if (row_bot < M && col_l < N) {
                    v2 += beta * (float)C[(int64_t)row_bot * ldc + col_l];
                }
                if (row_bot < M && col_r < N) {
                    v3 += beta * (float)C[(int64_t)row_bot * ldc + col_r];
                }
            }

            // saturating store
            if (row_top < M && col_l < N) {
                D[(int64_t)row_top * ldd + col_l] = baracuda::sat_cast_f32_to_s8(v0);
            }
            if (row_top < M && col_r < N) {
                D[(int64_t)row_top * ldd + col_r] = baracuda::sat_cast_f32_to_s8(v1);
            }
            if (row_bot < M && col_l < N) {
                D[(int64_t)row_bot * ldd + col_l] = baracuda::sat_cast_f32_to_s8(v2);
            }
            if (row_bot < M && col_r < N) {
                D[(int64_t)row_bot * ldd + col_r] = baracuda::sat_cast_f32_to_s8(v3);
            }
        }
    }
}

} // anonymous namespace

extern "C" {

// Run the S8 RRR Identity kernel.
//
// Status codes follow the baracuda-kernels-sys convention:
//   0 = success
//   1 = misaligned operand            (not enforced today — kernel is
//                                       robust to 1-byte alignment via
//                                       per-byte gmem reads)
//   2 = invalid problem               (M, N, or K non-positive)
//   3 = unsupported                   (e.g. caller passed a feature this
//                                       SKU doesn't implement yet)
//   4 = workspace too small or null   (this SKU has zero workspace)
//   5 = internal kernel error         (kernel launch failure)
int32_t baracuda_kernels_gemm_s8_rrr_sm80_run(
    int32_t m, int32_t n, int32_t k,
    const void *a, int64_t lda,
    const void *b, int64_t ldb,
    const void *c, int64_t ldc,
    void *d, int64_t ldd,
    float alpha, float beta,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void *stream
) {
    if (m <= 0 || n <= 0 || k <= 0) return 2;

    // grid: (ceil(N / N_TILE), ceil(M / M_TILE))
    dim3 grid((unsigned)((n + N_TILE - 1) / N_TILE),
              (unsigned)((m + M_TILE - 1) / M_TILE),
              1u);
    dim3 block((unsigned)NUM_THREADS, 1u, 1u);

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    gemm_s8_rrr_sm80_kernel<<<grid, block, 0, s>>>(
        m, n, k,
        reinterpret_cast<const int8_t *>(a), lda,
        reinterpret_cast<const int8_t *>(b), ldb,
        reinterpret_cast<const int8_t *>(c), ldc,
        reinterpret_cast<int8_t *>(d), ldd,
        alpha, beta
    );

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// This SKU has no caller-supplied workspace.
size_t baracuda_kernels_gemm_s8_rrr_sm80_workspace_size(
    int32_t /*m*/, int32_t /*n*/, int32_t /*k*/
) {
    return 0;
}

// Host-side implementability pre-check. Returns 0 if the kernel can
// launch with the given shape; non-zero with the standard status-code
// mapping otherwise. Today, only the basic shape sanity check; alignment
// is not enforced because the gmem→smem path is byte-granular.
int32_t baracuda_kernels_gemm_s8_rrr_sm80_can_implement(
    int32_t m, int32_t n, int32_t k,
    const void * /*a*/, int64_t /*lda*/,
    const void * /*b*/, int64_t /*ldb*/,
    const void * /*c*/, int64_t /*ldc*/,
    const void * /*d*/, int64_t /*ldd*/
) {
    if (m <= 0 || n <= 0 || k <= 0) return 2;
    return 0;
}

} // extern "C"
