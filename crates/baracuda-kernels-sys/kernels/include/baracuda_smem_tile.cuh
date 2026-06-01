// SPDX-FileCopyrightText: 2026 Eric Evans and the baracuda contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// baracuda_smem_tile.cuh — bank-conflict-padded 2D SMEM tile + cooperative
// tile load/store helpers for the "stage a 2D sub-block of a matrix in
// shared memory, then compute over it" kernel pattern (GEMM, conv im2col,
// attention QK^T, …).
//
// The footgun this captures: a tile of shape `[ROWS, COLS]` allocated as
// the natural `T smem[ROWS][COLS]` has every column `c` landing in SMEM
// bank `(c) % 32` for `sizeof(T) == 4`. When a warp's 32 threads stride
// down a *column* (the inner-product direction in a matmul), all 32 hit
// the same bank → a 32-way bank conflict, serializing the access. The
// canonical fix is one element of padding on the inner dimension:
// `T smem[ROWS][COLS + 1]`. The `+1` shifts each row's bank alignment by
// one, so a column access now spreads across all 32 banks. Forgetting the
// `+1` is a classic silent ~32× SMEM-throughput regression — making it a
// typed, named, default-on property of the tile type removes the footgun.
//
//   __shared__ baracuda::tile::SmemTile2D<float, 64, 16> a_tile;  // [64][16+1]
//   baracuda::tile::tile_load_row_major<256>(a_tile, A_ptr, lda);
//   // ... warp reads a_tile(r, k) down k without bank conflicts ...
//
// PAD defaults to 1 (correct for 4-byte elements). It is a template arg
// so callers with 8-byte elements (f64) that still conflict at PAD=1 can
// bump to PAD=2, and callers that genuinely want a dense tile (e.g. when
// the consumer is a warp-MMA `ldmatrix` path that does not stride columns
// scalar-wise) can pass PAD=0.
//
// Relationship to other helpers / libraries:
//   * `baracuda_smem_row_stager.cuh` (Phase 65a) stages a *1D* row of
//     runtime length — for the "one block per row" normalizer pattern.
//     This header is its 2D, compile-time-shaped sibling for matmul tiles.
//   * CUB's `cub::BlockLoad<T, BLOCK_THREADS, ITEMS_PER_THREAD>` loads a
//     1D segment into registers with a chosen access pattern. It does NOT
//     own a padded 2D SMEM tile — this helper complements it, it does not
//     replace it. Use CUB when you want register-resident 1D tiles; use
//     this when you want a padded 2D SMEM tile the whole block reads.
//
// LIMITATION — does not cover swizzled-SMEM tensor-core kernels.
// baracuda's MMA-fragment GEMMs (`baracuda_int8_rrr_sm80.cuh`,
// `baracuda_{fp8,int4,bin}_*_sm89.cuh`) declare *dense* `[M_TILE][K_TILE]`
// SMEM tiles and feed them to `ldmatrix` / warp-MMA, where the access
// pattern is fixed by the hardware fragment layout rather than scalar
// column strides — padding there would break the `ldmatrix` addressing.
// Those kernels are intentionally out of scope. This helper targets
// SIMT-style tiled kernels that read the tile with scalar column strides.
//
// Pure device-side, header-only, zero side effects on inclusion. The
// only host-visible members are the `constexpr` shape accessors.

#ifndef BARACUDA_SMEM_TILE_CUH
#define BARACUDA_SMEM_TILE_CUH

#include <cstddef>
#include <type_traits>
#include <cuda_runtime.h>

namespace baracuda {
namespace tile {

// =============================================================================
// SmemTile2D — bank-conflict-padded 2D shared-memory tile.
// =============================================================================
//
// Storage is `T data[ROWS][COLS + PAD]`. Indexing via `operator()(r, c)`
// hides the padding — callers address the logical `[ROWS, COLS]` shape and
// never see the extra column. `bytes()` reports the true allocation size
// (including padding) for callers sizing a dynamic-SMEM launch.
template <typename T, int ROWS, int COLS, int PAD = 1>
struct SmemTile2D {
    using value_type = T;

    T data[ROWS][COLS + PAD];

    __device__ __forceinline__ T&       operator()(int r, int c)       { return data[r][c]; }
    __device__ __forceinline__ const T& operator()(int r, int c) const { return data[r][c]; }

    static constexpr __device__ __host__ int rows() { return ROWS; }
    static constexpr __device__ __host__ int cols() { return COLS; }
    static constexpr __device__ __host__ int pad()  { return PAD; }
    static constexpr __device__ __host__ std::size_t bytes() {
        return static_cast<std::size_t>(ROWS) *
               static_cast<std::size_t>(COLS + PAD) * sizeof(T);
    }
};

// -----------------------------------------------------------------------------
// Internal: linear thread index within the block (supports 1D / 2D / 3D
// blocks). `BLOCK_THREADS` template arg on the public helpers must equal
// `blockDim.x * blockDim.y * blockDim.z`.
// -----------------------------------------------------------------------------
__device__ __forceinline__ int tile_linear_tid() {
    return threadIdx.x +
           blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
}

// Compile-time predicate: can this (tile element, transfer element) pair use
// the float4-vectorized fast path? Requires both sides to be `float` and the
// tile's inner extent to be a multiple of 4 (so whole rows split into float4
// groups). The runtime stride-alignment check is applied separately.
template <typename Tile, typename XferT>
__device__ __forceinline__ constexpr bool tile_can_vec4() {
    return std::is_same<XferT, float>::value &&
           std::is_same<typename Tile::value_type, float>::value &&
           (Tile::cols() % 4 == 0);
}

// =============================================================================
// tile_load_row_major — cooperative load of a [ROWS, COLS] tile from a
// row-major source matrix. `src` points at the tile's (0,0) element;
// `src_stride_row` is the element stride between consecutive source rows
// (the matrix's leading dimension). Synchronizes before returning so the
// staged tile is visible to every thread.
//
// f32 fast path: when the tile + source are `float`, `COLS % 4 == 0`, and
// `src_stride_row % 4 == 0`, global reads are issued as `float4` (coalesced,
// one transaction per 4 elements) and scattered to the padded SMEM tile with
// scalar stores. Caller must ensure `src` is 16-byte aligned for this path;
// if `src_stride_row` is not a multiple of 4 the scalar path is taken at
// runtime regardless of dtype.
// =============================================================================
template <int BLOCK_THREADS, typename Tile, typename SrcT>
__device__ __forceinline__ void tile_load_row_major(
    Tile& tile,
    const SrcT* __restrict__ src,
    int src_stride_row)
{
    constexpr int ROWS = Tile::rows();
    constexpr int COLS = Tile::cols();
    const int tid = tile_linear_tid();

    if constexpr (tile_can_vec4<Tile, SrcT>()) {
        if ((src_stride_row & 3) == 0) {
            constexpr int VCOLS = COLS / 4;            // float4 groups per row
            constexpr int VTOTAL = ROWS * VCOLS;
            for (int v = tid; v < VTOTAL; v += BLOCK_THREADS) {
                const int r = v / VCOLS;
                const int c = (v % VCOLS) * 4;
                const float4 packed =
                    *reinterpret_cast<const float4*>(&src[r * src_stride_row + c]);
                tile(r, c + 0) = packed.x;
                tile(r, c + 1) = packed.y;
                tile(r, c + 2) = packed.z;
                tile(r, c + 3) = packed.w;
            }
            __syncthreads();
            return;
        }
    }

    // Scalar fallback (non-float, COLS%4!=0, or unaligned stride).
    constexpr int TOTAL = ROWS * COLS;
    for (int i = tid; i < TOTAL; i += BLOCK_THREADS) {
        const int r = i / COLS;
        const int c = i % COLS;
        tile(r, c) = static_cast<typename Tile::value_type>(src[r * src_stride_row + c]);
    }
    __syncthreads();
}

// =============================================================================
// tile_store_row_major — cooperative writeback of a [ROWS, COLS] tile to a
// row-major destination. Symmetric to the load. Does NOT synchronize at the
// end — the caller decides whether a trailing `__syncthreads()` is needed
// (often the next loop iteration's load does not depend on this store).
//
// Same f32 fast path as the load (read 4 scalars from SMEM, write `float4`).
// =============================================================================
template <int BLOCK_THREADS, typename Tile, typename DstT>
__device__ __forceinline__ void tile_store_row_major(
    DstT* __restrict__ dst,
    const Tile& tile,
    int dst_stride_row)
{
    constexpr int ROWS = Tile::rows();
    constexpr int COLS = Tile::cols();
    const int tid = tile_linear_tid();

    if constexpr (tile_can_vec4<Tile, DstT>()) {
        if ((dst_stride_row & 3) == 0) {
            constexpr int VCOLS = COLS / 4;
            constexpr int VTOTAL = ROWS * VCOLS;
            for (int v = tid; v < VTOTAL; v += BLOCK_THREADS) {
                const int r = v / VCOLS;
                const int c = (v % VCOLS) * 4;
                float4 packed;
                packed.x = tile(r, c + 0);
                packed.y = tile(r, c + 1);
                packed.z = tile(r, c + 2);
                packed.w = tile(r, c + 3);
                *reinterpret_cast<float4*>(&dst[r * dst_stride_row + c]) = packed;
            }
            return;
        }
    }

    constexpr int TOTAL = ROWS * COLS;
    for (int i = tid; i < TOTAL; i += BLOCK_THREADS) {
        const int r = i / COLS;
        const int c = i % COLS;
        dst[r * dst_stride_row + c] = static_cast<DstT>(tile(r, c));
    }
}

// =============================================================================
// tile_load_col_major — cooperative load of a [ROWS, COLS] *logical* tile
// from a column-major source. `tile(r, c)` is filled from
// `src[c * src_stride_col + r]`, i.e. the source stores the same logical
// matrix transposed (consecutive elements down a column are contiguous in
// `src`, `src_stride_col` apart between columns). This is the layout the
// B / K operand takes in `C = A · Bᵀ` (GEMM RCR) and attention QKᵀ.
//
// Scalar only in this first ship: vectorizing a column-major read would
// require gathering 4 consecutive *rows* (contiguous in `src`) and scattering
// them down a tile column — a worthwhile follow-up but kept out of v1 to
// limit surface. Synchronizes before returning.
// =============================================================================
template <int BLOCK_THREADS, typename Tile, typename SrcT>
__device__ __forceinline__ void tile_load_col_major(
    Tile& tile,
    const SrcT* __restrict__ src,
    int src_stride_col)
{
    constexpr int ROWS = Tile::rows();
    constexpr int COLS = Tile::cols();
    constexpr int TOTAL = ROWS * COLS;
    const int tid = tile_linear_tid();

    for (int i = tid; i < TOTAL; i += BLOCK_THREADS) {
        const int r = i / COLS;
        const int c = i % COLS;
        tile(r, c) = static_cast<typename Tile::value_type>(src[c * src_stride_col + r]);
    }
    __syncthreads();
}

}  // namespace tile
}  // namespace baracuda

#endif  // BARACUDA_SMEM_TILE_CUH
