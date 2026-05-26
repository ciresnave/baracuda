// SPDX-FileCopyrightText: 2023-2024 The llama.cpp / ggml authors  (MIT)
// SPDX-FileCopyrightText: 2024-2026 Fuel project contributors      (MIT OR Apache-2.0)
// SPDX-FileCopyrightText: 2026 baracuda project contributors       (MIT OR Apache-2.0)
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// baracuda_mmvq_multim.cuh
//
// Phase 33 — Multi-M MMVQ via Q8_1 activation staging.
//
// This header ports llama.cpp's `mul_mat_vec_q<ncols_y, ...>` design into
// baracuda. The key idea: at compile time `ncols_y ∈ {1, 2, 4, 8}` is the
// number of activation vectors processed per kernel launch; each block
// accumulates `ncols_y × rows_per_cuda_block` partial sums so a **single
// weight load services up to 8 dot products** — recovering the gmem
// bandwidth that the M=1 path spent rereading the weight tensor once per
// token.
//
// The activations must first be staged into the Q8_1 block format by the
// `quantize_q8_1_*` kernel (companion file `gguf/quantize_q8_1.cu`).
// Staging is a one-shot ~µs prelude per inference step that amortizes
// across multiple matmuls reusing the same activation set.
//
// Scope
// -----
// Phase 33 ships **Q8_0** weights only (the simplest mapping — int8
// quants + half scale, dot via two `__dp4a` per vdr=2 chunk). The
// remaining 9 GGUF block formats (Q4_0/Q4_1/Q5_0/Q5_1/Q2_K..Q6_K)
// have proportionally more involved bit-unpacking and per-format
// `vec_dot_q*_q8_1` helpers; they will follow in a subsequent phase.
//
// Numerical equivalence with the M=1 FP path is **not** exact: the M=1
// path dequantizes the weight to fp32 first and multiplies by the fp
// activation; the staging path quantizes the activation to int8 first
// and uses an `__dp4a` SIMD dot, then a `d_x * d_y * sumi` rescale at
// the block boundary. The relative error is bounded by the ratio of the
// per-32-element activation range to 127 (typical ~ 1e-3 ringup), but
// model-quality comparisons (perplexity / generation outputs) are the
// preferred verifier.

#ifndef BARACUDA_MMVQ_MULTIM_CUH
#define BARACUDA_MMVQ_MULTIM_CUH

#include "baracuda_gguf.cuh"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace baracuda { namespace mmvq_multim {

using baracuda::gguf::WARP_SIZE;
using baracuda::gguf::mmvq_io;
using baracuda::gguf::block_q8_0;

// ----- Q8_1 block layout (matches llama.cpp / Fuel) ------------------------
//
// `block_q8_1` is the staging format produced by `quantize_q8_1`. It mirrors
// `block_q8_0` but adds a per-32-element activation sum (used by the bias
// term in Q4_1 / Q5_1; for Q8_0 the sum is ignored). 36 bytes per block.

inline constexpr int QK8_1 = 32;
inline constexpr int QI8_1 = QK8_1 / 4;       // = 8 int32-chunks
inline constexpr int VDR_Q8_0_Q8_1_MMVQ = 2;  // matches llama.cpp's choice

typedef struct __align__(4) {
    __half2 ds;       // ds.x = delta, ds.y = sum
    int8_t  qs[QK8_1];
} block_q8_1;
static_assert(sizeof(block_q8_1) == 2 * sizeof(__half) + QK8_1,
              "wrong block_q8_1 size/padding");

// ----- DP4A wrapper --------------------------------------------------------
//
// `__dp4a` is sm_61+; baracuda targets sm_80+ so the intrinsic is always
// available. Wrap it for clarity (and to match the Fuel naming).

static __device__ __forceinline__ int q8_1_dp4a(int a, int b, int c) {
    return __dp4a(a, b, c);
}

// ----- get_int_from_int8 / get_int_from_int8_aligned ----------------------
//
// Load 4 contiguous int8 quants as a packed int32. The "aligned" variant
// assumes 4-byte alignment of the source pointer (true for `block_q8_1.qs`
// and for `block_q8_0.qs` at offset 2 inside the 34-byte block).

static __device__ __forceinline__ int q8_get_int_from_int8(
    const int8_t * x8, const int i32)
{
    const uint16_t * x16 = reinterpret_cast<const uint16_t *>(
        x8 + sizeof(int) * i32);
    int x32 = 0;
    x32 |= x16[0] <<  0;
    x32 |= x16[1] << 16;
    return x32;
}

static __device__ __forceinline__ int q8_get_int_from_int8_aligned(
    const int8_t * x8, const int i32)
{
    return *reinterpret_cast<const int *>(x8 + sizeof(int) * i32);
}

// ----- Inner dot kernel: vec_dot_q8_0_q8_1_impl --------------------------
//
// Per 32-element block: `vdr` int8x4 SIMD MACs + one fp32 fixup.

template <int vdr>
static __device__ __forceinline__ float vec_dot_q8_0_q8_1_impl(
    const int * v, const int * u, float d8_0, float d8_1)
{
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        sumi = q8_1_dp4a(v[i], u[i], sumi);
    }
    return d8_0 * d8_1 * static_cast<float>(sumi);
}

// ----- Per-block load + dot wrapper --------------------------------------
//
// Pull `VDR_Q8_0_Q8_1_MMVQ` int32-chunks of weight + activation, call the
// impl. `iqs` is the int-chunk index inside the block (0..QI8_0-vdr+1).

static __device__ __forceinline__ float vec_dot_q8_0_q8_1(
    const void * __restrict__ vbq,
    const block_q8_1 * __restrict__ bq8_1,
    const int iqs)
{
    const block_q8_0 * bq8_0 = reinterpret_cast<const block_q8_0 *>(vbq);

    int v[VDR_Q8_0_Q8_1_MMVQ];
    int u[VDR_Q8_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        v[i] = q8_get_int_from_int8(bq8_0->qs, iqs + i);
        u[i] = q8_get_int_from_int8_aligned(bq8_1->qs, iqs + i);
    }

    const float d8_0 = __half2float(bq8_0->d);
    const float d8_1 = __low2float(bq8_1->ds);
    return vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(v, u, d8_0, d8_1);
}

// ----- Multi-M MMVQ kernel template --------------------------------------
//
// `ncols_y` is the number of activation vectors (= M). Each block
// accumulates `ncols_y × rows_per_cuda_block` partial sums; one weight
// load services `ncols_y` dot products → up to 8× gmem bandwidth save
// vs the M=1 path.
//
// Grid geometry (set by the launcher):
//   block.x = WARP_SIZE
//   block.y = nwarps (4 for ncols_y ≤ 4, 2 for ncols_y > 4)
//   grid.x  = ceil(nrows_x / rows_per_cuda_block)
//
// rows_per_cuda_block = 1 for ncols_y == 1, 2 otherwise (multi-row
// amortization is worth the extra accumulator pressure only when ncols_y
// already pays for the deeper reuse).
//
// `nrows_y` is in **unpacked** elements (= ncols_x in this 1D MMVQ shape).
// `nrows_dst` is the leading dim of the dst matrix (typically = nrows_x).

template <int ncols_y, int qk, int qi, typename block_q_t, int vdr,
          float (*vec_dot_q_cuda)(const void *, const block_q8_1 *, int)>
static __device__ __forceinline__ void mul_mat_vec_q_multim_tmpl(
    const void * __restrict__ vx,
    const void * __restrict__ vy,
    float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst)
{
    constexpr int nwarps              = (ncols_y <= 4) ? 4 : 2;
    constexpr int rows_per_cuda_block = (ncols_y == 1) ? 1 : 2;

    const int tid  = WARP_SIZE * threadIdx.y + threadIdx.x;
    const int row0 = rows_per_cuda_block * blockIdx.x;
    const int blocks_per_row_x = ncols_x / qk;
    const int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int blocks_per_iter = vdr * nwarps * WARP_SIZE / qi;

    // Partial sum for each thread.
    float tmp[ncols_y][rows_per_cuda_block] = {{0.0f}};

    const block_q_t  * x = reinterpret_cast<const block_q_t  *>(vx);
    const block_q8_1 * y = reinterpret_cast<const block_q8_1 *>(vy);

    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);          // matching y block
        const int kqs = vdr * (tid % (qi / vdr));    // int-chunk index within the block

#pragma unroll
        for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] += vec_dot_q_cuda(
                    &x[kbx + (row0 + i) * blocks_per_row_x],
                    &y[j * blocks_per_col_y + kby],
                    kqs);
            }
        }
    }

    // Cross-warp reduction via shared memory (nwarps > 1 case only).
    __shared__ float tmp_shared[nwarps - 1 > 0 ? nwarps - 1 : 1][ncols_y][rows_per_cuda_block][WARP_SIZE];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    // Sum partial sums and write back (only warp 0 reaches here).
#pragma unroll
    for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps - 1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
            }
            // Warp reduction.
#pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                tmp[j][i] += __shfl_xor_sync(0xffffffff, tmp[j][i], mask, 32);
            }
        }

        if (threadIdx.x < rows_per_cuda_block && (row0 + threadIdx.x) < nrows_x) {
            dst[j * nrows_dst + row0 + threadIdx.x] = tmp[j][threadIdx.x];
        }
    }
}

}} // namespace baracuda::mmvq_multim

#endif // BARACUDA_MMVQ_MULTIM_CUH
