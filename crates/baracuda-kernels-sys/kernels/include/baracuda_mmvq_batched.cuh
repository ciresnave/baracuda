// SPDX-FileCopyrightText: 2023-2024 The llama.cpp / ggml authors  (MIT)
// SPDX-FileCopyrightText: 2024-2026 Fuel project contributors      (MIT OR Apache-2.0)
// SPDX-FileCopyrightText: 2026 baracuda project contributors       (MIT OR Apache-2.0)
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// baracuda_mmvq_batched.cuh
//
// Phase 20.1 — batched MMVQ × N-experts kernel family. General-purpose
// MMVQ primitive that takes N weight matrices + routing semantics, used
// by MoE inference and any other workload that needs (token, expert)
// dispatch.
//
// Op shape:
//   For each dispatch `i ∈ [0, M_total)`:
//     token = sorted_token_ids[i]
//     expert = find e such that expert_offsets[e] <= i < expert_offsets[e+1]
//     w     = topk_weights[i] (or 1.0 if topk_weights == nullptr)
//     for row r ∈ [0, N_rows):
//       acc = w * dot(weights[expert, r, :], activations[token, :])
//       output[token, r] (+)= acc        // (+)= : atomicAdd if top_k > 1
//
// Coverage:
//   * 11 GGUF block formats × 3 activation dtypes (f32 / f16 / bf16)
//     = 33 quant variants. Output dtype = activation dtype.
//   * Pure FP non-quant: f32 / f16 / bf16. Weight + activation + output
//     same dtype = 3 variants.
//
// Dispatch model (template bool):
//   * `top_k == 1` (caller's hint, NO output-row aliasing) → regular
//     store. Each (token, row) is written exactly once.
//   * `top_k > 1` (potential aliasing — multiple dispatches share a
//     token) → atomicAdd via `baracuda::atomic::add<T>` (Phase 11.3
//     helper). Caller must zero-initialize `output` first.
//
// Grid geometry:
//   * One block per (dispatch_index, row) tuple.
//   * Block: (WARP_SIZE, 1, 1). Warp-shuffle reduction over the K axis.
//   * Grid:  (n_rows_per_expert, M_total). `blockIdx.x` = row, `blockIdx.y`
//     = dispatch_index. The block looks up `expert = dispatch_to_expert
//     [dispatch_index]` (precomputed by a tiny prelude kernel from
//     `expert_offsets[]`) and `token = sorted_token_ids[dispatch_index]`.
//
// Why a `dispatch_to_expert[]` prelude rather than per-block binary
// search of `expert_offsets[]`:
//   * Per-block binary search adds O(log N_experts) divergent control
//     flow per block. For M_total × n_rows blocks this multiplies up.
//   * The prelude kernel does O(M_total) integer compares once,
//     coalesced. Workspace cost: M_total × sizeof(i32).
//   * The prelude also drops the per-launch grid-y dependency on having
//     `expert_offsets` accessible from every block (avoiding a redundant
//     gmem read inside the hot kernel).

#ifndef BARACUDA_MMVQ_BATCHED_CUH
#define BARACUDA_MMVQ_BATCHED_CUH

#include "baracuda_atomic.cuh"
#include "baracuda_gguf.cuh"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace baracuda { namespace mmvq_batched {

using baracuda::gguf::QK_K;
using baracuda::gguf::K_SCALE_SIZE;
using baracuda::gguf::WARP_SIZE;
using baracuda::gguf::GGML_CUDA_DMMV_X;
using baracuda::gguf::K_QUANTS_PER_ITERATION;
using baracuda::gguf::dfloat;
using baracuda::gguf::dfloat2;
using baracuda::gguf::dequantize_kernel_t;
using baracuda::gguf::dequantize_q4_0;
using baracuda::gguf::dequantize_q4_1;
using baracuda::gguf::dequantize_q5_0;
using baracuda::gguf::dequantize_q5_1;
using baracuda::gguf::dequantize_q8_0;
using baracuda::gguf::mmvq_io;
using baracuda::gguf::block_q4_0;
using baracuda::gguf::block_q4_1;
using baracuda::gguf::block_q5_0;
using baracuda::gguf::block_q5_1;
using baracuda::gguf::block_q8_0;
using baracuda::gguf::block_q2_K;
using baracuda::gguf::block_q3_K;
using baracuda::gguf::block_q4_K;
using baracuda::gguf::block_q5_K;
using baracuda::gguf::block_q6_K;
using baracuda::gguf::block_q8_K;

// =============================================================================
// Prelude kernel: build `dispatch_to_expert[]` from `expert_offsets[]`.
//
// One grid-stride loop: thread i looks up the expert e such that
// `expert_offsets[e] <= i < expert_offsets[e+1]` via a linear scan
// (small N_experts; coalesced gmem read of `expert_offsets[]`).
// =============================================================================

__global__ inline void compute_dispatch_to_expert_kernel(
    const int32_t * __restrict__ expert_offsets,  // [N_experts + 1]
    int32_t n_experts,
    int32_t m_total,
    int32_t * __restrict__ dispatch_to_expert)    // [m_total]
{
    const int dispatch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (dispatch_idx >= m_total) return;

    // Linear scan — N_experts is small (typically 4-64).
    // Loop invariant: dispatch_idx >= expert_offsets[e] at end.
    int e = 0;
    while (e + 1 <= n_experts && expert_offsets[e + 1] <= dispatch_idx) {
        ++e;
    }
    dispatch_to_expert[dispatch_idx] = e;
}

// =============================================================================
// Atomic-or-direct store helper. Template-bool switch (Phase 17.2 pattern).
// =============================================================================

template <typename T, bool NEED_ATOMIC>
__device__ __forceinline__ void write_out(T * dst, float v) {
    if constexpr (NEED_ATOMIC) {
        T cur;
        baracuda::gguf::mmvq_io<T>::store(&cur, v);
        baracuda::atomic::add<T>(dst, cur);
    } else {
        baracuda::gguf::mmvq_io<T>::store(dst, v);
    }
}

// Direct float store for pure-FP variants that don't go through mmvq_io.
template <bool NEED_ATOMIC>
__device__ __forceinline__ void write_out_f32(float * dst, float v) {
    if constexpr (NEED_ATOMIC) {
        atomicAdd(dst, v);
    } else {
        *dst = v;
    }
}

// =============================================================================
// Type-0/1 (32-element block) batched MMVQ — shares the templated
// `dequantize_kernel` family with the single-MMVQ path.
//
// One block per (dispatch_idx, row). `blockIdx.x` = row, `blockIdx.y` =
// dispatch_idx. The weight base pointer for this dispatch is
// `weights + expert * n_rows_per_expert * blocks_per_row * sizeof(block)`.
// =============================================================================

template <int qk, int qr, dequantize_kernel_t dequantize_kernel,
          typename ActT, typename DstT, bool NEED_ATOMIC>
__device__ inline void mmvq_batched_type01_tmpl(
    const void   * __restrict__ weights,           // [N_experts, n_rows, K_packed]
    const ActT   * __restrict__ activations,       // [M_tokens, K]
    const int32_t * __restrict__ sorted_token_ids, // [M_total]
    const int32_t * __restrict__ dispatch_to_expert, // [M_total]
    const float  * __restrict__ topk_weights,      // [M_total] or nullptr
    DstT         * __restrict__ output,            // [M_tokens, n_rows]
    int n_rows_per_expert, int ncols)
{
    const int row = blockIdx.x;
    const int dispatch_idx = blockIdx.y;
    if (row >= n_rows_per_expert) return;

    const int expert = dispatch_to_expert[dispatch_idx];
    const int token = sorted_token_ids[dispatch_idx];

    // Per-expert weight base pointer (byte arithmetic — matches the
    // host-side `(const u8*)w + e * matrix_bytes` pattern).
    const int blocks_per_row = ncols / qk;
    const int64_t matrix_blocks =
        (int64_t)n_rows_per_expert * (int64_t)blocks_per_row;
    // type-0/1 block sizes vary per format; the templated `dequantize_kernel`
    // dereferences `block_q*_0 *` via index — we hand it `vx + expert *
    // matrix_blocks` and let it index from there. The ib in the inner loop
    // is row-local (`row * blocks_per_row + col/qk`), which is `vx`-relative.
    //
    // Forge the expert-shifted base by reinterpreting per-format struct
    // pointers via byte offsets at the launcher side; here we just walk
    // an `ib` offset.
    // The simpler approach: re-do the dot product loop with an explicit
    // ib offset of `expert * matrix_blocks`.

    const int tid = threadIdx.x;
    const int iter_stride = 2 * GGML_CUDA_DMMV_X;
    const int vals_per_iter = iter_stride / WARP_SIZE;
    const int y_offset = qr == 1 ? 1 : qk / 2;

    const ActT * y = activations + (int64_t)token * (int64_t)ncols;
    const int64_t expert_ib_offset = (int64_t)expert * matrix_blocks;

    float tmp = 0.0f;

    for (int i = 0; i < ncols; i += iter_stride) {
        const int col = i + vals_per_iter * tid;
        const int ib_local = (row * ncols + col) / qk;
        const int64_t ib = expert_ib_offset + (int64_t)ib_local;
        const int iqs = (col % qk) / qr;
        const int iybs = col - col % qk;

#pragma unroll
        for (int j = 0; j < vals_per_iter; j += 2) {
            dfloat2 v;
            dequantize_kernel(weights, (int)ib, iqs + j / qr, v);
            tmp += v.x * mmvq_io<ActT>::load(&y[iybs + iqs + j / qr + 0]);
            tmp += v.y * mmvq_io<ActT>::load(&y[iybs + iqs + j / qr + y_offset]);
        }
    }

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (tid == 0) {
        // Multiply by routing weight (caller may pass nullptr).
        const float w = (topk_weights != nullptr) ? topk_weights[dispatch_idx] : 1.0f;
        const float acc = w * tmp;
        DstT * dst_ptr = output + (int64_t)token * (int64_t)n_rows_per_expert + row;
        write_out<DstT, NEED_ATOMIC>(dst_ptr, acc);
    }
}

// NOTE: `dequantize_kernel` is templated on the integer `ib` index, so
// the `ib` we pass becomes the index into `(const block_qXY*)weights`.
// `dequantize_kernel` takes `int ib` — to safely address up to
// (N_experts × n_rows × blocks_per_row) blocks we cast via an int64_t
// intermediate but rely on the underlying pointer arithmetic in the
// dequantize helpers using i32 arithmetic. For the production sizes
// (N_experts ≤ 64, n_rows ≤ ~16k, blocks_per_row ≤ ~256) the product
// fits in i32. If extreme problem sizes ever exceed i32 we'd need to
// extend `dequantize_kernel_t` to int64_t — flagged as future work.

// =============================================================================
// k-quants batched MMVQ — per-format bespoke (same as Phase 14.5 /
// Phase 18.1 actstrided + f16/bf16 fanout).
//
// We don't shadow the single-MMVQ k-quant templates; we re-derive them
// with two changes vs `mmvq_q*_K_tmpl` in `gguf/mmvq.cu`:
//   1. weight base pointer is `x + expert * matrix_blocks`
//   2. write site uses `write_out<DstT, NEED_ATOMIC>` instead of
//      `mmvq_io<DstT>::store(&dst[row], …)`
//   3. activation base pointer is `activations + token * ncols`
//   4. accumulator multiplied by `topk_weights[dispatch_idx]` (or 1).
// Everything else (the dot-product math, the warp-shuffle reduction, the
// scale/min logic) is line-identical with the single-MMVQ kernels.
// =============================================================================

template <typename ActT, typename DstT, bool NEED_ATOMIC>
__device__ inline void mmvq_batched_q2_K_tmpl(
    const void   * __restrict__ vx,
    const ActT   * __restrict__ activations,
    const int32_t * __restrict__ sorted_token_ids,
    const int32_t * __restrict__ dispatch_to_expert,
    const float  * __restrict__ topk_weights,
    DstT         * __restrict__ output,
    int n_rows_per_expert, int ncols)
{
    static_assert(16 % K_QUANTS_PER_ITERATION == 0,
                  "16 must be divisible by K_QUANTS_PER_ITERATION");

    const int row = blockIdx.x;
    const int dispatch_idx = blockIdx.y;
    if (row >= n_rows_per_expert) return;

    const int expert = dispatch_to_expert[dispatch_idx];
    const int token = sorted_token_ids[dispatch_idx];

    const int num_blocks_per_row = ncols / QK_K;
    const int64_t expert_ib0 = (int64_t)expert * (int64_t)n_rows_per_expert * num_blocks_per_row;
    const int64_t ib0 = expert_ib0 + (int64_t)row * num_blocks_per_row;
    const block_q2_K * x = (const block_q2_K *)vx + ib0;

    const ActT * yy = activations + (int64_t)token * (int64_t)ncols;

    float tmp = 0;

    const int tid = threadIdx.x / K_QUANTS_PER_ITERATION;
    const int ix  = threadIdx.x % K_QUANTS_PER_ITERATION;

    const int step = 16 / K_QUANTS_PER_ITERATION;
    const int im = tid / step;
    const int in = tid - step * im;

    const int l0 = K_QUANTS_PER_ITERATION * in;
    const int q_offset = 32 * im + l0;
    const int s_offset = 8 * im;
    const int y_offset_b = 128 * im + l0;

    uint32_t aux[4];
    const uint8_t * d = (const uint8_t *)aux;
    const uint8_t * m = (const uint8_t *)(aux + 2);

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {
        const ActT * y = yy + i * QK_K + y_offset_b;
        const uint8_t * q = x[i].qs + q_offset;

        const float dall = __low2half(x[i].dm);
        const float dmin = __high2half(x[i].dm);

        const uint32_t * a = (const uint32_t *)(x[i].scales + s_offset);
        aux[0] = a[0] & 0x0f0f0f0f;
        aux[1] = a[1] & 0x0f0f0f0f;
        aux[2] = (a[0] >> 4) & 0x0f0f0f0f;
        aux[3] = (a[1] >> 4) & 0x0f0f0f0f;

        float sum1 = 0, sum2 = 0;
        for (int l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
            #define Y(off) mmvq_io<ActT>::load(&y[off])
            sum1 += Y(l+ 0) * d[0] * ((q[l+ 0] >> 0) & 3)
                  + Y(l+32) * d[2] * ((q[l+ 0] >> 2) & 3)
                  + Y(l+64) * d[4] * ((q[l+ 0] >> 4) & 3)
                  + Y(l+96) * d[6] * ((q[l+ 0] >> 6) & 3)
                  + Y(l+16) * d[1] * ((q[l+16] >> 0) & 3)
                  + Y(l+48) * d[3] * ((q[l+16] >> 2) & 3)
                  + Y(l+80) * d[5] * ((q[l+16] >> 4) & 3)
                  + Y(l+112) * d[7] * ((q[l+16] >> 6) & 3);
            sum2 += Y(l+ 0) * m[0] + Y(l+32) * m[2] + Y(l+64) * m[4] + Y(l+96) * m[6]
                  + Y(l+16) * m[1] + Y(l+48) * m[3] + Y(l+80) * m[5] + Y(l+112) * m[7];
            #undef Y
        }
        tmp += dall * sum1 - dmin * sum2;
    }

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (threadIdx.x == 0) {
        const float w = (topk_weights != nullptr) ? topk_weights[dispatch_idx] : 1.0f;
        DstT * dst_ptr = output + (int64_t)token * (int64_t)n_rows_per_expert + row;
        write_out<DstT, NEED_ATOMIC>(dst_ptr, w * tmp);
    }
}

template <typename ActT, typename DstT, bool NEED_ATOMIC>
__device__ inline void mmvq_batched_q3_K_tmpl(
    const void   * __restrict__ vx,
    const ActT   * __restrict__ activations,
    const int32_t * __restrict__ sorted_token_ids,
    const int32_t * __restrict__ dispatch_to_expert,
    const float  * __restrict__ topk_weights,
    DstT         * __restrict__ output,
    int n_rows_per_expert, int ncols)
{
    const int row = blockIdx.x;
    const int dispatch_idx = blockIdx.y;
    if (row >= n_rows_per_expert) return;

    const int expert = dispatch_to_expert[dispatch_idx];
    const int token = sorted_token_ids[dispatch_idx];

    const int num_blocks_per_row = ncols / QK_K;
    const int64_t expert_ib0 = (int64_t)expert * (int64_t)n_rows_per_expert * num_blocks_per_row;
    const int64_t ib0 = expert_ib0 + (int64_t)row * num_blocks_per_row;
    const block_q3_K * x = (const block_q3_K *)vx + ib0;

    const ActT * yy = activations + (int64_t)token * (int64_t)ncols;

    float tmp = 0;

    const uint16_t kmask1 = 0x0303;
    const uint16_t kmask2 = 0x0f0f;

    const int tid = threadIdx.x / K_QUANTS_PER_ITERATION;
    const int ix  = threadIdx.x % K_QUANTS_PER_ITERATION;

    const int n  = K_QUANTS_PER_ITERATION;
    const int step = 16 / K_QUANTS_PER_ITERATION;
    const int im = tid / step;
    const int in = tid - step * im;

    const uint8_t mb = 1 << (4 * im);

    const int l0 = n * in;
    const int q_offset =  32 * im + l0;
    const int y_offset_b = 128 * im + l0;

    uint16_t utmp[4];
    const int8_t * s = (const int8_t *)utmp;

    const uint16_t s_shift = 4 * im;

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {
        const ActT * y  = yy + i * QK_K + y_offset_b;
        const uint8_t * q = x[i].qs + q_offset;
        const uint8_t * h = x[i].hmask + l0;

        const uint16_t * a = (const uint16_t *)x[i].scales;
        utmp[0] = ((a[0] >> s_shift) & kmask2) | (((a[4] >> (s_shift + 0)) & kmask1) << 4);
        utmp[1] = ((a[1] >> s_shift) & kmask2) | (((a[5] >> (s_shift + 0)) & kmask1) << 4);
        utmp[2] = ((a[2] >> s_shift) & kmask2) | (((a[4] >> (s_shift + 2)) & kmask1) << 4);
        utmp[3] = ((a[3] >> s_shift) & kmask2) | (((a[5] >> (s_shift + 2)) & kmask1) << 4);

        const float d = x[i].d;

        float sum = 0;
        for (int l = 0; l < n; ++l) {
            #define Y(off) mmvq_io<ActT>::load(&y[off])
            sum += Y(l+ 0) * (s[0] - 32) * (((q[l] >> 0) & 3) - (h[l] & (mb << 0) ? 0 : 4))
                 + Y(l+32) * (s[2] - 32) * (((q[l] >> 2) & 3) - (h[l] & (mb << 1) ? 0 : 4))
                 + Y(l+64) * (s[4] - 32) * (((q[l] >> 4) & 3) - (h[l] & (mb << 2) ? 0 : 4))
                 + Y(l+96) * (s[6] - 32) * (((q[l] >> 6) & 3) - (h[l] & (mb << 3) ? 0 : 4));
            sum += Y(l+16) * (s[1] - 32) * (((q[l+16] >> 0) & 3) - (h[l+16] & (mb << 0) ? 0 : 4))
                 + Y(l+48) * (s[3] - 32) * (((q[l+16] >> 2) & 3) - (h[l+16] & (mb << 1) ? 0 : 4))
                 + Y(l+80) * (s[5] - 32) * (((q[l+16] >> 4) & 3) - (h[l+16] & (mb << 2) ? 0 : 4))
                 + Y(l+112) * (s[7] - 32) * (((q[l+16] >> 6) & 3) - (h[l+16] & (mb << 3) ? 0 : 4));
            #undef Y
        }
        tmp += d * sum;
    }

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (threadIdx.x == 0) {
        const float w = (topk_weights != nullptr) ? topk_weights[dispatch_idx] : 1.0f;
        DstT * dst_ptr = output + (int64_t)token * (int64_t)n_rows_per_expert + row;
        write_out<DstT, NEED_ATOMIC>(dst_ptr, w * tmp);
    }
}

template <typename ActT, typename DstT, bool NEED_ATOMIC>
__device__ inline void mmvq_batched_q4_K_tmpl(
    const void   * __restrict__ vx,
    const ActT   * __restrict__ activations,
    const int32_t * __restrict__ sorted_token_ids,
    const int32_t * __restrict__ dispatch_to_expert,
    const float  * __restrict__ topk_weights,
    DstT         * __restrict__ output,
    int n_rows_per_expert, int ncols)
{
    const int row = blockIdx.x;
    const int dispatch_idx = blockIdx.y;
    if (row >= n_rows_per_expert) return;

    const int expert = dispatch_to_expert[dispatch_idx];
    const int token = sorted_token_ids[dispatch_idx];

    const int num_blocks_per_row = ncols / QK_K;
    const int64_t expert_ib0 = (int64_t)expert * (int64_t)n_rows_per_expert * num_blocks_per_row;
    const int64_t ib0 = expert_ib0 + (int64_t)row * num_blocks_per_row;
    const block_q4_K * x = (const block_q4_K *)vx + ib0;

    const ActT * yy = activations + (int64_t)token * (int64_t)ncols;

    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int tid = threadIdx.x / K_QUANTS_PER_ITERATION;
    const int ix  = threadIdx.x % K_QUANTS_PER_ITERATION;

    const int step = 8 / K_QUANTS_PER_ITERATION;

    const int il  = tid / step;
    const int ir  = tid - step * il;
    const int n   = 2 * K_QUANTS_PER_ITERATION;

    const int im = il / 2;
    const int in = il % 2;

    const int l0 = n * (2 * ir + in);
    const int q_offset = 32 * im + l0;
    const int y_offset_b = 64 * im + l0;

    uint16_t aux[4];
    const uint8_t * sc = (const uint8_t *)aux;

    uint32_t q32[4];
    const uint8_t * q4 = (const uint8_t *)q32;

    float tmp = 0;

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {
        const ActT * y1 = yy + i * QK_K + y_offset_b;
        const ActT * y2 = y1 + 128;

        const float dall = __low2half(x[i].dm);
        const float dmin = __high2half(x[i].dm);

        const uint16_t * a = (const uint16_t *)x[i].scales;
        aux[0] = a[im+0] & kmask1;
        aux[1] = a[im+2] & kmask1;
        aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
        aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

        const uint32_t * q1 = (const uint32_t *)(x[i].qs + q_offset);
        const uint32_t * q2 = q1 + 16;

        q32[0] = q1[0] & 0x0f0f0f0f;
        q32[1] = q1[0] & 0xf0f0f0f0;
        q32[2] = q2[0] & 0x0f0f0f0f;
        q32[3] = q2[0] & 0xf0f0f0f0;

        float4 s = {0.f, 0.f, 0.f, 0.f};
        float smin = 0;
        for (int l = 0; l < 4; ++l) {
            #define Y1(off) mmvq_io<ActT>::load(&y1[off])
            #define Y2(off) mmvq_io<ActT>::load(&y2[off])
            s.x += Y1(l)    * q4[l+0]; s.y += Y1(l+32) * q4[l+ 4];
            s.z += Y2(l)    * q4[l+8]; s.w += Y2(l+32) * q4[l+12];
            smin += Y1(l) * sc[2] + Y1(l+32) * sc[3] + Y2(l) * sc[6] + Y2(l+32) * sc[7];
            #undef Y1
            #undef Y2
        }
        tmp += dall * (s.x * sc[0] + s.y * sc[1] * 1.f/16.f + s.z * sc[4] + s.w * sc[5] * 1.f/16.f) - dmin * smin;
    }

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (tid == 0) {
        const float w = (topk_weights != nullptr) ? topk_weights[dispatch_idx] : 1.0f;
        DstT * dst_ptr = output + (int64_t)token * (int64_t)n_rows_per_expert + row;
        write_out<DstT, NEED_ATOMIC>(dst_ptr, w * tmp);
    }
}

template <typename ActT, typename DstT, bool NEED_ATOMIC>
__device__ inline void mmvq_batched_q5_K_tmpl(
    const void   * __restrict__ vx,
    const ActT   * __restrict__ activations,
    const int32_t * __restrict__ sorted_token_ids,
    const int32_t * __restrict__ dispatch_to_expert,
    const float  * __restrict__ topk_weights,
    DstT         * __restrict__ output,
    int n_rows_per_expert, int ncols)
{
    const int row = blockIdx.x;
    const int dispatch_idx = blockIdx.y;
    if (row >= n_rows_per_expert) return;

    const int expert = dispatch_to_expert[dispatch_idx];
    const int token = sorted_token_ids[dispatch_idx];

    const int num_blocks_per_row = ncols / QK_K;
    const int64_t expert_ib0 = (int64_t)expert * (int64_t)n_rows_per_expert * num_blocks_per_row;
    const int64_t ib0 = expert_ib0 + (int64_t)row * num_blocks_per_row;
    const block_q5_K * x = (const block_q5_K *)vx + ib0;

    const ActT * yy = activations + (int64_t)token * (int64_t)ncols;

    float tmp = 0;

    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int tid = threadIdx.x / 2;
    const int ix  = threadIdx.x % 2;

    const int il  = tid / 4;
    const int ir  = tid - 4 * il;
    const int n   = 2;

    const int im = il / 2;
    const int in = il % 2;

    const int l0 = n * (2 * ir + in);
    const int q_offset = 32 * im + l0;
    const int y_offset_b = 64 * im + l0;

    const uint8_t hm1  = 1 << (2 * im);
    const uint8_t hm2  = hm1 << 4;

    uint16_t aux[4];
    const uint8_t * sc = (const uint8_t *)aux;

    uint16_t q16[8];
    const uint8_t * q4 = (const uint8_t *)q16;

    for (int i = ix; i < num_blocks_per_row; i += 2) {
        const uint8_t * ql1 = x[i].qs + q_offset;
        const uint8_t * qh  = x[i].qh + l0;
        const ActT * y1  = yy + i * QK_K + y_offset_b;
        const ActT * y2  = y1 + 128;

        const float dall = __low2half(x[i].dm);
        const float dmin = __high2half(x[i].dm);

        const uint16_t * a = (const uint16_t *)x[i].scales;
        aux[0] = a[im+0] & kmask1;
        aux[1] = a[im+2] & kmask1;
        aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
        aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

        float4 sum = {0.f, 0.f, 0.f, 0.f};
        float smin = 0;
        const uint16_t * q1 = (const uint16_t *)ql1;
        const uint16_t * q2 = q1 + 32;
        q16[0] = q1[0] & 0x0f0f;
        q16[1] = q1[8] & 0x0f0f;
        q16[2] = (q1[0] >> 4) & 0x0f0f;
        q16[3] = (q1[8] >> 4) & 0x0f0f;
        q16[4] = q2[0] & 0x0f0f;
        q16[5] = q2[8] & 0x0f0f;
        q16[6] = (q2[0] >> 4) & 0x0f0f;
        q16[7] = (q2[8] >> 4) & 0x0f0f;
        for (int l = 0; l < n; ++l) {
            #define Y1(off) mmvq_io<ActT>::load(&y1[off])
            #define Y2(off) mmvq_io<ActT>::load(&y2[off])
            sum.x += Y1(l+ 0) * (q4[l +0] + (qh[l+ 0] & (hm1 << 0) ? 16 : 0))
                   + Y1(l+16) * (q4[l +2] + (qh[l+16] & (hm1 << 0) ? 16 : 0));
            sum.y += Y1(l+32) * (q4[l +4] + (qh[l+ 0] & (hm1 << 1) ? 16 : 0))
                   + Y1(l+48) * (q4[l +6] + (qh[l+16] & (hm1 << 1) ? 16 : 0));
            sum.z += Y2(l+ 0) * (q4[l +8] + (qh[l+ 0] & (hm2 << 0) ? 16 : 0))
                   + Y2(l+16) * (q4[l+10] + (qh[l+16] & (hm2 << 0) ? 16 : 0));
            sum.w += Y2(l+32) * (q4[l+12] + (qh[l+ 0] & (hm2 << 1) ? 16 : 0))
                   + Y2(l+48) * (q4[l+14] + (qh[l+16] & (hm2 << 1) ? 16 : 0));
            smin += (Y1(l) + Y1(l+16)) * sc[2] + (Y1(l+32) + Y1(l+48)) * sc[3]
                  + (Y2(l) + Y2(l+16)) * sc[6] + (Y2(l+32) + Y2(l+48)) * sc[7];
            #undef Y1
            #undef Y2
        }
        tmp += dall * (sum.x * sc[0] + sum.y * sc[1] + sum.z * sc[4] + sum.w * sc[5]) - dmin * smin;
    }

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (threadIdx.x == 0) {
        const float w = (topk_weights != nullptr) ? topk_weights[dispatch_idx] : 1.0f;
        DstT * dst_ptr = output + (int64_t)token * (int64_t)n_rows_per_expert + row;
        write_out<DstT, NEED_ATOMIC>(dst_ptr, w * tmp);
    }
}

template <typename ActT, typename DstT, bool NEED_ATOMIC>
__device__ inline void mmvq_batched_q6_K_tmpl(
    const void   * __restrict__ vx,
    const ActT   * __restrict__ activations,
    const int32_t * __restrict__ sorted_token_ids,
    const int32_t * __restrict__ dispatch_to_expert,
    const float  * __restrict__ topk_weights,
    DstT         * __restrict__ output,
    int n_rows_per_expert, int ncols)
{
    static_assert(16 % K_QUANTS_PER_ITERATION == 0,
                  "16 must be divisible by K_QUANTS_PER_ITERATION");

    const int row = blockIdx.x;
    const int dispatch_idx = blockIdx.y;
    if (row >= n_rows_per_expert) return;

    const int expert = dispatch_to_expert[dispatch_idx];
    const int token = sorted_token_ids[dispatch_idx];

    const int num_blocks_per_row = ncols / QK_K;
    const int64_t expert_ib0 = (int64_t)expert * (int64_t)n_rows_per_expert * num_blocks_per_row;
    const int64_t ib0 = expert_ib0 + (int64_t)row * num_blocks_per_row;
    const block_q6_K * x = (const block_q6_K *)vx + ib0;

    const ActT * yy = activations + (int64_t)token * (int64_t)ncols;

    const int tid = threadIdx.x / K_QUANTS_PER_ITERATION;
    const int ix  = threadIdx.x % K_QUANTS_PER_ITERATION;

    const int step = 16 / K_QUANTS_PER_ITERATION;
    const int im = tid / step;
    const int in = tid - step * im;

    const int l0 = 4 * in;
    const int is = in / 4;

    const int ql_offset = 64 * im + l0;
    const int qh_offset = 32 * im + l0;
    const int s_offset  =  8 * im + is;
    const int y_offset_b = 128 * im + l0;

    float tmp = 0;

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {
        const ActT * y  = yy + i * QK_K + y_offset_b;
        const uint8_t * ql = x[i].ql + ql_offset;
        const uint8_t * qh = x[i].qh + qh_offset;
        const int8_t  * s  = x[i].scales + s_offset;

        const float d = x[i].d;

        float sum = 0;
        for (int l = 0; l < 4; ++l) {
            #define Y(off) mmvq_io<ActT>::load(&y[off])
            sum += Y(l+ 0) * s[0] * d * ((int8_t)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32)
                 + Y(l+32) * s[2] * d * ((int8_t)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32)
                 + Y(l+64) * s[4] * d * ((int8_t)((ql[l+ 0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32)
                 + Y(l+96) * s[6] * d * ((int8_t)((ql[l+32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            #undef Y
        }
        tmp += sum;
    }

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (tid == 0) {
        const float w = (topk_weights != nullptr) ? topk_weights[dispatch_idx] : 1.0f;
        DstT * dst_ptr = output + (int64_t)token * (int64_t)n_rows_per_expert + row;
        write_out<DstT, NEED_ATOMIC>(dst_ptr, w * tmp);
    }
}

template <typename ActT, typename DstT, bool NEED_ATOMIC>
__device__ inline void mmvq_batched_q8_K_tmpl(
    const void   * __restrict__ vx,
    const ActT   * __restrict__ activations,
    const int32_t * __restrict__ sorted_token_ids,
    const int32_t * __restrict__ dispatch_to_expert,
    const float  * __restrict__ topk_weights,
    DstT         * __restrict__ output,
    int n_rows_per_expert, int ncols)
{
    static_assert(QK_K % 16 == 0, "QK_K must be a multiple of 16");
    static_assert(16 % K_QUANTS_PER_ITERATION == 0,
                  "16 must be divisible by K_QUANTS_PER_ITERATION");

    const int row = blockIdx.x;
    const int dispatch_idx = blockIdx.y;
    if (row >= n_rows_per_expert) return;

    const int expert = dispatch_to_expert[dispatch_idx];
    const int token = sorted_token_ids[dispatch_idx];

    const int num_blocks_per_row = ncols / QK_K;
    const int64_t expert_ib0 = (int64_t)expert * (int64_t)n_rows_per_expert * num_blocks_per_row;
    const int64_t ib0 = expert_ib0 + (int64_t)row * num_blocks_per_row;
    const block_q8_K * x = (const block_q8_K *)vx + ib0;

    const ActT * yy = activations + (int64_t)token * (int64_t)ncols;

    const int tid = threadIdx.x / K_QUANTS_PER_ITERATION;
    const int ix  = threadIdx.x % K_QUANTS_PER_ITERATION;
    constexpr int CHUNK = QK_K / 16;
    const int l0 = tid * CHUNK;

    float tmp = 0.0f;

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {
        const float    d = x[i].d;
        const int8_t * q = x[i].qs + l0;
        const ActT   * y = yy + i * QK_K + l0;

        float fsum = 0.0f;
#pragma unroll
        for (int l = 0; l < CHUNK; ++l) {
            fsum += (float)q[l] * mmvq_io<ActT>::load(&y[l]);
        }
        tmp += d * fsum;
    }

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (threadIdx.x == 0) {
        const float w = (topk_weights != nullptr) ? topk_weights[dispatch_idx] : 1.0f;
        DstT * dst_ptr = output + (int64_t)token * (int64_t)n_rows_per_expert + row;
        write_out<DstT, NEED_ATOMIC>(dst_ptr, w * tmp);
    }
}

// =============================================================================
// Pure-FP batched MMVQ — no quantization. Weight + activation + output
// share the same dtype `T`. Accumulator stays f32 in every variant.
//
// Same grid geometry as the quant variants. Block: WARP_SIZE threads;
// warp-shuffle reduction over K. The K axis is chunked into `vals_per_iter
// = 2` per thread per iteration (matches the type-0/1 cadence — 32
// threads × 2 elements = 64 K values per iteration).
// =============================================================================

template <typename T, bool NEED_ATOMIC>
__device__ inline void mmvq_batched_fp_tmpl(
    const T      * __restrict__ weights,           // [N_experts, n_rows, K]
    const T      * __restrict__ activations,       // [M_tokens, K]
    const int32_t * __restrict__ sorted_token_ids,
    const int32_t * __restrict__ dispatch_to_expert,
    const float  * __restrict__ topk_weights,
    T            * __restrict__ output,            // [M_tokens, n_rows]
    int n_rows_per_expert, int ncols)
{
    const int row = blockIdx.x;
    const int dispatch_idx = blockIdx.y;
    if (row >= n_rows_per_expert) return;

    const int expert = dispatch_to_expert[dispatch_idx];
    const int token = sorted_token_ids[dispatch_idx];

    const int tid = threadIdx.x;

    const int64_t weight_row_base =
        (int64_t)expert * (int64_t)n_rows_per_expert * (int64_t)ncols
        + (int64_t)row * (int64_t)ncols;
    const T * w_row = weights + weight_row_base;
    const T * y = activations + (int64_t)token * (int64_t)ncols;

    float tmp = 0.0f;

    // Coalesced K-axis stride: each thread strides by `WARP_SIZE`.
    for (int k = tid; k < ncols; k += WARP_SIZE) {
        const float wv = mmvq_io<T>::load(&w_row[k]);
        const float yv = mmvq_io<T>::load(&y[k]);
        tmp += wv * yv;
    }

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (tid == 0) {
        const float wgt = (topk_weights != nullptr) ? topk_weights[dispatch_idx] : 1.0f;
        T * dst_ptr = output + (int64_t)token * (int64_t)n_rows_per_expert + row;
        write_out<T, NEED_ATOMIC>(dst_ptr, wgt * tmp);
    }
}

}} // namespace baracuda::mmvq_batched

#endif // BARACUDA_MMVQ_BATCHED_CUH
