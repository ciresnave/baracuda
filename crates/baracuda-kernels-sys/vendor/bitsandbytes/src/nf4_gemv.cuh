// SPDX-FileCopyrightText: 2022-2024 Tim Dettmers and the bitsandbytes contributors  (MIT)
// SPDX-FileCopyrightText: 2026 baracuda project contributors                          (MIT OR Apache-2.0)
// SPDX-License-Identifier: MIT
//
// nf4_gemv.cuh — Phase 53 NF4 dequant + GEMV kernel templates.
//
// All kernels use a single-output-row-per-block layout — `gridDim.x =
// N`, `blockDim.x = WARP_SIZE` (32). The kernel iterates K columns in
// 32-element strides, accumulating in `f32` registers. A final warp
// shuffle reduces the partial sums into lane 0, which writes the
// output cell.
//
// For multi-M variants, the same weight load is reused across the
// `M` activation rows in registers, dropping gmem bandwidth on the
// weight side by `M`× (matches the Phase 33 GGUF multi-M reuse
// pattern). The accumulator is one f32 per M, kept in registers.

#ifndef BARACUDA_NF4_GEMV_CUH
#define BARACUDA_NF4_GEMV_CUH

#include "nf4_kernel.cuh"

#include <cstdint>
#include <cuda_runtime.h>

namespace baracuda {
namespace nf4 {

// =============================================================================
// Dequant kernel: unpack NF4 weights into a dense `T` matrix.
//
// Shape: weight `[N/2, K]` u8 (pair-packed) → output `[N, K]` of T.
// =============================================================================
template <typename T>
__global__ void nf4_dequantize_kernel(
    const uint8_t * __restrict__ w_packed,
    const float   * __restrict__ absmax,
    T             * __restrict__ out,
    int N, int K, int block_size)
{
    // One thread per output element. Grid: (ceil(N/16), ceil(K/16)).
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || k >= K) return;

    int idx = nf4_load_idx(w_packed, n, k, K);
    float code = nf4_codebook(idx);
    int   amax_off = nf4_absmax_off(n, k, K, block_size);
    float scale = absmax[amax_off];
    float val   = code * scale;

    int out_off = n * K + k;
    out[out_off] = load_act_nf4<T>::store_cast(val);
}

// =============================================================================
// GEMV M=1 — single activation vector decode.
//
// For each output row `n` ∈ [0, N), compute:
//   out[n] = Σ_k codebook[W_q[n, k]] * absmax[n, k/block_size] * y[k]
//
// Launch: gridDim.x = N, blockDim.x = WARP_SIZE.
// Each thread strides K in WARP_SIZE-wide chunks; final warp reduce.
// =============================================================================
template <typename T>
__global__ void nf4_gemv_m1_kernel(
    const uint8_t * __restrict__ w_packed,
    const float   * __restrict__ absmax,
    const T       * __restrict__ y,         // [K]
    T             * __restrict__ out,       // [N]
    int N, int K, int block_size)
{
    int n   = blockIdx.x;
    if (n >= N) return;
    int tid = threadIdx.x;

    float acc = 0.0f;
    // Stride K in warp-wide steps.
    for (int k = tid; k < K; k += 32) {
        int   idx       = nf4_load_idx(w_packed, n, k, K);
        float code      = nf4_codebook(idx);
        int   amax_off  = nf4_absmax_off(n, k, K, block_size);
        float scale     = absmax[amax_off];
        float y_val     = load_act_nf4<T>::load(&y[k]);
        acc += code * scale * y_val;
    }
    acc = warp_reduce_sum_nf4(acc);
    if (tid == 0) {
        out[n] = load_act_nf4<T>::store_cast(acc);
    }
}

// =============================================================================
// GEMV multi-M — `M` activation vectors decoded against one weight
// matrix (weight reuse across the M dimension).
//
// Output: out `[M, N]` row-major in T.
// =============================================================================
template <typename T, int M>
__global__ void nf4_gemv_multim_kernel(
    const uint8_t * __restrict__ w_packed,
    const float   * __restrict__ absmax,
    const T       * __restrict__ y,         // [M, K]
    T             * __restrict__ out,       // [M, N]
    int N, int K, int block_size)
{
    int n   = blockIdx.x;
    if (n >= N) return;
    int tid = threadIdx.x;

    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.0f;

    for (int k = tid; k < K; k += 32) {
        int   idx      = nf4_load_idx(w_packed, n, k, K);
        float code     = nf4_codebook(idx);
        int   amax_off = nf4_absmax_off(n, k, K, block_size);
        float scale    = absmax[amax_off];
        float wval     = code * scale;
#pragma unroll
        for (int m = 0; m < M; ++m) {
            float y_val = load_act_nf4<T>::load(&y[m * K + k]);
            acc[m] += wval * y_val;
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
        float v = warp_reduce_sum_nf4(acc[m]);
        if (tid == 0) {
            out[m * N + n] = load_act_nf4<T>::store_cast(v);
        }
    }
}

} // namespace nf4
} // namespace baracuda

#endif // BARACUDA_NF4_GEMV_CUH
