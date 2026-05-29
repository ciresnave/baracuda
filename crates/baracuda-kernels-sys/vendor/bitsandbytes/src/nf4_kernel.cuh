// SPDX-FileCopyrightText: 2022-2024 Tim Dettmers and the bitsandbytes contributors  (MIT)
// SPDX-FileCopyrightText: 2026 baracuda project contributors                          (MIT OR Apache-2.0)
// SPDX-License-Identifier: MIT
//
// nf4_kernel.cuh — Phase 53.
//
// Provides the NF4 (NormalFloat 4-bit) codebook + pair-packed-nibble
// decode helpers used by the baracuda dequant + GEMV launchers in
// `kernels/quantize/nf4_launcher.cu`.
//
// The 16 codebook entries are bit-identical to bitsandbytes' upstream
// (Section 3.1 of Dettmers et al. 2023, arXiv:2305.14314). They are the
// inverse-CDF-of-N(0,1) evaluated at the 16-quantile midpoints with the
// zero quantile pinned exactly to 0.0.
//
// Pack layout (matches bitsandbytes' upstream `Linear4bit` pack):
//   weight[n, k] (4-bit code) lives at byte (n/2)*K + k of the packed
//   buffer; if (n & 1) == 0 the LOW nibble of the byte holds the code,
//   otherwise the HIGH nibble. K is in elements (1 nibble per element).
//
// Per-block absmax storage:
//   weight matrix `[N, K]` is partitioned by K into block_size-wide
//   spans. One f32 absmax per (n, block_idx) pair, laid out as
//   absmax[n * num_blocks_per_row + b]. block_size is typically 64.

#ifndef BARACUDA_NF4_KERNEL_CUH
#define BARACUDA_NF4_KERNEL_CUH

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace baracuda {
namespace nf4 {

// =============================================================================
// NF4 codebook (16 entries; Section 3.1 of arXiv:2305.14314).
//
// The values are the inverse CDF of N(0, 1) evaluated at:
//   q_i = (Phi^-1((i + 0.5) / 16) - Phi^-1(0.5/16)) /
//         (Phi^-1(15.5/16) - Phi^-1(0.5/16))
// shifted so that q[8] is exactly 0 and rescaled to [-1, 1]. The
// asymmetry (8 negative + 7 positive + zero) reflects the design
// choice of pinning zero exactly to one of the codes.
//
// Reproduced verbatim from bitsandbytes' upstream `kernels.cu`.
// =============================================================================
__device__ __forceinline__ float nf4_codebook(int idx) {
    // 16 entries — kept as a switch instead of a __constant__ array
    // so the compiler can fold the lookup into a sequence of small
    // immediates / fp predicated selects in the hot path (matches
    // upstream bitsandbytes' codegen pattern). Order: idx 0..15.
    switch (idx & 0xF) {
        case  0: return -1.0f;
        case  1: return -0.6961928009986877f;
        case  2: return -0.5250730514526367f;
        case  3: return -0.39491748809814453f;
        case  4: return -0.28444138169288635f;
        case  5: return -0.18477343022823334f;
        case  6: return -0.09105003625154495f;
        case  7: return  0.0f;
        case  8: return  0.07958029955625534f;
        case  9: return  0.16093020141124725f;
        case 10: return  0.24611230194568634f;
        case 11: return  0.33791524171829224f;
        case 12: return  0.44070982933044434f;
        case 13: return  0.5626170039176941f;
        case 14: return  0.7229568362236023f;
        default: return  1.0f;     // case 15
    }
}

// Decode the low + high nibble of a packed byte into two f32 codebook
// entries. Returns (low_code_value, high_code_value).
__device__ __forceinline__ void nf4_unpack_pair(
    uint8_t packed, float * lo_val, float * hi_val)
{
    int lo_idx = static_cast<int>(packed & 0x0F);
    int hi_idx = static_cast<int>((packed >> 4) & 0x0F);
    *lo_val = nf4_codebook(lo_idx);
    *hi_val = nf4_codebook(hi_idx);
}

// =============================================================================
// Activation load helpers (matches the baracuda GGUF MMVQ pattern).
// =============================================================================
template <typename T> struct load_act_nf4;

template <> struct load_act_nf4<float> {
    __device__ __forceinline__ static float load(const float * p) { return *p; }
    __device__ __forceinline__ static float store_cast(float v) { return v; }
};
template <> struct load_act_nf4<__half> {
    __device__ __forceinline__ static float load(const __half * p) {
        return __half2float(*p);
    }
    __device__ __forceinline__ static __half store_cast(float v) {
        return __float2half(v);
    }
};
template <> struct load_act_nf4<__nv_bfloat16> {
    __device__ __forceinline__ static float load(const __nv_bfloat16 * p) {
        return __bfloat162float(*p);
    }
    __device__ __forceinline__ static __nv_bfloat16 store_cast(float v) {
        return __float2bfloat16(v);
    }
};

// =============================================================================
// Per-output-row index helper.
//
// Given a row `n` (0..N) and column `k` (0..K), returns the codebook
// index packed in the weight buffer. Cost: 1 byte load + 1 nibble
// extract.
//
//   `w_ptr` = base pointer to the packed weight bytes ([N/2, K] u8).
//   `n`     = output row.
//   `k`     = column (within K).
//   `K`     = total column count (= weight stride along the row dim).
// =============================================================================
__device__ __forceinline__ int nf4_load_idx(
    const uint8_t * __restrict__ w_ptr, int n, int k, int K)
{
    // Pack layout: row `n` lives at byte (n/2)*K + k. Within that byte,
    // even row → LOW nibble, odd row → HIGH nibble.
    int byte_row = n >> 1;
    int byte_off = byte_row * K + k;
    uint8_t b = w_ptr[byte_off];
    if ((n & 1) == 0) {
        return static_cast<int>(b & 0x0F);
    } else {
        return static_cast<int>((b >> 4) & 0x0F);
    }
}

// =============================================================================
// Compute the absmax offset for (n, k). `block_size` is the per-block
// element count (typically 64).
//
//   absmax layout: row-major [N, num_blocks_per_row] f32, where
//   num_blocks_per_row = K / block_size.
// =============================================================================
__device__ __forceinline__ int nf4_absmax_off(
    int n, int k, int K, int block_size)
{
    int num_blocks_per_row = K / block_size;
    int b = k / block_size;
    return n * num_blocks_per_row + b;
}

// =============================================================================
// Warp reduction primitive (sum) — 32-wide.
// =============================================================================
__device__ __forceinline__ float warp_reduce_sum_nf4(float v) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, mask, 32);
    }
    return v;
}

} // namespace nf4
} // namespace baracuda

#endif // BARACUDA_NF4_KERNEL_CUH
