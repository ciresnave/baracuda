// SPDX-FileCopyrightText: 2023-2024 The llama.cpp / ggml authors    (MIT)
// SPDX-FileCopyrightText: 2024-2026 attention.rs (guoqingbao)       (MIT OR Apache-2.0)
// SPDX-FileCopyrightText: 2024-2026 Fuel project contributors       (MIT OR Apache-2.0)
// SPDX-FileCopyrightText: 2026 baracuda project contributors        (MIT OR Apache-2.0)
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// baracuda_moe.cuh
//
// Mixture-of-Experts (MoE) inference forward — fused per-token dispatch
// + expert matmul + accumulate. Phase 8 Milestone 8.5 — Category V.
//
// Three kernel variants (one templated body each) live behind this
// header; the companion `.cu` files in `kernels/moe/` provide one
// `extern "C" int32_t baracuda_kernels_moe_<variant>_<dtype>_run(...)`
// launcher per (variant × supported dtype combo).
//
// Lineage:
//   attention.rs/src/kernels/src/moe_gemm_{gguf,wmma,wmma_gguf}.cu
//     → fuel-cuda-kernels/src/moe/                                  (MIT OR Apache-2.0)
//       → baracuda-kernels-sys/kernels/include/baracuda_moe.cuh     (THIS FILE)
//
// Adaptation summary:
//   * Vendored the three kernel bodies verbatim (template parameters,
//     shared-memory layout, WMMA fragment usage). The non-trivial
//     compute logic stays; only the FFI shape, status-code returns,
//     and GGUF surface dependency are adapted.
//   * Vendored the q8_1-staging support family (`block_q8_1`,
//     `quantize_q8_1`, `vec_dot_*_q8_1`, `warp_reduce_*`,
//     `ggml_cuda_dp4a`, `get_int_from_*`) into this header — these
//     were intentionally excluded from `baracuda_gguf.cuh` (the dequant
//     + FP-activation-MMVQ-only header for Milestone 8.4). The MoE
//     scalar GGUF path needs them; concentrating them here keeps the
//     8.4 surface clean.
//   * Vendored warp-level single-block dequantize functions
//     (`dequantize_block_q[2-6]_K`) distinct from baracuda_gguf.cuh's
//     `dequantize_block_q*_K_tmpl` variants. The MoE WMMA+GGUF kernel
//     calls them once per N-row per K-iter on a shared-mem block
//     pointer (one block = `i = 0` per call), versus the dequant-
//     launcher pattern where `i = blockIdx.x` selects one block out
//     of `nb32` per CTA. Same compute, different call site contract.
//   * Replaced Fuel's `extern "C" void moe_gemm_<variant>(...)` direct
//     FFI exports with the standard baracuda
//     `extern "C" int32_t baracuda_kernels_moe_<variant>_<dtype>_run(...)`
//     launcher convention (defined in the companion `.cu` files in
//     `kernels/moe/`). Status codes:
//       0  success
//       2  invalid problem (nullptr operand, shape mismatch, ...)
//       5  internal launch failure (cudaPeekAtLastError != 0).
//   * GGUF block formats use the baracuda layouts from
//     `baracuda_gguf.cuh` (NOT Fuel's `moe/gguf.cuh`). The block
//     struct ABI is identical — both header lines vendor llama.cpp
//     verbatim — but the namespace and constant routing go through
//     `baracuda::gguf::` here.
//   * Internal expert-offset helpers (`calculate_expert_offsets` /
//     `calculate_expert_offsets_light`) are inlined here as
//     `static __device__/__host__` to avoid an extra `.cuh` and an
//     extra translation unit. Thrust-based scan variant requires
//     `<thrust/scan.h>` which the launcher TU pulls in.
//
// Block format coverage (matches Fuel exactly — adding Q4_0/Q4_1/
// Q5_0/Q5_1 would require wiring 4 new vec_dot entries Fuel itself
// doesn't ship for MoE):
//   ScalarGguf : Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K
//   WmmaGguf   : Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K
//   Wmma       : N/A (FP weights, no GGUF block format)

#ifndef BARACUDA_MOE_CUH
#define BARACUDA_MOE_CUH

#include "baracuda_gguf.cuh"

#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace baracuda { namespace moe {

// =============================================================================
// Bring the baracuda GGUF block layouts + constants into scope.
// =============================================================================

using baracuda::gguf::QK_K;
using baracuda::gguf::K_SCALE_SIZE;
using baracuda::gguf::WARP_SIZE;
using baracuda::gguf::block_q4_K;
using baracuda::gguf::block_q5_K;
using baracuda::gguf::block_q2_K;
using baracuda::gguf::block_q3_K;
using baracuda::gguf::block_q6_K;
using baracuda::gguf::block_q8_0;

// =============================================================================
// q8_1-staging family (MoE-only — distinct from the dequant + FP-activation
// MMVQ surface in baracuda_gguf.cuh). Vendored from llama.cpp via Fuel.
// =============================================================================

#define MOE_QK8_0 32
#define MOE_QK8_1 32
#define MOE_QR8_1 1
#define MOE_QI8_0 (MOE_QK8_0 / 4)
#define MOE_QI8_1 (MOE_QK8_1 / (4 * MOE_QR8_1))

typedef struct {
    half2  ds;             // ds.x = delta, ds.y = sum
    int8_t qs[MOE_QK8_0];  // quants
} block_q8_1;
static_assert(sizeof(block_q8_1) == 2 * sizeof(uint16_t) + MOE_QK8_0,
              "wrong q8_1 block size/padding");

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq,
                                  const block_q8_1 * __restrict__ bq8_1,
                                  const int & iqs);

// Warp reductions.
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
}

static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, mask, 32));
    }
    return x;
}

// 4-byte unaligned / aligned int loaders.
static __device__ __forceinline__ int get_int_from_int8(const int8_t * x8, const int & i32) {
    const uint16_t * x16 = (const uint16_t *)(x8 + sizeof(int) * i32);
    int x32 = 0;
    x32 |= x16[0] <<  0;
    x32 |= x16[1] << 16;
    return x32;
}
static __device__ __forceinline__ int get_int_from_uint8(const uint8_t * x8, const int & i32) {
    const uint16_t * x16 = (const uint16_t *)(x8 + sizeof(int) * i32);
    int x32 = 0;
    x32 |= x16[0] <<  0;
    x32 |= x16[1] << 16;
    return x32;
}
static __device__ __forceinline__ int get_int_from_int8_aligned(const int8_t * x8, const int & i32) {
    return *((const int *)(x8 + sizeof(int) * i32));
}
static __device__ __forceinline__ int get_int_from_uint8_aligned(const uint8_t * x8, const int & i32) {
    return *((const int *)(x8 + sizeof(int) * i32));
}

// 4xINT8 dot-product → INT32 (DP4A on sm_61+).
#define MOE_MIN_CC_DP4A 610
static __device__ __forceinline__ int moe_dp4a(const int a, const int b, int c) {
#if __CUDA_ARCH__ >= MOE_MIN_CC_DP4A
    return __dp4a(a, b, c);
#else
    const int8_t * a8 = (const int8_t *)&a;
    const int8_t * b8 = (const int8_t *)&b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif
}

// FP32 → q8_1 quantizer (one block per warp). Per-warp reduction of
// max-abs + sum, then per-thread store.
static __global__ void moe_quantize_q8_1(
    const float * __restrict__ x, void * __restrict__ vy, const int kx, const int kx_padded)
{
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix >= kx_padded) return;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    const int i_padded = iy * kx_padded + ix;
    block_q8_1 * y = (block_q8_1 *)vy;

    const int ib  = i_padded / MOE_QK8_1;
    const int iqs = i_padded % MOE_QK8_1;

    const float xi = ix < kx ? x[iy * kx + ix] : 0.0f;
    float amax = fabsf(xi);
    float sum  = xi;

    amax = warp_reduce_max(amax);
    sum  = warp_reduce_sum(sum);

    const float d = amax / 127;
    const int8_t q = amax == 0.0f ? 0 : (int8_t)roundf(xi / d);

    y[ib].qs[iqs] = q;
    if (iqs > 0) return;
    reinterpret_cast<half&>(y[ib].ds.x) = d;
    reinterpret_cast<half&>(y[ib].ds.y) = sum;
}

// ---- vec_dot impls (contiguous v/x against staged q8_1 y) ------------

#define MOE_VDR_Q8_0_Q8_1_MMVQ 2

template <int vdr>
static __device__ __forceinline__ float moe_vec_dot_q8_0_q8_1_impl(
    const int * v, const int * u, const float & d8_0, const float & d8_1)
{
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        sumi = moe_dp4a(v[i], u[i], sumi);
    }
    return d8_0 * d8_1 * sumi;
}

#define MOE_VDR_Q2_K_Q8_1_MMVQ 1
static __device__ __forceinline__ float moe_vec_dot_q2_K_q8_1_impl_mmvq(
    const int & v, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const half2 & dm2, const float * __restrict__ d8)
{
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {     // QR2_K == 4
        const int sc = scales[2 * i];
        const int vi = (v >> (2 * i)) & 0x03030303;
        sumf_d += d8[i] * (moe_dp4a(vi, u[i], 0) * (sc & 0xF));
        int m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;
        sumf_m += d8[i] * moe_dp4a(m, u[i], 0);
    }
    const float2 dm2f = __half22float2(dm2);
    return dm2f.x * sumf_d - dm2f.y * sumf_m;
}

#define MOE_VDR_Q3_K_Q8_1_MMVQ 1
static __device__ __forceinline__ float moe_vec_dot_q3_K_q8_1_impl_mmvq(
    const int & vl, const int & vh, const int * __restrict__ u,
    const uint8_t * __restrict__ scales, const int & scale_offset,
    const float & d3, const float * __restrict__ d8)
{
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {     // QR3_K == 4
        const int isc          = scale_offset + 2 * i;
        const int isc_low      = isc % (QK_K / 32);
        const int sc_shift_low = 4 * (isc / (QK_K / 32));
        const int sc_low       = (scales[isc_low] >> sc_shift_low) & 0xF;

        const int isc_high      = isc % (QK_K / 64);
        const int sc_shift_high = 2 * (isc / (QK_K / 64));
        const int sc_high       = ((scales[(QK_K / 32) + isc_high] >> sc_shift_high) & 3) << 4;

        const int sc = (sc_low | sc_high) - 32;
        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        const int vi  = __vsubss4(vil, vih);
        sumf += d8[i] * (moe_dp4a(vi, u[i], 0) * sc);
    }
    return d3 * sumf;
}

#define MOE_VDR_Q4_K_Q8_1_MMVQ 2
static __device__ __forceinline__ float moe_vec_dot_q4_K_q8_1_impl_vmmq(
    const int * __restrict__ v, const int * __restrict__ u,
    const uint8_t * __restrict__ sc, const uint8_t * __restrict__ m,
    const half2 & dm4, const float * __restrict__ d8)
{
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {     // QR4_K == 2
        const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;
        const int dot1 = moe_dp4a(v1i, u[2 * i + 1], moe_dp4a(v0i, u[2 * i + 0], 0));
        const int dot2 = moe_dp4a(0x01010101, u[2 * i + 1],
                                  moe_dp4a(0x01010101, u[2 * i + 0], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 *  m[i]);
    }
    const float2 dm4f = __half22float2(dm4);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

#define MOE_VDR_Q5_K_Q8_1_MMVQ 2
static __device__ __forceinline__ float moe_vec_dot_q5_K_q8_1_impl_vmmq(
    const int * __restrict__ vl, const int * __restrict__ vh, const int * __restrict__ u,
    const uint8_t * __restrict__ sc, const uint8_t * __restrict__ m,
    const half2 & dm5, const float * __restrict__ d8)
{
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {     // QR5_K == 2
        const int vl0i = (vl[0] >> (4 * i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0F0F0F0F;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = moe_dp4a(v0i, u[2 * i + 0], moe_dp4a(v1i, u[2 * i + 1], 0));
        const int dot2 = moe_dp4a(0x01010101, u[2 * i + 0],
                                  moe_dp4a(0x01010101, u[2 * i + 1], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 *  m[i]);
    }
    const float2 dm5f = __half22float2(dm5);
    return dm5f.x * sumf_d - dm5f.y * sumf_m;
}

#define MOE_VDR_Q6_K_Q8_1_MMVQ 1
static __device__ __forceinline__ float moe_vec_dot_q6_K_q8_1_impl_mmvq(
    const int & vl, const int & vh, const int * __restrict__ u,
    const int8_t * __restrict__ scales, const float & d, const float * __restrict__ d8)
{
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {     // QR6_K == 2
        const int sc  = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi  = __vsubss4((vil | vih), 0x20202020);  // (vil | vih) - 32
        sumf += d8[i] * (moe_dp4a(vi, u[i], 0) * sc);
    }
    return d * sumf;
}

// Public vec_dot entry points (block-q-aware wrappers).

static __device__ __forceinline__ float moe_vec_dot_q8_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs)
{
    const block_q8_0 * bq8_0 = (const block_q8_0 *)vbq;
    int v[MOE_VDR_Q8_0_Q8_1_MMVQ];
    int u[MOE_VDR_Q8_0_Q8_1_MMVQ];
#pragma unroll
    for (int i = 0; i < MOE_VDR_Q8_0_Q8_1_MMVQ; ++i) {
        v[i] = get_int_from_int8(bq8_0->qs, iqs + i);
        u[i] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
    }
    return moe_vec_dot_q8_0_q8_1_impl<MOE_VDR_Q8_0_Q8_1_MMVQ>(
        v, u, __half2float(bq8_0->d), __low2float(bq8_1->ds));
}

static __device__ __forceinline__ float moe_vec_dot_q2_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs)
{
    const block_q2_K * bq2_K = (const block_q2_K *)vbq;
    const int bq8_offset   = 4 * (iqs / MOE_QI8_1);
    const int scale_offset = iqs - iqs % MOE_QI8_1 + (iqs % MOE_QI8_1) / (MOE_QI8_1 / 2);
    const uint8_t * scales = bq2_K->scales + scale_offset;

    const int v = get_int_from_uint8_aligned(bq2_K->qs, iqs);
    int   u[4];
    float d8[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + i].qs, iqs % MOE_QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }
    return moe_vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
}

static __device__ __forceinline__ float moe_vec_dot_q3_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs)
{
    const block_q3_K * bq3_K = (const block_q3_K *)vbq;
    // QR3_K = 4, QI3_K = QK_K / (4 * QR3_K) = 16. Fuel: bq8_offset =
    // QR3_K * (iqs / (QI3_K/2)) = 4 * (iqs / 8).
    const int bq8_offset   = 4 * (iqs / 8);
    const int scale_offset = iqs - iqs % MOE_QI8_1 + (iqs % MOE_QI8_1) / (MOE_QI8_1 / 2);

    const float d  = __half2float(bq3_K->d);
    const int   vl = get_int_from_uint8(bq3_K->qs, iqs);
    // Fuel: vh = ~get_int_from_uint8(hmask, iqs % (QI3_K/2)) >> bq8_offset
    // = ~get_int_from_uint8(hmask, iqs % 8) >> bq8_offset.
    const int   vh = ~get_int_from_uint8(bq3_K->hmask, iqs % 8) >> bq8_offset;

    int    u[4];
    float d8[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + i].qs, iqs % MOE_QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }
    return moe_vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, bq3_K->scales, scale_offset, d, d8);
}

static __device__ __forceinline__ float moe_vec_dot_q4_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs)
{
    const block_q4_K * bq4_K = (const block_q4_K *)vbq;
    int    v[2];
    int    u[2 * 2];                                            // QR4_K == 2
    float d8[2];

    const int bq8_offset = 2 * ((iqs / 2) / (MOE_QI8_1 / 2));
    const int * q4 = (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = q4[0];
    v[1] = q4[4];

    const uint16_t * scales = (const uint16_t *)bq4_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset / 2;
    if (j < 2) {
        aux[0] = scales[j + 0] & 0x3f3f;
        aux[1] = scales[j + 2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

    for (int i = 0; i < 2; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int * q8 = (const int *)bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return moe_vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
}

static __device__ __forceinline__ float moe_vec_dot_q5_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs)
{
    const block_q5_K * bq5_K = (const block_q5_K *)vbq;
    int   vl[2];
    int   vh[2];
    int    u[2 * 2];
    float d8[2];

    const int bq8_offset = 2 * ((iqs / 2) / (MOE_QI8_1 / 2));
    const int * ql = (const int *)(bq5_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const int * qh = (const int *)(bq5_K->qh + 4 * ((iqs / 2) % 4));
    vl[0] = ql[0];
    vl[1] = ql[4];
    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    const uint16_t * scales = (const uint16_t *)bq5_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset / 2;
    if (j < 2) {
        aux[0] = scales[j + 0] & 0x3f3f;
        aux[1] = scales[j + 2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int * q8 = (const int *)bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return moe_vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);
}

static __device__ __forceinline__ float moe_vec_dot_q6_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs)
{
    const block_q6_K * bq6_K = (const block_q6_K *)vbq;
    // QI6_K = 32 (= QK_K / (4 * QR6_K)) — already #define'd in baracuda_gguf.cuh;
    // do NOT redeclare here or the macro expands and breaks the const-int decl.
    const int bq8_offset = 2 * 2 * (iqs / (QI6_K / 2))
                         + (iqs % (QI6_K / 2)) / (QI6_K / 4);    // 2*QR6_K = 4
    const int scale_offset = (QI6_K / 4) * (iqs / (QI6_K / 2))
                           + (iqs % (QI6_K / 2)) / (QI6_K / 8);
    const int vh_shift = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));

    const int vl = get_int_from_uint8(bq6_K->ql, iqs);
    const int vh = get_int_from_uint8(bq6_K->qh,
                                      (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4))
                   >> vh_shift;
    const int8_t * scales = bq6_K->scales + scale_offset;

    int    u[2];
    float d8[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + 2 * i].qs, iqs % MOE_QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + 2 * i].ds);
    }
    return moe_vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, u, scales, __half2float(bq6_K->d), d8);
}

// =============================================================================
// Warp-level single-block dequantize functions for the WMMA + GGUF path.
// These dequantize ONE GGUF block per call (`i = 0`) starting at the
// passed-in `vx` block pointer, writing `qk` FP elements to `yy`. They
// share compute with baracuda_gguf.cuh's `dequantize_block_q*_K_tmpl`
// (which select one block out of `nb32` per CTA using `blockIdx.x`),
// but the call-site contract differs — see header notes above.
//
// Type conversion lane: the WMMA+GGUF kernel operates on `half` /
// `nv_bfloat16` activation/output. These routines compute in `half`
// arithmetic and convert via `moe_from_half<dst_t>(...)`.
// =============================================================================

template<typename dst_t>
static __device__ __forceinline__ dst_t moe_from_half(half val) { return val; }

template<>
__device__ __forceinline__ nv_bfloat16 moe_from_half<nv_bfloat16>(half val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __float2bfloat16(__half2float(val));
#else
    return __nv_bfloat16(__half2float(val));
#endif
}

template<>
__device__ __forceinline__ float moe_from_half<float>(half val) {
    return __half2float(val);
}

template<typename dst_t>
__device__ __forceinline__ void moe_dequantize_block_q2_K(
    const void * __restrict__ vx, dst_t * __restrict__ yy)
{
    const int i = 0;
    const block_q2_K * x = (const block_q2_K *)vx;

    const int tid = threadIdx.x;
    const int n   = tid / 32;
    const int l   = tid - 32 * n;
    const int is  = 8 * n + l / 16;

    const uint8_t q = x[i].qs[32 * n + l];
    dst_t * y = yy + i * QK_K + 128 * n;

    half dall = __low2half(x[i].dm);
    half dmin = __high2half(x[i].dm);
    y[l +  0] = moe_from_half<dst_t>(__hsub(__hmul(dall, __int2half_rn((x[i].scales[is + 0] & 0xF) * ((q >> 0) & 3))),
                                            __hmul(dmin, __int2half_rn(x[i].scales[is + 0] >> 4))));
    y[l + 32] = moe_from_half<dst_t>(__hsub(__hmul(dall, __int2half_rn((x[i].scales[is + 2] & 0xF) * ((q >> 2) & 3))),
                                            __hmul(dmin, __int2half_rn(x[i].scales[is + 2] >> 4))));
    y[l + 64] = moe_from_half<dst_t>(__hsub(__hmul(dall, __int2half_rn((x[i].scales[is + 4] & 0xF) * ((q >> 4) & 3))),
                                            __hmul(dmin, __int2half_rn(x[i].scales[is + 4] >> 4))));
    y[l + 96] = moe_from_half<dst_t>(__hsub(__hmul(dall, __int2half_rn((x[i].scales[is + 6] & 0xF) * ((q >> 6) & 3))),
                                            __hmul(dmin, __int2half_rn(x[i].scales[is + 6] >> 4))));
}

template<typename dst_t>
__device__ __forceinline__ void moe_dequantize_block_q3_K(
    const void * __restrict__ vx, dst_t * __restrict__ yy)
{
    const int i = 0;
    const block_q3_K * x = (const block_q3_K *)vx;

    const int r   = threadIdx.x / 4;
    const int tid = r / 2;
    const int is0 = r % 2;
    const int l0  = 16 * is0 + 4 * (threadIdx.x % 4);
    const int n   = tid / 4;
    const int j   = tid - 4 * n;

    uint8_t m = 1 << (4 * n + j);
    int is    = 8 * n + 2 * j + is0;
    int shift = 2 * j;

    int8_t us = is <  4 ? (x[i].scales[is - 0] & 0xF) | (((x[i].scales[is + 8] >> 0) & 3) << 4) :
                is <  8 ? (x[i].scales[is - 0] & 0xF) | (((x[i].scales[is + 4] >> 2) & 3) << 4) :
                is < 12 ? (x[i].scales[is - 8] >>  4) | (((x[i].scales[is + 0] >> 4) & 3) << 4) :
                          (x[i].scales[is - 8] >>  4) | (((x[i].scales[is - 4] >> 6) & 3) << 4);
    half d_all = x[i].d;
    half dl    = __hmul(d_all, __int2half_rn(us - 32));

    dst_t * y = yy + i * QK_K + 128 * n + 32 * j;
    const uint8_t * q  = x[i].qs + 32 * n;
    const uint8_t * hm = x[i].hmask;

    for (int l = l0; l < l0 + 4; ++l) {
        y[l] = moe_from_half<dst_t>(
            __hmul(dl, __int2half_rn((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4))));
    }
}

static inline __device__ void moe_get_scale_min_k4(
    int j, const uint8_t * q, uint8_t & d, uint8_t & m)
{
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

template<typename dst_t>
__device__ __forceinline__ void moe_dequantize_block_q4_K(
    const void * __restrict__ vx, dst_t * __restrict__ yy)
{
    const block_q4_K * x = (const block_q4_K *)vx;
    const int i = 0;

    const int tid = threadIdx.x;
    const int il  = tid / 8;
    const int ir  = tid % 8;
    const int is  = 2 * il;
    const int n   = 4;

    dst_t * y = yy + i * QK_K + 64 * il + n * ir;
    const half dall = __low2half(x[i].dm);
    const half dmin = __high2half(x[i].dm);
    const uint8_t * q = x[i].qs + 32 * il + n * ir;

    uint8_t sc, m;
    moe_get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const half d1 = __hmul(dall, __int2half_rn(sc));
    const half m1 = __hmul(dmin, __int2half_rn(m));
    moe_get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const half d2 = __hmul(dall, __int2half_rn(sc));
    const half m2 = __hmul(dmin, __int2half_rn(m));
    for (int l = 0; l < n; ++l) {
        y[l +  0] = moe_from_half<dst_t>(__hsub(__hmul(d1, __int2half_rn(q[l] & 0xF)), m1));
        y[l + 32] = moe_from_half<dst_t>(__hsub(__hmul(d2, __int2half_rn(q[l] >>  4)), m2));
    }
}

template<typename dst_t>
__device__ __forceinline__ void moe_dequantize_block_q5_K(
    const void * __restrict__ vx, dst_t * __restrict__ yy)
{
    const block_q5_K * x = (const block_q5_K *)vx;
    const int i = 0;

    const int tid = threadIdx.x;
    const int il  = tid / 16;
    const int ir  = tid % 16;
    const int is  = 2 * il;

    dst_t * y = yy + i * QK_K + 64 * il + 2 * ir;
    const half dall = __low2half(x[i].dm);
    const half dmin = __high2half(x[i].dm);
    const uint8_t * ql = x[i].qs + 32 * il + 2 * ir;
    const uint8_t * qh = x[i].qh + 2 * ir;

    uint8_t sc, m;
    moe_get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const half d1 = __hmul(dall, __int2half_rn(sc));
    const half m1 = __hmul(dmin, __int2half_rn(m));
    moe_get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const half d2 = __hmul(dall, __int2half_rn(sc));
    const half m2 = __hmul(dmin, __int2half_rn(m));

    uint8_t hm = 1 << (2 * il);
    y[ 0] = moe_from_half<dst_t>(__hsub(__hmul(d1, __int2half_rn((ql[0] & 0xF) + (qh[0] & hm ? 16 : 0))), m1));
    y[ 1] = moe_from_half<dst_t>(__hsub(__hmul(d1, __int2half_rn((ql[1] & 0xF) + (qh[1] & hm ? 16 : 0))), m1));
    hm <<= 1;
    y[32] = moe_from_half<dst_t>(__hsub(__hmul(d2, __int2half_rn((ql[0] >> 4) + (qh[0] & hm ? 16 : 0))), m2));
    y[33] = moe_from_half<dst_t>(__hsub(__hmul(d2, __int2half_rn((ql[1] >> 4) + (qh[1] & hm ? 16 : 0))), m2));
}

template<typename dst_t>
__device__ __forceinline__ void moe_dequantize_block_q6_K(
    const void * __restrict__ vx, dst_t * __restrict__ yy)
{
    const block_q6_K * x = (const block_q6_K *)vx;
    const int i = 0;

    const int tid = threadIdx.x;
    const int ip  = tid / 32;
    const int il  = tid - 32 * ip;
    const int is  = 8 * ip + il / 16;

    dst_t * y = yy + i * QK_K + 128 * ip + il;
    const half d = x[i].d;
    const uint8_t * ql = x[i].ql + 64 * ip + il;
    const uint8_t   qh = x[i].qh[32 * ip + il];
    const int8_t  * sc = x[i].scales + is;

    y[ 0] = moe_from_half<dst_t>(__hmul(d, __int2half_rn(sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32))));
    y[32] = moe_from_half<dst_t>(__hmul(d, __int2half_rn(sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32))));
    y[64] = moe_from_half<dst_t>(__hmul(d, __int2half_rn(sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32))));
    y[96] = moe_from_half<dst_t>(__hmul(d, __int2half_rn(sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32))));
}

// Q8_0 warp-level single-block dequant (inline — Fuel ships the body
// directly in the kernel switch, but factoring it here keeps the
// kernel dispatch readable).
template<typename dst_t>
__device__ __forceinline__ void moe_dequantize_block_q8_0(
    const void * __restrict__ vx, dst_t * __restrict__ dequant_out)
{
    const int laneId = threadIdx.x;
    const uint8_t * quant_in = (const uint8_t *)vx;
    const half  * d_ptr      = (const half  *)quant_in;
    const int8_t * qs        = (const int8_t *)(quant_in + 2);

    half d_val = (laneId == 0) ? *d_ptr : (half)0.0f;
    d_val = __shfl_sync(0xFFFFFFFF, d_val, 0);
    float d_f = __half2float(d_val);

    if (laneId < MOE_QK8_0) {
        dequant_out[laneId] = dst_t((float)qs[laneId] * d_f);
    }
}

// Top-level dispatch — selects which warp-level dequant to call based
// on the qtype tag (matches Fuel's switch ordering: Q8_0=0, Q4K=1,
// Q2K=2, Q3k=3, Q5K=4, Q6K=5).
template<typename T>
__forceinline__ __device__ void moe_dequantize_block_warp(
    T * dequant_out, const uint8_t * quant_in, int gguf_dtype)
{
    switch (gguf_dtype) {
        case 0: moe_dequantize_block_q8_0<T>(quant_in, dequant_out); break;
        case 1: moe_dequantize_block_q4_K<T>(quant_in, dequant_out); break;
        case 2: moe_dequantize_block_q2_K<T>(quant_in, dequant_out); break;
        case 3: moe_dequantize_block_q3_K<T>(quant_in, dequant_out); break;
        case 4: moe_dequantize_block_q5_K<T>(quant_in, dequant_out); break;
        case 5: moe_dequantize_block_q6_K<T>(quant_in, dequant_out); break;
        default: break;
    }
}

// =============================================================================
// from_float — write-back conversion lane used by the WMMA (no-GGUF)
// kernel's cooperative-store loop. Matches Fuel's helper signature.
// =============================================================================

__device__ __forceinline__ void moe_from_float(half & dst, float src) {
    dst = __float2half(src);
}
__device__ __forceinline__ void moe_from_float(__nv_bfloat16 & dst, float src) {
    dst = __float2bfloat16(src);
}

// =============================================================================
// Kernel templates.
// =============================================================================

namespace device {

// Scalar GGUF MoE GEMM. One warp per output (m_idx, n) cell. K is
// tiled by `vdr * WARP_SIZE / qi` blocks per iteration. Activations
// are staged through q8_1 (allocated + filled by the launcher).
template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
__global__ void moe_gemm_gguf_kernel(
    const void   * __restrict__ all_weights,
    const void   * __restrict__ all_inputs_q8_1,
    const int32_t * __restrict__ sorted_token_ids,
    const int32_t * __restrict__ expert_ids,
    const float   * __restrict__ topk_weights,
    float         * __restrict__ all_outputs,
    int num_experts, int topk,
    int size_m, int size_n, int size_k,
    int k_padded)
{
    const int laneId = threadIdx.x;
    const int wrapId = threadIdx.y;
    const int nWraps = blockDim.y;
    const int row    = blockIdx.x * nWraps + wrapId;
    const int m_idx  = blockIdx.y;

    if (row >= size_n || m_idx >= size_m) return;

    const size_t weight_expert_stride_bytes = (size_t)(size_n * size_k) / qk * sizeof(block_q_t);
    const size_t input_task_stride_bytes    = (size_t)k_padded / MOE_QK8_1 * sizeof(block_q8_1);
    const size_t output_task_stride_elems   = (size_t)size_n;

    const int token_id = sorted_token_ids[m_idx];
    const int expert   = expert_ids[m_idx];
    if (expert < 0 || expert >= num_experts) return;

    const float scale = (topk_weights) ? topk_weights[token_id] : 1.0f;

    const block_q_t * __restrict__ w_expert =
        (const block_q_t *)((const char *)all_weights + (size_t)expert * weight_expert_stride_bytes);

    const int input_index = topk_weights ? token_id : (token_id / topk);
    const block_q8_1 * __restrict__ y_ptr =
        (const block_q8_1 *)((const char *)all_inputs_q8_1 + (size_t)input_index * input_task_stride_bytes);

    const int blocks_per_row_x = size_k / qk;
    const int blocks_per_iter  = vdr * WARP_SIZE / qi;

    extern __shared__ int8_t shared_bytes[];
    block_q_t * w_shared_row = reinterpret_cast<block_q_t *>(shared_bytes);
    for (int i = laneId; i < blocks_per_row_x; i += WARP_SIZE) {
        w_shared_row[wrapId * blocks_per_row_x + i] = w_expert[row * blocks_per_row_x + i];
    }
    __syncthreads();

    float acc = 0.0f;
#pragma unroll
    for (int kbx = laneId / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / MOE_QK8_1);
        const int kqs = vdr * (laneId % (qi / vdr));
        acc += vec_dot_q_cuda(
            &w_shared_row[wrapId * blocks_per_row_x + kbx],
            &y_ptr[kby],
            kqs);
    }

    float v = warp_reduce_sum(acc) * scale;
    if (laneId == 0) {
        float * __restrict__ out_ptr =
            all_outputs + ((size_t)token_id) * output_task_stride_elems;
        out_ptr[row] = v;
    }
}

// WMMA (no-GGUF) grouped MoE GEMM. Tiles inputs+weights into shared
// memory, accumulates the per-warp 16×16 fragment across the full K,
// then cooperatively stores to global. Output must be zero-init by
// the launcher (top-k > 1 yields multiple writes per token row when
// `topk_weights == nullptr`).
constexpr int WMMA_K        = 16;
constexpr int VEC_SIZE      = 8;     // float4 = 128 bits = 8 fp16/bf16
constexpr int M_BLK         = 32;
constexpr int N_BLK         = 32;
constexpr int K_BLK         = WMMA_K;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int BLOCK_THREADS   = WARPS_PER_BLOCK * 32;
using VecT = float4;

template<typename T, int WMMA_M, int WMMA_N, int WARPS_N>
__global__ void moe_gemm_wmma_kernel(
    const T       * __restrict__ input,
    const T       * __restrict__ weights,
    const int32_t * __restrict__ sorted_token_ids,
    const int32_t * __restrict__ expert_offsets,
    const float   * __restrict__ topk_weights,
    T             * __restrict__ output,
    const int num_experts, const int topk,
    const int32_t size_m,
    const int32_t size_n,
    const int32_t size_k)
{
    using namespace nvcuda::wmma;

    const int expert_id   = blockIdx.x;
    const int n_tile_idx  = blockIdx.y;
    if (expert_id < 0 || expert_id >= num_experts) return;

    const int segment_start = expert_offsets[expert_id];
    const int segment_end   = expert_offsets[expert_id + 1];
    const int num_rows_in_segment = segment_end - segment_start;
    if (num_rows_in_segment == 0) return;

    const int n_base = n_tile_idx * N_BLK;
    if (n_base >= size_n) return;

    const T * expert_w = weights + (size_t)expert_id * (size_t)size_n * (size_t)size_k;

    extern __shared__ uint8_t smem_bytes[];
    T * A_sh = reinterpret_cast<T *>(smem_bytes);
    T * B_sh = reinterpret_cast<T *>(A_sh + M_BLK * K_BLK);
    uint8_t * C_ptr = reinterpret_cast<uint8_t *>(B_sh + N_BLK * K_BLK);
    size_t offset = reinterpret_cast<uintptr_t>(C_ptr) % alignof(float);
    if (offset != 0) C_ptr += (alignof(float) - offset);
    float * C_sh = reinterpret_cast<float *>(C_ptr);

    const int threadId = threadIdx.x;
    const int warpId   = threadId / 32;
    const int laneId   = threadId % 32;
    const int warp_m_idx = warpId / WARPS_N;
    const int warp_n_idx = warpId % WARPS_N;
    (void)laneId;

    constexpr int B_ELEMS_PER_BLOCK = N_BLK * K_BLK;
    constexpr int VEC_ELEMS_B       = B_ELEMS_PER_BLOCK / VEC_SIZE;
    constexpr int A_ELEMS_PER_BLOCK = M_BLK * K_BLK;
    constexpr int VEC_ELEMS_A       = A_ELEMS_PER_BLOCK / VEC_SIZE;

    VecT zero_vec;
    zero_vec.x = zero_vec.y = zero_vec.z = zero_vec.w = 0.0f;

    for (int m_base = 0; m_base < num_rows_in_segment; m_base += M_BLK) {
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
        fill_fragment(c_frag, 0.0f);

        for (int k_base = 0; k_base < size_k; k_base += K_BLK) {
            // Load B (weights) tile
            for (int i = threadId; i < VEC_ELEMS_B; i += BLOCK_THREADS) {
                int idx     = i * VEC_SIZE;
                int n_local = idx / K_BLK;
                int k_local = idx % K_BLK;
                int n_global = n_base + n_local;
                int k_global = k_base + k_local;
                if (n_global < size_n && k_global < size_k) {
                    *reinterpret_cast<VecT *>(&B_sh[n_local * K_BLK + k_local]) =
                        *reinterpret_cast<const VecT *>(&expert_w[(size_t)n_global * size_k + k_global]);
                } else {
                    *reinterpret_cast<VecT *>(&B_sh[n_local * K_BLK + k_local]) = zero_vec;
                }
            }
            // Load A (inputs) tile
            for (int i = threadId; i < VEC_ELEMS_A; i += BLOCK_THREADS) {
                int idx     = i * VEC_SIZE;
                int m_local = idx / K_BLK;
                int k_local = idx % K_BLK;
                int m_seg = m_base + m_local;
                int k_global = k_base + k_local;
                if (m_seg < num_rows_in_segment && k_global < size_k) {
                    int token_pair_index = segment_start + m_seg;
                    int token_index      = sorted_token_ids[token_pair_index];
                    int input_index      = token_index / (topk_weights ? 1 : topk);
                    *reinterpret_cast<VecT *>(&A_sh[m_local * K_BLK + k_local]) =
                        *reinterpret_cast<const VecT *>(&input[(size_t)input_index * size_k + k_global]);
                } else {
                    *reinterpret_cast<VecT *>(&A_sh[m_local * K_BLK + k_local]) = zero_vec;
                }
            }
            __syncthreads();

            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, T, row_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, T, col_major> b_frag;
            const T * A_sh_ptr = A_sh + (warp_m_idx * WMMA_M * K_BLK);
            const T * B_sh_ptr = B_sh + (warp_n_idx * WMMA_N * K_BLK);
            load_matrix_sync(a_frag, A_sh_ptr, K_BLK);
            load_matrix_sync(b_frag, B_sh_ptr, K_BLK);
            mma_sync(c_frag, a_frag, b_frag, c_frag);
            __syncthreads();
        }

        float * C_sh_ptr = C_sh + (warp_m_idx * WMMA_M * N_BLK) + (warp_n_idx * WMMA_N);
        store_matrix_sync(C_sh_ptr, c_frag, N_BLK, mem_row_major);
        __syncthreads();

        const int C_ELEMS_PER_BLOCK = M_BLK * N_BLK;
        for (int i = threadId; i < C_ELEMS_PER_BLOCK; i += BLOCK_THREADS) {
            int m_local_c = i / N_BLK;
            int n_local_c = i % N_BLK;
            int m_seg     = m_base + m_local_c;
            int n_global  = n_base + n_local_c;
            if (m_seg < num_rows_in_segment && n_global < size_n) {
                int token_pair_index = segment_start + m_seg;
                if (token_pair_index < size_m) {
                    int token_index = sorted_token_ids[token_pair_index];
                    float val = C_sh[m_local_c * N_BLK + n_local_c];
                    if (topk_weights) val *= topk_weights[token_index];
                    moe_from_float(output[(size_t)token_index * size_n + n_global], val);
                }
            }
        }
    }
}

// WMMA + GGUF MoE prefill GEMM. One GGUF block per N-row is loaded
// into shared memory and dequantized warp-cooperatively into the B
// tile; the A (activation) tile is loaded directly from FP global.
// A 16×16×16 mma.sync accumulates over `qk / WMMA_K` k-subtiles.
constexpr int WMMA_GGUF_M     = 16;
constexpr int WMMA_GGUF_N     = 16;
constexpr int WMMA_GGUF_K     = 16;
constexpr int WARPS_M_GGUF    = 2;
constexpr int WARPS_N_GGUF    = 2;
constexpr int WARPS_PER_BLOCK_GGUF = WARPS_M_GGUF * WARPS_N_GGUF;
constexpr int M_BLK_GGUF      = WARPS_M_GGUF * WMMA_GGUF_M;  // 32
constexpr int N_BLK_GGUF      = WARPS_N_GGUF * WMMA_GGUF_N;  // 32

template<typename T, int qk, typename block_q_t, int wrap_size>
__global__ void moe_gemm_wmma_gguf_prefill_kernel(
    const T       * __restrict__ input,
    const uint8_t * __restrict__ weights,
    const int32_t * __restrict__ sorted_token_ids,
    const int32_t * __restrict__ expert_offsets,
    const float   * __restrict__ topk_weights,
    float         * __restrict__ output,
    const int num_experts, const int topk,
    const int32_t size_m,
    const int32_t size_n,
    const int32_t size_k,
    const int gguf_dtype)
{
    using namespace nvcuda::wmma;

    const int expert_id  = blockIdx.x;
    const int n_tile_idx = blockIdx.y;
    if (expert_id < 0 || expert_id >= num_experts) return;

    const int segment_start = expert_offsets[expert_id];
    const int segment_end   = expert_offsets[expert_id + 1];
    const int num_rows_in_segment = segment_end - segment_start;
    if (num_rows_in_segment == 0) return;
    constexpr int BLOCK_THREADS_LOCAL = WARPS_PER_BLOCK_GGUF * wrap_size;

    const int n_base = n_tile_idx * N_BLK_GGUF;
    if (n_base >= size_n) return;

    const size_t block_size_bytes        = sizeof(block_q_t);
    const size_t expert_w_row_stride_bytes = (size_k / qk) * block_size_bytes;
    const uint8_t * expert_w = weights + (size_t)expert_id * size_n * expert_w_row_stride_bytes;

    extern __shared__ uint8_t smem_bytes[];
    T * A_sh = reinterpret_cast<T *>(smem_bytes);
    size_t A_sh_bytes = (size_t)M_BLK_GGUF * qk * sizeof(T);
    uint8_t * B_sh_ptr = smem_bytes + A_sh_bytes;
    size_t B_sh_bytes  = (size_t)N_BLK_GGUF * qk * sizeof(T);
    uint8_t * B_quant_sh_ptr = B_sh_ptr + B_sh_bytes;
    size_t B_quant_sh_bytes  = (size_t)N_BLK_GGUF * block_size_bytes;
    uint8_t * C_sh_ptr = B_quant_sh_ptr + B_quant_sh_bytes;
    size_t C_sh_offset = reinterpret_cast<uintptr_t>(C_sh_ptr) % alignof(float);
    if (C_sh_offset != 0) C_sh_ptr += (alignof(float) - C_sh_offset);

    T       * B_sh       = reinterpret_cast<T *>(B_sh_ptr);
    uint8_t * B_quant_sh = reinterpret_cast<uint8_t *>(B_quant_sh_ptr);
    float   * C_sh       = reinterpret_cast<float *>(C_sh_ptr);

    const int laneId   = threadIdx.x;
    const int warpId   = threadIdx.y;
    const int threadId = warpId * wrap_size + laneId;
    const int warp_m_idx = warpId / WARPS_N_GGUF;
    const int warp_n_idx = warpId % WARPS_N_GGUF;

    const size_t A_ELEMS_PER_BLOCK = (size_t)M_BLK_GGUF * qk;
    const size_t VEC_ELEMS_A       = A_ELEMS_PER_BLOCK / VEC_SIZE;
    VecT zero_vec;
    zero_vec.x = zero_vec.y = zero_vec.z = zero_vec.w = 0.0f;

    for (int m_base = 0; m_base < num_rows_in_segment; m_base += M_BLK_GGUF) {
        fragment<accumulator, WMMA_GGUF_M, WMMA_GGUF_N, WMMA_GGUF_K, float> c_frag;
        fill_fragment(c_frag, 0.0f);

        for (int k_base = 0; k_base < size_k; k_base += qk) {
            // Load A tile
            #pragma unroll
            for (size_t i = threadId; i < VEC_ELEMS_A; i += BLOCK_THREADS_LOCAL) {
                size_t idx     = i * VEC_SIZE;
                size_t m_local = idx / qk;
                size_t k_local = idx % qk;
                int m_seg = m_base + m_local;
                int k_global = k_base + k_local;
                if (m_seg < num_rows_in_segment && k_global < size_k) {
                    int token_pair_index = segment_start + m_seg;
                    int token_index      = sorted_token_ids[token_pair_index];
                    int input_index      = token_index / (topk_weights ? 1 : topk);
                    *reinterpret_cast<VecT *>(&A_sh[m_local * qk + k_local]) =
                        *reinterpret_cast<const VecT *>(&input[(size_t)input_index * size_k + k_global]);
                } else {
                    *reinterpret_cast<VecT *>(&A_sh[m_local * qk + k_local]) = zero_vec;
                }
            }

            // Load B quantized tile (one block per N-row per warp).
            const size_t k_base_offset_bytes = (k_base / qk) * block_size_bytes;
            constexpr int ROWS_PER_WARP = N_BLK_GGUF / WARPS_PER_BLOCK_GGUF;
            #pragma unroll
            for (int row = 0; row < ROWS_PER_WARP; ++row) {
                int n_local  = warpId * ROWS_PER_WARP + row;
                int n_global = n_base + n_local;
                if (n_local < N_BLK_GGUF && n_global < size_n) {
                    block_q_t * dest_ptr = reinterpret_cast<block_q_t *>(B_quant_sh + n_local * block_size_bytes);
                    const block_q_t * src_ptr = reinterpret_cast<const block_q_t *>(
                        expert_w + (size_t)n_global * expert_w_row_stride_bytes + k_base_offset_bytes);
                    *dest_ptr = *src_ptr;
                }
            }
            __syncthreads();

            // Dequantize B (warp-cooperative — one warp per N-row block).
            #pragma unroll
            for (int row = 0; row < ROWS_PER_WARP; ++row) {
                int n_local  = warpId * ROWS_PER_WARP + row;
                int n_global = n_base + n_local;
                if (n_local < N_BLK_GGUF && n_global < size_n) {
                    const uint8_t * quant_ptr = B_quant_sh + n_local * block_size_bytes;
                    T * dequant_ptr           = B_sh + n_local * qk;
                    moe_dequantize_block_warp<T>(dequant_ptr, quant_ptr, gguf_dtype);
                }
            }
            __syncthreads();

            // Inner WMMA loop over `qk / WMMA_K` k-subtiles.
            #pragma unroll
            for (int k_tile = 0; k_tile < qk; k_tile += WMMA_GGUF_K) {
                fragment<matrix_a, WMMA_GGUF_M, WMMA_GGUF_N, WMMA_GGUF_K, T, row_major> a_frag;
                fragment<matrix_b, WMMA_GGUF_M, WMMA_GGUF_N, WMMA_GGUF_K, T, col_major> b_frag;
                const T * A_sh_ptr = A_sh + (warp_m_idx * WMMA_GGUF_M * qk) + k_tile;
                const T * B_sh_ptr = B_sh + (warp_n_idx * WMMA_GGUF_N * qk) + k_tile;
                load_matrix_sync(a_frag, A_sh_ptr, qk);
                load_matrix_sync(b_frag, B_sh_ptr, qk);
                mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
        }

        // Store C
        float * C_sh_ptr_warp =
            C_sh + (warp_m_idx * WMMA_GGUF_M * N_BLK_GGUF) + (warp_n_idx * WMMA_GGUF_N);
        store_matrix_sync(C_sh_ptr_warp, c_frag, N_BLK_GGUF, mem_row_major);
        __syncthreads();

        const int C_ELEMS_PER_BLOCK = M_BLK_GGUF * N_BLK_GGUF;
        #pragma unroll
        for (int i = threadId; i < C_ELEMS_PER_BLOCK; i += BLOCK_THREADS_LOCAL) {
            int m_local_c = i / N_BLK_GGUF;
            int n_local_c = i % N_BLK_GGUF;
            int m_seg = m_base + m_local_c;
            int n_global = n_base + n_local_c;
            if (m_seg < num_rows_in_segment && n_global < size_n) {
                int token_pair_index = segment_start + m_seg;
                if (token_pair_index < size_m) {
                    int token_index = sorted_token_ids[token_pair_index];
                    float val = C_sh[m_local_c * N_BLK_GGUF + n_local_c];
                    if (topk_weights) val *= topk_weights[token_index];
                    output[(size_t)token_index * size_n + n_global] = val;
                }
            }
        }
    }
}

} // namespace device

// =============================================================================
// Expert-offset helpers used by the WMMA paths. The launcher allocates
// `expert_counts[num_experts]` + `expert_offsets[num_experts + 1]` and
// fills them via these helpers before the main kernel launch.
// =============================================================================

__global__ inline void moe_count_tokens_per_expert_kernel(
    const int32_t * expert_ids, int32_t * expert_counts, int size_m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size_m) {
        int32_t expert_id = expert_ids[i];
        atomicAdd(&expert_counts[expert_id], 1);
    }
}

// Hillis-Steele scan kernel — exclusive prefix sum of `counts` into
// `offsets[1..num_experts+1]`, with `offsets[0] = 0`. One block,
// `scan_threads` threads (power of two ≥ num_experts).
__global__ inline void moe_expert_prefix_sum_kernel(
    const int32_t * __restrict__ counts,
    int32_t       * __restrict__ offsets,
    int num_experts)
{
    extern __shared__ int32_t temp_storage[];
    int tid = threadIdx.x;
    int val = (tid < num_experts) ? counts[tid] : 0;
    temp_storage[tid] = val;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int temp_val = 0;
        if (tid >= offset) temp_val = temp_storage[tid - offset];
        __syncthreads();
        if (tid >= offset) temp_storage[tid] += temp_val;
        __syncthreads();
    }

    if (tid < num_experts) {
        offsets[tid + 1] = temp_storage[tid];
        if (tid == 0) offsets[0] = 0;
    }
}

}} // namespace baracuda::moe

#endif  // BARACUDA_MOE_CUH
