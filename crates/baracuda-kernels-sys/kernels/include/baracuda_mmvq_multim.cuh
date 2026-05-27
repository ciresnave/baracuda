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
// Phase 33 shipped **Q8_0** weights only.
// Phase 34 extends the multi-M dispatch to the remaining 9 type-0/1
// and k-quants block formats (Q4_0/Q4_1/Q5_0/Q5_1/Q2_K..Q6_K). Each
// format gets a `vec_dot_q*_q8_1` helper (ported line-for-line from
// llama.cpp / Fuel) + a `mul_mat_vec_q_multim_tmpl<ncols_y, ...>`
// instantiation. Q8_K MMVQ remains bespoke (Phase 11.4).
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
using baracuda::gguf::block_q4_0;
using baracuda::gguf::block_q4_1;
using baracuda::gguf::block_q5_0;
using baracuda::gguf::block_q5_1;
using baracuda::gguf::block_q2_K;
using baracuda::gguf::block_q3_K;
using baracuda::gguf::block_q4_K;
using baracuda::gguf::block_q5_K;
using baracuda::gguf::block_q6_K;
using baracuda::gguf::QK_K;

// ----- Q8_1 block layout (matches llama.cpp / Fuel) ------------------------
//
// `block_q8_1` is the staging format produced by `quantize_q8_1`. It mirrors
// `block_q8_0` but adds a per-32-element activation sum (used by the bias
// term in Q4_1 / Q5_1; for Q8_0 the sum is ignored). 36 bytes per block.

inline constexpr int QK8_1 = 32;
inline constexpr int QI8_1 = QK8_1 / 4;       // = 8 int32-chunks
inline constexpr int VDR_Q8_0_Q8_1_MMVQ = 2;  // matches llama.cpp's choice

// Phase 34 per-format VDR constants (matches llama.cpp's `_MMVQ` choice).
inline constexpr int VDR_Q4_0_Q8_1_MMVQ = 2;
inline constexpr int VDR_Q4_1_Q8_1_MMVQ = 2;
inline constexpr int VDR_Q5_0_Q8_1_MMVQ = 2;
inline constexpr int VDR_Q5_1_Q8_1_MMVQ = 2;
inline constexpr int VDR_Q2_K_Q8_1_MMVQ = 1;
inline constexpr int VDR_Q3_K_Q8_1_MMVQ = 1;
inline constexpr int VDR_Q4_K_Q8_1_MMVQ = 2;
inline constexpr int VDR_Q5_K_Q8_1_MMVQ = 2;
inline constexpr int VDR_Q6_K_Q8_1_MMVQ = 1;

// Re-export type-0/1 + k-quant block-index constants. baracuda_gguf.cuh
// defines these as `#define`s; we keep them visible here for the
// `mul_mat_vec_q_multim_tmpl` instantiation calls.
// (QK4_0=32, QI4_0=4, QK4_1=32, QI4_1=4, QK5_0=32, QI5_0=4, QK5_1=32,
//  QI5_1=4, QI2_K=16, QI3_K=16, QI4_K=32, QI5_K=32, QI6_K=16.)

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

// Phase 34: uint8 variants — Q4_0/Q4_1/Q5_0/Q5_1/Q2_K..Q6_K all store `qs`
// (and friends) as `uint8_t`. Same packing pattern as the int8 readers.
static __device__ __forceinline__ int q8_get_int_from_uint8(
    const uint8_t * x8, const int i32)
{
    const uint16_t * x16 = reinterpret_cast<const uint16_t *>(
        x8 + sizeof(int) * i32);
    int x32 = 0;
    x32 |= x16[0] <<  0;
    x32 |= x16[1] << 16;
    return x32;
}

static __device__ __forceinline__ int q8_get_int_from_uint8_aligned(
    const uint8_t * x8, const int i32)
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

// =============================================================================
// Phase 34 — per-format vec_dot helpers + per-block wrappers.
//
// Each helper is the same arithmetic as Fuel / llama.cpp's
// `vec_dot_q*_q8_1_impl` (the line-references in the body comments
// give exact upstream coordinates). The per-block `vec_dot_q*_q8_1`
// wrapper signature matches what `mul_mat_vec_q_multim_tmpl` expects:
//   `float (const void * vbq, const block_q8_1 * bq8_1, int iqs)`.
// =============================================================================

// ----- Q4_0 ---------------------------------------------------------------
//
// 4-bit type-0; QK=32, QI=4, VDR=2. Per-block math:
//   sumi = Σ_i dp4a(lo_nibbles[i], u[2i+0]) + dp4a(hi_nibbles[i], u[2i+1])
//   return d4 * (sumi * ds_x - (8*vdr/QI4_0) * ds_y)
// The "-(8*vdr/QI4_0) * ds_y" term replaces the per-quant "-8" bias.
template <int vdr>
static __device__ __forceinline__ float vec_dot_q4_0_q8_1_multim_impl(
    const int * v, const int * u, float d4, __half2 ds8)
{
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;
        sumi = q8_1_dp4a(vi0, u[2*i+0], sumi);
        sumi = q8_1_dp4a(vi1, u[2*i+1], sumi);
    }
    const float2 ds8f = __half22float2(ds8);
    return d4 * (sumi * ds8f.x - (8.0f * vdr / QI4_0) * ds8f.y);
}

static __device__ __forceinline__ float vec_dot_q4_0_q8_1(
    const void * __restrict__ vbq,
    const block_q8_1 * __restrict__ bq8_1,
    const int iqs)
{
    const block_q4_0 * bq4_0 = reinterpret_cast<const block_q4_0 *>(vbq);
    int v[VDR_Q4_0_Q8_1_MMVQ];
    int u[2 * VDR_Q4_0_Q8_1_MMVQ];
#pragma unroll
    for (int i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
        v[i]       = q8_get_int_from_uint8(bq4_0->qs, iqs + i);
        u[2*i + 0] = q8_get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i + 1] = q8_get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_0);
    }
    const float d = __half2float(bq4_0->d);
    return vec_dot_q4_0_q8_1_multim_impl<VDR_Q4_0_Q8_1_MMVQ>(v, u, d, bq8_1->ds);
}

// ----- Q4_1 ---------------------------------------------------------------
//
// 4-bit type-1; QK=32, QI=4, VDR=2. Adds a per-block min `m_x` carried
// in `dm.y`. Math: `sumi * d4*d8 + m4*s8 / (QI8_1/(vdr*QR4_1))`.
template <int vdr>
static __device__ __forceinline__ float vec_dot_q4_1_q8_1_multim_impl(
    const int * v, const int * u, __half2 dm4, __half2 ds8)
{
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;
        sumi = q8_1_dp4a(vi0, u[2*i+0], sumi);
        sumi = q8_1_dp4a(vi1, u[2*i+1], sumi);
    }
    const float2 dm4f = __half22float2(dm4);
    const float2 ds8f = __half22float2(ds8);
    const float d4d8 = dm4f.x * ds8f.x;
    const float m4s8 = dm4f.y * ds8f.y;
    return sumi * d4d8 + m4s8 / (QI8_1 / (vdr * QR4_1));
}

static __device__ __forceinline__ float vec_dot_q4_1_q8_1(
    const void * __restrict__ vbq,
    const block_q8_1 * __restrict__ bq8_1,
    const int iqs)
{
    const block_q4_1 * bq4_1 = reinterpret_cast<const block_q4_1 *>(vbq);
    int v[VDR_Q4_1_Q8_1_MMVQ];
    int u[2 * VDR_Q4_1_Q8_1_MMVQ];
#pragma unroll
    for (int i = 0; i < VDR_Q4_1_Q8_1_MMVQ; ++i) {
        v[i]       = q8_get_int_from_uint8_aligned(bq4_1->qs, iqs + i);
        u[2*i + 0] = q8_get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i + 1] = q8_get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_1);
    }
    return vec_dot_q4_1_q8_1_multim_impl<VDR_Q4_1_Q8_1_MMVQ>(v, u, bq4_1->dm, bq8_1->ds);
}

// ----- Q5_0 ---------------------------------------------------------------
//
// 5-bit type-0; QK=32, QI=4, VDR=2. 5th bit lives in `qh` (4 bytes/block).
// Math: same dot pattern as Q4_0 but the OR-in of the high bit pre-shifts
// each nibble up; final "−16 per quant" bias becomes
// `-(16*vdr/QI5_0) * ds8_y`.
template <int vdr>
static __device__ __forceinline__ float vec_dot_q5_0_q8_1_multim_impl(
    const int * vl, const int * vh, const int * u, float d5, __half2 ds8)
{
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        int vi0 = (vl[i] >>  0) & 0x0F0F0F0F;
        vi0    |= (vh[i] <<  4) & 0x00000010;
        vi0    |= (vh[i] << 11) & 0x00001000;
        vi0    |= (vh[i] << 18) & 0x00100000;
        vi0    |= (vh[i] << 25) & 0x10000000;
        sumi = q8_1_dp4a(vi0, u[2*i+0], sumi);

        int vi1 = (vl[i] >>  4) & 0x0F0F0F0F;
        vi1    |= (vh[i] >> 12) & 0x00000010;
        vi1    |= (vh[i] >>  5) & 0x00001000;
        vi1    |= (vh[i] <<  2) & 0x00100000;
        vi1    |= (vh[i] <<  9) & 0x10000000;
        sumi = q8_1_dp4a(vi1, u[2*i+1], sumi);
    }
    const float2 ds8f = __half22float2(ds8);
    return d5 * (sumi * ds8f.x - (16.0f * vdr / QI5_0) * ds8f.y);
}

static __device__ __forceinline__ float vec_dot_q5_0_q8_1(
    const void * __restrict__ vbq,
    const block_q8_1 * __restrict__ bq8_1,
    const int iqs)
{
    const block_q5_0 * bq5_0 = reinterpret_cast<const block_q5_0 *>(vbq);
    int vl[VDR_Q5_0_Q8_1_MMVQ];
    int vh[VDR_Q5_0_Q8_1_MMVQ];
    int  u[2 * VDR_Q5_0_Q8_1_MMVQ];
#pragma unroll
    for (int i = 0; i < VDR_Q5_0_Q8_1_MMVQ; ++i) {
        vl[i]      = q8_get_int_from_uint8(bq5_0->qs, iqs + i);
        vh[i]      = q8_get_int_from_uint8(bq5_0->qh, 0) >> (4 * (iqs + i));
        u[2*i + 0] = q8_get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i + 1] = q8_get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_0);
    }
    const float d = __half2float(bq5_0->d);
    return vec_dot_q5_0_q8_1_multim_impl<VDR_Q5_0_Q8_1_MMVQ>(vl, vh, u, d, bq8_1->ds);
}

// ----- Q5_1 ---------------------------------------------------------------
//
// 5-bit type-1; QK=32, QI=4, VDR=2. Same 5th-bit unpacking as Q5_0;
// d4d8/m4s8 split as in Q4_1.
template <int vdr>
static __device__ __forceinline__ float vec_dot_q5_1_q8_1_multim_impl(
    const int * vl, const int * vh, const int * u, __half2 dm5, __half2 ds8)
{
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        int vi0 = (vl[i] >>  0) & 0x0F0F0F0F;
        vi0    |= (vh[i] <<  4) & 0x00000010;
        vi0    |= (vh[i] << 11) & 0x00001000;
        vi0    |= (vh[i] << 18) & 0x00100000;
        vi0    |= (vh[i] << 25) & 0x10000000;
        sumi = q8_1_dp4a(vi0, u[2*i+0], sumi);

        int vi1 = (vl[i] >>  4) & 0x0F0F0F0F;
        vi1    |= (vh[i] >> 12) & 0x00000010;
        vi1    |= (vh[i] >>  5) & 0x00001000;
        vi1    |= (vh[i] <<  2) & 0x00100000;
        vi1    |= (vh[i] <<  9) & 0x10000000;
        sumi = q8_1_dp4a(vi1, u[2*i+1], sumi);
    }
    const float2 dm5f = __half22float2(dm5);
    const float2 ds8f = __half22float2(ds8);
    const float d5d8 = dm5f.x * ds8f.x;
    const float m5s8 = dm5f.y * ds8f.y;
    return sumi * d5d8 + m5s8 / (QI5_1 / vdr);
}

static __device__ __forceinline__ float vec_dot_q5_1_q8_1(
    const void * __restrict__ vbq,
    const block_q8_1 * __restrict__ bq8_1,
    const int iqs)
{
    const block_q5_1 * bq5_1 = reinterpret_cast<const block_q5_1 *>(vbq);
    int vl[VDR_Q5_1_Q8_1_MMVQ];
    int vh[VDR_Q5_1_Q8_1_MMVQ];
    int  u[2 * VDR_Q5_1_Q8_1_MMVQ];
#pragma unroll
    for (int i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
        vl[i]      = q8_get_int_from_uint8_aligned(bq5_1->qs, iqs + i);
        vh[i]      = q8_get_int_from_uint8_aligned(bq5_1->qh, 0) >> (4 * (iqs + i));
        u[2*i + 0] = q8_get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i + 1] = q8_get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_1);
    }
    return vec_dot_q5_1_q8_1_multim_impl<VDR_Q5_1_Q8_1_MMVQ>(vl, vh, u, bq5_1->dm, bq8_1->ds);
}

// =============================================================================
// k-quants (256-element super-blocks).
//
// For k-quants the per-block math walks several Q8_1 sub-blocks (QR2_K=4,
// QR3_K=4, QR4_K=2, QR5_K=2, QR6_K=2). The dot wrapper computes the
// `bq8_offset` + `scale_offset` from `iqs` exactly as upstream
// `vec_dot_q*_q8_1` does (mmvq.cu in llama.cpp).
// =============================================================================

// ----- Q2_K ---------------------------------------------------------------
//
// 2-bit k-quant; QI2_K=16, VDR=1, QR2_K=4. Per-iteration multiplies the
// 2-bit quant by the 4-bit scale (sc & 0xF) and subtracts a 4-bit bias
// (sc >> 4) — see Fuel quantized.cu:1995 (`vec_dot_q2_K_q8_1_impl_mmvq`).
static __device__ __forceinline__ float vec_dot_q2_K_q8_1_multim_impl(
    int v, const int * u, const uint8_t * scales,
    __half2 dm2, const float * d8)
{
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        const int sc = scales[2*i];
        const int vi = (v >> (2*i)) & 0x03030303;
        sumf_d += d8[i] * (q8_1_dp4a(vi, u[i], 0) * (sc & 0xF));
        int m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;
        sumf_m += d8[i] * q8_1_dp4a(m, u[i], 0);
    }
    const float2 dm2f = __half22float2(dm2);
    return dm2f.x * sumf_d - dm2f.y * sumf_m;
}

static __device__ __forceinline__ float vec_dot_q2_K_q8_1(
    const void * __restrict__ vbq,
    const block_q8_1 * __restrict__ bq8_1,
    const int iqs)
{
    const block_q2_K * bq2_K = reinterpret_cast<const block_q2_K *>(vbq);
    const int bq8_offset   = QR2_K * (iqs / QI8_1);
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);
    const uint8_t * scales = bq2_K->scales + scale_offset;
    const int v = q8_get_int_from_uint8_aligned(bq2_K->qs, iqs);
    int    u[QR2_K];
    float d8[QR2_K];
#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        u[i]  = q8_get_int_from_int8_aligned(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }
    return vec_dot_q2_K_q8_1_multim_impl(v, u, scales, bq2_K->dm, d8);
}

// ----- Q3_K ---------------------------------------------------------------
//
// 3-bit k-quant; QI3_K=16, VDR=1, QR3_K=4. Combines a 2-bit quant (low),
// a 1-bit high-mask (inverted via `__vsubss4`), and a 6-bit scale split
// across `scales[0..QK_K/32]` (low 4 bits) + `scales[QK_K/32..]` (high
// 2 bits). See Fuel quantized.cu:2059.
static __device__ __forceinline__ float vec_dot_q3_K_q8_1_multim_impl(
    int vl, int vh, const int * u, const uint8_t * scales,
    int scale_offset, float d3, const float * d8)
{
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        const int isc = scale_offset + 2*i;
        const int isc_low       = isc % (QK_K / 32);
        const int sc_shift_low  = 4 * (isc / (QK_K / 32));
        const int sc_low        = (scales[isc_low] >> sc_shift_low) & 0xF;
        const int isc_high      = isc % (QK_K / 64);
        const int sc_shift_high = 2 * (isc / (QK_K / 64));
        const int sc_high       = ((scales[(QK_K / 32) + isc_high] >> sc_shift_high) & 3) << 4;
        const int sc            = (sc_low | sc_high) - 32;
        const int vil = (vl >> (2*i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        const int vi  = __vsubss4(vil, vih);
        sumf += d8[i] * (q8_1_dp4a(vi, u[i], 0) * sc);
    }
    return d3 * sumf;
}

static __device__ __forceinline__ float vec_dot_q3_K_q8_1(
    const void * __restrict__ vbq,
    const block_q8_1 * __restrict__ bq8_1,
    const int iqs)
{
    const block_q3_K * bq3_K = reinterpret_cast<const block_q3_K *>(vbq);
    const int bq8_offset   = QR3_K * (iqs / (QI3_K / 2));
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);
    const float d = __half2float(bq3_K->d);
    const int vl = q8_get_int_from_uint8(bq3_K->qs, iqs);
    // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
    const int vh = ~q8_get_int_from_uint8(bq3_K->hmask, iqs % (QI3_K / 2)) >> bq8_offset;
    int    u[QR3_K];
    float d8[QR3_K];
#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        u[i]  = q8_get_int_from_int8_aligned(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }
    return vec_dot_q3_K_q8_1_multim_impl(vl, vh, u, bq3_K->scales, scale_offset, d, d8);
}

// ----- Q4_K ---------------------------------------------------------------
//
// 4-bit k-quant; QI4_K=32, VDR=2, QR4_K=2. The 6-bit scales / 6-bit mins
// for the 8 sub-blocks live packed into `scales[3*QK_K/64] = scales[12]`.
// `get_scale_min_k4` (in baracuda_gguf.cuh) unpacks one (sc, m) pair.
// See Fuel quantized.cu:2116 (`vec_dot_q4_K_q8_1_impl_vmmq`).
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_multim_impl(
    const int * v, const int * u, const uint8_t * sc,
    const uint8_t * m, __half2 dm4, const float * d8)
{
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;
        const int dot1 = q8_1_dp4a(v1i, u[2*i+1], q8_1_dp4a(v0i, u[2*i+0], 0));
        const int dot2 = q8_1_dp4a(0x01010101, u[2*i+1], q8_1_dp4a(0x01010101, u[2*i+0], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm4f = __half22float2(dm4);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1(
    const void * __restrict__ vbq,
    const block_q8_1 * __restrict__ bq8_1,
    const int iqs)
{
    const block_q4_K * bq4_K = reinterpret_cast<const block_q4_K *>(vbq);
    int   v[2];
    int   u[2 * QR4_K];
    float d8[QR4_K];

    const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
    const int * q4 = reinterpret_cast<const int *>(
        bq4_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = q4[0];
    v[1] = q4[4];

    const uint16_t * scales = reinterpret_cast<const uint16_t *>(bq4_K->scales);
    uint16_t aux[2];
    const int j = bq8_offset / 2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = reinterpret_cast<const uint8_t *>(aux);
    const uint8_t * m  = sc + 2;

    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int * q8 = reinterpret_cast<const int *>(bq8i->qs) + ((iqs / 2) % 4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }
    return vec_dot_q4_K_q8_1_multim_impl(v, u, sc, m, bq4_K->dm, d8);
}

// ----- Q5_K ---------------------------------------------------------------
//
// 5-bit k-quant; QI5_K=32, VDR=2, QR5_K=2. Same scale/min unpacking as
// Q4_K, with an extra `qh` byte stream holding the 5th bit. See Fuel
// quantized.cu:2172 (`vec_dot_q5_K_q8_1_impl_vmmq`).
static __device__ __forceinline__ float vec_dot_q5_K_q8_1_multim_impl(
    const int * vl, const int * vh, const int * u, const uint8_t * sc,
    const uint8_t * m, __half2 dm5, const float * d8)
{
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const int vl0i = (vl[0] >> (4*i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4*i)) & 0x0F0F0F0F;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = q8_1_dp4a(v0i, u[2*i+0], q8_1_dp4a(v1i, u[2*i+1], 0));
        const int dot2 = q8_1_dp4a(0x01010101, u[2*i+0], q8_1_dp4a(0x01010101, u[2*i+1], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm5f = __half22float2(dm5);
    return dm5f.x * sumf_d - dm5f.y * sumf_m;
}

static __device__ __forceinline__ float vec_dot_q5_K_q8_1(
    const void * __restrict__ vbq,
    const block_q8_1 * __restrict__ bq8_1,
    const int iqs)
{
    const block_q5_K * bq5_K = reinterpret_cast<const block_q5_K *>(vbq);
    int   vl[2];
    int   vh[2];
    int    u[2 * QR5_K];
    float d8[QR5_K];

    const int bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
    const int * ql = reinterpret_cast<const int *>(
        bq5_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const int * qh = reinterpret_cast<const int *>(
        bq5_K->qh + 4 * ((iqs / 2) % 4));
    vl[0] = ql[0];
    vl[1] = ql[4];
    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    const uint16_t * scales = reinterpret_cast<const uint16_t *>(bq5_K->scales);
    uint16_t aux[2];
    const int j = bq8_offset / 2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = reinterpret_cast<const uint8_t *>(aux);
    const uint8_t * m  = sc + 2;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int * q8 = reinterpret_cast<const int *>(bq8i->qs) + ((iqs / 2) % 4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }
    return vec_dot_q5_K_q8_1_multim_impl(vl, vh, u, sc, m, bq5_K->dm, d8);
}

// ----- Q6_K ---------------------------------------------------------------
//
// 6-bit k-quant; QI6_K=16, VDR=1, QR6_K=2. 4-bit + 2-bit packing, with
// a per-block signed 8-bit `scales` array. See Fuel quantized.cu:2235.
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_multim_impl(
    int vl, int vh, const int * u, const int8_t * scales,
    float d, const float * d8)
{
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc = scales[4*i];
        const int vil = (vl >> (4*i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4*i)) << 4) & 0x30303030;
        const int vi  = __vsubss4((vil | vih), 0x20202020); // − 32
        sumf += d8[i] * (q8_1_dp4a(vi, u[i], 0) * sc);
    }
    return d * sumf;
}

static __device__ __forceinline__ float vec_dot_q6_K_q8_1(
    const void * __restrict__ vbq,
    const block_q8_1 * __restrict__ bq8_1,
    const int iqs)
{
    const block_q6_K * bq6_K = reinterpret_cast<const block_q6_K *>(vbq);
    const int bq8_offset   = 2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
    const int scale_offset = (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
    const int vh_shift     = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));
    const int vl = q8_get_int_from_uint8(bq6_K->ql, iqs);
    const int vh = q8_get_int_from_uint8(bq6_K->qh,
        (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4)) >> vh_shift;
    const int8_t * scales = bq6_K->scales + scale_offset;
    int    u[QR6_K];
    float d8[QR6_K];
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        u[i]  = q8_get_int_from_int8_aligned(bq8_1[bq8_offset + 2*i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + 2*i].ds);
    }
    const float d = __half2float(bq6_K->d);
    return vec_dot_q6_K_q8_1_multim_impl(vl, vh, u, scales, d, d8);
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
