// SPDX-FileCopyrightText: 2023-2024 The llama.cpp / ggml authors  (MIT)
// SPDX-FileCopyrightText: 2024-2026 Fuel project contributors      (MIT OR Apache-2.0)
// SPDX-FileCopyrightText: 2026 baracuda project contributors       (MIT OR Apache-2.0)
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// baracuda_gguf.cuh
//
// GGUF (k-quants + type-0/1 quants) dequantize + dequantize-mul-mat-vec
// kernels. Phase 8 Milestone 8.4 — Category P.
//
// Lineage:
//   ggml-cuda.cu (llama.cpp, MIT)
//     → fuel-cuda-kernels/src/quantized.cu (Fuel project, MIT OR Apache-2.0)
//       → baracuda-kernels-sys/kernels/include/baracuda_gguf.cuh (THIS FILE)
//
// Adaptation summary:
//   * Block-format struct definitions, dequantize-block templates,
//     dequantize_mul_mat_vec kernels: vendored line-for-line from
//     fuel-cuda-kernels (which itself vendored from llama.cpp).
//   * Removed Fuel's `extern "C" __global__` direct-as-FFI exports;
//     baracuda routes the dequant + MMVQ kernels through the standard
//     `extern "C" int32_t baracuda_kernels_<op>_<qtype>_run(...)`
//     launcher convention (defined in the companion .cu files in
//     kernels/gguf/).
//   * Removed the q8_1-staging MMQ / MMVQ tile-based matmul family.
//     baracuda's GGUF surface ships the FP-activation MMVQ
//     (dequantize_mul_mat_vec) variant only — single FP activation
//     vector in, FP output vector out. The q8_1-staging MMQ path is
//     deferred (would need a sibling quantize_q8_1 launch + a 2-stage
//     plan, plus the full MMQ tile-load template machinery).
//   * Removed the indexed-MoE forward kernels (Fuel-specific; not on
//     baracuda's roadmap yet).
//   * GGML_QKK_64 path elided — baracuda fixes QK_K = 256, matching
//     every modern GGUF file produced by llama.cpp >= 2024.
//   * Q4_0 + Phi-2 bug check: searched fuel-cuda-kernels git log and
//     upstream commit `25805fff` ("Fix some NaNs with GGML quantized
//     #3428"); neither carries a downstream-only Q4_0 dequant patch.
//     The Phi-2 Q4_0 inference path uses the same dequant routine as
//     every other model; no special-case fix-up needed during vendor.
//
// Shared helpers (`__host__ __device__` where called from both launcher
// and kernel sides):
//   * Block-size constants (QK4_0, QK_K, ...) — pure compile-time #define.
//   * `ceil_div_host` — used in both .cu launchers; not shared with
//     device code (the device-side equivalents use integer div directly).
//
// Status codes returned by the launchers in the companion .cu files
// follow the baracuda convention:
//   0  success
//   2  invalid problem (e.g. nullptr operand, numel <= 0, ncols not a
//      multiple of the block size)
//   5  internal kernel error (cudaPeekAtLastError after launch != 0)

#ifndef BARACUDA_GGUF_CUH
#define BARACUDA_GGUF_CUH

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace baracuda { namespace gguf {

// =============================================================================
// Block-format constants & layouts. Vendored verbatim from llama.cpp /
// fuel-cuda-kernels (the static_asserts ensure ABI-compatible binary
// layout with the rest of the GGUF ecosystem).
// =============================================================================

typedef uint16_t ggml_fp16_t;
typedef float    dfloat;     // dequantize float (matches llama.cpp naming)
typedef float2   dfloat2;

inline constexpr int QK_K = 256;
inline constexpr int K_SCALE_SIZE = 12;
inline constexpr int K_QUANTS_PER_ITERATION = 2;
inline constexpr int WARP_SIZE = 32;
inline constexpr int GGML_CUDA_DMMV_X = 32;
inline constexpr int CUDA_DEQUANTIZE_BLOCK_SIZE = 256;

// ---- Type-0/1 (32-element block) layouts ----

#define QK4_0 32
#define QR4_0 2
#define QI4_0 (QK4_0 / (4 * QR4_0))
typedef struct {
    half    d;              // delta
    uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(ggml_fp16_t) + QK4_0 / 2,
              "wrong q4_0 block size/padding");

#define QK4_1 32
#define QR4_1 2
#define QI4_1 (QK4_1 / (4 * QR4_1))
typedef struct {
    half2   dm;             // dm.x = delta, dm.y = min
    uint8_t qs[QK4_1 / 2];  // nibbles / quants
} block_q4_1;
static_assert(sizeof(block_q4_1) == sizeof(ggml_fp16_t) * 2 + QK4_1 / 2,
              "wrong q4_1 block size/padding");

#define QK5_0 32
#define QR5_0 2
#define QI5_0 (QK5_0 / (4 * QR5_0))
typedef struct {
    half d;                 // delta
    uint8_t qh[4];          // 5-th bit of quants
    uint8_t qs[QK5_0 / 2];  // nibbles / quants
} block_q5_0;
static_assert(sizeof(block_q5_0) == sizeof(ggml_fp16_t) + sizeof(uint32_t) + QK5_0 / 2,
              "wrong q5_0 block size/padding");

#define QK5_1 32
#define QR5_1 2
#define QI5_1 (QK5_1 / (4 * QR5_1))
typedef struct {
    half2 dm;               // dm.x = delta, dm.y = min
    uint8_t qh[4];          // 5-th bit of quants
    uint8_t qs[QK5_1 / 2];  // nibbles / quants
} block_q5_1;
static_assert(sizeof(block_q5_1) == 2 * sizeof(ggml_fp16_t) + sizeof(uint32_t) + QK5_1 / 2,
              "wrong q5_1 block size/padding");

#define QK8_0 32
#define QR8_0 1
#define QI8_0 (QK8_0 / (4 * QR8_0))
typedef struct {
    half    d;              // delta
    int8_t  qs[QK8_0];      // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(ggml_fp16_t) + QK8_0,
              "wrong q8_0 block size/padding");

// ---- k-quants (256-element block) layouts ----

#define QR2_K 4
#define QI2_K (QK_K / (4*QR2_K))
typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    half2 dm;                // super-block scale for quantized scales/mins
} block_q2_K;
static_assert(sizeof(block_q2_K) == 2*sizeof(ggml_fp16_t) + QK_K/16 + QK_K/4,
              "wrong q2_K block size/padding");

#define QR3_K 4
#define QI3_K (QK_K / (4*QR3_K))
typedef struct {
    uint8_t hmask[QK_K/8];        // quants - high bit
    uint8_t qs[QK_K/4];           // quants - low 2 bits
    uint8_t scales[K_SCALE_SIZE]; // scales, quantized with 6 bits
    half d;                       // super-block scale
} block_q3_K;

#define QR4_K 2
#define QI4_K (QK_K / (4*QR4_K))
typedef struct {
    half2 dm;                  // super-block scale for quantized scales/mins
    uint8_t scales[3*QK_K/64]; // scales, quantized with 6 bits
    uint8_t qs[QK_K/2];        // 4-bit quants
} block_q4_K;
static_assert(sizeof(block_q4_K) == 2*sizeof(ggml_fp16_t) + 3*QK_K/64 + QK_K/2,
              "wrong q4_K block size/padding");

#define QR5_K 2
#define QI5_K (QK_K / (4*QR5_K))
typedef struct {
    half2 dm;                     // super-block scale for quantized scales/mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];           // quants, high bit
    uint8_t qs[QK_K/2];           // quants, low 4 bits
} block_q5_K;
static_assert(sizeof(block_q5_K) == 2*sizeof(ggml_fp16_t) + K_SCALE_SIZE + QK_K/2 + QK_K/8,
              "wrong q5_K block size/padding");

#define QR6_K 2
#define QI6_K (QK_K / (4*QR6_K))
typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales
    half    d;               // delta
} block_q6_K;
static_assert(sizeof(block_q6_K) == sizeof(ggml_fp16_t) + 13*QK_K/16,
              "wrong q6_K block size/padding");

typedef struct {
    float   d;              // delta
    int8_t  qs[QK_K];       // quants
    int16_t bsums[QK_K/16]; // sum of quants in groups of 16
} block_q8_K;
static_assert(sizeof(block_q8_K) == sizeof(float) + QK_K + QK_K/16*sizeof(int16_t),
              "wrong q8_K block size/padding");

// =============================================================================
// Per-block dequant primitives (used by the type-0/1 dequantize_block path
// and by the FP-activation MMVQ path).
// =============================================================================

typedef void (*dequantize_kernel_t)(const void * vx, const int ib, const int iqs, dfloat2 & v);

static __device__ __forceinline__ void dequantize_q4_0(
    const void * vx, const int ib, const int iqs, dfloat2 & v)
{
    const block_q4_0 * x = (const block_q4_0 *) vx;
    const dfloat d = x[ib].d;
    const int vui = x[ib].qs[iqs];
    v.x = vui & 0xF;
    v.y = vui >> 4;
    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_1(
    const void * vx, const int ib, const int iqs, dfloat2 & v)
{
    const block_q4_1 * x = (const block_q4_1 *) vx;
    const dfloat d = __low2half(x[ib].dm);
    const dfloat m = __high2half(x[ib].dm);
    const int vui = x[ib].qs[iqs];
    v.x = vui & 0xF;
    v.y = vui >> 4;
    v.x = (v.x * d) + m;
    v.y = (v.y * d) + m;
}

static __device__ __forceinline__ void dequantize_q5_0(
    const void * vx, const int ib, const int iqs, dfloat2 & v)
{
    const block_q5_0 * x = (const block_q5_0 *) vx;
    const dfloat d = x[ib].d;
    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));
    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;
    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);
    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void dequantize_q5_1(
    const void * vx, const int ib, const int iqs, dfloat2 & v)
{
    const block_q5_1 * x = (const block_q5_1 *) vx;
    const dfloat d = __low2half(x[ib].dm);
    const dfloat m = __high2half(x[ib].dm);
    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));
    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;
    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);
    v.x = (v.x * d) + m;
    v.y = (v.y * d) + m;
}

static __device__ __forceinline__ void dequantize_q8_0(
    const void * vx, const int ib, const int iqs, dfloat2 & v)
{
    const block_q8_0 * x = (const block_q8_0 *) vx;
    const dfloat d = x[ib].d;
    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];
    v.x *= d;
    v.y *= d;
}

// =============================================================================
// Full-block dequantize_block kernels (type-0/1, 32-element blocks).
// =============================================================================

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __device__ void dequantize_block(
    const void * __restrict__ vx, dst_t * __restrict__ y, const int k)
{
    const int i = 2*(blockDim.x*blockIdx.x + threadIdx.x);
    if (i >= k) {
        return;
    }
    const int ib = i/qk;
    const int iqs = (i%qk)/qr;
    const int iybs = i - i%qk;
    const int y_offset = qr == 1 ? 1 : qk/2;
    dfloat2 v;
    dequantize_kernel(vx, ib, iqs, v);
    y[iybs + iqs + 0]        = v.x;
    y[iybs + iqs + y_offset] = v.y;
}

template<typename dst_t>
static __device__ void dequantize_block_q4_0_tmpl(
    const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32)
{
    const int64_t i = blockIdx.x;
    const int tid = threadIdx.x;
    const int il  = tid/8;
    const int ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) return;

    dst_t * y = yy + 256*i + 32*ir + 4*il;
    const block_q4_0 * x = (const block_q4_0 *)vx + ib;
    const float d = __half2float(x->d);
    const float dm = -8*d;
    const uint8_t * q = x->qs + 4*il;

    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = d * (q[l] & 0xF) + dm;
        y[l+16] = d * (q[l] >>  4) + dm;
    }
}

template<typename dst_t>
static __device__ void dequantize_block_q4_1_tmpl(
    const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32)
{
    const int64_t i = blockIdx.x;
    const int tid = threadIdx.x;
    const int il  = tid/8;
    const int ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) return;

    dst_t * y = yy + 256*i + 32*ir + 4*il;
    const block_q4_1 * x = (const block_q4_1 *)vx + ib;
    const float2 d = __half22float2(x->dm);
    const uint8_t * q = x->qs + 4*il;

    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = d.x * (q[l] & 0xF) + d.y;
        y[l+16] = d.x * (q[l] >>  4) + d.y;
    }
}

template<typename dst_t>
static __device__ void dequantize_block_q5_0_tmpl(
    const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32)
{
    return dequantize_block<QK5_0, QR5_0, dequantize_q5_0>(vx, yy, nb32);
}

template<typename dst_t>
static __device__ void dequantize_block_q5_1_tmpl(
    const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32)
{
    return dequantize_block<QK5_1, QR5_1, dequantize_q5_1>(vx, yy, nb32);
}

template<typename dst_t>
static __device__ void dequantize_block_q8_0_tmpl(
    const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32)
{
    const int i = blockIdx.x;
    const int tid = threadIdx.x;
    const int il  = tid/8;
    const int ir  = tid%8;
    const int ib = 8*i + ir;
    if (ib >= nb32) return;

    dst_t * y = yy + 256*i + 32*ir + 8*il;
    const block_q8_0 * x = (const block_q8_0 *)vx + ib;
    const float d = __half2float(x->d);
    const int8_t * q = x->qs + 8*il;

    for (int l = 0; l < 8; ++l) {
        y[l] = d * q[l];
    }
}

// =============================================================================
// k-quants dequantize_block kernels (256-element blocks, QK_K = 256 path).
// =============================================================================

template<typename dst_t>
static __device__ void dequantize_block_q2_K_tmpl(
    const void * __restrict__ vx, dst_t * __restrict__ yy)
{
    const int i = blockIdx.x;
    const block_q2_K * x = (const block_q2_K *) vx;
    const int tid = threadIdx.x;
    const int n   = tid/32;
    const int l   = tid - 32*n;
    const int is  = 8*n + l/16;

    const uint8_t q = x[i].qs[32*n + l];
    dst_t * y = yy + i*QK_K + 128*n;

    float dall = __low2half(x[i].dm);
    float dmin = __high2half(x[i].dm);
    y[l+ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[l+32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x[i].scales[is+2] >> 4);
    y[l+64] = dall * (x[i].scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+4] >> 4);
    y[l+96] = dall * (x[i].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[i].scales[is+6] >> 4);
}

template<typename dst_t>
static __device__ void dequantize_block_q3_K_tmpl(
    const void * __restrict__ vx, dst_t * __restrict__ yy)
{
    const int i = blockIdx.x;
    const block_q3_K * x = (const block_q3_K *) vx;
    const int r = threadIdx.x/4;
    const int tid = r/2;
    const int is0 = r%2;
    const int l0 = 16*is0 + 4*(threadIdx.x%4);
    const int n = tid / 4;
    const int j = tid - 4*n;

    uint8_t m = 1 << (4*n + j);
    int is = 8*n + 2*j + is0;
    int shift = 2*j;

    int8_t us = is <  4 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x[i].scales[is-8] >>  4) | (((x[i].scales[is+0] >> 4) & 3) << 4) :
                          (x[i].scales[is-8] >>  4) | (((x[i].scales[is-4] >> 6) & 3) << 4);
    float d_all = x[i].d;
    float dl = d_all * (us - 32);

    dst_t * y = yy + i*QK_K + 128*n + 32*j;
    const uint8_t * q = x[i].qs + 32*n;
    const uint8_t * hm = x[i].hmask;

    for (int l = l0; l < l0+4; ++l) {
        y[l] = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
    }
}

static inline __device__ void get_scale_min_k4(
    int j, const uint8_t * q, uint8_t & d, uint8_t & m)
{
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

template<typename dst_t>
static __device__ void dequantize_block_q4_K_tmpl(
    const void * __restrict__ vx, dst_t * __restrict__ yy)
{
    const block_q4_K * x = (const block_q4_K *) vx;
    const int i = blockIdx.x;
    const int tid = threadIdx.x;
    const int il  = tid/8;
    const int ir  = tid%8;
    const int is  = 2*il;
    const int n   = 4;

    dst_t * y = yy + i*QK_K + 64*il + n*ir;
    const float dall = __low2half(x[i].dm);
    const float dmin = __high2half(x[i].dm);
    const uint8_t * q = x[i].qs + 32*il + n*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = d1 * (q[l] & 0xF) - m1;
        y[l +32] = d2 * (q[l] >>  4) - m2;
    }
}

template<typename dst_t>
static __device__ void dequantize_block_q5_K_tmpl(
    const void * __restrict__ vx, dst_t * __restrict__ yy)
{
    const block_q5_K * x = (const block_q5_K *) vx;
    const int i = blockIdx.x;
    const int tid = threadIdx.x;
    const int il  = tid/16;
    const int ir  = tid%16;
    const int is  = 2*il;

    dst_t * y = yy + i*QK_K + 64*il + 2*ir;
    const float dall = __low2half(x[i].dm);
    const float dmin = __high2half(x[i].dm);

    const uint8_t * ql = x[i].qs + 32*il + 2*ir;
    const uint8_t * qh = x[i].qh + 2*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;

    uint8_t hm  = 1 << (2*il);
    y[ 0] = d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1;
    y[ 1] = d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1;
    hm <<= 1;
    y[32] = d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2;
    y[33] = d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2;
}

template<typename dst_t>
static __device__ void dequantize_block_q6_K_tmpl(
    const void * __restrict__ vx, dst_t * __restrict__ yy)
{
    const block_q6_K * x = (const block_q6_K *) vx;
    const int64_t i = blockIdx.x;
    const int64_t tid = threadIdx.x;
    const int64_t ip  = tid/32;
    const int64_t il  = tid - 32*ip;
    const int64_t is  = 8*ip + il/16;

    dst_t * y = yy + i*QK_K + 128*ip + il;
    const float d = x[i].d;

    const uint8_t * ql = x[i].ql + 64*ip + il;
    const uint8_t   qh = x[i].qh[32*ip + il];
    const int8_t  * sc = x[i].scales + is;

    y[ 0] = d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32);
}

template<typename dst_t>
static __device__ void dequantize_block_q8_K_tmpl(
    const void * __restrict__ vx, dst_t * __restrict__ yy)
{
    const block_q8_K * x = (const block_q8_K *) vx;
    const int i = blockIdx.x;
    const int tid = threadIdx.x;
    const int il  = tid/8;
    const int ir  = tid%8;
    const int n   = 8;

    dst_t * y = yy + i*QK_K + 64*il + n*ir;
    const int8_t * q = x[i].qs + 64*il + n*ir;

    for (int l = 0; l < n; ++l) {
        y[l] = q[l] * x[i].d;
    }
}

// =============================================================================
// Dequantize-mul-mat-vec (FP-activation MMVQ) kernels. Block: WARP_SIZE x mmv_y.
// =============================================================================

template <int qk, int qr, dequantize_kernel_t dequantize_kernel>
static __device__ void dequantize_mul_mat_vec(
    const void * __restrict__ vx, const dfloat * __restrict__ y,
    float * __restrict__ dst, const int ncols, const int nrows)
{
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    if (row >= nrows) return;

    const int tid = threadIdx.x;
    const int iter_stride = 2*GGML_CUDA_DMMV_X;
    const int vals_per_iter = iter_stride / WARP_SIZE;
    const int y_offset = qr == 1 ? 1 : qk/2;

    float tmp = 0.0f;

    for (int i = 0; i < ncols; i += iter_stride) {
        const int col = i + vals_per_iter*tid;
        const int ib = (row*ncols + col)/qk;
        const int iqs = (col%qk)/qr;
        const int iybs = col - col%qk;

#pragma unroll
        for (int j = 0; j < vals_per_iter; j += 2) {
            dfloat2 v;
            dequantize_kernel(vx, ib, iqs + j/qr, v);
            tmp += v.x * y[iybs + iqs + j/qr + 0];
            tmp += v.y * y[iybs + iqs + j/qr + y_offset];
        }
    }

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

// k-quants MMVQ kernels are not template-shared with the type-0/1 path —
// they live as `extern "C" __global__` functions in mmvq.cu (per-qtype
// specialized math), wrapped by baracuda's `_run` launchers.
//
// Q8_K MMVQ (Phase 11.4) — added by baracuda to close the Fuel feedback
// gap. Upstream llama.cpp ships only the dequant kernel for Q8_K (treats
// it as a CPU-side intermediate); baracuda exposes a fused MMVQ to avoid
// the 2× memory traffic of materializing the dequantized weight first.
// Math: per super-block, dot(qs[0..256], y[0..256]) × d, accumulated.

// =============================================================================
// Activation-strided MMVQ — Phase 14.5.
// =============================================================================
//
// Same math + thread geometry as `dequantize_mul_mat_vec`, but every read
// of `y[col]` becomes `y[col * stride_y]` (element stride, signed i64).
// Use cases:
//   * `stride_y == 1` → identical to the contig variant (sanity arm).
//   * `stride_y == 0` → broadcast: every thread reads `y[0]`.
//   * other values    → strided view into a larger activation tensor
//                       (e.g. GQA where the kv-head axis has stride 0
//                       and the kernel is launched once per Q-head batch
//                       slot at the host level).
//
// W-allocation-sharing: the launcher accepts `w_start_byte_offset`
// (host-side i64). The launcher does the pointer arithmetic on the host
// (`vx_adjusted = (const u8*)vx + w_start_byte_offset`) so the kernel is
// unchanged on the W side. Zero perf cost when offset = 0.

template <int qk, int qr, dequantize_kernel_t dequantize_kernel>
static __device__ void dequantize_mul_mat_vec_actstrided(
    const void * __restrict__ vx, const dfloat * __restrict__ y,
    float * __restrict__ dst, const int ncols, const int nrows,
    const int64_t stride_y)
{
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    if (row >= nrows) return;

    const int tid = threadIdx.x;
    const int iter_stride = 2*GGML_CUDA_DMMV_X;
    const int vals_per_iter = iter_stride / WARP_SIZE;
    const int y_offset = qr == 1 ? 1 : qk/2;

    float tmp = 0.0f;

    for (int i = 0; i < ncols; i += iter_stride) {
        const int col = i + vals_per_iter*tid;
        const int ib = (row*ncols + col)/qk;
        const int iqs = (col%qk)/qr;
        const int iybs = col - col%qk;

#pragma unroll
        for (int j = 0; j < vals_per_iter; j += 2) {
            dfloat2 v;
            dequantize_kernel(vx, ib, iqs + j/qr, v);
            const int64_t y0 = (int64_t)(iybs + iqs + j/qr + 0) * stride_y;
            const int64_t y1 = (int64_t)(iybs + iqs + j/qr + y_offset) * stride_y;
            tmp += v.x * y[y0];
            tmp += v.y * y[y1];
        }
    }

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

}} // namespace baracuda::gguf

#endif  // BARACUDA_GGUF_CUH
