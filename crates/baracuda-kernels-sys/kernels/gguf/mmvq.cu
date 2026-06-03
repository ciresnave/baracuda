// SPDX-FileCopyrightText: 2023-2024 The llama.cpp / ggml authors  (MIT)
// SPDX-FileCopyrightText: 2024-2026 Fuel project contributors      (MIT OR Apache-2.0)
// SPDX-FileCopyrightText: 2026 baracuda project contributors       (MIT OR Apache-2.0)
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// baracuda-kernels Phase 8 Milestone 8.4 — GGUF dequantize-mul-mat-vec
// (FP-activation MMVQ) launchers.
//
// Vendored from llama.cpp via fuel-cuda-kernels. See
// `kernels/include/baracuda_gguf.cuh` for lineage notes.
//
// Op shape:  out[r] = Σ_c W_q[r, c] · y[c],
//   where W_q is GGUF-packed (rows × packed_cols bytes), y is FP32, out is FP32.
//   `ncols` is the unpacked column count (must be a multiple of the block size).
//
// Dtype coverage:
//   Block formats : Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 (type-0/1)
//                   Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K (k-quants, QK_K=256)
//   Activation    : f32   (f16 / bf16 deferred)
//   Output        : f32
//
// Q8_K MMVQ (Phase 11.4): NOT vendored from llama.cpp / Fuel — upstream
// ships only the dequant kernel for Q8_K and treats it as a CPU-side
// intermediate. baracuda adds a fused MMVQ here to close the Fuel team's
// feedback gap (avoids the 2× memory traffic of dequant-then-GEMV). The
// kernel is bespoke; math is straightforward since Q8_K is a single
// f32 scale × 256 signed-byte quants per super-block.
//
// k-quants kernels below: vendored verbatim from
// fuel-cuda-kernels/src/quantized.cu lines 1246..1816 (QK_K == 256 path).
// The Fuel implementation is itself a direct port of llama.cpp.

#include "../include/baracuda_gguf.cuh"

using namespace baracuda::gguf;

// =============================================================================
// Type-0/1 MMVQ kernels — template-instantiated from baracuda_gguf.cuh.
// =============================================================================

extern "C" __global__ void baracuda_gguf_mmvq_q4_0_f32_kernel(
    const void * __restrict__ vx, const float * __restrict__ y,
    float * __restrict__ dst, const int ncols, const int nrows)
{
    dequantize_mul_mat_vec<QK4_0, QR4_0, dequantize_q4_0, float, float>(vx, y, dst, ncols, nrows);
}

extern "C" __global__ void baracuda_gguf_mmvq_q4_0_f16_kernel(
    const void * __restrict__ vx, const __half * __restrict__ y,
    __half * __restrict__ dst, const int ncols, const int nrows)
{
    dequantize_mul_mat_vec<QK4_0, QR4_0, dequantize_q4_0, __half, __half>(vx, y, dst, ncols, nrows);
}

extern "C" __global__ void baracuda_gguf_mmvq_q4_0_bf16_kernel(
    const void * __restrict__ vx, const __nv_bfloat16 * __restrict__ y,
    __nv_bfloat16 * __restrict__ dst, const int ncols, const int nrows)
{
    dequantize_mul_mat_vec<QK4_0, QR4_0, dequantize_q4_0, __nv_bfloat16, __nv_bfloat16>(
        vx, y, dst, ncols, nrows);
}

extern "C" __global__ void baracuda_gguf_mmvq_q4_1_f32_kernel(
    const void * __restrict__ vx, const float * __restrict__ y,
    float * __restrict__ dst, const int ncols, const int nrows)
{
    dequantize_mul_mat_vec<QK4_1, QR4_1, dequantize_q4_1, float, float>(vx, y, dst, ncols, nrows);
}

extern "C" __global__ void baracuda_gguf_mmvq_q4_1_f16_kernel(
    const void * __restrict__ vx, const __half * __restrict__ y,
    __half * __restrict__ dst, const int ncols, const int nrows)
{
    dequantize_mul_mat_vec<QK4_1, QR4_1, dequantize_q4_1, __half, __half>(vx, y, dst, ncols, nrows);
}

extern "C" __global__ void baracuda_gguf_mmvq_q4_1_bf16_kernel(
    const void * __restrict__ vx, const __nv_bfloat16 * __restrict__ y,
    __nv_bfloat16 * __restrict__ dst, const int ncols, const int nrows)
{
    dequantize_mul_mat_vec<QK4_1, QR4_1, dequantize_q4_1, __nv_bfloat16, __nv_bfloat16>(
        vx, y, dst, ncols, nrows);
}

extern "C" __global__ void baracuda_gguf_mmvq_q5_0_f32_kernel(
    const void * __restrict__ vx, const float * __restrict__ y,
    float * __restrict__ dst, const int ncols, const int nrows)
{
    dequantize_mul_mat_vec<QK5_0, QR5_0, dequantize_q5_0, float, float>(vx, y, dst, ncols, nrows);
}

extern "C" __global__ void baracuda_gguf_mmvq_q5_0_f16_kernel(
    const void * __restrict__ vx, const __half * __restrict__ y,
    __half * __restrict__ dst, const int ncols, const int nrows)
{
    dequantize_mul_mat_vec<QK5_0, QR5_0, dequantize_q5_0, __half, __half>(vx, y, dst, ncols, nrows);
}

extern "C" __global__ void baracuda_gguf_mmvq_q5_0_bf16_kernel(
    const void * __restrict__ vx, const __nv_bfloat16 * __restrict__ y,
    __nv_bfloat16 * __restrict__ dst, const int ncols, const int nrows)
{
    dequantize_mul_mat_vec<QK5_0, QR5_0, dequantize_q5_0, __nv_bfloat16, __nv_bfloat16>(
        vx, y, dst, ncols, nrows);
}

extern "C" __global__ void baracuda_gguf_mmvq_q5_1_f32_kernel(
    const void * __restrict__ vx, const float * __restrict__ y,
    float * __restrict__ dst, const int ncols, const int nrows)
{
    dequantize_mul_mat_vec<QK5_1, QR5_1, dequantize_q5_1, float, float>(vx, y, dst, ncols, nrows);
}

extern "C" __global__ void baracuda_gguf_mmvq_q5_1_f16_kernel(
    const void * __restrict__ vx, const __half * __restrict__ y,
    __half * __restrict__ dst, const int ncols, const int nrows)
{
    dequantize_mul_mat_vec<QK5_1, QR5_1, dequantize_q5_1, __half, __half>(vx, y, dst, ncols, nrows);
}

extern "C" __global__ void baracuda_gguf_mmvq_q5_1_bf16_kernel(
    const void * __restrict__ vx, const __nv_bfloat16 * __restrict__ y,
    __nv_bfloat16 * __restrict__ dst, const int ncols, const int nrows)
{
    dequantize_mul_mat_vec<QK5_1, QR5_1, dequantize_q5_1, __nv_bfloat16, __nv_bfloat16>(
        vx, y, dst, ncols, nrows);
}

extern "C" __global__ void baracuda_gguf_mmvq_q8_0_f32_kernel(
    const void * __restrict__ vx, const float * __restrict__ y,
    float * __restrict__ dst, const int ncols, const int nrows)
{
    dequantize_mul_mat_vec<QK8_0, QR8_0, dequantize_q8_0, float, float>(vx, y, dst, ncols, nrows);
}

extern "C" __global__ void baracuda_gguf_mmvq_q8_0_f16_kernel(
    const void * __restrict__ vx, const __half * __restrict__ y,
    __half * __restrict__ dst, const int ncols, const int nrows)
{
    dequantize_mul_mat_vec<QK8_0, QR8_0, dequantize_q8_0, __half, __half>(vx, y, dst, ncols, nrows);
}

extern "C" __global__ void baracuda_gguf_mmvq_q8_0_bf16_kernel(
    const void * __restrict__ vx, const __nv_bfloat16 * __restrict__ y,
    __nv_bfloat16 * __restrict__ dst, const int ncols, const int nrows)
{
    dequantize_mul_mat_vec<QK8_0, QR8_0, dequantize_q8_0, __nv_bfloat16, __nv_bfloat16>(
        vx, y, dst, ncols, nrows);
}

// =============================================================================
// k-quants MMVQ kernels — vendored verbatim from fuel-cuda-kernels
// (QK_K == 256 path; non-256 path elided since baracuda's GGUF surface
// pins QK_K = 256).
// =============================================================================

template <typename ActT, typename DstT>
static __device__ void mmvq_q2_K_tmpl(
    const void * __restrict__ vx, const ActT * __restrict__ yy,
    DstT * __restrict__ dst, const int ncols, int nrows)
{
    static_assert(16 % K_QUANTS_PER_ITERATION == 0,
                  "16 must be divisible by K_QUANTS_PER_ITERATION");

    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    if (row > nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q2_K * x = (const block_q2_K *)vx + ib0;

    float tmp = 0;

    const int tid = threadIdx.x/K_QUANTS_PER_ITERATION;
    const int ix  = threadIdx.x%K_QUANTS_PER_ITERATION;

    const int step = 16/K_QUANTS_PER_ITERATION;

    const int im = tid/step;
    const int in = tid - step*im;

    const int l0 = K_QUANTS_PER_ITERATION*in;
    const int q_offset = 32*im + l0;
    const int s_offset = 8*im;
    const int y_offset = 128*im + l0;

    uint32_t aux[4];
    const uint8_t * d = (const uint8_t *)aux;
    const uint8_t * m = (const uint8_t *)(aux + 2);

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const ActT    * y = yy + i * QK_K + y_offset;
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
        mmvq_io<DstT>::store(&dst[row], tmp);
    }
}

extern "C" __global__ void baracuda_gguf_mmvq_q2_K_f32_kernel(
    const void * __restrict__ vx, const float * __restrict__ yy,
    float * __restrict__ dst, const int ncols, int nrows)
{
    mmvq_q2_K_tmpl<float, float>(vx, yy, dst, ncols, nrows);
}

extern "C" __global__ void baracuda_gguf_mmvq_q2_K_f16_kernel(
    const void * __restrict__ vx, const __half * __restrict__ yy,
    __half * __restrict__ dst, const int ncols, int nrows)
{
    mmvq_q2_K_tmpl<__half, __half>(vx, yy, dst, ncols, nrows);
}

extern "C" __global__ void baracuda_gguf_mmvq_q2_K_bf16_kernel(
    const void * __restrict__ vx, const __nv_bfloat16 * __restrict__ yy,
    __nv_bfloat16 * __restrict__ dst, const int ncols, int nrows)
{
    mmvq_q2_K_tmpl<__nv_bfloat16, __nv_bfloat16>(vx, yy, dst, ncols, nrows);
}

template <typename ActT, typename DstT>
static __device__ void mmvq_q3_K_tmpl(
    const void * __restrict__ vx, const ActT * __restrict__ yy,
    DstT * __restrict__ dst, const int ncols, int nrows)
{
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    if (row > nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q3_K * x = (const block_q3_K *)vx + ib0;

    float tmp = 0;

    const uint16_t kmask1 = 0x0303;
    const uint16_t kmask2 = 0x0f0f;

    const int tid = threadIdx.x/K_QUANTS_PER_ITERATION;
    const int ix  = threadIdx.x%K_QUANTS_PER_ITERATION;

    const int n  = K_QUANTS_PER_ITERATION;
    const int step = 16/K_QUANTS_PER_ITERATION;
    const int im = tid/step;
    const int in = tid - step*im;

    const uint8_t m = 1 << (4*im);

    const int l0 = n*in;
    const int q_offset =  32*im + l0;
    const int y_offset = 128*im + l0;

    uint16_t utmp[4];
    const int8_t * s = (const int8_t *)utmp;

    const uint16_t s_shift = 4*im;

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const ActT    * y  = yy + i * QK_K + y_offset;
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
            sum += Y(l+ 0) * (s[0] - 32) * (((q[l] >> 0) & 3) - (h[l] & (m << 0) ? 0 : 4))
                 + Y(l+32) * (s[2] - 32) * (((q[l] >> 2) & 3) - (h[l] & (m << 1) ? 0 : 4))
                 + Y(l+64) * (s[4] - 32) * (((q[l] >> 4) & 3) - (h[l] & (m << 2) ? 0 : 4))
                 + Y(l+96) * (s[6] - 32) * (((q[l] >> 6) & 3) - (h[l] & (m << 3) ? 0 : 4));
            sum += Y(l+16) * (s[1] - 32) * (((q[l+16] >> 0) & 3) - (h[l+16] & (m << 0) ? 0 : 4))
                 + Y(l+48) * (s[3] - 32) * (((q[l+16] >> 2) & 3) - (h[l+16] & (m << 1) ? 0 : 4))
                 + Y(l+80) * (s[5] - 32) * (((q[l+16] >> 4) & 3) - (h[l+16] & (m << 2) ? 0 : 4))
                 + Y(l+112) * (s[7] - 32) * (((q[l+16] >> 6) & 3) - (h[l+16] & (m << 3) ? 0 : 4));
            #undef Y
        }
        tmp += d * sum;
    }

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (threadIdx.x == 0) {
        mmvq_io<DstT>::store(&dst[row], tmp);
    }
}

extern "C" __global__ void baracuda_gguf_mmvq_q3_K_f32_kernel(
    const void * __restrict__ vx, const float * __restrict__ yy,
    float * __restrict__ dst, const int ncols, int nrows)
{
    mmvq_q3_K_tmpl<float, float>(vx, yy, dst, ncols, nrows);
}

extern "C" __global__ void baracuda_gguf_mmvq_q3_K_f16_kernel(
    const void * __restrict__ vx, const __half * __restrict__ yy,
    __half * __restrict__ dst, const int ncols, int nrows)
{
    mmvq_q3_K_tmpl<__half, __half>(vx, yy, dst, ncols, nrows);
}

extern "C" __global__ void baracuda_gguf_mmvq_q3_K_bf16_kernel(
    const void * __restrict__ vx, const __nv_bfloat16 * __restrict__ yy,
    __nv_bfloat16 * __restrict__ dst, const int ncols, int nrows)
{
    mmvq_q3_K_tmpl<__nv_bfloat16, __nv_bfloat16>(vx, yy, dst, ncols, nrows);
}

template <typename ActT, typename DstT>
static __device__ void mmvq_q4_K_tmpl(
    const void * __restrict__ vx, const ActT * __restrict__ yy,
    DstT * __restrict__ dst, const int ncols, int nrows)
{
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    if (row > nrows) return;
    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q4_K * x = (const block_q4_K *)vx + ib0;

    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int tid = threadIdx.x/K_QUANTS_PER_ITERATION;
    const int ix  = threadIdx.x%K_QUANTS_PER_ITERATION;

    const int step = 8/K_QUANTS_PER_ITERATION;

    const int il  = tid/step;
    const int ir  = tid - step*il;
    const int n   = 2 * K_QUANTS_PER_ITERATION;

    const int im = il/2;
    const int in = il%2;

    const int l0 = n*(2*ir + in);
    const int q_offset = 32*im + l0;
    const int y_offset = 64*im + l0;

    uint16_t aux[4];
    const uint8_t * sc = (const uint8_t *)aux;

    uint32_t q32[4];
    const uint8_t * q4 = (const uint8_t *)q32;

    float tmp = 0;

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const ActT * y1 = yy + i*QK_K + y_offset;
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
        mmvq_io<DstT>::store(&dst[row], tmp);
    }
}

extern "C" __global__ void baracuda_gguf_mmvq_q4_K_f32_kernel(
    const void * __restrict__ vx, const float * __restrict__ yy,
    float * __restrict__ dst, const int ncols, int nrows)
{
    mmvq_q4_K_tmpl<float, float>(vx, yy, dst, ncols, nrows);
}

extern "C" __global__ void baracuda_gguf_mmvq_q4_K_f16_kernel(
    const void * __restrict__ vx, const __half * __restrict__ yy,
    __half * __restrict__ dst, const int ncols, int nrows)
{
    mmvq_q4_K_tmpl<__half, __half>(vx, yy, dst, ncols, nrows);
}

extern "C" __global__ void baracuda_gguf_mmvq_q4_K_bf16_kernel(
    const void * __restrict__ vx, const __nv_bfloat16 * __restrict__ yy,
    __nv_bfloat16 * __restrict__ dst, const int ncols, int nrows)
{
    mmvq_q4_K_tmpl<__nv_bfloat16, __nv_bfloat16>(vx, yy, dst, ncols, nrows);
}

template <typename ActT, typename DstT>
static __device__ void mmvq_q5_K_tmpl(
    const void * __restrict__ vx, const ActT * __restrict__ yy,
    DstT * __restrict__ dst, const int ncols)
{
    const int row = blockIdx.x;
    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q5_K * x = (const block_q5_K *)vx + ib0;

    float tmp = 0;

    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int tid = threadIdx.x/2;
    const int ix  = threadIdx.x%2;

    const int il  = tid/4;
    const int ir  = tid - 4*il;
    const int n   = 2;

    const int im = il/2;
    const int in = il%2;

    const int l0 = n*(2*ir + in);
    const int q_offset = 32*im + l0;
    const int y_offset = 64*im + l0;

    const uint8_t hm1  = 1 << (2*im);
    const uint8_t hm2  = hm1 << 4;

    uint16_t aux[4];
    const uint8_t * sc = (const uint8_t *)aux;

    uint16_t q16[8];
    const uint8_t * q4 = (const uint8_t *)q16;

    for (int i = ix; i < num_blocks_per_row; i += 2) {

        const uint8_t * ql1 = x[i].qs + q_offset;
        const uint8_t * qh  = x[i].qh + l0;
        const ActT    * y1  = yy + i*QK_K + y_offset;
        const ActT    * y2  = y1 + 128;

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
        mmvq_io<DstT>::store(&dst[row], tmp);
    }
}

extern "C" __global__ void baracuda_gguf_mmvq_q5_K_f32_kernel(
    const void * __restrict__ vx, const float * __restrict__ yy,
    float * __restrict__ dst, const int ncols)
{
    mmvq_q5_K_tmpl<float, float>(vx, yy, dst, ncols);
}

extern "C" __global__ void baracuda_gguf_mmvq_q5_K_f16_kernel(
    const void * __restrict__ vx, const __half * __restrict__ yy,
    __half * __restrict__ dst, const int ncols)
{
    mmvq_q5_K_tmpl<__half, __half>(vx, yy, dst, ncols);
}

extern "C" __global__ void baracuda_gguf_mmvq_q5_K_bf16_kernel(
    const void * __restrict__ vx, const __nv_bfloat16 * __restrict__ yy,
    __nv_bfloat16 * __restrict__ dst, const int ncols)
{
    mmvq_q5_K_tmpl<__nv_bfloat16, __nv_bfloat16>(vx, yy, dst, ncols);
}

// -----------------------------------------------------------------------------
// Q8_K MMVQ — Phase 11.4. Bespoke (not vendored). Math:
//   out[r] = Σ_sb d[r,sb] · Σ_{l=0..255} qs[r,sb,l] · y[sb*256 + l]
// Each block-row's super-block contributes its full 256-element dot
// product, scaled by the f32 per-super-block scale `d`.
//
// Geometry: 1 row per CUDA block, 32 threads per block (one warp).
// `K_QUANTS_PER_ITERATION = 2` follows the rest of the k-quants family:
// the warp's 32 threads split into 2 stride groups of 16, and each
// 16-thread group covers the 256-element super-block as 16 chunks of
// 16 quants (one chunk per thread). Each thread reads its 16 quants
// as four vectorized int8x4 loads via `char4` to keep memory traffic
// at 1B per quant (32 threads × 16 bytes = 512B per super-block load).
// -----------------------------------------------------------------------------

template <typename ActT, typename DstT>
static __device__ void mmvq_q8_K_tmpl(
    const void * __restrict__ vx, const ActT * __restrict__ yy,
    DstT * __restrict__ dst, const int ncols, int nrows)
{
    static_assert(QK_K % 16 == 0, "QK_K must be a multiple of 16");
    static_assert(16 % K_QUANTS_PER_ITERATION == 0,
                  "16 must be divisible by K_QUANTS_PER_ITERATION");

    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    if (row > nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q8_K * x = (const block_q8_K *)vx + ib0;

    // 32 threads → 16 chunk-owners × K_QUANTS_PER_ITERATION stride groups.
    const int tid = threadIdx.x / K_QUANTS_PER_ITERATION;  // 0..15
    const int ix  = threadIdx.x % K_QUANTS_PER_ITERATION;  // 0..1
    constexpr int CHUNK = QK_K / 16;                       // = 16 quants
    const int l0 = tid * CHUNK;                            // 0,16,...,240

    float tmp = 0.0f;

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {
        const float    d = x[i].d;
        const int8_t * q = x[i].qs + l0;
        const ActT   * y = yy + i*QK_K + l0;

        // Accumulate this thread's CHUNK-element chunk of the super-block
        // in f32, then fold in the f32 super-block scale once.
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
        mmvq_io<DstT>::store(&dst[row], tmp);
    }
}

extern "C" __global__ void baracuda_gguf_mmvq_q8_K_f32_kernel(
    const void * __restrict__ vx, const float * __restrict__ yy,
    float * __restrict__ dst, const int ncols, int nrows)
{
    mmvq_q8_K_tmpl<float, float>(vx, yy, dst, ncols, nrows);
}

extern "C" __global__ void baracuda_gguf_mmvq_q8_K_f16_kernel(
    const void * __restrict__ vx, const __half * __restrict__ yy,
    __half * __restrict__ dst, const int ncols, int nrows)
{
    mmvq_q8_K_tmpl<__half, __half>(vx, yy, dst, ncols, nrows);
}

extern "C" __global__ void baracuda_gguf_mmvq_q8_K_bf16_kernel(
    const void * __restrict__ vx, const __nv_bfloat16 * __restrict__ yy,
    __nv_bfloat16 * __restrict__ dst, const int ncols, int nrows)
{
    mmvq_q8_K_tmpl<__nv_bfloat16, __nv_bfloat16>(vx, yy, dst, ncols, nrows);
}

template <typename ActT, typename DstT>
static __device__ void mmvq_q6_K_tmpl(
    const void * __restrict__ vx, const ActT * __restrict__ yy,
    DstT * __restrict__ dst, const int ncols, int nrows)
{
    static_assert(16 % K_QUANTS_PER_ITERATION == 0,
                  "16 must be divisible by K_QUANTS_PER_ITERATION");

    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    if (row > nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q6_K * x = (const block_q6_K *)vx + ib0;

    const int tid = threadIdx.x/K_QUANTS_PER_ITERATION;
    const int ix  = threadIdx.x%K_QUANTS_PER_ITERATION;

    const int step = 16/K_QUANTS_PER_ITERATION;

    const int im = tid/step;
    const int in = tid - step*im;

    const int l0 = 4 * in;
    const int is = in / 4;

    const int ql_offset = 64*im + l0;
    const int qh_offset = 32*im + l0;
    const int s_offset  =  8*im + is;
    const int y_offset = 128*im + l0;

    float tmp = 0;

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const ActT    * y  = yy + i * QK_K + y_offset;
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
        mmvq_io<DstT>::store(&dst[row], tmp);
    }
}

extern "C" __global__ void baracuda_gguf_mmvq_q6_K_f32_kernel(
    const void * __restrict__ vx, const float * __restrict__ yy,
    float * __restrict__ dst, const int ncols, int nrows)
{
    mmvq_q6_K_tmpl<float, float>(vx, yy, dst, ncols, nrows);
}

extern "C" __global__ void baracuda_gguf_mmvq_q6_K_f16_kernel(
    const void * __restrict__ vx, const __half * __restrict__ yy,
    __half * __restrict__ dst, const int ncols, int nrows)
{
    mmvq_q6_K_tmpl<__half, __half>(vx, yy, dst, ncols, nrows);
}

extern "C" __global__ void baracuda_gguf_mmvq_q6_K_bf16_kernel(
    const void * __restrict__ vx, const __nv_bfloat16 * __restrict__ yy,
    __nv_bfloat16 * __restrict__ dst, const int ncols, int nrows)
{
    mmvq_q6_K_tmpl<__nv_bfloat16, __nv_bfloat16>(vx, yy, dst, ncols, nrows);
}

// =============================================================================
// Launchers. Same grid math as Fuel's dequantize_mul_mat_vec():
//   block_dim = (WARP_SIZE, GGML_CUDA_MMV_Y, 1)   // = (32, 1, 1)
//   grid_dim  = (ceil_div(nrows, GGML_CUDA_MMV_Y), 1, 1)
// Q5_K uses a 1-D grid of `nrows`, blockDim = (32, 1, 1) — matches Fuel.
// =============================================================================

namespace {

inline int ceil_div_host(int p, int q) {
    return (p + q - 1) / q;
}

inline int32_t status_from_launch(cudaError_t err) {
    if (err != cudaSuccess) return 5;
    return 0;
}

constexpr int GGML_CUDA_MMV_Y = 1;

template <typename Kernel, typename ActT, typename DstT>
inline int32_t launch_type01_mmvq(
    Kernel kernel,
    int qk,
    int ncols,
    int nrows,
    const void * x,
    const ActT * y,
    DstT * dst,
    cudaStream_t stream)
{
    if (ncols <= 0 || nrows <= 0 || (ncols % qk) != 0) return 2;
    const int block_num_y = ceil_div_host(nrows, GGML_CUDA_MMV_Y);
    dim3 grid(block_num_y, 1, 1);
    dim3 block(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    kernel<<<grid, block, 0, stream>>>(x, y, dst, ncols, nrows);
    return status_from_launch(cudaPeekAtLastError());
}

} // anonymous namespace

// ---- Type-0/1 launchers ---------------------------------------------------

extern "C" int32_t baracuda_kernels_mmvq_q4_0_run(
    int32_t ncols, int32_t nrows,
    const void * __restrict__ x,
    const void * __restrict__ y,
    void       * __restrict__ dst,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!x || !y || !dst) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    return launch_type01_mmvq(
        baracuda_gguf_mmvq_q4_0_f32_kernel, QK4_0, ncols, nrows, x,
        static_cast<const float*>(y), static_cast<float*>(dst), stream);
}

extern "C" int32_t baracuda_kernels_mmvq_q4_1_run(
    int32_t ncols, int32_t nrows,
    const void * __restrict__ x,
    const void * __restrict__ y,
    void       * __restrict__ dst,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!x || !y || !dst) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    return launch_type01_mmvq(
        baracuda_gguf_mmvq_q4_1_f32_kernel, QK4_1, ncols, nrows, x,
        static_cast<const float*>(y), static_cast<float*>(dst), stream);
}

extern "C" int32_t baracuda_kernels_mmvq_q5_0_run(
    int32_t ncols, int32_t nrows,
    const void * __restrict__ x,
    const void * __restrict__ y,
    void       * __restrict__ dst,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!x || !y || !dst) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    return launch_type01_mmvq(
        baracuda_gguf_mmvq_q5_0_f32_kernel, QK5_0, ncols, nrows, x,
        static_cast<const float*>(y), static_cast<float*>(dst), stream);
}

extern "C" int32_t baracuda_kernels_mmvq_q5_1_run(
    int32_t ncols, int32_t nrows,
    const void * __restrict__ x,
    const void * __restrict__ y,
    void       * __restrict__ dst,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!x || !y || !dst) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    return launch_type01_mmvq(
        baracuda_gguf_mmvq_q5_1_f32_kernel, QK5_1, ncols, nrows, x,
        static_cast<const float*>(y), static_cast<float*>(dst), stream);
}

extern "C" int32_t baracuda_kernels_mmvq_q8_0_run(
    int32_t ncols, int32_t nrows,
    const void * __restrict__ x,
    const void * __restrict__ y,
    void       * __restrict__ dst,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!x || !y || !dst) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    return launch_type01_mmvq(
        baracuda_gguf_mmvq_q8_0_f32_kernel, QK8_0, ncols, nrows, x,
        static_cast<const float*>(y), static_cast<float*>(dst), stream);
}

// ---- k-quants launchers ---------------------------------------------------

extern "C" int32_t baracuda_kernels_mmvq_q2_K_run(
    int32_t ncols, int32_t nrows,
    const void * __restrict__ x,
    const void * __restrict__ y,
    void       * __restrict__ dst,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!x || !y || !dst) return 2;
    if (ncols <= 0 || nrows <= 0 || (ncols % QK_K) != 0) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    const int block_num_y = ceil_div_host(nrows, GGML_CUDA_MMV_Y);
    dim3 grid(block_num_y, 1, 1);
    dim3 block(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    baracuda_gguf_mmvq_q2_K_f32_kernel<<<grid, block, 0, stream>>>(
        x, static_cast<const float*>(y), static_cast<float*>(dst), ncols, nrows);
    return status_from_launch(cudaPeekAtLastError());
}

extern "C" int32_t baracuda_kernels_mmvq_q3_K_run(
    int32_t ncols, int32_t nrows,
    const void * __restrict__ x,
    const void * __restrict__ y,
    void       * __restrict__ dst,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!x || !y || !dst) return 2;
    if (ncols <= 0 || nrows <= 0 || (ncols % QK_K) != 0) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    const int block_num_y = ceil_div_host(nrows, GGML_CUDA_MMV_Y);
    dim3 grid(block_num_y, 1, 1);
    dim3 block(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    baracuda_gguf_mmvq_q3_K_f32_kernel<<<grid, block, 0, stream>>>(
        x, static_cast<const float*>(y), static_cast<float*>(dst), ncols, nrows);
    return status_from_launch(cudaPeekAtLastError());
}

extern "C" int32_t baracuda_kernels_mmvq_q4_K_run(
    int32_t ncols, int32_t nrows,
    const void * __restrict__ x,
    const void * __restrict__ y,
    void       * __restrict__ dst,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!x || !y || !dst) return 2;
    if (ncols <= 0 || nrows <= 0 || (ncols % QK_K) != 0) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    const int block_num_y = ceil_div_host(nrows, GGML_CUDA_MMV_Y);
    dim3 grid(block_num_y, 1, 1);
    dim3 block(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    baracuda_gguf_mmvq_q4_K_f32_kernel<<<grid, block, 0, stream>>>(
        x, static_cast<const float*>(y), static_cast<float*>(dst), ncols, nrows);
    return status_from_launch(cudaPeekAtLastError());
}

extern "C" int32_t baracuda_kernels_mmvq_q5_K_run(
    int32_t ncols, int32_t nrows,
    const void * __restrict__ x,
    const void * __restrict__ y,
    void       * __restrict__ dst,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!x || !y || !dst) return 2;
    if (ncols <= 0 || nrows <= 0 || (ncols % QK_K) != 0) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    // Q5_K uses 1-D grid of `nrows`, block_dim = (32, 1, 1) — matches Fuel.
    dim3 grid((unsigned)nrows, 1, 1);
    dim3 block(WARP_SIZE, 1, 1);
    baracuda_gguf_mmvq_q5_K_f32_kernel<<<grid, block, 0, stream>>>(
        x, static_cast<const float*>(y), static_cast<float*>(dst), ncols);
    return status_from_launch(cudaPeekAtLastError());
}

extern "C" int32_t baracuda_kernels_mmvq_q6_K_run(
    int32_t ncols, int32_t nrows,
    const void * __restrict__ x,
    const void * __restrict__ y,
    void       * __restrict__ dst,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!x || !y || !dst) return 2;
    if (ncols <= 0 || nrows <= 0 || (ncols % QK_K) != 0) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    const int block_num_y = ceil_div_host(nrows, GGML_CUDA_MMV_Y);
    dim3 grid(block_num_y, 1, 1);
    dim3 block(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    baracuda_gguf_mmvq_q6_K_f32_kernel<<<grid, block, 0, stream>>>(
        x, static_cast<const float*>(y), static_cast<float*>(dst), ncols, nrows);
    return status_from_launch(cudaPeekAtLastError());
}

// Q8_K MMVQ launcher — Phase 11.4. Same grid math as the other k-quants.
extern "C" int32_t baracuda_kernels_mmvq_q8_K_run(
    int32_t ncols, int32_t nrows,
    const void * __restrict__ x,
    const void * __restrict__ y,
    void       * __restrict__ dst,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!x || !y || !dst) return 2;
    if (ncols <= 0 || nrows <= 0 || (ncols % QK_K) != 0) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    const int block_num_y = ceil_div_host(nrows, GGML_CUDA_MMV_Y);
    dim3 grid(block_num_y, 1, 1);
    dim3 block(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    baracuda_gguf_mmvq_q8_K_f32_kernel<<<grid, block, 0, stream>>>(
        x, static_cast<const float*>(y), static_cast<float*>(dst), ncols, nrows);
    return status_from_launch(cudaPeekAtLastError());
}

// =============================================================================
// Phase 18.1 — f16 / bf16 activation MMVQ launchers (contig).
//
// One launcher per (block-format, activation-dtype) pair. Activation and
// output share dtype (PyTorch convention: `y: T` → `dst: T`). Internal
// accumulator stays f32 in every variant — see the `mmvq_io<T>` cast helpers
// in `baracuda_gguf.cuh`.
// =============================================================================

// ---- Type-0/1 f16 / bf16 contig launchers --------------------------------

extern "C" int32_t baracuda_kernels_mmvq_q4_0_f16_run(
    int32_t ncols, int32_t nrows,
    const void * __restrict__ x, const void * __restrict__ y, void * __restrict__ dst,
    void * /*workspace*/, size_t /*workspace_bytes*/, void * stream_ptr)
{
    if (!x || !y || !dst) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    return launch_type01_mmvq(baracuda_gguf_mmvq_q4_0_f16_kernel, QK4_0, ncols, nrows, x,
        static_cast<const __half*>(y), static_cast<__half*>(dst), stream);
}

extern "C" int32_t baracuda_kernels_mmvq_q4_0_bf16_run(
    int32_t ncols, int32_t nrows,
    const void * __restrict__ x, const void * __restrict__ y, void * __restrict__ dst,
    void * /*workspace*/, size_t /*workspace_bytes*/, void * stream_ptr)
{
    if (!x || !y || !dst) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    return launch_type01_mmvq(baracuda_gguf_mmvq_q4_0_bf16_kernel, QK4_0, ncols, nrows, x,
        static_cast<const __nv_bfloat16*>(y), static_cast<__nv_bfloat16*>(dst), stream);
}

extern "C" int32_t baracuda_kernels_mmvq_q4_1_f16_run(
    int32_t ncols, int32_t nrows,
    const void * __restrict__ x, const void * __restrict__ y, void * __restrict__ dst,
    void * /*workspace*/, size_t /*workspace_bytes*/, void * stream_ptr)
{
    if (!x || !y || !dst) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    return launch_type01_mmvq(baracuda_gguf_mmvq_q4_1_f16_kernel, QK4_1, ncols, nrows, x,
        static_cast<const __half*>(y), static_cast<__half*>(dst), stream);
}

extern "C" int32_t baracuda_kernels_mmvq_q4_1_bf16_run(
    int32_t ncols, int32_t nrows,
    const void * __restrict__ x, const void * __restrict__ y, void * __restrict__ dst,
    void * /*workspace*/, size_t /*workspace_bytes*/, void * stream_ptr)
{
    if (!x || !y || !dst) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    return launch_type01_mmvq(baracuda_gguf_mmvq_q4_1_bf16_kernel, QK4_1, ncols, nrows, x,
        static_cast<const __nv_bfloat16*>(y), static_cast<__nv_bfloat16*>(dst), stream);
}

extern "C" int32_t baracuda_kernels_mmvq_q5_0_f16_run(
    int32_t ncols, int32_t nrows,
    const void * __restrict__ x, const void * __restrict__ y, void * __restrict__ dst,
    void * /*workspace*/, size_t /*workspace_bytes*/, void * stream_ptr)
{
    if (!x || !y || !dst) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    return launch_type01_mmvq(baracuda_gguf_mmvq_q5_0_f16_kernel, QK5_0, ncols, nrows, x,
        static_cast<const __half*>(y), static_cast<__half*>(dst), stream);
}

extern "C" int32_t baracuda_kernels_mmvq_q5_0_bf16_run(
    int32_t ncols, int32_t nrows,
    const void * __restrict__ x, const void * __restrict__ y, void * __restrict__ dst,
    void * /*workspace*/, size_t /*workspace_bytes*/, void * stream_ptr)
{
    if (!x || !y || !dst) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    return launch_type01_mmvq(baracuda_gguf_mmvq_q5_0_bf16_kernel, QK5_0, ncols, nrows, x,
        static_cast<const __nv_bfloat16*>(y), static_cast<__nv_bfloat16*>(dst), stream);
}

extern "C" int32_t baracuda_kernels_mmvq_q5_1_f16_run(
    int32_t ncols, int32_t nrows,
    const void * __restrict__ x, const void * __restrict__ y, void * __restrict__ dst,
    void * /*workspace*/, size_t /*workspace_bytes*/, void * stream_ptr)
{
    if (!x || !y || !dst) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    return launch_type01_mmvq(baracuda_gguf_mmvq_q5_1_f16_kernel, QK5_1, ncols, nrows, x,
        static_cast<const __half*>(y), static_cast<__half*>(dst), stream);
}

extern "C" int32_t baracuda_kernels_mmvq_q5_1_bf16_run(
    int32_t ncols, int32_t nrows,
    const void * __restrict__ x, const void * __restrict__ y, void * __restrict__ dst,
    void * /*workspace*/, size_t /*workspace_bytes*/, void * stream_ptr)
{
    if (!x || !y || !dst) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    return launch_type01_mmvq(baracuda_gguf_mmvq_q5_1_bf16_kernel, QK5_1, ncols, nrows, x,
        static_cast<const __nv_bfloat16*>(y), static_cast<__nv_bfloat16*>(dst), stream);
}

extern "C" int32_t baracuda_kernels_mmvq_q8_0_f16_run(
    int32_t ncols, int32_t nrows,
    const void * __restrict__ x, const void * __restrict__ y, void * __restrict__ dst,
    void * /*workspace*/, size_t /*workspace_bytes*/, void * stream_ptr)
{
    if (!x || !y || !dst) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    return launch_type01_mmvq(baracuda_gguf_mmvq_q8_0_f16_kernel, QK8_0, ncols, nrows, x,
        static_cast<const __half*>(y), static_cast<__half*>(dst), stream);
}

extern "C" int32_t baracuda_kernels_mmvq_q8_0_bf16_run(
    int32_t ncols, int32_t nrows,
    const void * __restrict__ x, const void * __restrict__ y, void * __restrict__ dst,
    void * /*workspace*/, size_t /*workspace_bytes*/, void * stream_ptr)
{
    if (!x || !y || !dst) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    return launch_type01_mmvq(baracuda_gguf_mmvq_q8_0_bf16_kernel, QK8_0, ncols, nrows, x,
        static_cast<const __nv_bfloat16*>(y), static_cast<__nv_bfloat16*>(dst), stream);
}

// ---- k-quants f16 / bf16 contig launchers --------------------------------

#define BCDA_MMVQ_KQUANT_LAUNCHER(qtype, kernel, fdtype, ctype) \
extern "C" int32_t baracuda_kernels_mmvq_##qtype##_##fdtype##_run( \
    int32_t ncols, int32_t nrows, \
    const void * __restrict__ x, const void * __restrict__ y, void * __restrict__ dst, \
    void * /*workspace*/, size_t /*workspace_bytes*/, void * stream_ptr) \
{ \
    if (!x || !y || !dst) return 2; \
    if (ncols <= 0 || nrows <= 0 || (ncols % QK_K) != 0) return 2; \
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr); \
    const int block_num_y = ceil_div_host(nrows, GGML_CUDA_MMV_Y); \
    dim3 grid(block_num_y, 1, 1); \
    dim3 block(WARP_SIZE, GGML_CUDA_MMV_Y, 1); \
    kernel<<<grid, block, 0, stream>>>( \
        x, static_cast<const ctype*>(y), static_cast<ctype*>(dst), ncols, nrows); \
    return status_from_launch(cudaPeekAtLastError()); \
}

BCDA_MMVQ_KQUANT_LAUNCHER(q2_K, baracuda_gguf_mmvq_q2_K_f16_kernel, f16, __half)
BCDA_MMVQ_KQUANT_LAUNCHER(q2_K, baracuda_gguf_mmvq_q2_K_bf16_kernel, bf16, __nv_bfloat16)
BCDA_MMVQ_KQUANT_LAUNCHER(q3_K, baracuda_gguf_mmvq_q3_K_f16_kernel, f16, __half)
BCDA_MMVQ_KQUANT_LAUNCHER(q3_K, baracuda_gguf_mmvq_q3_K_bf16_kernel, bf16, __nv_bfloat16)
BCDA_MMVQ_KQUANT_LAUNCHER(q4_K, baracuda_gguf_mmvq_q4_K_f16_kernel, f16, __half)
BCDA_MMVQ_KQUANT_LAUNCHER(q4_K, baracuda_gguf_mmvq_q4_K_bf16_kernel, bf16, __nv_bfloat16)
BCDA_MMVQ_KQUANT_LAUNCHER(q6_K, baracuda_gguf_mmvq_q6_K_f16_kernel, f16, __half)
BCDA_MMVQ_KQUANT_LAUNCHER(q6_K, baracuda_gguf_mmvq_q6_K_bf16_kernel, bf16, __nv_bfloat16)
BCDA_MMVQ_KQUANT_LAUNCHER(q8_K, baracuda_gguf_mmvq_q8_K_f16_kernel, f16, __half)
BCDA_MMVQ_KQUANT_LAUNCHER(q8_K, baracuda_gguf_mmvq_q8_K_bf16_kernel, bf16, __nv_bfloat16)

#undef BCDA_MMVQ_KQUANT_LAUNCHER

// Q5_K — 1D grid of `nrows`, kernel signature has no `nrows` arg.
#define BCDA_MMVQ_Q5K_LAUNCHER(fdtype, ctype, kernel) \
extern "C" int32_t baracuda_kernels_mmvq_q5_K_##fdtype##_run( \
    int32_t ncols, int32_t nrows, \
    const void * __restrict__ x, const void * __restrict__ y, void * __restrict__ dst, \
    void * /*workspace*/, size_t /*workspace_bytes*/, void * stream_ptr) \
{ \
    if (!x || !y || !dst) return 2; \
    if (ncols <= 0 || nrows <= 0 || (ncols % QK_K) != 0) return 2; \
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr); \
    dim3 grid((unsigned)nrows, 1, 1); \
    dim3 block(WARP_SIZE, 1, 1); \
    kernel<<<grid, block, 0, stream>>>( \
        x, static_cast<const ctype*>(y), static_cast<ctype*>(dst), ncols); \
    return status_from_launch(cudaPeekAtLastError()); \
}

BCDA_MMVQ_Q5K_LAUNCHER(f16, __half, baracuda_gguf_mmvq_q5_K_f16_kernel)
BCDA_MMVQ_Q5K_LAUNCHER(bf16, __nv_bfloat16, baracuda_gguf_mmvq_q5_K_bf16_kernel)

#undef BCDA_MMVQ_Q5K_LAUNCHER

// =============================================================================
// Phase 14.5 — activation-strided + w-offset MMVQ siblings.
//
// Three runtime params added vs. the contig FFI:
//   * `w_start_byte_offset` (i64) — host-side affine offset into W's
//     allocation. The launcher does `(const u8*)x + w_start_byte_offset`
//     before launching; the kernel is unchanged on the W side. Zero cost
//     when offset = 0.
//   * `stride_y` (i64) — element stride along the activation's ncols
//     axis. Signed. `stride_y == 1` matches the contig kernel; `0`
//     broadcasts the single-element activation across every col; other
//     values read from a strided view.
//
// Per-format strided kernels live next to the contig kernels below. The
// type-0/1 family reuses the templated `dequantize_mul_mat_vec_actstrided`
// helper; the k-quants family has bespoke per-format strided kernels.
// =============================================================================

// ---- Type-0/1 strided kernels ---------------------------------------------

extern "C" __global__ void baracuda_gguf_mmvq_q4_0_f32_actstrided_kernel(
    const void * __restrict__ vx, const float * __restrict__ y,
    float * __restrict__ dst, const int ncols, const int nrows,
    const int64_t stride_y)
{
    dequantize_mul_mat_vec_actstrided<QK4_0, QR4_0, dequantize_q4_0, float, float>(
        vx, y, dst, ncols, nrows, stride_y);
}

extern "C" __global__ void baracuda_gguf_mmvq_q4_0_f16_actstrided_kernel(
    const void * __restrict__ vx, const __half * __restrict__ y,
    __half * __restrict__ dst, const int ncols, const int nrows,
    const int64_t stride_y)
{
    dequantize_mul_mat_vec_actstrided<QK4_0, QR4_0, dequantize_q4_0, __half, __half>(
        vx, y, dst, ncols, nrows, stride_y);
}

extern "C" __global__ void baracuda_gguf_mmvq_q4_0_bf16_actstrided_kernel(
    const void * __restrict__ vx, const __nv_bfloat16 * __restrict__ y,
    __nv_bfloat16 * __restrict__ dst, const int ncols, const int nrows,
    const int64_t stride_y)
{
    dequantize_mul_mat_vec_actstrided<QK4_0, QR4_0, dequantize_q4_0, __nv_bfloat16, __nv_bfloat16>(
        vx, y, dst, ncols, nrows, stride_y);
}

extern "C" __global__ void baracuda_gguf_mmvq_q4_1_f32_actstrided_kernel(
    const void * __restrict__ vx, const float * __restrict__ y,
    float * __restrict__ dst, const int ncols, const int nrows,
    const int64_t stride_y)
{
    dequantize_mul_mat_vec_actstrided<QK4_1, QR4_1, dequantize_q4_1, float, float>(
        vx, y, dst, ncols, nrows, stride_y);
}

extern "C" __global__ void baracuda_gguf_mmvq_q4_1_f16_actstrided_kernel(
    const void * __restrict__ vx, const __half * __restrict__ y,
    __half * __restrict__ dst, const int ncols, const int nrows,
    const int64_t stride_y)
{
    dequantize_mul_mat_vec_actstrided<QK4_1, QR4_1, dequantize_q4_1, __half, __half>(
        vx, y, dst, ncols, nrows, stride_y);
}

extern "C" __global__ void baracuda_gguf_mmvq_q4_1_bf16_actstrided_kernel(
    const void * __restrict__ vx, const __nv_bfloat16 * __restrict__ y,
    __nv_bfloat16 * __restrict__ dst, const int ncols, const int nrows,
    const int64_t stride_y)
{
    dequantize_mul_mat_vec_actstrided<QK4_1, QR4_1, dequantize_q4_1, __nv_bfloat16, __nv_bfloat16>(
        vx, y, dst, ncols, nrows, stride_y);
}

extern "C" __global__ void baracuda_gguf_mmvq_q5_0_f32_actstrided_kernel(
    const void * __restrict__ vx, const float * __restrict__ y,
    float * __restrict__ dst, const int ncols, const int nrows,
    const int64_t stride_y)
{
    dequantize_mul_mat_vec_actstrided<QK5_0, QR5_0, dequantize_q5_0, float, float>(
        vx, y, dst, ncols, nrows, stride_y);
}

extern "C" __global__ void baracuda_gguf_mmvq_q5_0_f16_actstrided_kernel(
    const void * __restrict__ vx, const __half * __restrict__ y,
    __half * __restrict__ dst, const int ncols, const int nrows,
    const int64_t stride_y)
{
    dequantize_mul_mat_vec_actstrided<QK5_0, QR5_0, dequantize_q5_0, __half, __half>(
        vx, y, dst, ncols, nrows, stride_y);
}

extern "C" __global__ void baracuda_gguf_mmvq_q5_0_bf16_actstrided_kernel(
    const void * __restrict__ vx, const __nv_bfloat16 * __restrict__ y,
    __nv_bfloat16 * __restrict__ dst, const int ncols, const int nrows,
    const int64_t stride_y)
{
    dequantize_mul_mat_vec_actstrided<QK5_0, QR5_0, dequantize_q5_0, __nv_bfloat16, __nv_bfloat16>(
        vx, y, dst, ncols, nrows, stride_y);
}

extern "C" __global__ void baracuda_gguf_mmvq_q5_1_f32_actstrided_kernel(
    const void * __restrict__ vx, const float * __restrict__ y,
    float * __restrict__ dst, const int ncols, const int nrows,
    const int64_t stride_y)
{
    dequantize_mul_mat_vec_actstrided<QK5_1, QR5_1, dequantize_q5_1, float, float>(
        vx, y, dst, ncols, nrows, stride_y);
}

extern "C" __global__ void baracuda_gguf_mmvq_q5_1_f16_actstrided_kernel(
    const void * __restrict__ vx, const __half * __restrict__ y,
    __half * __restrict__ dst, const int ncols, const int nrows,
    const int64_t stride_y)
{
    dequantize_mul_mat_vec_actstrided<QK5_1, QR5_1, dequantize_q5_1, __half, __half>(
        vx, y, dst, ncols, nrows, stride_y);
}

extern "C" __global__ void baracuda_gguf_mmvq_q5_1_bf16_actstrided_kernel(
    const void * __restrict__ vx, const __nv_bfloat16 * __restrict__ y,
    __nv_bfloat16 * __restrict__ dst, const int ncols, const int nrows,
    const int64_t stride_y)
{
    dequantize_mul_mat_vec_actstrided<QK5_1, QR5_1, dequantize_q5_1, __nv_bfloat16, __nv_bfloat16>(
        vx, y, dst, ncols, nrows, stride_y);
}

extern "C" __global__ void baracuda_gguf_mmvq_q8_0_f32_actstrided_kernel(
    const void * __restrict__ vx, const float * __restrict__ y,
    float * __restrict__ dst, const int ncols, const int nrows,
    const int64_t stride_y)
{
    dequantize_mul_mat_vec_actstrided<QK8_0, QR8_0, dequantize_q8_0, float, float>(
        vx, y, dst, ncols, nrows, stride_y);
}

extern "C" __global__ void baracuda_gguf_mmvq_q8_0_f16_actstrided_kernel(
    const void * __restrict__ vx, const __half * __restrict__ y,
    __half * __restrict__ dst, const int ncols, const int nrows,
    const int64_t stride_y)
{
    dequantize_mul_mat_vec_actstrided<QK8_0, QR8_0, dequantize_q8_0, __half, __half>(
        vx, y, dst, ncols, nrows, stride_y);
}

extern "C" __global__ void baracuda_gguf_mmvq_q8_0_bf16_actstrided_kernel(
    const void * __restrict__ vx, const __nv_bfloat16 * __restrict__ y,
    __nv_bfloat16 * __restrict__ dst, const int ncols, const int nrows,
    const int64_t stride_y)
{
    dequantize_mul_mat_vec_actstrided<QK8_0, QR8_0, dequantize_q8_0, __nv_bfloat16, __nv_bfloat16>(
        vx, y, dst, ncols, nrows, stride_y);
}

// ---- k-quants strided kernels ---------------------------------------------
//
// Bespoke per-format, but mechanically just `y[idx]` → `y[idx * stride_y]`.
// The y-pointer pre-adjustments in the contig versions (e.g. `yy + i*QK_K + y_offset`)
// become column-index expressions multiplied through by `stride_y`.

template <typename ActT, typename DstT>
static __device__ void mmvq_q2_K_actstrided_tmpl(
    const void * __restrict__ vx, const ActT * __restrict__ yy,
    DstT * __restrict__ dst, const int ncols, int nrows,
    const int64_t stride_y)
{
    static_assert(16 % K_QUANTS_PER_ITERATION == 0,
                  "16 must be divisible by K_QUANTS_PER_ITERATION");

    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    if (row > nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q2_K * x = (const block_q2_K *)vx + ib0;

    float tmp = 0;

    const int tid = threadIdx.x/K_QUANTS_PER_ITERATION;
    const int ix  = threadIdx.x%K_QUANTS_PER_ITERATION;

    const int step = 16/K_QUANTS_PER_ITERATION;

    const int im = tid/step;
    const int in = tid - step*im;

    const int l0 = K_QUANTS_PER_ITERATION*in;
    const int q_offset = 32*im + l0;
    const int s_offset = 8*im;
    const int y_offset = 128*im + l0;

    uint32_t aux[4];
    const uint8_t * d = (const uint8_t *)aux;
    const uint8_t * m = (const uint8_t *)(aux + 2);

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const int64_t y_base = ((int64_t)i * QK_K + y_offset) * stride_y;
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
            #define Y(off) mmvq_io<ActT>::load(&yy[y_base + (int64_t)(off) * stride_y])
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
        mmvq_io<DstT>::store(&dst[row], tmp);
    }
}

extern "C" __global__ void baracuda_gguf_mmvq_q2_K_f32_actstrided_kernel(
    const void * __restrict__ vx, const float * __restrict__ yy,
    float * __restrict__ dst, const int ncols, int nrows,
    const int64_t stride_y)
{
    mmvq_q2_K_actstrided_tmpl<float, float>(vx, yy, dst, ncols, nrows, stride_y);
}

extern "C" __global__ void baracuda_gguf_mmvq_q2_K_f16_actstrided_kernel(
    const void * __restrict__ vx, const __half * __restrict__ yy,
    __half * __restrict__ dst, const int ncols, int nrows,
    const int64_t stride_y)
{
    mmvq_q2_K_actstrided_tmpl<__half, __half>(vx, yy, dst, ncols, nrows, stride_y);
}

extern "C" __global__ void baracuda_gguf_mmvq_q2_K_bf16_actstrided_kernel(
    const void * __restrict__ vx, const __nv_bfloat16 * __restrict__ yy,
    __nv_bfloat16 * __restrict__ dst, const int ncols, int nrows,
    const int64_t stride_y)
{
    mmvq_q2_K_actstrided_tmpl<__nv_bfloat16, __nv_bfloat16>(vx, yy, dst, ncols, nrows, stride_y);
}

template <typename ActT, typename DstT>
static __device__ void mmvq_q3_K_actstrided_tmpl(
    const void * __restrict__ vx, const ActT * __restrict__ yy,
    DstT * __restrict__ dst, const int ncols, int nrows,
    const int64_t stride_y)
{
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    if (row > nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q3_K * x = (const block_q3_K *)vx + ib0;

    float tmp = 0;

    const uint16_t kmask1 = 0x0303;
    const uint16_t kmask2 = 0x0f0f;

    const int tid = threadIdx.x/K_QUANTS_PER_ITERATION;
    const int ix  = threadIdx.x%K_QUANTS_PER_ITERATION;

    const int n  = K_QUANTS_PER_ITERATION;
    const int step = 16/K_QUANTS_PER_ITERATION;
    const int im = tid/step;
    const int in = tid - step*im;

    const uint8_t mb = 1 << (4*im);

    const int l0 = n*in;
    const int q_offset =  32*im + l0;
    const int y_offset = 128*im + l0;

    uint16_t utmp[4];
    const int8_t * s = (const int8_t *)utmp;

    const uint16_t s_shift = 4*im;

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const int64_t y_base = ((int64_t)i * QK_K + y_offset) * stride_y;
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
            #define Y(off) mmvq_io<ActT>::load(&yy[y_base + (int64_t)(off) * stride_y])
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
        mmvq_io<DstT>::store(&dst[row], tmp);
    }
}

extern "C" __global__ void baracuda_gguf_mmvq_q3_K_f32_actstrided_kernel(
    const void * __restrict__ vx, const float * __restrict__ yy,
    float * __restrict__ dst, const int ncols, int nrows,
    const int64_t stride_y)
{
    mmvq_q3_K_actstrided_tmpl<float, float>(vx, yy, dst, ncols, nrows, stride_y);
}

extern "C" __global__ void baracuda_gguf_mmvq_q3_K_f16_actstrided_kernel(
    const void * __restrict__ vx, const __half * __restrict__ yy,
    __half * __restrict__ dst, const int ncols, int nrows,
    const int64_t stride_y)
{
    mmvq_q3_K_actstrided_tmpl<__half, __half>(vx, yy, dst, ncols, nrows, stride_y);
}

extern "C" __global__ void baracuda_gguf_mmvq_q3_K_bf16_actstrided_kernel(
    const void * __restrict__ vx, const __nv_bfloat16 * __restrict__ yy,
    __nv_bfloat16 * __restrict__ dst, const int ncols, int nrows,
    const int64_t stride_y)
{
    mmvq_q3_K_actstrided_tmpl<__nv_bfloat16, __nv_bfloat16>(vx, yy, dst, ncols, nrows, stride_y);
}

template <typename ActT, typename DstT>
static __device__ void mmvq_q4_K_actstrided_tmpl(
    const void * __restrict__ vx, const ActT * __restrict__ yy,
    DstT * __restrict__ dst, const int ncols, int nrows,
    const int64_t stride_y)
{
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    if (row > nrows) return;
    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q4_K * x = (const block_q4_K *)vx + ib0;

    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int tid = threadIdx.x/K_QUANTS_PER_ITERATION;
    const int ix  = threadIdx.x%K_QUANTS_PER_ITERATION;

    const int step = 8/K_QUANTS_PER_ITERATION;

    const int il  = tid/step;
    const int ir  = tid - step*il;
    const int n   = 2 * K_QUANTS_PER_ITERATION;

    const int im = il/2;
    const int in = il%2;

    const int l0 = n*(2*ir + in);
    const int q_offset = 32*im + l0;
    const int y_offset = 64*im + l0;

    uint16_t aux[4];
    const uint8_t * sc = (const uint8_t *)aux;

    uint32_t q32[4];
    const uint8_t * q4 = (const uint8_t *)q32;

    float tmp = 0;

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const int64_t y1_base = ((int64_t)i * QK_K + y_offset) * stride_y;
        const int64_t y2_base = y1_base + (int64_t)128 * stride_y;

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

        float4 sv = {0.f, 0.f, 0.f, 0.f};
        float smin = 0;
        for (int l = 0; l < 4; ++l) {
            #define Y1(off) mmvq_io<ActT>::load(&yy[y1_base + (int64_t)(off) * stride_y])
            #define Y2(off) mmvq_io<ActT>::load(&yy[y2_base + (int64_t)(off) * stride_y])
            sv.x += Y1(l)    * q4[l+0]; sv.y += Y1(l+32) * q4[l+ 4];
            sv.z += Y2(l)    * q4[l+8]; sv.w += Y2(l+32) * q4[l+12];
            smin += Y1(l) * sc[2] + Y1(l+32) * sc[3] + Y2(l) * sc[6] + Y2(l+32) * sc[7];
            #undef Y1
            #undef Y2
        }
        tmp += dall * (sv.x * sc[0] + sv.y * sc[1] * 1.f/16.f + sv.z * sc[4] + sv.w * sc[5] * 1.f/16.f) - dmin * smin;
    }

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (tid == 0) {
        mmvq_io<DstT>::store(&dst[row], tmp);
    }
}

extern "C" __global__ void baracuda_gguf_mmvq_q4_K_f32_actstrided_kernel(
    const void * __restrict__ vx, const float * __restrict__ yy,
    float * __restrict__ dst, const int ncols, int nrows,
    const int64_t stride_y)
{
    mmvq_q4_K_actstrided_tmpl<float, float>(vx, yy, dst, ncols, nrows, stride_y);
}

extern "C" __global__ void baracuda_gguf_mmvq_q4_K_f16_actstrided_kernel(
    const void * __restrict__ vx, const __half * __restrict__ yy,
    __half * __restrict__ dst, const int ncols, int nrows,
    const int64_t stride_y)
{
    mmvq_q4_K_actstrided_tmpl<__half, __half>(vx, yy, dst, ncols, nrows, stride_y);
}

extern "C" __global__ void baracuda_gguf_mmvq_q4_K_bf16_actstrided_kernel(
    const void * __restrict__ vx, const __nv_bfloat16 * __restrict__ yy,
    __nv_bfloat16 * __restrict__ dst, const int ncols, int nrows,
    const int64_t stride_y)
{
    mmvq_q4_K_actstrided_tmpl<__nv_bfloat16, __nv_bfloat16>(vx, yy, dst, ncols, nrows, stride_y);
}

template <typename ActT, typename DstT>
static __device__ void mmvq_q5_K_actstrided_tmpl(
    const void * __restrict__ vx, const ActT * __restrict__ yy,
    DstT * __restrict__ dst, const int ncols,
    const int64_t stride_y)
{
    const int row = blockIdx.x;
    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q5_K * x = (const block_q5_K *)vx + ib0;

    float tmp = 0;

    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int tid = threadIdx.x/2;
    const int ix  = threadIdx.x%2;

    const int il  = tid/4;
    const int ir  = tid - 4*il;
    const int n   = 2;

    const int im = il/2;
    const int in = il%2;

    const int l0 = n*(2*ir + in);
    const int q_offset = 32*im + l0;
    const int y_offset = 64*im + l0;

    const uint8_t hm1  = 1 << (2*im);
    const uint8_t hm2  = hm1 << 4;

    uint16_t aux[4];
    const uint8_t * sc = (const uint8_t *)aux;

    uint16_t q16[8];
    const uint8_t * q4 = (const uint8_t *)q16;

    for (int i = ix; i < num_blocks_per_row; i += 2) {

        const uint8_t * ql1 = x[i].qs + q_offset;
        const uint8_t * qh  = x[i].qh + l0;
        const int64_t y1_base = ((int64_t)i * QK_K + y_offset) * stride_y;
        const int64_t y2_base = y1_base + (int64_t)128 * stride_y;

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
            #define Y1(off) mmvq_io<ActT>::load(&yy[y1_base + (int64_t)(off) * stride_y])
            #define Y2(off) mmvq_io<ActT>::load(&yy[y2_base + (int64_t)(off) * stride_y])
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
        mmvq_io<DstT>::store(&dst[row], tmp);
    }
}

extern "C" __global__ void baracuda_gguf_mmvq_q5_K_f32_actstrided_kernel(
    const void * __restrict__ vx, const float * __restrict__ yy,
    float * __restrict__ dst, const int ncols,
    const int64_t stride_y)
{
    mmvq_q5_K_actstrided_tmpl<float, float>(vx, yy, dst, ncols, stride_y);
}

extern "C" __global__ void baracuda_gguf_mmvq_q5_K_f16_actstrided_kernel(
    const void * __restrict__ vx, const __half * __restrict__ yy,
    __half * __restrict__ dst, const int ncols,
    const int64_t stride_y)
{
    mmvq_q5_K_actstrided_tmpl<__half, __half>(vx, yy, dst, ncols, stride_y);
}

extern "C" __global__ void baracuda_gguf_mmvq_q5_K_bf16_actstrided_kernel(
    const void * __restrict__ vx, const __nv_bfloat16 * __restrict__ yy,
    __nv_bfloat16 * __restrict__ dst, const int ncols,
    const int64_t stride_y)
{
    mmvq_q5_K_actstrided_tmpl<__nv_bfloat16, __nv_bfloat16>(vx, yy, dst, ncols, stride_y);
}

template <typename ActT, typename DstT>
static __device__ void mmvq_q6_K_actstrided_tmpl(
    const void * __restrict__ vx, const ActT * __restrict__ yy,
    DstT * __restrict__ dst, const int ncols, int nrows,
    const int64_t stride_y)
{
    static_assert(16 % K_QUANTS_PER_ITERATION == 0,
                  "16 must be divisible by K_QUANTS_PER_ITERATION");

    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    if (row > nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q6_K * x = (const block_q6_K *)vx + ib0;

    const int tid = threadIdx.x/K_QUANTS_PER_ITERATION;
    const int ix  = threadIdx.x%K_QUANTS_PER_ITERATION;

    const int step = 16/K_QUANTS_PER_ITERATION;

    const int im = tid/step;
    const int in = tid - step*im;

    const int l0 = 4 * in;
    const int is = in / 4;

    const int ql_offset = 64*im + l0;
    const int qh_offset = 32*im + l0;
    const int s_offset  =  8*im + is;
    const int y_offset = 128*im + l0;

    float tmp = 0;

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const int64_t y_base = ((int64_t)i * QK_K + y_offset) * stride_y;
        const uint8_t * ql = x[i].ql + ql_offset;
        const uint8_t * qh = x[i].qh + qh_offset;
        const int8_t  * s  = x[i].scales + s_offset;

        const float d = x[i].d;

        float sum = 0;
        for (int l = 0; l < 4; ++l) {
            #define Y(off) mmvq_io<ActT>::load(&yy[y_base + (int64_t)(off) * stride_y])
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
        mmvq_io<DstT>::store(&dst[row], tmp);
    }
}

extern "C" __global__ void baracuda_gguf_mmvq_q6_K_f32_actstrided_kernel(
    const void * __restrict__ vx, const float * __restrict__ yy,
    float * __restrict__ dst, const int ncols, int nrows,
    const int64_t stride_y)
{
    mmvq_q6_K_actstrided_tmpl<float, float>(vx, yy, dst, ncols, nrows, stride_y);
}

extern "C" __global__ void baracuda_gguf_mmvq_q6_K_f16_actstrided_kernel(
    const void * __restrict__ vx, const __half * __restrict__ yy,
    __half * __restrict__ dst, const int ncols, int nrows,
    const int64_t stride_y)
{
    mmvq_q6_K_actstrided_tmpl<__half, __half>(vx, yy, dst, ncols, nrows, stride_y);
}

extern "C" __global__ void baracuda_gguf_mmvq_q6_K_bf16_actstrided_kernel(
    const void * __restrict__ vx, const __nv_bfloat16 * __restrict__ yy,
    __nv_bfloat16 * __restrict__ dst, const int ncols, int nrows,
    const int64_t stride_y)
{
    mmvq_q6_K_actstrided_tmpl<__nv_bfloat16, __nv_bfloat16>(vx, yy, dst, ncols, nrows, stride_y);
}

template <typename ActT, typename DstT>
static __device__ void mmvq_q8_K_actstrided_tmpl(
    const void * __restrict__ vx, const ActT * __restrict__ yy,
    DstT * __restrict__ dst, const int ncols, int nrows,
    const int64_t stride_y)
{
    static_assert(QK_K % 16 == 0, "QK_K must be a multiple of 16");
    static_assert(16 % K_QUANTS_PER_ITERATION == 0,
                  "16 must be divisible by K_QUANTS_PER_ITERATION");

    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    if (row > nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q8_K * x = (const block_q8_K *)vx + ib0;

    const int tid = threadIdx.x / K_QUANTS_PER_ITERATION;
    const int ix  = threadIdx.x % K_QUANTS_PER_ITERATION;
    constexpr int CHUNK = QK_K / 16;
    const int l0 = tid * CHUNK;

    float tmp = 0.0f;

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {
        const float    d = x[i].d;
        const int8_t * q = x[i].qs + l0;
        const int64_t y_base = ((int64_t)i * QK_K + l0) * stride_y;

        float fsum = 0.0f;

#pragma unroll
        for (int l = 0; l < CHUNK; ++l) {
            fsum += (float)q[l] * mmvq_io<ActT>::load(&yy[y_base + (int64_t)l * stride_y]);
        }
        tmp += d * fsum;
    }

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (threadIdx.x == 0) {
        mmvq_io<DstT>::store(&dst[row], tmp);
    }
}

extern "C" __global__ void baracuda_gguf_mmvq_q8_K_f32_actstrided_kernel(
    const void * __restrict__ vx, const float * __restrict__ yy,
    float * __restrict__ dst, const int ncols, int nrows,
    const int64_t stride_y)
{
    mmvq_q8_K_actstrided_tmpl<float, float>(vx, yy, dst, ncols, nrows, stride_y);
}

extern "C" __global__ void baracuda_gguf_mmvq_q8_K_f16_actstrided_kernel(
    const void * __restrict__ vx, const __half * __restrict__ yy,
    __half * __restrict__ dst, const int ncols, int nrows,
    const int64_t stride_y)
{
    mmvq_q8_K_actstrided_tmpl<__half, __half>(vx, yy, dst, ncols, nrows, stride_y);
}

extern "C" __global__ void baracuda_gguf_mmvq_q8_K_bf16_actstrided_kernel(
    const void * __restrict__ vx, const __nv_bfloat16 * __restrict__ yy,
    __nv_bfloat16 * __restrict__ dst, const int ncols, int nrows,
    const int64_t stride_y)
{
    mmvq_q8_K_actstrided_tmpl<__nv_bfloat16, __nv_bfloat16>(vx, yy, dst, ncols, nrows, stride_y);
}

// =============================================================================
// Strided MMVQ launchers — Phase 14.5.
// =============================================================================

namespace {

template <typename Kernel, typename ActT, typename DstT>
inline int32_t launch_type01_mmvq_strided(
    Kernel kernel,
    int qk,
    int ncols,
    int nrows,
    const void * x,
    const ActT * y,
    DstT * dst,
    int64_t stride_y,
    cudaStream_t stream)
{
    if (ncols <= 0 || nrows <= 0 || (ncols % qk) != 0) return 2;
    const int block_num_y = ceil_div_host(nrows, GGML_CUDA_MMV_Y);
    dim3 grid(block_num_y, 1, 1);
    dim3 block(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    kernel<<<grid, block, 0, stream>>>(x, y, dst, ncols, nrows, stride_y);
    return status_from_launch(cudaPeekAtLastError());
}

} // anonymous namespace

// ---- Type-0/1 strided launchers ------------------------------------------

extern "C" int32_t baracuda_kernels_mmvq_q4_0_actstrided_run(
    int32_t ncols, int32_t nrows,
    const void * __restrict__ x,
    int64_t w_start_byte_offset,
    int64_t stride_y,
    const void * __restrict__ y,
    void       * __restrict__ dst,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!x || !y || !dst) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    const void * x_off = reinterpret_cast<const void *>(
        reinterpret_cast<const uint8_t *>(x) + w_start_byte_offset);
    return launch_type01_mmvq_strided(
        baracuda_gguf_mmvq_q4_0_f32_actstrided_kernel, QK4_0, ncols, nrows, x_off,
        static_cast<const float*>(y), static_cast<float*>(dst), stride_y, stream);
}

extern "C" int32_t baracuda_kernels_mmvq_q4_1_actstrided_run(
    int32_t ncols, int32_t nrows,
    const void * __restrict__ x,
    int64_t w_start_byte_offset,
    int64_t stride_y,
    const void * __restrict__ y,
    void       * __restrict__ dst,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!x || !y || !dst) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    const void * x_off = reinterpret_cast<const void *>(
        reinterpret_cast<const uint8_t *>(x) + w_start_byte_offset);
    return launch_type01_mmvq_strided(
        baracuda_gguf_mmvq_q4_1_f32_actstrided_kernel, QK4_1, ncols, nrows, x_off,
        static_cast<const float*>(y), static_cast<float*>(dst), stride_y, stream);
}

extern "C" int32_t baracuda_kernels_mmvq_q5_0_actstrided_run(
    int32_t ncols, int32_t nrows,
    const void * __restrict__ x,
    int64_t w_start_byte_offset,
    int64_t stride_y,
    const void * __restrict__ y,
    void       * __restrict__ dst,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!x || !y || !dst) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    const void * x_off = reinterpret_cast<const void *>(
        reinterpret_cast<const uint8_t *>(x) + w_start_byte_offset);
    return launch_type01_mmvq_strided(
        baracuda_gguf_mmvq_q5_0_f32_actstrided_kernel, QK5_0, ncols, nrows, x_off,
        static_cast<const float*>(y), static_cast<float*>(dst), stride_y, stream);
}

extern "C" int32_t baracuda_kernels_mmvq_q5_1_actstrided_run(
    int32_t ncols, int32_t nrows,
    const void * __restrict__ x,
    int64_t w_start_byte_offset,
    int64_t stride_y,
    const void * __restrict__ y,
    void       * __restrict__ dst,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!x || !y || !dst) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    const void * x_off = reinterpret_cast<const void *>(
        reinterpret_cast<const uint8_t *>(x) + w_start_byte_offset);
    return launch_type01_mmvq_strided(
        baracuda_gguf_mmvq_q5_1_f32_actstrided_kernel, QK5_1, ncols, nrows, x_off,
        static_cast<const float*>(y), static_cast<float*>(dst), stride_y, stream);
}

extern "C" int32_t baracuda_kernels_mmvq_q8_0_actstrided_run(
    int32_t ncols, int32_t nrows,
    const void * __restrict__ x,
    int64_t w_start_byte_offset,
    int64_t stride_y,
    const void * __restrict__ y,
    void       * __restrict__ dst,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!x || !y || !dst) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    const void * x_off = reinterpret_cast<const void *>(
        reinterpret_cast<const uint8_t *>(x) + w_start_byte_offset);
    return launch_type01_mmvq_strided(
        baracuda_gguf_mmvq_q8_0_f32_actstrided_kernel, QK8_0, ncols, nrows, x_off,
        static_cast<const float*>(y), static_cast<float*>(dst), stride_y, stream);
}

// ---- k-quants strided launchers ------------------------------------------

extern "C" int32_t baracuda_kernels_mmvq_q2_K_actstrided_run(
    int32_t ncols, int32_t nrows,
    const void * __restrict__ x,
    int64_t w_start_byte_offset,
    int64_t stride_y,
    const void * __restrict__ y,
    void       * __restrict__ dst,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!x || !y || !dst) return 2;
    if (ncols <= 0 || nrows <= 0 || (ncols % QK_K) != 0) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    const void * x_off = reinterpret_cast<const void *>(
        reinterpret_cast<const uint8_t *>(x) + w_start_byte_offset);
    const int block_num_y = ceil_div_host(nrows, GGML_CUDA_MMV_Y);
    dim3 grid(block_num_y, 1, 1);
    dim3 block(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    baracuda_gguf_mmvq_q2_K_f32_actstrided_kernel<<<grid, block, 0, stream>>>(
        x_off, static_cast<const float*>(y), static_cast<float*>(dst), ncols, nrows, stride_y);
    return status_from_launch(cudaPeekAtLastError());
}

extern "C" int32_t baracuda_kernels_mmvq_q3_K_actstrided_run(
    int32_t ncols, int32_t nrows,
    const void * __restrict__ x,
    int64_t w_start_byte_offset,
    int64_t stride_y,
    const void * __restrict__ y,
    void       * __restrict__ dst,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!x || !y || !dst) return 2;
    if (ncols <= 0 || nrows <= 0 || (ncols % QK_K) != 0) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    const void * x_off = reinterpret_cast<const void *>(
        reinterpret_cast<const uint8_t *>(x) + w_start_byte_offset);
    const int block_num_y = ceil_div_host(nrows, GGML_CUDA_MMV_Y);
    dim3 grid(block_num_y, 1, 1);
    dim3 block(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    baracuda_gguf_mmvq_q3_K_f32_actstrided_kernel<<<grid, block, 0, stream>>>(
        x_off, static_cast<const float*>(y), static_cast<float*>(dst), ncols, nrows, stride_y);
    return status_from_launch(cudaPeekAtLastError());
}

extern "C" int32_t baracuda_kernels_mmvq_q4_K_actstrided_run(
    int32_t ncols, int32_t nrows,
    const void * __restrict__ x,
    int64_t w_start_byte_offset,
    int64_t stride_y,
    const void * __restrict__ y,
    void       * __restrict__ dst,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!x || !y || !dst) return 2;
    if (ncols <= 0 || nrows <= 0 || (ncols % QK_K) != 0) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    const void * x_off = reinterpret_cast<const void *>(
        reinterpret_cast<const uint8_t *>(x) + w_start_byte_offset);
    const int block_num_y = ceil_div_host(nrows, GGML_CUDA_MMV_Y);
    dim3 grid(block_num_y, 1, 1);
    dim3 block(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    baracuda_gguf_mmvq_q4_K_f32_actstrided_kernel<<<grid, block, 0, stream>>>(
        x_off, static_cast<const float*>(y), static_cast<float*>(dst), ncols, nrows, stride_y);
    return status_from_launch(cudaPeekAtLastError());
}

extern "C" int32_t baracuda_kernels_mmvq_q5_K_actstrided_run(
    int32_t ncols, int32_t nrows,
    const void * __restrict__ x,
    int64_t w_start_byte_offset,
    int64_t stride_y,
    const void * __restrict__ y,
    void       * __restrict__ dst,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!x || !y || !dst) return 2;
    if (ncols <= 0 || nrows <= 0 || (ncols % QK_K) != 0) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    const void * x_off = reinterpret_cast<const void *>(
        reinterpret_cast<const uint8_t *>(x) + w_start_byte_offset);
    dim3 grid((unsigned)nrows, 1, 1);
    dim3 block(WARP_SIZE, 1, 1);
    baracuda_gguf_mmvq_q5_K_f32_actstrided_kernel<<<grid, block, 0, stream>>>(
        x_off, static_cast<const float*>(y), static_cast<float*>(dst), ncols, stride_y);
    return status_from_launch(cudaPeekAtLastError());
}

extern "C" int32_t baracuda_kernels_mmvq_q6_K_actstrided_run(
    int32_t ncols, int32_t nrows,
    const void * __restrict__ x,
    int64_t w_start_byte_offset,
    int64_t stride_y,
    const void * __restrict__ y,
    void       * __restrict__ dst,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!x || !y || !dst) return 2;
    if (ncols <= 0 || nrows <= 0 || (ncols % QK_K) != 0) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    const void * x_off = reinterpret_cast<const void *>(
        reinterpret_cast<const uint8_t *>(x) + w_start_byte_offset);
    const int block_num_y = ceil_div_host(nrows, GGML_CUDA_MMV_Y);
    dim3 grid(block_num_y, 1, 1);
    dim3 block(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    baracuda_gguf_mmvq_q6_K_f32_actstrided_kernel<<<grid, block, 0, stream>>>(
        x_off, static_cast<const float*>(y), static_cast<float*>(dst), ncols, nrows, stride_y);
    return status_from_launch(cudaPeekAtLastError());
}

extern "C" int32_t baracuda_kernels_mmvq_q8_K_actstrided_run(
    int32_t ncols, int32_t nrows,
    const void * __restrict__ x,
    int64_t w_start_byte_offset,
    int64_t stride_y,
    const void * __restrict__ y,
    void       * __restrict__ dst,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!x || !y || !dst) return 2;
    if (ncols <= 0 || nrows <= 0 || (ncols % QK_K) != 0) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    const void * x_off = reinterpret_cast<const void *>(
        reinterpret_cast<const uint8_t *>(x) + w_start_byte_offset);
    const int block_num_y = ceil_div_host(nrows, GGML_CUDA_MMV_Y);
    dim3 grid(block_num_y, 1, 1);
    dim3 block(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    baracuda_gguf_mmvq_q8_K_f32_actstrided_kernel<<<grid, block, 0, stream>>>(
        x_off, static_cast<const float*>(y), static_cast<float*>(dst), ncols, nrows, stride_y);
    return status_from_launch(cudaPeekAtLastError());
}

// =============================================================================
// Phase 18.1 — f16 / bf16 activation MMVQ launchers (activation-strided).
// =============================================================================

// ---- Type-0/1 f16 / bf16 strided launchers --------------------------------

#define BCDA_MMVQ_T01_STR_LAUNCHER(qtype, qk, kernel, fdtype, ctype) \
extern "C" int32_t baracuda_kernels_mmvq_##qtype##_actstrided_##fdtype##_run( \
    int32_t ncols, int32_t nrows, \
    const void * __restrict__ x, \
    int64_t w_start_byte_offset, int64_t stride_y, \
    const void * __restrict__ y, void * __restrict__ dst, \
    void * /*workspace*/, size_t /*workspace_bytes*/, void * stream_ptr) \
{ \
    if (!x || !y || !dst) return 2; \
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr); \
    const void * x_off = reinterpret_cast<const void *>( \
        reinterpret_cast<const uint8_t *>(x) + w_start_byte_offset); \
    return launch_type01_mmvq_strided(kernel, qk, ncols, nrows, x_off, \
        static_cast<const ctype*>(y), static_cast<ctype*>(dst), stride_y, stream); \
}

BCDA_MMVQ_T01_STR_LAUNCHER(q4_0, QK4_0, baracuda_gguf_mmvq_q4_0_f16_actstrided_kernel,  f16,  __half)
BCDA_MMVQ_T01_STR_LAUNCHER(q4_0, QK4_0, baracuda_gguf_mmvq_q4_0_bf16_actstrided_kernel, bf16, __nv_bfloat16)
BCDA_MMVQ_T01_STR_LAUNCHER(q4_1, QK4_1, baracuda_gguf_mmvq_q4_1_f16_actstrided_kernel,  f16,  __half)
BCDA_MMVQ_T01_STR_LAUNCHER(q4_1, QK4_1, baracuda_gguf_mmvq_q4_1_bf16_actstrided_kernel, bf16, __nv_bfloat16)
BCDA_MMVQ_T01_STR_LAUNCHER(q5_0, QK5_0, baracuda_gguf_mmvq_q5_0_f16_actstrided_kernel,  f16,  __half)
BCDA_MMVQ_T01_STR_LAUNCHER(q5_0, QK5_0, baracuda_gguf_mmvq_q5_0_bf16_actstrided_kernel, bf16, __nv_bfloat16)
BCDA_MMVQ_T01_STR_LAUNCHER(q5_1, QK5_1, baracuda_gguf_mmvq_q5_1_f16_actstrided_kernel,  f16,  __half)
BCDA_MMVQ_T01_STR_LAUNCHER(q5_1, QK5_1, baracuda_gguf_mmvq_q5_1_bf16_actstrided_kernel, bf16, __nv_bfloat16)
BCDA_MMVQ_T01_STR_LAUNCHER(q8_0, QK8_0, baracuda_gguf_mmvq_q8_0_f16_actstrided_kernel,  f16,  __half)
BCDA_MMVQ_T01_STR_LAUNCHER(q8_0, QK8_0, baracuda_gguf_mmvq_q8_0_bf16_actstrided_kernel, bf16, __nv_bfloat16)

#undef BCDA_MMVQ_T01_STR_LAUNCHER

// ---- k-quants f16 / bf16 strided launchers --------------------------------

#define BCDA_MMVQ_KQUANT_STR_LAUNCHER(qtype, kernel, fdtype, ctype) \
extern "C" int32_t baracuda_kernels_mmvq_##qtype##_actstrided_##fdtype##_run( \
    int32_t ncols, int32_t nrows, \
    const void * __restrict__ x, \
    int64_t w_start_byte_offset, int64_t stride_y, \
    const void * __restrict__ y, void * __restrict__ dst, \
    void * /*workspace*/, size_t /*workspace_bytes*/, void * stream_ptr) \
{ \
    if (!x || !y || !dst) return 2; \
    if (ncols <= 0 || nrows <= 0 || (ncols % QK_K) != 0) return 2; \
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr); \
    const void * x_off = reinterpret_cast<const void *>( \
        reinterpret_cast<const uint8_t *>(x) + w_start_byte_offset); \
    const int block_num_y = ceil_div_host(nrows, GGML_CUDA_MMV_Y); \
    dim3 grid(block_num_y, 1, 1); \
    dim3 block(WARP_SIZE, GGML_CUDA_MMV_Y, 1); \
    kernel<<<grid, block, 0, stream>>>(x_off, \
        static_cast<const ctype*>(y), static_cast<ctype*>(dst), ncols, nrows, stride_y); \
    return status_from_launch(cudaPeekAtLastError()); \
}

BCDA_MMVQ_KQUANT_STR_LAUNCHER(q2_K, baracuda_gguf_mmvq_q2_K_f16_actstrided_kernel,  f16,  __half)
BCDA_MMVQ_KQUANT_STR_LAUNCHER(q2_K, baracuda_gguf_mmvq_q2_K_bf16_actstrided_kernel, bf16, __nv_bfloat16)
BCDA_MMVQ_KQUANT_STR_LAUNCHER(q3_K, baracuda_gguf_mmvq_q3_K_f16_actstrided_kernel,  f16,  __half)
BCDA_MMVQ_KQUANT_STR_LAUNCHER(q3_K, baracuda_gguf_mmvq_q3_K_bf16_actstrided_kernel, bf16, __nv_bfloat16)
BCDA_MMVQ_KQUANT_STR_LAUNCHER(q4_K, baracuda_gguf_mmvq_q4_K_f16_actstrided_kernel,  f16,  __half)
BCDA_MMVQ_KQUANT_STR_LAUNCHER(q4_K, baracuda_gguf_mmvq_q4_K_bf16_actstrided_kernel, bf16, __nv_bfloat16)
BCDA_MMVQ_KQUANT_STR_LAUNCHER(q6_K, baracuda_gguf_mmvq_q6_K_f16_actstrided_kernel,  f16,  __half)
BCDA_MMVQ_KQUANT_STR_LAUNCHER(q6_K, baracuda_gguf_mmvq_q6_K_bf16_actstrided_kernel, bf16, __nv_bfloat16)
BCDA_MMVQ_KQUANT_STR_LAUNCHER(q8_K, baracuda_gguf_mmvq_q8_K_f16_actstrided_kernel,  f16,  __half)
BCDA_MMVQ_KQUANT_STR_LAUNCHER(q8_K, baracuda_gguf_mmvq_q8_K_bf16_actstrided_kernel, bf16, __nv_bfloat16)

#undef BCDA_MMVQ_KQUANT_STR_LAUNCHER

// Q5_K strided — 1D grid of `nrows`, kernel signature has no `nrows` arg.
#define BCDA_MMVQ_Q5K_STR_LAUNCHER(fdtype, ctype, kernel) \
extern "C" int32_t baracuda_kernels_mmvq_q5_K_actstrided_##fdtype##_run( \
    int32_t ncols, int32_t nrows, \
    const void * __restrict__ x, \
    int64_t w_start_byte_offset, int64_t stride_y, \
    const void * __restrict__ y, void * __restrict__ dst, \
    void * /*workspace*/, size_t /*workspace_bytes*/, void * stream_ptr) \
{ \
    if (!x || !y || !dst) return 2; \
    if (ncols <= 0 || nrows <= 0 || (ncols % QK_K) != 0) return 2; \
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr); \
    const void * x_off = reinterpret_cast<const void *>( \
        reinterpret_cast<const uint8_t *>(x) + w_start_byte_offset); \
    dim3 grid((unsigned)nrows, 1, 1); \
    dim3 block(WARP_SIZE, 1, 1); \
    kernel<<<grid, block, 0, stream>>>(x_off, \
        static_cast<const ctype*>(y), static_cast<ctype*>(dst), ncols, stride_y); \
    return status_from_launch(cudaPeekAtLastError()); \
}

BCDA_MMVQ_Q5K_STR_LAUNCHER(f16,  __half,         baracuda_gguf_mmvq_q5_K_f16_actstrided_kernel)
BCDA_MMVQ_Q5K_STR_LAUNCHER(bf16, __nv_bfloat16,  baracuda_gguf_mmvq_q5_K_bf16_actstrided_kernel)

#undef BCDA_MMVQ_Q5K_STR_LAUNCHER

// =============================================================================
// _can_implement companions -- host-side validators (Phase 66-prep).
// Mirror each _run signature minus workspace/stream; output pointers demoted
// to const void*. Returns 0 (ok) / 2 (invalid arg) / 3 (unsupported).
// =============================================================================

extern "C" int32_t baracuda_kernels_mmvq_q2_K_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/, const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && (ncols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q2_K_actstrided_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/,
    int64_t /*w_start_byte_offset*/, int64_t /*stride_y*/,
    const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && (ncols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q2_K_actstrided_bf16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/,
    int64_t /*w_start_byte_offset*/, int64_t /*stride_y*/,
    const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && (ncols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q2_K_actstrided_f16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/,
    int64_t /*w_start_byte_offset*/, int64_t /*stride_y*/,
    const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && (ncols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q2_K_bf16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/, const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && (ncols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q2_K_f16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/, const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && (ncols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q3_K_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/, const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && (ncols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q3_K_actstrided_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/,
    int64_t /*w_start_byte_offset*/, int64_t /*stride_y*/,
    const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && (ncols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q3_K_actstrided_bf16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/,
    int64_t /*w_start_byte_offset*/, int64_t /*stride_y*/,
    const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && (ncols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q3_K_actstrided_f16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/,
    int64_t /*w_start_byte_offset*/, int64_t /*stride_y*/,
    const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && (ncols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q3_K_bf16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/, const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && (ncols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q3_K_f16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/, const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && (ncols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q4_0_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/, const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && ncols < 64) return 2;
    if (ncols > 0 && (ncols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q4_0_actstrided_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/,
    int64_t /*w_start_byte_offset*/, int64_t /*stride_y*/,
    const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && ncols < 64) return 2;
    if (ncols > 0 && (ncols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q4_0_actstrided_bf16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/,
    int64_t /*w_start_byte_offset*/, int64_t /*stride_y*/,
    const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && ncols < 64) return 2;
    if (ncols > 0 && (ncols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q4_0_actstrided_f16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/,
    int64_t /*w_start_byte_offset*/, int64_t /*stride_y*/,
    const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && ncols < 64) return 2;
    if (ncols > 0 && (ncols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q4_0_bf16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/, const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && ncols < 64) return 2;
    if (ncols > 0 && (ncols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q4_0_f16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/, const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && ncols < 64) return 2;
    if (ncols > 0 && (ncols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q4_1_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/, const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && ncols < 64) return 2;
    if (ncols > 0 && (ncols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q4_1_actstrided_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/,
    int64_t /*w_start_byte_offset*/, int64_t /*stride_y*/,
    const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && ncols < 64) return 2;
    if (ncols > 0 && (ncols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q4_1_actstrided_bf16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/,
    int64_t /*w_start_byte_offset*/, int64_t /*stride_y*/,
    const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && ncols < 64) return 2;
    if (ncols > 0 && (ncols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q4_1_actstrided_f16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/,
    int64_t /*w_start_byte_offset*/, int64_t /*stride_y*/,
    const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && ncols < 64) return 2;
    if (ncols > 0 && (ncols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q4_1_bf16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/, const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && ncols < 64) return 2;
    if (ncols > 0 && (ncols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q4_1_f16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/, const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && ncols < 64) return 2;
    if (ncols > 0 && (ncols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q4_K_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/, const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && (ncols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q4_K_actstrided_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/,
    int64_t /*w_start_byte_offset*/, int64_t /*stride_y*/,
    const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && (ncols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q4_K_actstrided_bf16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/,
    int64_t /*w_start_byte_offset*/, int64_t /*stride_y*/,
    const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && (ncols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q4_K_actstrided_f16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/,
    int64_t /*w_start_byte_offset*/, int64_t /*stride_y*/,
    const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && (ncols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q4_K_bf16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/, const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && (ncols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q4_K_f16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/, const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && (ncols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q5_0_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/, const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && ncols < 64) return 2;
    if (ncols > 0 && (ncols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q5_0_actstrided_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/,
    int64_t /*w_start_byte_offset*/, int64_t /*stride_y*/,
    const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && ncols < 64) return 2;
    if (ncols > 0 && (ncols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q5_0_actstrided_bf16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/,
    int64_t /*w_start_byte_offset*/, int64_t /*stride_y*/,
    const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && ncols < 64) return 2;
    if (ncols > 0 && (ncols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q5_0_actstrided_f16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/,
    int64_t /*w_start_byte_offset*/, int64_t /*stride_y*/,
    const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && ncols < 64) return 2;
    if (ncols > 0 && (ncols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q5_0_bf16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/, const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && ncols < 64) return 2;
    if (ncols > 0 && (ncols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q5_0_f16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/, const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && ncols < 64) return 2;
    if (ncols > 0 && (ncols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q5_1_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/, const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && ncols < 64) return 2;
    if (ncols > 0 && (ncols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q5_1_actstrided_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/,
    int64_t /*w_start_byte_offset*/, int64_t /*stride_y*/,
    const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && ncols < 64) return 2;
    if (ncols > 0 && (ncols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q5_1_actstrided_bf16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/,
    int64_t /*w_start_byte_offset*/, int64_t /*stride_y*/,
    const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && ncols < 64) return 2;
    if (ncols > 0 && (ncols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q5_1_actstrided_f16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/,
    int64_t /*w_start_byte_offset*/, int64_t /*stride_y*/,
    const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && ncols < 64) return 2;
    if (ncols > 0 && (ncols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q5_1_bf16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/, const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && ncols < 64) return 2;
    if (ncols > 0 && (ncols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q5_1_f16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/, const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && ncols < 64) return 2;
    if (ncols > 0 && (ncols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q5_K_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/, const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && (ncols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q5_K_actstrided_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/,
    int64_t /*w_start_byte_offset*/, int64_t /*stride_y*/,
    const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && (ncols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q5_K_actstrided_bf16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/,
    int64_t /*w_start_byte_offset*/, int64_t /*stride_y*/,
    const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && (ncols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q5_K_actstrided_f16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/,
    int64_t /*w_start_byte_offset*/, int64_t /*stride_y*/,
    const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && (ncols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q5_K_bf16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/, const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && (ncols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q5_K_f16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/, const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && (ncols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q6_K_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/, const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && (ncols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q6_K_actstrided_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/,
    int64_t /*w_start_byte_offset*/, int64_t /*stride_y*/,
    const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && (ncols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q6_K_actstrided_bf16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/,
    int64_t /*w_start_byte_offset*/, int64_t /*stride_y*/,
    const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && (ncols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q6_K_actstrided_f16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/,
    int64_t /*w_start_byte_offset*/, int64_t /*stride_y*/,
    const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && (ncols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q6_K_bf16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/, const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && (ncols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q6_K_f16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/, const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && (ncols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q8_0_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/, const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && ncols < 64) return 2;
    if (ncols > 0 && (ncols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q8_0_actstrided_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/,
    int64_t /*w_start_byte_offset*/, int64_t /*stride_y*/,
    const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && ncols < 64) return 2;
    if (ncols > 0 && (ncols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q8_0_actstrided_bf16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/,
    int64_t /*w_start_byte_offset*/, int64_t /*stride_y*/,
    const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && ncols < 64) return 2;
    if (ncols > 0 && (ncols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q8_0_actstrided_f16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/,
    int64_t /*w_start_byte_offset*/, int64_t /*stride_y*/,
    const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && ncols < 64) return 2;
    if (ncols > 0 && (ncols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q8_0_bf16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/, const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && ncols < 64) return 2;
    if (ncols > 0 && (ncols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q8_0_f16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/, const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && ncols < 64) return 2;
    if (ncols > 0 && (ncols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q8_K_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/, const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && (ncols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q8_K_actstrided_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/,
    int64_t /*w_start_byte_offset*/, int64_t /*stride_y*/,
    const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && (ncols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q8_K_actstrided_bf16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/,
    int64_t /*w_start_byte_offset*/, int64_t /*stride_y*/,
    const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && (ncols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q8_K_actstrided_f16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/,
    int64_t /*w_start_byte_offset*/, int64_t /*stride_y*/,
    const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && (ncols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q8_K_bf16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/, const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && (ncols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q8_K_f16_can_implement(
    int32_t ncols, int32_t nrows,
    const void * /*x*/, const void * /*y*/, const void * /*dst*/)
{
    if (ncols < 0 || nrows < 0) return 2;
    if (ncols > 0 && (ncols % QK_K) != 0) return 2;
    return 0;
}

