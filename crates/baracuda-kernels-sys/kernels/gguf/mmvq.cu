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
//                   Q2_K, Q3_K, Q4_K, Q5_K, Q6_K (k-quants, QK_K=256)
//   Activation    : f32   (f16 / bf16 deferred)
//   Output        : f32
//
// Q8_K MMVQ is intentionally NOT shipped — llama.cpp / Fuel reserves
// Q8_K as a CPU-side intermediate and never wires a `dequantize_mul_mat_vec_q8_k`
// kernel. The Rust plan's GgufBlockFormat::Q8_K dispatches at MMVQ
// time will return Error::Unsupported (matches Fuel's behavior).
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
    dequantize_mul_mat_vec<QK4_0, QR4_0, dequantize_q4_0>(vx, y, dst, ncols, nrows);
}

extern "C" __global__ void baracuda_gguf_mmvq_q4_1_f32_kernel(
    const void * __restrict__ vx, const float * __restrict__ y,
    float * __restrict__ dst, const int ncols, const int nrows)
{
    dequantize_mul_mat_vec<QK4_1, QR4_1, dequantize_q4_1>(vx, y, dst, ncols, nrows);
}

extern "C" __global__ void baracuda_gguf_mmvq_q5_0_f32_kernel(
    const void * __restrict__ vx, const float * __restrict__ y,
    float * __restrict__ dst, const int ncols, const int nrows)
{
    dequantize_mul_mat_vec<QK5_0, QR5_0, dequantize_q5_0>(vx, y, dst, ncols, nrows);
}

extern "C" __global__ void baracuda_gguf_mmvq_q5_1_f32_kernel(
    const void * __restrict__ vx, const float * __restrict__ y,
    float * __restrict__ dst, const int ncols, const int nrows)
{
    dequantize_mul_mat_vec<QK5_1, QR5_1, dequantize_q5_1>(vx, y, dst, ncols, nrows);
}

extern "C" __global__ void baracuda_gguf_mmvq_q8_0_f32_kernel(
    const void * __restrict__ vx, const float * __restrict__ y,
    float * __restrict__ dst, const int ncols, const int nrows)
{
    dequantize_mul_mat_vec<QK8_0, QR8_0, dequantize_q8_0>(vx, y, dst, ncols, nrows);
}

// =============================================================================
// k-quants MMVQ kernels — vendored verbatim from fuel-cuda-kernels
// (QK_K == 256 path; non-256 path elided since baracuda's GGUF surface
// pins QK_K = 256).
// =============================================================================

extern "C" __global__ void baracuda_gguf_mmvq_q2_K_f32_kernel(
    const void * __restrict__ vx, const float * __restrict__ yy,
    float * __restrict__ dst, const int ncols, int nrows)
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

        const float   * y = yy + i * QK_K + y_offset;
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
            sum1 += y[l+ 0] * d[0] * ((q[l+ 0] >> 0) & 3)
                  + y[l+32] * d[2] * ((q[l+ 0] >> 2) & 3)
                  + y[l+64] * d[4] * ((q[l+ 0] >> 4) & 3)
                  + y[l+96] * d[6] * ((q[l+ 0] >> 6) & 3)
                  + y[l+16] * d[1] * ((q[l+16] >> 0) & 3)
                  + y[l+48] * d[3] * ((q[l+16] >> 2) & 3)
                  + y[l+80] * d[5] * ((q[l+16] >> 4) & 3)
                  +y[l+112] * d[7] * ((q[l+16] >> 6) & 3);
            sum2 += y[l+ 0] * m[0] + y[l+32] * m[2] + y[l+64] * m[4] + y[ l+96] * m[6]
                  + y[l+16] * m[1] + y[l+48] * m[3] + y[l+80] * m[5] + y[l+112] * m[7];
        }
        tmp += dall * sum1 - dmin * sum2;
    }

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (threadIdx.x == 0) {
        dst[row] = tmp;
    }
}

extern "C" __global__ void baracuda_gguf_mmvq_q3_K_f32_kernel(
    const void * __restrict__ vx, const float * __restrict__ yy,
    float * __restrict__ dst, const int ncols, int nrows)
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

        const float   * y  = yy + i * QK_K + y_offset;
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
            sum += y[l+ 0] * (s[0] - 32) * (((q[l] >> 0) & 3) - (h[l] & (m << 0) ? 0 : 4))
                 + y[l+32] * (s[2] - 32) * (((q[l] >> 2) & 3) - (h[l] & (m << 1) ? 0 : 4))
                 + y[l+64] * (s[4] - 32) * (((q[l] >> 4) & 3) - (h[l] & (m << 2) ? 0 : 4))
                 + y[l+96] * (s[6] - 32) * (((q[l] >> 6) & 3) - (h[l] & (m << 3) ? 0 : 4));
            sum += y[l+16] * (s[1] - 32) * (((q[l+16] >> 0) & 3) - (h[l+16] & (m << 0) ? 0 : 4))
                 + y[l+48] * (s[3] - 32) * (((q[l+16] >> 2) & 3) - (h[l+16] & (m << 1) ? 0 : 4))
                 + y[l+80] * (s[5] - 32) * (((q[l+16] >> 4) & 3) - (h[l+16] & (m << 2) ? 0 : 4))
                + y[l+112] * (s[7] - 32) * (((q[l+16] >> 6) & 3) - (h[l+16] & (m << 3) ? 0 : 4));
        }
        tmp += d * sum;
    }

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (threadIdx.x == 0) {
        dst[row] = tmp;
    }
}

extern "C" __global__ void baracuda_gguf_mmvq_q4_K_f32_kernel(
    const void * __restrict__ vx, const float * __restrict__ yy,
    float * __restrict__ dst, const int ncols, int nrows)
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

        const float   * y1 = yy + i*QK_K + y_offset;
        const float   * y2 = y1 + 128;

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
            s.x += y1[l] * q4[l+0]; s.y += y1[l+32] * q4[l+ 4];
            s.z += y2[l] * q4[l+8]; s.w += y2[l+32] * q4[l+12];
            smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
        }
        tmp += dall * (s.x * sc[0] + s.y * sc[1] * 1.f/16.f + s.z * sc[4] + s.w * sc[5] * 1.f/16.f) - dmin * smin;
    }

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

extern "C" __global__ void baracuda_gguf_mmvq_q5_K_f32_kernel(
    const void * __restrict__ vx, const float * __restrict__ yy,
    float * __restrict__ dst, const int ncols)
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
        const float   * y1  = yy + i*QK_K + y_offset;
        const float   * y2  = y1 + 128;

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
            sum.x += y1[l+ 0] * (q4[l +0] + (qh[l+ 0] & (hm1 << 0) ? 16 : 0))
                   + y1[l+16] * (q4[l +2] + (qh[l+16] & (hm1 << 0) ? 16 : 0));
            sum.y += y1[l+32] * (q4[l +4] + (qh[l+ 0] & (hm1 << 1) ? 16 : 0))
                   + y1[l+48] * (q4[l +6] + (qh[l+16] & (hm1 << 1) ? 16 : 0));
            sum.z += y2[l+ 0] * (q4[l +8] + (qh[l+ 0] & (hm2 << 0) ? 16 : 0))
                   + y2[l+16] * (q4[l+10] + (qh[l+16] & (hm2 << 0) ? 16 : 0));
            sum.w += y2[l+32] * (q4[l+12] + (qh[l+ 0] & (hm2 << 1) ? 16 : 0))
                   + y2[l+48] * (q4[l+14] + (qh[l+16] & (hm2 << 1) ? 16 : 0));
            smin += (y1[l] + y1[l+16]) * sc[2] + (y1[l+32] + y1[l+48]) * sc[3]
                  + (y2[l] + y2[l+16]) * sc[6] + (y2[l+32] + y2[l+48]) * sc[7];
        }
        tmp += dall * (sum.x * sc[0] + sum.y * sc[1] + sum.z * sc[4] + sum.w * sc[5]) - dmin * smin;
    }

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (threadIdx.x == 0) {
        dst[row] = tmp;
    }
}

extern "C" __global__ void baracuda_gguf_mmvq_q6_K_f32_kernel(
    const void * __restrict__ vx, const float * __restrict__ yy,
    float * __restrict__ dst, const int ncols, int nrows)
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

        const float   * y  = yy + i * QK_K + y_offset;
        const uint8_t * ql = x[i].ql + ql_offset;
        const uint8_t * qh = x[i].qh + qh_offset;
        const int8_t  * s  = x[i].scales + s_offset;

        const float d = x[i].d;

        float sum = 0;
        for (int l = 0; l < 4; ++l) {
            sum += y[l+ 0] * s[0] * d * ((int8_t)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32)
                 + y[l+32] * s[2] * d * ((int8_t)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32)
                 + y[l+64] * s[4] * d * ((int8_t)((ql[l+ 0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32)
                 + y[l+96] * s[6] * d * ((int8_t)((ql[l+32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
        }
        tmp += sum;
    }

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
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

template <typename Kernel>
inline int32_t launch_type01_mmvq(
    Kernel kernel,
    int qk,
    int ncols,
    int nrows,
    const void * x,
    const float * y,
    float * dst,
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
