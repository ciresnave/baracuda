// SPDX-FileCopyrightText: 2023-2024 The llama.cpp / ggml authors  (MIT)
// SPDX-FileCopyrightText: 2024-2026 Fuel project contributors      (MIT OR Apache-2.0)
// SPDX-FileCopyrightText: 2026 baracuda project contributors       (MIT OR Apache-2.0)
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// mmvq_multim.cu — Phase 33 multi-M MMVQ launchers.
//
// One kernel template (`mul_mat_vec_q_multim_tmpl<ncols_y, ...>`) per
// supported M ∈ {1, 2, 4, 8}; one launcher entry point per (qtype, M).
//
// Op shape:
//   For each m ∈ [0, M):  out[m, r] = Σ_c W_q[r, c] · y[m, c]
//   where W_q is a single packed weight matrix (shared across all M
//   activations) and y is the Q8_1-staged activation buffer produced
//   by `quantize_q8_1_*_run` upstream.
//
// Scope
// -----
// Phase 33 shipped Q8_0 only.
// Phase 34 adds Q4_0/Q4_1/Q5_0/Q5_1/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K — 9 formats,
// 4 M-sizes each = 36 new launchers + 36 FFI symbols.

#include "../include/baracuda_mmvq_multim.cuh"

#include <cstdint>
#include <cuda_runtime.h>

using namespace baracuda::mmvq_multim;
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
// Note: QK* / QI* / WARP_SIZE are `#define` macros (vendored from
// llama.cpp's layout); they cannot be `using`-imported. They're visible
// transitively through `baracuda_gguf.cuh`.

namespace {

inline int32_t status_from_launch(cudaError_t err) {
    return (err != cudaSuccess) ? 5 : 0;
}

inline int ceil_div_host(int a, int b) { return (a + b - 1) / b; }

template <int ncols_y>
__global__ void mul_mat_vec_q8_0_q8_1_multim_kernel(
    const void * __restrict__ vx, const void * __restrict__ vy,
    float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst)
{
    mul_mat_vec_q_multim_tmpl<
        ncols_y, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

template <int ncols_y>
int32_t launch_q8_0_multim(
    int32_t ncols_x, int32_t nrows_x,
    const void * vx, const void * vy_q8_1,
    float * dst,
    cudaStream_t stream)
{
    if (ncols_x <= 0 || nrows_x <= 0) { return 2; }
    if ((ncols_x % QK8_0) != 0)        { return 2; }

    constexpr int nwarps              = (ncols_y <= 4) ? 4 : 2;
    constexpr int rows_per_cuda_block = (ncols_y == 1) ? 1 : 2;

    const int nrows_y    = ncols_x;       // K dim of activation, padded to QK8_1=32 upstream
    const int nrows_dst  = nrows_x;       // dst is [M, nrows_x]; nrows_x is the leading dim

    dim3 block(WARP_SIZE, nwarps, 1);
    dim3 grid(ceil_div_host(nrows_x, rows_per_cuda_block), 1, 1);

    mul_mat_vec_q8_0_q8_1_multim_kernel<ncols_y><<<grid, block, 0, stream>>>(
        vx, vy_q8_1, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
    return status_from_launch(cudaPeekAtLastError());
}

} // anonymous namespace

// ---- FFI surface ----------------------------------------------------------
//
// Symbol naming: `baracuda_kernels_mmvq_multim_<qtype>_m<M>_run`. Output
// dst layout = `[M, nrows_x]` (row-major; the kernel writes
// `dst[j * nrows_dst + r] = ...` for j ∈ [0, M), r ∈ [0, nrows_x)).
//
// Inputs:
//   ncols_x        = K dimension (# unpacked weight columns; must be a
//                    multiple of 32).
//   nrows_x        = output rows (= packed weight matrix rows).
//   w_ptr / w_off  = packed weight bytes (one matrix). w_off allows the
//                    weight to start mid-buffer (e.g. when multiple
//                    matrices share an allocation).
//   activations_q8_1 = staged Q8_1 activation buffer, shape
//                      `[M × (ncols_x / 32)]` blocks (see
//                      `baracuda_kernels_quantize_q8_1_workspace_bytes`).
//   dst            = output buffer, shape `[M × nrows_x]` f32.

#define BCDA_DEFINE_MULTIM_LAUNCHER(M) \
extern "C" int32_t baracuda_kernels_mmvq_multim_q8_0_m##M##_run( \
    int32_t ncols_x, int32_t nrows_x, \
    const void * __restrict__ w_ptr, \
    int64_t w_start_byte_offset, \
    const void * __restrict__ activations_q8_1, \
    void       * __restrict__ dst, \
    void * /*workspace*/, size_t /*workspace_bytes*/, \
    void * stream_ptr) \
{ \
    if (!w_ptr || !activations_q8_1 || !dst) { return 2; } \
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr); \
    const void * w_off_ptr = reinterpret_cast<const void *>( \
        reinterpret_cast<const uint8_t *>(w_ptr) + w_start_byte_offset); \
    return launch_q8_0_multim<M>( \
        ncols_x, nrows_x, w_off_ptr, activations_q8_1, \
        static_cast<float *>(dst), stream); \
}

BCDA_DEFINE_MULTIM_LAUNCHER(1)
BCDA_DEFINE_MULTIM_LAUNCHER(2)
BCDA_DEFINE_MULTIM_LAUNCHER(4)
BCDA_DEFINE_MULTIM_LAUNCHER(8)

#undef BCDA_DEFINE_MULTIM_LAUNCHER

// =============================================================================
// Phase 34 — multi-M launchers for 9 remaining GGUF block formats.
//
// Per-format kernel + launcher + FFI macro family. The kernel /
// launcher pattern mirrors the Q8_0 path above; each format
// substitutes `(block_q_t, QK_, QI_, VDR_, vec_dot_)`. The FFI symbol
// follows `baracuda_kernels_mmvq_multim_<qtype>_m<M>_run`.
//
// Note: `(ncols_x % QK_) == 0` is enforced; for k-quants QK_=256
// (Q2_K/Q3_K/Q4_K/Q5_K/Q6_K), and for type-0/1 QK_=32.
// =============================================================================

namespace {

#define BCDA_MULTIM_PERFMT(QTYPE, BLOCK_T, QK_, QI_, VDR_, VEC_DOT_)        \
    template <int ncols_y>                                                  \
    __global__ void mul_mat_vec_##QTYPE##_q8_1_multim_kernel(               \
        const void * __restrict__ vx, const void * __restrict__ vy,         \
        float * __restrict__ dst,                                           \
        const int ncols_x, const int nrows_x,                               \
        const int nrows_y, const int nrows_dst)                             \
    {                                                                       \
        mul_mat_vec_q_multim_tmpl<                                          \
            ncols_y, QK_, QI_, BLOCK_T, VDR_, VEC_DOT_>                     \
            (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);            \
    }                                                                       \
                                                                            \
    template <int ncols_y>                                                  \
    int32_t launch_##QTYPE##_multim(                                        \
        int32_t ncols_x, int32_t nrows_x,                                   \
        const void * vx, const void * vy_q8_1,                              \
        float * dst,                                                        \
        cudaStream_t stream)                                                \
    {                                                                       \
        if (ncols_x <= 0 || nrows_x <= 0) { return 2; }                     \
        if ((ncols_x % (QK_)) != 0)       { return 2; }                     \
        constexpr int nwarps              = (ncols_y <= 4) ? 4 : 2;         \
        constexpr int rows_per_cuda_block = (ncols_y == 1) ? 1 : 2;         \
        const int nrows_y   = ncols_x;                                      \
        const int nrows_dst = nrows_x;                                      \
        dim3 block(WARP_SIZE, nwarps, 1);                                   \
        dim3 grid(ceil_div_host(nrows_x, rows_per_cuda_block), 1, 1);       \
        mul_mat_vec_##QTYPE##_q8_1_multim_kernel<ncols_y>                   \
            <<<grid, block, 0, stream>>>(                                   \
                vx, vy_q8_1, dst, ncols_x, nrows_x, nrows_y, nrows_dst);    \
        return status_from_launch(cudaPeekAtLastError());                   \
    }

BCDA_MULTIM_PERFMT(q4_0, block_q4_0, QK4_0, QI4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1)
BCDA_MULTIM_PERFMT(q4_1, block_q4_1, QK4_1, QI4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1)
BCDA_MULTIM_PERFMT(q5_0, block_q5_0, QK5_0, QI5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1)
BCDA_MULTIM_PERFMT(q5_1, block_q5_1, QK5_1, QI5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1)
BCDA_MULTIM_PERFMT(q2_K, block_q2_K, QK_K,  QI2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1)
BCDA_MULTIM_PERFMT(q3_K, block_q3_K, QK_K,  QI3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1)
BCDA_MULTIM_PERFMT(q4_K, block_q4_K, QK_K,  QI4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1)
BCDA_MULTIM_PERFMT(q5_K, block_q5_K, QK_K,  QI5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1)
BCDA_MULTIM_PERFMT(q6_K, block_q6_K, QK_K,  QI6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1)

#undef BCDA_MULTIM_PERFMT

} // anonymous namespace

#define BCDA_DEFINE_MULTIM_LAUNCHER_FMT(QTYPE, M)                            \
extern "C" int32_t baracuda_kernels_mmvq_multim_##QTYPE##_m##M##_run(        \
    int32_t ncols_x, int32_t nrows_x,                                        \
    const void * __restrict__ w_ptr,                                         \
    int64_t w_start_byte_offset,                                             \
    const void * __restrict__ activations_q8_1,                              \
    void       * __restrict__ dst,                                           \
    void * /*workspace*/, size_t /*workspace_bytes*/,                        \
    void * stream_ptr)                                                       \
{                                                                            \
    if (!w_ptr || !activations_q8_1 || !dst) { return 2; }                   \
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);             \
    const void * w_off_ptr = reinterpret_cast<const void *>(                 \
        reinterpret_cast<const uint8_t *>(w_ptr) + w_start_byte_offset);     \
    return launch_##QTYPE##_multim<M>(                                       \
        ncols_x, nrows_x, w_off_ptr, activations_q8_1,                       \
        static_cast<float *>(dst), stream);                                  \
}

#define BCDA_DEFINE_MULTIM_LAUNCHERS_ALL_M(QTYPE)        \
    BCDA_DEFINE_MULTIM_LAUNCHER_FMT(QTYPE, 1)            \
    BCDA_DEFINE_MULTIM_LAUNCHER_FMT(QTYPE, 2)            \
    BCDA_DEFINE_MULTIM_LAUNCHER_FMT(QTYPE, 4)            \
    BCDA_DEFINE_MULTIM_LAUNCHER_FMT(QTYPE, 8)

BCDA_DEFINE_MULTIM_LAUNCHERS_ALL_M(q4_0)
BCDA_DEFINE_MULTIM_LAUNCHERS_ALL_M(q4_1)
BCDA_DEFINE_MULTIM_LAUNCHERS_ALL_M(q5_0)
BCDA_DEFINE_MULTIM_LAUNCHERS_ALL_M(q5_1)
BCDA_DEFINE_MULTIM_LAUNCHERS_ALL_M(q2_K)
BCDA_DEFINE_MULTIM_LAUNCHERS_ALL_M(q3_K)
BCDA_DEFINE_MULTIM_LAUNCHERS_ALL_M(q4_K)
BCDA_DEFINE_MULTIM_LAUNCHERS_ALL_M(q5_K)
BCDA_DEFINE_MULTIM_LAUNCHERS_ALL_M(q6_K)

#undef BCDA_DEFINE_MULTIM_LAUNCHERS_ALL_M
#undef BCDA_DEFINE_MULTIM_LAUNCHER_FMT
