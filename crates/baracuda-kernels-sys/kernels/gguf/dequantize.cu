// SPDX-FileCopyrightText: 2023-2024 The llama.cpp / ggml authors  (MIT)
// SPDX-FileCopyrightText: 2024-2026 Fuel project contributors      (MIT OR Apache-2.0)
// SPDX-FileCopyrightText: 2026 baracuda project contributors       (MIT OR Apache-2.0)
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// baracuda-kernels Phase 8 Milestone 8.4 — GGUF dequantize launchers.
//
// Vendored from llama.cpp via fuel-cuda-kernels. See
// `kernels/include/baracuda_gguf.cuh` for full lineage notes.
//
// Trailblazer dtype matrix:
//   Block formats : Q4_0, Q4_1, Q5_0, Q5_1, Q8_0          (type-0/1, 32-elem blocks)
//                   Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K    (k-quants, 256-elem blocks)
//   Output FP     : f32   (f16 output deferred to a follow-up)
//
// One `_run` launcher per (block-format) combo; output is always f32 today.
// The Rust plan in `crates/baracuda-kernels/src/quantize/gguf/dequantize.rs`
// dispatches across block formats.

#include "../include/baracuda_gguf.cuh"

using namespace baracuda::gguf;

// =============================================================================
// __global__ kernel entry points (one per block-format). f32 output only.
// =============================================================================

extern "C" __global__ void baracuda_gguf_dequantize_q4_0_f32_kernel(
    const void * __restrict__ vx, float * __restrict__ y, const int nb32)
{
    dequantize_block_q4_0_tmpl<float>(vx, y, nb32);
}

extern "C" __global__ void baracuda_gguf_dequantize_q4_1_f32_kernel(
    const void * __restrict__ vx, float * __restrict__ y, const int nb32)
{
    dequantize_block_q4_1_tmpl<float>(vx, y, nb32);
}

extern "C" __global__ void baracuda_gguf_dequantize_q5_0_f32_kernel(
    const void * __restrict__ vx, float * __restrict__ y, const int k)
{
    dequantize_block_q5_0_tmpl<float>(vx, y, k);
}

extern "C" __global__ void baracuda_gguf_dequantize_q5_1_f32_kernel(
    const void * __restrict__ vx, float * __restrict__ y, const int k)
{
    dequantize_block_q5_1_tmpl<float>(vx, y, k);
}

extern "C" __global__ void baracuda_gguf_dequantize_q8_0_f32_kernel(
    const void * __restrict__ vx, float * __restrict__ y, const int nb32)
{
    dequantize_block_q8_0_tmpl<float>(vx, y, nb32);
}

extern "C" __global__ void baracuda_gguf_dequantize_q2_K_f32_kernel(
    const void * __restrict__ vx, float * __restrict__ y)
{
    dequantize_block_q2_K_tmpl<float>(vx, y);
}

extern "C" __global__ void baracuda_gguf_dequantize_q3_K_f32_kernel(
    const void * __restrict__ vx, float * __restrict__ y)
{
    dequantize_block_q3_K_tmpl<float>(vx, y);
}

extern "C" __global__ void baracuda_gguf_dequantize_q4_K_f32_kernel(
    const void * __restrict__ vx, float * __restrict__ y)
{
    dequantize_block_q4_K_tmpl<float>(vx, y);
}

extern "C" __global__ void baracuda_gguf_dequantize_q5_K_f32_kernel(
    const void * __restrict__ vx, float * __restrict__ y)
{
    dequantize_block_q5_K_tmpl<float>(vx, y);
}

extern "C" __global__ void baracuda_gguf_dequantize_q6_K_f32_kernel(
    const void * __restrict__ vx, float * __restrict__ y)
{
    dequantize_block_q6_K_tmpl<float>(vx, y);
}

extern "C" __global__ void baracuda_gguf_dequantize_q8_K_f32_kernel(
    const void * __restrict__ vx, float * __restrict__ y)
{
    dequantize_block_q8_K_tmpl<float>(vx, y);
}

// =============================================================================
// Launcher helpers. Same grid math as Fuel's dequantize_f32() —
// nb := ceil(numel/256) is the number of 256-element output blocks.
//
// type-0/1 (32-elem block) launches: blockDim = (32, 1, 1).
//   nb32 = numel / 32        (q4_0, q4_1, q8_0)
//   nb32 = numel             (q5_0, q5_1; the dequantize_block template
//                             processes 2 outputs per thread × 256 threads
//                             with block_dim = CUDA_DEQUANTIZE_BLOCK_SIZE).
//   For q5_0 / q5_1 we use blockDim = (256, 1, 1) and grid
//   = ceil_div(numel, 2 * 256). Mirrors Fuel layout.
//
// k-quants (256-elem block) launches: blockDim varies per qtype
//   (Q2_K, Q3_K, Q5_K, Q6_K → 64; Q4_K → 32; Q8_K → 32),
//   grid = nb (one block per output super-block).
// =============================================================================

namespace {

inline int ceil_div_host(int p, int q) {
    return (p + q - 1) / q;
}

inline int32_t status_from_launch(cudaError_t err) {
    if (err != cudaSuccess) return 5;
    return 0;
}

} // anonymous namespace

// ---- Type-0/1 launchers ---------------------------------------------------

extern "C" int32_t baracuda_kernels_dequantize_q4_0_run(
    int64_t numel,
    const void * __restrict__ x,
    void * __restrict__ y,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!x || !y) return 2;
    if (numel <= 0 || (numel % 32) != 0) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    const int nb = ceil_div_host((int)numel, 256);
    const int nb32 = (int)(numel / 32);
    baracuda_gguf_dequantize_q4_0_f32_kernel<<<dim3(nb), dim3(32), 0, stream>>>(
        x, static_cast<float*>(y), nb32);
    return status_from_launch(cudaPeekAtLastError());
}

extern "C" int32_t baracuda_kernels_dequantize_q4_1_run(
    int64_t numel,
    const void * __restrict__ x,
    void * __restrict__ y,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!x || !y) return 2;
    if (numel <= 0 || (numel % 32) != 0) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    const int nb = ceil_div_host((int)numel, 256);
    const int nb32 = (int)(numel / 32);
    baracuda_gguf_dequantize_q4_1_f32_kernel<<<dim3(nb), dim3(32), 0, stream>>>(
        x, static_cast<float*>(y), nb32);
    return status_from_launch(cudaPeekAtLastError());
}

extern "C" int32_t baracuda_kernels_dequantize_q5_0_run(
    int64_t numel,
    const void * __restrict__ x,
    void * __restrict__ y,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!x || !y) return 2;
    if (numel <= 0 || (numel % 32) != 0) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    const int num_blocks = ceil_div_host((int)numel, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE);
    baracuda_gguf_dequantize_q5_0_f32_kernel<<<dim3(num_blocks), dim3(CUDA_DEQUANTIZE_BLOCK_SIZE), 0, stream>>>(
        x, static_cast<float*>(y), (int)numel);
    return status_from_launch(cudaPeekAtLastError());
}

extern "C" int32_t baracuda_kernels_dequantize_q5_1_run(
    int64_t numel,
    const void * __restrict__ x,
    void * __restrict__ y,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!x || !y) return 2;
    if (numel <= 0 || (numel % 32) != 0) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    const int num_blocks = ceil_div_host((int)numel, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE);
    baracuda_gguf_dequantize_q5_1_f32_kernel<<<dim3(num_blocks), dim3(CUDA_DEQUANTIZE_BLOCK_SIZE), 0, stream>>>(
        x, static_cast<float*>(y), (int)numel);
    return status_from_launch(cudaPeekAtLastError());
}

extern "C" int32_t baracuda_kernels_dequantize_q8_0_run(
    int64_t numel,
    const void * __restrict__ x,
    void * __restrict__ y,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!x || !y) return 2;
    if (numel <= 0 || (numel % 32) != 0) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    const int nb = ceil_div_host((int)numel, 256);
    const int nb32 = (int)(numel / 32);
    baracuda_gguf_dequantize_q8_0_f32_kernel<<<dim3(nb), dim3(32), 0, stream>>>(
        x, static_cast<float*>(y), nb32);
    return status_from_launch(cudaPeekAtLastError());
}

// ---- k-quants launchers ---------------------------------------------------

extern "C" int32_t baracuda_kernels_dequantize_q2_K_run(
    int64_t numel,
    const void * __restrict__ x,
    void * __restrict__ y,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!x || !y) return 2;
    if (numel <= 0 || (numel % QK_K) != 0) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    const int nb = (int)(numel / QK_K);
    baracuda_gguf_dequantize_q2_K_f32_kernel<<<dim3(nb), dim3(64), 0, stream>>>(
        x, static_cast<float*>(y));
    return status_from_launch(cudaPeekAtLastError());
}

extern "C" int32_t baracuda_kernels_dequantize_q3_K_run(
    int64_t numel,
    const void * __restrict__ x,
    void * __restrict__ y,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!x || !y) return 2;
    if (numel <= 0 || (numel % QK_K) != 0) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    const int nb = (int)(numel / QK_K);
    baracuda_gguf_dequantize_q3_K_f32_kernel<<<dim3(nb), dim3(64), 0, stream>>>(
        x, static_cast<float*>(y));
    return status_from_launch(cudaPeekAtLastError());
}

extern "C" int32_t baracuda_kernels_dequantize_q4_K_run(
    int64_t numel,
    const void * __restrict__ x,
    void * __restrict__ y,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!x || !y) return 2;
    if (numel <= 0 || (numel % QK_K) != 0) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    const int nb = (int)(numel / QK_K);
    baracuda_gguf_dequantize_q4_K_f32_kernel<<<dim3(nb), dim3(32), 0, stream>>>(
        x, static_cast<float*>(y));
    return status_from_launch(cudaPeekAtLastError());
}

extern "C" int32_t baracuda_kernels_dequantize_q5_K_run(
    int64_t numel,
    const void * __restrict__ x,
    void * __restrict__ y,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!x || !y) return 2;
    if (numel <= 0 || (numel % QK_K) != 0) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    const int nb = (int)(numel / QK_K);
    baracuda_gguf_dequantize_q5_K_f32_kernel<<<dim3(nb), dim3(64), 0, stream>>>(
        x, static_cast<float*>(y));
    return status_from_launch(cudaPeekAtLastError());
}

extern "C" int32_t baracuda_kernels_dequantize_q6_K_run(
    int64_t numel,
    const void * __restrict__ x,
    void * __restrict__ y,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!x || !y) return 2;
    if (numel <= 0 || (numel % QK_K) != 0) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    const int nb = (int)(numel / QK_K);
    baracuda_gguf_dequantize_q6_K_f32_kernel<<<dim3(nb), dim3(64), 0, stream>>>(
        x, static_cast<float*>(y));
    return status_from_launch(cudaPeekAtLastError());
}

extern "C" int32_t baracuda_kernels_dequantize_q8_K_run(
    int64_t numel,
    const void * __restrict__ x,
    void * __restrict__ y,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!x || !y) return 2;
    if (numel <= 0 || (numel % QK_K) != 0) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    const int nb = (int)(numel / QK_K);
    baracuda_gguf_dequantize_q8_K_f32_kernel<<<dim3(nb), dim3(32), 0, stream>>>(
        x, static_cast<float*>(y));
    return status_from_launch(cudaPeekAtLastError());
}

// =============================================================================
// _can_implement companions — host-side dispatch validation.
// =============================================================================
//
// Type-0/1 formats (Q4_0/1, Q5_0/1, Q8_0) accept numel % 32 == 0.
// k-quants (Q2..Q6, Q8_K) accept numel % QK_K == 0.

#define BARACUDA_KERNELS_DEQUANT_TYPE01_CAN_IMPL(NAME)                          \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                 \
        int64_t numel,                                                          \
        const void * /*x*/,                                                     \
        const void * /*y*/)                                                     \
    {                                                                           \
        if (numel <= 0 || (numel % 32) != 0) return 2;                          \
        return 0;                                                               \
    }

#define BARACUDA_KERNELS_DEQUANT_KQUANT_CAN_IMPL(NAME)                          \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                 \
        int64_t numel,                                                          \
        const void * /*x*/,                                                     \
        const void * /*y*/)                                                     \
    {                                                                           \
        if (numel <= 0 || (numel % QK_K) != 0) return 2;                        \
        return 0;                                                               \
    }

BARACUDA_KERNELS_DEQUANT_TYPE01_CAN_IMPL(dequantize_q4_0)
BARACUDA_KERNELS_DEQUANT_TYPE01_CAN_IMPL(dequantize_q4_1)
BARACUDA_KERNELS_DEQUANT_TYPE01_CAN_IMPL(dequantize_q5_0)
BARACUDA_KERNELS_DEQUANT_TYPE01_CAN_IMPL(dequantize_q5_1)
BARACUDA_KERNELS_DEQUANT_TYPE01_CAN_IMPL(dequantize_q8_0)
BARACUDA_KERNELS_DEQUANT_KQUANT_CAN_IMPL(dequantize_q2_K)
BARACUDA_KERNELS_DEQUANT_KQUANT_CAN_IMPL(dequantize_q3_K)
BARACUDA_KERNELS_DEQUANT_KQUANT_CAN_IMPL(dequantize_q4_K)
BARACUDA_KERNELS_DEQUANT_KQUANT_CAN_IMPL(dequantize_q5_K)
BARACUDA_KERNELS_DEQUANT_KQUANT_CAN_IMPL(dequantize_q6_K)
BARACUDA_KERNELS_DEQUANT_KQUANT_CAN_IMPL(dequantize_q8_K)
