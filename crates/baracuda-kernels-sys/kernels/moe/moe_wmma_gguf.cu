// SPDX-FileCopyrightText: 2023-2024 The llama.cpp / ggml authors    (MIT)
// SPDX-FileCopyrightText: 2024-2026 attention.rs (guoqingbao)       (MIT OR Apache-2.0)
// SPDX-FileCopyrightText: 2024-2026 Fuel project contributors       (MIT OR Apache-2.0)
// SPDX-FileCopyrightText: 2026 baracuda project contributors        (MIT OR Apache-2.0)
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// baracuda-kernels Phase 8 Milestone 8.5 — MoE WMMA + GGUF launchers.
//
// The combined hot path for quantized LLM inference. WMMA tensor cores
// for the matmul accumulation; per-N-row GGUF block dequant into
// shared memory for the weight tile.
//
// Op shape (forward):
//   input  [size_m, size_k] — f16 or bf16 (dense)
//   weights [num_experts, size_n, size_k]   — GGUF-packed (uint8_t bytes)
//   output [size_m, size_n] — f32 (zero-init not required; written directly)
//
// Block-format coverage: Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K (Fuel parity).
// `gguf_dtype` ids match Fuel: 0=Q8_0, 1=Q4_K, 2=Q2_K, 3=Q3_K, 4=Q5_K, 5=Q6_K.

#include "../include/baracuda_moe.cuh"

using namespace baracuda::moe;

namespace {

inline int ceil_div_host(int p, int q) { return (p + q - 1) / q; }

inline int32_t status_from_launch(cudaError_t err) {
    return err != cudaSuccess ? 5 : 0;
}

// Reuse the same expert-offset helper as moe_wmma.cu (private to the
// TU — both files would otherwise share a static-linkage helper).
inline void calculate_expert_offsets_inline(
    const int32_t * d_expert_ids,
    int size_m,
    int32_t * d_expert_counts,
    int32_t * d_expert_offsets,
    int num_experts,
    cudaStream_t stream)
{
    cudaMemsetAsync(d_expert_counts, 0, num_experts * sizeof(int32_t), stream);
    int threads = 256;
    int blocks  = ceil_div_host(size_m, threads);
    baracuda::moe::moe_count_tokens_per_expert_kernel<<<blocks, threads, 0, stream>>>(
        d_expert_ids, d_expert_counts, size_m);

    int scan_threads = num_experts;
    if (scan_threads < 32) scan_threads = 32;
    {
        int p = 1;
        while (p < scan_threads) p <<= 1;
        scan_threads = p > 1024 ? 1024 : p;
    }
    size_t smem_size = (size_t)scan_threads * sizeof(int32_t);
    baracuda::moe::moe_expert_prefix_sum_kernel<<<1, scan_threads, smem_size, stream>>>(
        d_expert_counts, d_expert_offsets, num_experts);
}

#define LAUNCH_MOE_WMMA_GGUF_PREFILL(DTYPE, gguf_type)                                            \
    do {                                                                                          \
        using namespace baracuda::moe;                                                            \
        using namespace baracuda::moe::device;                                                    \
        if ((gguf_type) == 0) {                                                                   \
            dim3 block(32, WARPS_PER_BLOCK_GGUF, 1);                                              \
            moe_gemm_wmma_gguf_prefill_kernel<DTYPE, MOE_QK8_0, block_q8_0, 32>                   \
                <<<grid, block, smem_bytes, stream>>>(                                            \
                reinterpret_cast<const DTYPE *>(input),                                           \
                reinterpret_cast<const uint8_t *>(weights),                                       \
                sorted_token_ids, d_expert_offsets, topk_weights,                                 \
                output, num_experts, topk, size_m, size_n, size_k, (gguf_type));                  \
        } else if ((gguf_type) == 1) {                                                            \
            dim3 block(32, WARPS_PER_BLOCK_GGUF, 1);                                              \
            moe_gemm_wmma_gguf_prefill_kernel<DTYPE, QK_K, block_q4_K, 32>                        \
                <<<grid, block, smem_bytes, stream>>>(                                            \
                reinterpret_cast<const DTYPE *>(input),                                           \
                reinterpret_cast<const uint8_t *>(weights),                                       \
                sorted_token_ids, d_expert_offsets, topk_weights,                                 \
                output, num_experts, topk, size_m, size_n, size_k, (gguf_type));                  \
        } else if ((gguf_type) == 2) {                                                            \
            dim3 block(64, WARPS_PER_BLOCK_GGUF, 1);                                              \
            moe_gemm_wmma_gguf_prefill_kernel<DTYPE, QK_K, block_q2_K, 64>                        \
                <<<grid, block, smem_bytes, stream>>>(                                            \
                reinterpret_cast<const DTYPE *>(input),                                           \
                reinterpret_cast<const uint8_t *>(weights),                                       \
                sorted_token_ids, d_expert_offsets, topk_weights,                                 \
                output, num_experts, topk, size_m, size_n, size_k, (gguf_type));                  \
        } else if ((gguf_type) == 3) {                                                            \
            dim3 block(64, WARPS_PER_BLOCK_GGUF, 1);                                              \
            moe_gemm_wmma_gguf_prefill_kernel<DTYPE, QK_K, block_q3_K, 64>                        \
                <<<grid, block, smem_bytes, stream>>>(                                            \
                reinterpret_cast<const DTYPE *>(input),                                           \
                reinterpret_cast<const uint8_t *>(weights),                                       \
                sorted_token_ids, d_expert_offsets, topk_weights,                                 \
                output, num_experts, topk, size_m, size_n, size_k, (gguf_type));                  \
        } else if ((gguf_type) == 4) {                                                            \
            dim3 block(64, WARPS_PER_BLOCK_GGUF, 1);                                              \
            moe_gemm_wmma_gguf_prefill_kernel<DTYPE, QK_K, block_q5_K, 64>                        \
                <<<grid, block, smem_bytes, stream>>>(                                            \
                reinterpret_cast<const DTYPE *>(input),                                           \
                reinterpret_cast<const uint8_t *>(weights),                                       \
                sorted_token_ids, d_expert_offsets, topk_weights,                                 \
                output, num_experts, topk, size_m, size_n, size_k, (gguf_type));                  \
        } else if ((gguf_type) == 5) {                                                            \
            dim3 block(64, WARPS_PER_BLOCK_GGUF, 1);                                              \
            moe_gemm_wmma_gguf_prefill_kernel<DTYPE, QK_K, block_q6_K, 64>                        \
                <<<grid, block, smem_bytes, stream>>>(                                            \
                reinterpret_cast<const DTYPE *>(input),                                           \
                reinterpret_cast<const uint8_t *>(weights),                                       \
                sorted_token_ids, d_expert_offsets, topk_weights,                                 \
                output, num_experts, topk, size_m, size_n, size_k, (gguf_type));                  \
        }                                                                                         \
    } while (0)

template<typename DTYPE>
inline int32_t launch_moe_wmma_gguf(
    const void    * input,
    const uint8_t * weights,
    const int32_t * sorted_token_ids,
    const int32_t * expert_ids,
    const float   * topk_weights,
    float         * output,
    int32_t * expert_counts,
    int32_t * d_expert_offsets,
    int num_experts, int topk,
    int size_m, int size_n, int size_k,
    int gguf_dtype,
    cudaStream_t stream)
{
    using baracuda::moe::device::N_BLK_GGUF;

    calculate_expert_offsets_inline(expert_ids, size_m, expert_counts,
                                    d_expert_offsets, num_experts, stream);

    int grid_n = ceil_div_host(size_n, N_BLK_GGUF);
    dim3 grid((unsigned)num_experts, (unsigned)grid_n, 1);

    size_t qk_runtime              = QK_K;
    size_t block_size_bytes_rt     = sizeof(baracuda::moe::block_q6_K);
    if (gguf_dtype == 0) {
        block_size_bytes_rt = sizeof(baracuda::moe::block_q8_0);
        qk_runtime          = 32;
    } else if (gguf_dtype == 1) {
        block_size_bytes_rt = sizeof(baracuda::moe::block_q4_K);
    } else if (gguf_dtype == 2) {
        block_size_bytes_rt = sizeof(baracuda::moe::block_q2_K);
    } else if (gguf_dtype == 3) {
        block_size_bytes_rt = sizeof(baracuda::moe::block_q3_K);
    } else if (gguf_dtype == 4) {
        block_size_bytes_rt = sizeof(baracuda::moe::block_q5_K);
    } // 5 = q6_K (default)

    size_t A_sh_bytes       = (size_t)device::M_BLK_GGUF * qk_runtime * sizeof(DTYPE);
    size_t B_sh_bytes       = (size_t)device::N_BLK_GGUF * qk_runtime * sizeof(DTYPE);
    size_t B_quant_sh_bytes = (size_t)device::N_BLK_GGUF * block_size_bytes_rt;
    size_t C_sh_bytes       = (size_t)device::M_BLK_GGUF * device::N_BLK_GGUF * sizeof(float);

    size_t smem_bytes  = A_sh_bytes + B_sh_bytes + B_quant_sh_bytes;
    size_t C_sh_offset = smem_bytes % alignof(float);
    if (C_sh_offset != 0) smem_bytes += (alignof(float) - C_sh_offset);
    smem_bytes += C_sh_bytes;

    LAUNCH_MOE_WMMA_GGUF_PREFILL(DTYPE, gguf_dtype);
    return status_from_launch(cudaPeekAtLastError());
}

} // anonymous namespace

extern "C" int32_t baracuda_kernels_moe_wmma_gguf_f16_run(
    const void    * input,
    const void    * weights,                          // packed bytes
    const int32_t * sorted_token_ids,
    const int32_t * expert_ids,
    const float   * topk_weights,                     // device or nullptr
    void          * output,                           // f32
    int32_t * expert_counts,                          // prealloc [num_experts]
    int32_t * expert_offsets,                         // prealloc [num_experts + 1]
    int32_t num_experts,
    int32_t topk,
    int32_t size_m,
    int32_t size_n,
    int32_t size_k,
    int32_t gguf_dtype,                               // 0=Q8_0, 1=Q4_K, 2=Q2_K, 3=Q3_K, 4=Q5_K, 5=Q6_K
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!input || !weights || !sorted_token_ids || !expert_ids || !output ||
        !expert_counts || !expert_offsets) return 2;
    if (num_experts <= 0 || topk <= 0 || size_m <= 0 || size_n <= 0 || size_k <= 0) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    return launch_moe_wmma_gguf<half>(
        input, static_cast<const uint8_t *>(weights),
        sorted_token_ids, expert_ids, topk_weights,
        static_cast<float *>(output), expert_counts, expert_offsets,
        num_experts, topk, size_m, size_n, size_k, gguf_dtype, stream);
}

extern "C" int32_t baracuda_kernels_moe_wmma_gguf_bf16_run(
    const void    * input,
    const void    * weights,
    const int32_t * sorted_token_ids,
    const int32_t * expert_ids,
    const float   * topk_weights,
    void          * output,
    int32_t * expert_counts,
    int32_t * expert_offsets,
    int32_t num_experts,
    int32_t topk,
    int32_t size_m,
    int32_t size_n,
    int32_t size_k,
    int32_t gguf_dtype,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!input || !weights || !sorted_token_ids || !expert_ids || !output ||
        !expert_counts || !expert_offsets) return 2;
    if (num_experts <= 0 || topk <= 0 || size_m <= 0 || size_n <= 0 || size_k <= 0) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    return launch_moe_wmma_gguf<nv_bfloat16>(
        input, static_cast<const uint8_t *>(weights),
        sorted_token_ids, expert_ids, topk_weights,
        static_cast<float *>(output), expert_counts, expert_offsets,
        num_experts, topk, size_m, size_n, size_k, gguf_dtype, stream);
}
