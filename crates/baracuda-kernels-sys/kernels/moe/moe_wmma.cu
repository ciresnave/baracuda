// SPDX-FileCopyrightText: 2024-2026 attention.rs (guoqingbao)       (MIT OR Apache-2.0)
// SPDX-FileCopyrightText: 2024-2026 Fuel project contributors       (MIT OR Apache-2.0)
// SPDX-FileCopyrightText: 2026 baracuda project contributors        (MIT OR Apache-2.0)
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// baracuda-kernels Phase 8 Milestone 8.5 — MoE WMMA launchers (FP weights).
//
// Op shape (forward, FP weights):
//   input  [size_m, size_k]                — f16 or bf16
//   weights [num_experts, size_n, size_k]  — f16 or bf16
//   output [size_m, size_n]                — f16 or bf16 (zero-init by caller)
//
// Two instantiations per dtype: a "prefill" variant (M_tile=16, N_tile=16,
// WARPS_N=2 — matches Fuel) and a "decode" variant (M_tile=8, N_tile=32,
// WARPS_N=1). The `is_prefill` flag selects between them.
//
// Expert offsets are computed internally via two device kernels:
//   1. `moe_count_tokens_per_expert_kernel` — atomicAdd histogram.
//   2. `moe_expert_prefix_sum_kernel` — single-block Hillis-Steele scan
//      (requires `num_experts <= 1024`).

#include "../include/baracuda_moe.cuh"

using namespace baracuda::moe;

namespace {

inline int ceil_div_host(int p, int q) { return (p + q - 1) / q; }

inline int32_t status_from_launch(cudaError_t err) {
    return err != cudaSuccess ? 5 : 0;
}

// In-kernel-only scan; reuses the prefix-sum kernel from the header.
// Caller passes pre-allocated `d_expert_counts[num_experts]` and
// `d_expert_offsets[num_experts + 1]` scratch buffers.
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
    // Round up to next power of 2.
    {
        int p = 1;
        while (p < scan_threads) p <<= 1;
        scan_threads = p > 1024 ? 1024 : p;
    }
    size_t smem_size = (size_t)scan_threads * sizeof(int32_t);
    baracuda::moe::moe_expert_prefix_sum_kernel<<<1, scan_threads, smem_size, stream>>>(
        d_expert_counts, d_expert_offsets, num_experts);
}

#define LAUNCH_MOE_WMMA(DTYPE, WMMA_M_, WMMA_N_, WARPS_N_)                                   \
    baracuda::moe::device::moe_gemm_wmma_kernel<DTYPE, (WMMA_M_), (WMMA_N_), (WARPS_N_)>     \
        <<<grid, block, smem_bytes, stream>>>(                                               \
            reinterpret_cast<const DTYPE *>(input),                                          \
            reinterpret_cast<const DTYPE *>(weights),                                        \
            sorted_token_ids,                                                                \
            d_expert_offsets,                                                                \
            topk_weights,                                                                    \
            reinterpret_cast<DTYPE *>(output),                                               \
            num_experts, topk,                                                               \
            size_m, size_n, size_k)

template<typename DTYPE>
inline int32_t launch_moe_wmma(
    const void * input,
    const void * weights,
    const int32_t * sorted_token_ids,
    const int32_t * expert_ids,
    const float   * topk_weights,
    void          * output,
    int32_t * expert_counts,
    int32_t * d_expert_offsets,
    int num_experts, int topk,
    int size_m, int size_n, int size_k,
    bool is_prefill,
    cudaStream_t stream)
{
    using baracuda::moe::device::M_BLK;
    using baracuda::moe::device::N_BLK;
    using baracuda::moe::device::K_BLK;
    using baracuda::moe::device::BLOCK_THREADS;

    calculate_expert_offsets_inline(expert_ids, size_m, expert_counts,
                                    d_expert_offsets, num_experts, stream);

    int grid_n = ceil_div_host(size_n, N_BLK);
    dim3 grid((unsigned)num_experts, (unsigned)grid_n, 1);
    dim3 block((unsigned)BLOCK_THREADS, 1, 1);

    size_t A_sh_bytes = (size_t)M_BLK * K_BLK * sizeof(DTYPE);
    size_t B_sh_bytes = (size_t)N_BLK * K_BLK * sizeof(DTYPE);
    size_t C_sh_bytes = (size_t)M_BLK * N_BLK * sizeof(float);
    size_t AB_bytes   = A_sh_bytes + B_sh_bytes;
    size_t pad        = (16 - (AB_bytes % 16)) % 16;
    size_t smem_bytes = AB_bytes + pad + C_sh_bytes;

    if (is_prefill) {
        LAUNCH_MOE_WMMA(DTYPE, 16, 16, 2);
    } else {
        LAUNCH_MOE_WMMA(DTYPE, 8, 32, 1);
    }
    return status_from_launch(cudaPeekAtLastError());
}

} // anonymous namespace

extern "C" int32_t baracuda_kernels_moe_wmma_f16_run(
    const void    * input,                  // [size_m, size_k] f16
    const void    * weights,                // [num_experts, size_n, size_k] f16
    const int32_t * sorted_token_ids,
    const int32_t * expert_ids,
    const float   * topk_weights,           // device or nullptr
    void          * output,                 // [size_m, size_n] f16 (zero-init by caller)
    int32_t * expert_counts,                // prealloc [num_experts]
    int32_t * expert_offsets,               // prealloc [num_experts + 1]
    int32_t num_experts,
    int32_t topk,
    int32_t size_m,
    int32_t size_n,
    int32_t size_k,
    int32_t is_prefill,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!input || !weights || !sorted_token_ids || !expert_ids || !output ||
        !expert_counts || !expert_offsets) return 2;
    if (num_experts <= 0 || topk <= 0 || size_m <= 0 || size_n <= 0 || size_k <= 0) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    return launch_moe_wmma<half>(input, weights, sorted_token_ids, expert_ids,
                                 topk_weights, output, expert_counts, expert_offsets,
                                 num_experts, topk, size_m, size_n, size_k,
                                 is_prefill != 0, stream);
}

extern "C" int32_t baracuda_kernels_moe_wmma_bf16_run(
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
    int32_t is_prefill,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!input || !weights || !sorted_token_ids || !expert_ids || !output ||
        !expert_counts || !expert_offsets) return 2;
    if (num_experts <= 0 || topk <= 0 || size_m <= 0 || size_n <= 0 || size_k <= 0) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    return launch_moe_wmma<nv_bfloat16>(input, weights, sorted_token_ids, expert_ids,
                                        topk_weights, output, expert_counts, expert_offsets,
                                        num_experts, topk, size_m, size_n, size_k,
                                        is_prefill != 0, stream);
}

extern "C" int32_t baracuda_kernels_moe_wmma_f16_can_implement(
    int32_t num_experts, int32_t topk,
    int32_t size_m, int32_t size_n, int32_t size_k,
    int32_t /*is_prefill*/)
{
    if (num_experts <= 0 || topk <= 0) return 2;
    if (size_m <= 0 || size_n <= 0 || size_k <= 0) return 2;
    if (topk > num_experts) return 2;
    if (num_experts > 1024) return 3;  // single-block prefix-sum cap
    return 0;
}

extern "C" int32_t baracuda_kernels_moe_wmma_bf16_can_implement(
    int32_t num_experts, int32_t topk,
    int32_t size_m, int32_t size_n, int32_t size_k,
    int32_t /*is_prefill*/)
{
    if (num_experts <= 0 || topk <= 0) return 2;
    if (size_m <= 0 || size_n <= 0 || size_k <= 0) return 2;
    if (topk > num_experts) return 2;
    if (num_experts > 1024) return 3;
    return 0;
}
