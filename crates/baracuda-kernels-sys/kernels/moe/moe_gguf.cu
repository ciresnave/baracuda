// SPDX-FileCopyrightText: 2023-2024 The llama.cpp / ggml authors    (MIT)
// SPDX-FileCopyrightText: 2024-2026 attention.rs (guoqingbao)       (MIT OR Apache-2.0)
// SPDX-FileCopyrightText: 2024-2026 Fuel project contributors       (MIT OR Apache-2.0)
// SPDX-FileCopyrightText: 2026 baracuda project contributors        (MIT OR Apache-2.0)
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// baracuda-kernels Phase 8 Milestone 8.5 — MoE GGUF scalar launcher.
//
// Op shape (forward):
//   activations [T, D_model] (f32, dense)
//   expert weights [num_experts, D_expert, D_model] (GGUF-packed)
//   expert_indices [T, top_k] (i32) — already pre-sorted by expert
//   expert_weights [T, top_k] (f32) — per-token expert mixing weights
//   output [T, D_model] (f32)
//
// The scalar GGUF path stages activations through q8_1 (allocated +
// filled internally by this launcher), then dispatches one warp per
// (token, n) output cell to vec_dot the q8_1-staged activations
// against the GGUF-packed expert weights.
//
// Block-format coverage: Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K
// (matches Fuel's `moe_gemm_gguf` switch exactly).
//
// `gguf_dtype` discriminant matches Fuel (NOT baracuda's
// `GgufBlockFormat` repr):
//   0 = Q8_0, 1 = Q4_K, 2 = Q2_K, 3 = Q3_K, 4 = Q5_K, 5 = Q6_K.
// The Rust plan in `crates/baracuda-kernels/src/moe/mod.rs` maps
// `GgufBlockFormat -> i32` before calling the launcher.

#include "../include/baracuda_moe.cuh"

using namespace baracuda::moe;

namespace {

inline int ceil_div_host(int p, int q) { return (p + q - 1) / q; }
inline int pad_host(int s, int p)      { return p == 0 ? s : ((s + p - 1) / p) * p; }

inline int32_t status_from_launch(cudaError_t err) {
    return err != cudaSuccess ? 5 : 0;
}

constexpr int MATRIX_ROW_PADDING_HOST = 512;

// LAUNCH_MOE_GGUF expands the templated kernel invocation. Mirrors
// Fuel's macro but routes through baracuda's namespaced types.
#define LAUNCH_MOE_GGUF(qk, qi, block_q_t, vdr, vec_dot_fn)                                      \
    do {                                                                                         \
        const int shared_bytes = size_k / (qk) * (int)sizeof(block_q_t) * nWraps + 1024;         \
        baracuda::moe::device::moe_gemm_gguf_kernel<                                             \
            (qk), (qi), block_q_t, (vdr), vec_dot_fn>                                            \
            <<<grid_dim, block_dim, shared_bytes, stream>>>(                                     \
            weights, y_q8_1,                                                                     \
            sorted_token_ids, expert_ids, topk_weights,                                          \
            outputs,                                                                             \
            num_experts, topk,                                                                   \
            size_m, size_n, size_k,                                                              \
            kx_padded);                                                                          \
    } while (0)

} // anonymous namespace

extern "C" int32_t baracuda_kernels_moe_scalar_gguf_run(
    const float   * __restrict__ inputs,             // [size_m_input, size_k] f32, dense
    const void    * __restrict__ weights,            // [num_experts, size_n, size_k] GGUF-packed
    const int32_t * __restrict__ sorted_token_ids,   // [size_m] device
    const int32_t * __restrict__ expert_ids,         // [size_m] device
    const float   * __restrict__ topk_weights,       // [size_m] device or nullptr
    float         * __restrict__ outputs,            // [size_m_input, size_n] f32
    int32_t num_experts,
    int32_t topk,
    int32_t size_m,
    int32_t size_n,
    int32_t size_k,
    int32_t gguf_dtype,                              // 0=Q8_0, 1=Q4_K, 2=Q2_K, 3=Q3_K, 4=Q5_K, 5=Q6_K
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void * stream_ptr)
{
    if (!inputs || !weights || !sorted_token_ids || !expert_ids || !outputs) return 2;
    if (num_experts <= 0 || topk <= 0 || size_m <= 0 || size_n <= 0 || size_k <= 0) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    const int kx_padded = pad_host(size_k, MATRIX_ROW_PADDING_HOST);
    const int QUANTIZE_BLOCK_SIZE = 256;
    const int num_blocks = ceil_div_host(kx_padded, QUANTIZE_BLOCK_SIZE);
    const int m = topk_weights ? size_m : size_m / topk;

    dim3 grid_dim_quant((unsigned)num_blocks, (unsigned)m, 1);
    dim3 block_dim_quant((unsigned)QUANTIZE_BLOCK_SIZE, 1, 1);
    const size_t y_size_in_bytes =
        (size_t)m * ((size_t)kx_padded / 32 * sizeof(baracuda::moe::block_q8_1));
    void * y_q8_1 = nullptr;
    cudaError_t err = cudaMallocAsync(&y_q8_1, y_size_in_bytes, stream);
    if (err != cudaSuccess) return 5;
    baracuda::moe::moe_quantize_q8_1<<<grid_dim_quant, block_dim_quant, 0, stream>>>(
        inputs, y_q8_1, size_k, kx_padded);

    const int nWraps = 4;
    dim3 grid_dim((unsigned)ceil_div_host(size_n, nWraps), (unsigned)size_m, 1);
    dim3 block_dim((unsigned)32, (unsigned)nWraps, 1);  // (WARP_SIZE, nWraps, 1)

    switch (gguf_dtype) {
        case 0: { // Q8_0
            using bt = baracuda::moe::block_q8_0;
            LAUNCH_MOE_GGUF(32, MOE_QI8_0, bt, MOE_VDR_Q8_0_Q8_1_MMVQ, baracuda::moe::moe_vec_dot_q8_0_q8_1);
            break;
        }
        case 1: { // Q4_K
            using bt = baracuda::moe::block_q4_K;
            LAUNCH_MOE_GGUF(QK_K, (QK_K / 8), bt, MOE_VDR_Q4_K_Q8_1_MMVQ, baracuda::moe::moe_vec_dot_q4_K_q8_1);
            break;
        }
        case 2: { // Q2_K
            using bt = baracuda::moe::block_q2_K;
            LAUNCH_MOE_GGUF(QK_K, (QK_K / 16), bt, MOE_VDR_Q2_K_Q8_1_MMVQ, baracuda::moe::moe_vec_dot_q2_K_q8_1);
            break;
        }
        case 3: { // Q3_K
            using bt = baracuda::moe::block_q3_K;
            LAUNCH_MOE_GGUF(QK_K, (QK_K / 16), bt, MOE_VDR_Q3_K_Q8_1_MMVQ, baracuda::moe::moe_vec_dot_q3_K_q8_1);
            break;
        }
        case 4: { // Q5_K
            using bt = baracuda::moe::block_q5_K;
            LAUNCH_MOE_GGUF(QK_K, (QK_K / 8), bt, MOE_VDR_Q5_K_Q8_1_MMVQ, baracuda::moe::moe_vec_dot_q5_K_q8_1);
            break;
        }
        case 5: { // Q6_K
            using bt = baracuda::moe::block_q6_K;
            LAUNCH_MOE_GGUF(QK_K, (QK_K / 16), bt, MOE_VDR_Q6_K_Q8_1_MMVQ, baracuda::moe::moe_vec_dot_q6_K_q8_1);
            break;
        }
        default:
            cudaFreeAsync(y_q8_1, stream);
            return 2;
    }

    cudaFreeAsync(y_q8_1, stream);
    return status_from_launch(cudaPeekAtLastError());
}

extern "C" int32_t baracuda_kernels_moe_scalar_gguf_can_implement(
    const void * /*inputs*/,
    const void * /*weights*/,
    const int32_t * /*sorted_token_ids*/,
    const int32_t * /*expert_ids*/,
    const float   * /*topk_weights*/,
    const void * /*outputs*/,
    int32_t num_experts,
    int32_t topk,
    int32_t size_m,
    int32_t size_n,
    int32_t size_k,
    int32_t gguf_dtype)
{
    if (num_experts <= 0 || topk <= 0 || size_m <= 0 || size_n <= 0 || size_k <= 0) return 2;
    // Only the same six gguf_dtype values handled by _run are valid.
    if (gguf_dtype < 0 || gguf_dtype > 5) return 2;
    return 0;
}
