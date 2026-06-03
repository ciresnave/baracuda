// SPDX-FileCopyrightText: 2023-2024 The llama.cpp / ggml authors  (MIT)
// SPDX-FileCopyrightText: 2024-2026 Fuel project contributors      (MIT OR Apache-2.0)
// SPDX-FileCopyrightText: 2026 baracuda project contributors       (MIT OR Apache-2.0)
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// baracuda-kernels Phase 20.1 — batched MMVQ × N-experts launchers.
//
// See `kernels/include/baracuda_mmvq_batched.cuh` for the kernel templates
// and the routing-array consumption model. This .cu file:
//   1. Instantiates `__global__` kernels per (block format, activation
//      dtype, NEED_ATOMIC) triple.
//   2. Exposes one `extern "C" int32_t baracuda_kernels_mmvq_<fmt>_batched
//      _<adtype>_run(...)` launcher per public surface, which:
//        a. validates problem dims,
//        b. fires the small `compute_dispatch_to_expert_kernel` prelude
//           into the caller-provided workspace,
//        c. fires the appropriate batched MMVQ kernel based on the
//           `top_k` parameter (`top_k == 1` → NEED_ATOMIC=false fast
//           path; `top_k > 1` → NEED_ATOMIC=true scatter-add).
//
// FFI symbol count: 33 (11 block formats × 3 act dtypes) + 3 (pure-FP)
// = 36.
//
// Workspace contract: caller passes `workspace` + `workspace_bytes`.
// We need `m_total * sizeof(i32)` bytes for the `dispatch_to_expert`
// buffer. `m_total` is derived from `workspace_bytes / sizeof(i32)`
// — the Rust plan layer sizes the workspace to exactly that. Status 4
// is returned when workspace_bytes is too small (a defensive bound).

#include "../include/baracuda_mmvq_batched.cuh"

#include <cstdint>
#include <cuda_runtime.h>

using namespace baracuda::mmvq_batched;

// =============================================================================
// `__global__` kernel instantiations.
// Two NEED_ATOMIC variants per (block-format, act-dtype) triple →
// 5 type-0/1 × 3 dtypes × 2 = 30
// 6 k-quant    × 3 dtypes × 2 = 36
// 3 pure-FP             × 2 =  6
// Plus the dispatch_to_expert prelude (already a `__global__` in the header).
// =============================================================================

// ---- Type-0/1 (32-element blocks) ----------------------------------------

#define BCDA_BATCHED_T01_KERNEL(qtype, qk, qr, deq_fn, act_t, suffix, na, na_suffix) \
extern "C" __global__ void baracuda_mmvq_batched_##qtype##_##suffix##_##na_suffix##_kernel( \
    const void * __restrict__ weights, const act_t * __restrict__ activations, \
    const int32_t * __restrict__ sorted_token_ids, \
    const int32_t * __restrict__ dispatch_to_expert, \
    const float * __restrict__ topk_weights, \
    act_t * __restrict__ output, int n_rows_per_expert, int ncols) \
{ \
    mmvq_batched_type01_tmpl<qk, qr, deq_fn, act_t, act_t, na>( \
        weights, activations, sorted_token_ids, dispatch_to_expert, \
        topk_weights, output, n_rows_per_expert, ncols); \
}

#define BCDA_BATCHED_T01_FANOUT(qtype, qk, qr, deq_fn) \
    BCDA_BATCHED_T01_KERNEL(qtype, qk, qr, deq_fn, float,         f32,  false, store) \
    BCDA_BATCHED_T01_KERNEL(qtype, qk, qr, deq_fn, float,         f32,  true,  atomic) \
    BCDA_BATCHED_T01_KERNEL(qtype, qk, qr, deq_fn, __half,        f16,  false, store) \
    BCDA_BATCHED_T01_KERNEL(qtype, qk, qr, deq_fn, __half,        f16,  true,  atomic) \
    BCDA_BATCHED_T01_KERNEL(qtype, qk, qr, deq_fn, __nv_bfloat16, bf16, false, store) \
    BCDA_BATCHED_T01_KERNEL(qtype, qk, qr, deq_fn, __nv_bfloat16, bf16, true,  atomic)

BCDA_BATCHED_T01_FANOUT(q4_0, QK4_0, QR4_0, dequantize_q4_0)
BCDA_BATCHED_T01_FANOUT(q4_1, QK4_1, QR4_1, dequantize_q4_1)
BCDA_BATCHED_T01_FANOUT(q5_0, QK5_0, QR5_0, dequantize_q5_0)
BCDA_BATCHED_T01_FANOUT(q5_1, QK5_1, QR5_1, dequantize_q5_1)
BCDA_BATCHED_T01_FANOUT(q8_0, QK8_0, QR8_0, dequantize_q8_0)

#undef BCDA_BATCHED_T01_KERNEL
#undef BCDA_BATCHED_T01_FANOUT

// ---- k-quants (256-element blocks) ---------------------------------------

#define BCDA_BATCHED_KQUANT_KERNEL(qtype, act_t, suffix, na, na_suffix) \
extern "C" __global__ void baracuda_mmvq_batched_##qtype##_##suffix##_##na_suffix##_kernel( \
    const void * __restrict__ weights, const act_t * __restrict__ activations, \
    const int32_t * __restrict__ sorted_token_ids, \
    const int32_t * __restrict__ dispatch_to_expert, \
    const float * __restrict__ topk_weights, \
    act_t * __restrict__ output, int n_rows_per_expert, int ncols) \
{ \
    mmvq_batched_##qtype##_tmpl<act_t, act_t, na>( \
        weights, activations, sorted_token_ids, dispatch_to_expert, \
        topk_weights, output, n_rows_per_expert, ncols); \
}

#define BCDA_BATCHED_KQUANT_FANOUT(qtype) \
    BCDA_BATCHED_KQUANT_KERNEL(qtype, float,         f32,  false, store) \
    BCDA_BATCHED_KQUANT_KERNEL(qtype, float,         f32,  true,  atomic) \
    BCDA_BATCHED_KQUANT_KERNEL(qtype, __half,        f16,  false, store) \
    BCDA_BATCHED_KQUANT_KERNEL(qtype, __half,        f16,  true,  atomic) \
    BCDA_BATCHED_KQUANT_KERNEL(qtype, __nv_bfloat16, bf16, false, store) \
    BCDA_BATCHED_KQUANT_KERNEL(qtype, __nv_bfloat16, bf16, true,  atomic)

BCDA_BATCHED_KQUANT_FANOUT(q2_K)
BCDA_BATCHED_KQUANT_FANOUT(q3_K)
BCDA_BATCHED_KQUANT_FANOUT(q4_K)
BCDA_BATCHED_KQUANT_FANOUT(q5_K)
BCDA_BATCHED_KQUANT_FANOUT(q6_K)
BCDA_BATCHED_KQUANT_FANOUT(q8_K)

#undef BCDA_BATCHED_KQUANT_KERNEL
#undef BCDA_BATCHED_KQUANT_FANOUT

// ---- Pure FP -------------------------------------------------------------

#define BCDA_BATCHED_FP_KERNEL(act_t, suffix, na, na_suffix) \
extern "C" __global__ void baracuda_mmvq_batched_fp_##suffix##_##na_suffix##_kernel( \
    const act_t * __restrict__ weights, const act_t * __restrict__ activations, \
    const int32_t * __restrict__ sorted_token_ids, \
    const int32_t * __restrict__ dispatch_to_expert, \
    const float * __restrict__ topk_weights, \
    act_t * __restrict__ output, int n_rows_per_expert, int ncols) \
{ \
    mmvq_batched_fp_tmpl<act_t, na>( \
        weights, activations, sorted_token_ids, dispatch_to_expert, \
        topk_weights, output, n_rows_per_expert, ncols); \
}

BCDA_BATCHED_FP_KERNEL(float,         f32,  false, store)
BCDA_BATCHED_FP_KERNEL(float,         f32,  true,  atomic)
BCDA_BATCHED_FP_KERNEL(__half,        f16,  false, store)
BCDA_BATCHED_FP_KERNEL(__half,        f16,  true,  atomic)
BCDA_BATCHED_FP_KERNEL(__nv_bfloat16, bf16, false, store)
BCDA_BATCHED_FP_KERNEL(__nv_bfloat16, bf16, true,  atomic)

#undef BCDA_BATCHED_FP_KERNEL

// =============================================================================
// Launcher helpers.
// =============================================================================

namespace {

inline int ceil_div_host(int p, int q) { return (p + q - 1) / q; }

inline int32_t status_from_launch(cudaError_t err) {
    return (err != cudaSuccess) ? 5 : 0;
}

inline int32_t launch_dispatch_to_expert(
    const int32_t * expert_offsets, int32_t n_experts, int32_t m_total,
    int32_t * dispatch_to_expert, cudaStream_t stream)
{
    if (m_total <= 0) return 0;
    const int block = 256;
    const int grid = ceil_div_host(m_total, block);
    compute_dispatch_to_expert_kernel<<<grid, block, 0, stream>>>(
        expert_offsets, n_experts, m_total, dispatch_to_expert);
    return status_from_launch(cudaPeekAtLastError());
}

inline bool validate_problem(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * weights, const void * activations,
    const int32_t * sorted_token_ids, const int32_t * expert_offsets,
    void * output)
{
    if (n_experts <= 0 || n_rows_per_expert <= 0 || n_cols <= 0) return false;
    if (!weights || !activations || !sorted_token_ids || !expert_offsets ||
        !output) return false;
    return true;
}

inline int32_t derive_m_total(size_t workspace_bytes) {
    if (workspace_bytes < sizeof(int32_t)) return 0;
    return (int32_t)(workspace_bytes / sizeof(int32_t));
}

} // anonymous namespace

// =============================================================================
// Launcher macros. Each invocation creates an `extern "C" int32_t
// baracuda_kernels_mmvq_<qtype>_batched_<dtype_suffix>_run(...)` symbol.
//
// `dtype_suffix` controls the FFI name + the kernel base name pick:
//   * f32   variant → un-suffixed FFI (baracuda_kernels_mmvq_<qtype>_batched_run)
//   * f16   variant → ..._batched_f16_run
//   * bf16  variant → ..._batched_bf16_run
// per project convention (matches Phase 18.1's f32 = un-suffixed pattern).
// =============================================================================

// f32 = un-suffixed FFI symbol; pick the `<base>_f32_store|atomic_kernel`.
#define BCDA_BATCHED_QUANT_RUN_F32(qtype) \
extern "C" int32_t baracuda_kernels_mmvq_##qtype##_batched_run( \
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols, \
    const void * weights, const void * activations, \
    const int32_t * sorted_token_ids, const int32_t * expert_offsets, \
    const float * topk_weights, void * output, int32_t top_k, \
    void * workspace, size_t workspace_bytes, void * stream_ptr) \
{ \
    if (!validate_problem(n_experts, n_rows_per_expert, n_cols, \
        weights, activations, sorted_token_ids, expert_offsets, output)) return 2; \
    const int32_t m_total = derive_m_total(workspace_bytes); \
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr); \
    if (m_total <= 0) return 0; \
    const size_t need = (size_t)m_total * sizeof(int32_t); \
    if (workspace == nullptr || workspace_bytes < need) return 4; \
    int32_t * d2e = static_cast<int32_t *>(workspace); \
    int32_t st = launch_dispatch_to_expert( \
        expert_offsets, n_experts, m_total, d2e, stream); \
    if (st != 0) return st; \
    dim3 grid((unsigned)n_rows_per_expert, (unsigned)m_total, 1); \
    dim3 block(WARP_SIZE, 1, 1); \
    if (top_k <= 1) { \
        baracuda_mmvq_batched_##qtype##_f32_store_kernel<<<grid, block, 0, stream>>>( \
            weights, static_cast<const float *>(activations), \
            sorted_token_ids, d2e, topk_weights, \
            static_cast<float *>(output), n_rows_per_expert, n_cols); \
    } else { \
        baracuda_mmvq_batched_##qtype##_f32_atomic_kernel<<<grid, block, 0, stream>>>( \
            weights, static_cast<const float *>(activations), \
            sorted_token_ids, d2e, topk_weights, \
            static_cast<float *>(output), n_rows_per_expert, n_cols); \
    } \
    return status_from_launch(cudaPeekAtLastError()); \
}

#define BCDA_BATCHED_QUANT_RUN_SUFFIX(qtype, act_t, suffix) \
extern "C" int32_t baracuda_kernels_mmvq_##qtype##_batched_##suffix##_run( \
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols, \
    const void * weights, const void * activations, \
    const int32_t * sorted_token_ids, const int32_t * expert_offsets, \
    const float * topk_weights, void * output, int32_t top_k, \
    void * workspace, size_t workspace_bytes, void * stream_ptr) \
{ \
    if (!validate_problem(n_experts, n_rows_per_expert, n_cols, \
        weights, activations, sorted_token_ids, expert_offsets, output)) return 2; \
    const int32_t m_total = derive_m_total(workspace_bytes); \
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr); \
    if (m_total <= 0) return 0; \
    const size_t need = (size_t)m_total * sizeof(int32_t); \
    if (workspace == nullptr || workspace_bytes < need) return 4; \
    int32_t * d2e = static_cast<int32_t *>(workspace); \
    int32_t st = launch_dispatch_to_expert( \
        expert_offsets, n_experts, m_total, d2e, stream); \
    if (st != 0) return st; \
    dim3 grid((unsigned)n_rows_per_expert, (unsigned)m_total, 1); \
    dim3 block(WARP_SIZE, 1, 1); \
    if (top_k <= 1) { \
        baracuda_mmvq_batched_##qtype##_##suffix##_store_kernel<<<grid, block, 0, stream>>>( \
            weights, static_cast<const act_t *>(activations), \
            sorted_token_ids, d2e, topk_weights, \
            static_cast<act_t *>(output), n_rows_per_expert, n_cols); \
    } else { \
        baracuda_mmvq_batched_##qtype##_##suffix##_atomic_kernel<<<grid, block, 0, stream>>>( \
            weights, static_cast<const act_t *>(activations), \
            sorted_token_ids, d2e, topk_weights, \
            static_cast<act_t *>(output), n_rows_per_expert, n_cols); \
    } \
    return status_from_launch(cudaPeekAtLastError()); \
}

#define BCDA_BATCHED_QUANT_FANOUT_RUN(qtype) \
    BCDA_BATCHED_QUANT_RUN_F32(qtype) \
    BCDA_BATCHED_QUANT_RUN_SUFFIX(qtype, __half,        f16) \
    BCDA_BATCHED_QUANT_RUN_SUFFIX(qtype, __nv_bfloat16, bf16)

BCDA_BATCHED_QUANT_FANOUT_RUN(q4_0)
BCDA_BATCHED_QUANT_FANOUT_RUN(q4_1)
BCDA_BATCHED_QUANT_FANOUT_RUN(q5_0)
BCDA_BATCHED_QUANT_FANOUT_RUN(q5_1)
BCDA_BATCHED_QUANT_FANOUT_RUN(q8_0)
BCDA_BATCHED_QUANT_FANOUT_RUN(q2_K)
BCDA_BATCHED_QUANT_FANOUT_RUN(q3_K)
BCDA_BATCHED_QUANT_FANOUT_RUN(q4_K)
BCDA_BATCHED_QUANT_FANOUT_RUN(q5_K)
BCDA_BATCHED_QUANT_FANOUT_RUN(q6_K)
BCDA_BATCHED_QUANT_FANOUT_RUN(q8_K)

#undef BCDA_BATCHED_QUANT_RUN_F32
#undef BCDA_BATCHED_QUANT_RUN_SUFFIX
#undef BCDA_BATCHED_QUANT_FANOUT_RUN

// ---- Pure-FP launchers ---------------------------------------------------

#define BCDA_BATCHED_FP_RUN(act_t, suffix) \
extern "C" int32_t baracuda_kernels_mmvq_batched_##suffix##_run( \
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols, \
    const void * weights, const void * activations, \
    const int32_t * sorted_token_ids, const int32_t * expert_offsets, \
    const float * topk_weights, void * output, int32_t top_k, \
    void * workspace, size_t workspace_bytes, void * stream_ptr) \
{ \
    if (!validate_problem(n_experts, n_rows_per_expert, n_cols, \
        weights, activations, sorted_token_ids, expert_offsets, output)) return 2; \
    const int32_t m_total = derive_m_total(workspace_bytes); \
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr); \
    if (m_total <= 0) return 0; \
    const size_t need = (size_t)m_total * sizeof(int32_t); \
    if (workspace == nullptr || workspace_bytes < need) return 4; \
    int32_t * d2e = static_cast<int32_t *>(workspace); \
    int32_t st = launch_dispatch_to_expert( \
        expert_offsets, n_experts, m_total, d2e, stream); \
    if (st != 0) return st; \
    dim3 grid((unsigned)n_rows_per_expert, (unsigned)m_total, 1); \
    dim3 block(WARP_SIZE, 1, 1); \
    if (top_k <= 1) { \
        baracuda_mmvq_batched_fp_##suffix##_store_kernel<<<grid, block, 0, stream>>>( \
            static_cast<const act_t *>(weights), \
            static_cast<const act_t *>(activations), \
            sorted_token_ids, d2e, topk_weights, \
            static_cast<act_t *>(output), n_rows_per_expert, n_cols); \
    } else { \
        baracuda_mmvq_batched_fp_##suffix##_atomic_kernel<<<grid, block, 0, stream>>>( \
            static_cast<const act_t *>(weights), \
            static_cast<const act_t *>(activations), \
            sorted_token_ids, d2e, topk_weights, \
            static_cast<act_t *>(output), n_rows_per_expert, n_cols); \
    } \
    return status_from_launch(cudaPeekAtLastError()); \
}

BCDA_BATCHED_FP_RUN(float,         f32)
BCDA_BATCHED_FP_RUN(__half,        f16)
BCDA_BATCHED_FP_RUN(__nv_bfloat16, bf16)

#undef BCDA_BATCHED_FP_RUN

// =============================================================================
// _can_implement companions -- host-side validators (Phase 66-prep).
// Mirror each _run signature minus workspace/stream; output pointers demoted
// to const void*. Returns 0 (ok) / 2 (invalid arg) / 3 (unsupported).
// =============================================================================

extern "C" int32_t baracuda_kernels_mmvq_batched_bf16_can_implement(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * /*weights*/, const void * /*activations*/,
    const int32_t * /*sorted_token_ids*/, const int32_t * /*expert_offsets*/,
    const float * /*topk_weights*/, const void * /*output*/, int32_t /*top_k*/)
{
    if (n_experts < 0 || n_rows_per_expert < 0 || n_cols < 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_batched_f16_can_implement(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * /*weights*/, const void * /*activations*/,
    const int32_t * /*sorted_token_ids*/, const int32_t * /*expert_offsets*/,
    const float * /*topk_weights*/, const void * /*output*/, int32_t /*top_k*/)
{
    if (n_experts < 0 || n_rows_per_expert < 0 || n_cols < 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_batched_f32_can_implement(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * /*weights*/, const void * /*activations*/,
    const int32_t * /*sorted_token_ids*/, const int32_t * /*expert_offsets*/,
    const float * /*topk_weights*/, const void * /*output*/, int32_t /*top_k*/)
{
    if (n_experts < 0 || n_rows_per_expert < 0 || n_cols < 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q2_K_batched_can_implement(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * /*weights*/, const void * /*activations*/,
    const int32_t * /*sorted_token_ids*/, const int32_t * /*expert_offsets*/,
    const float * /*topk_weights*/, const void * /*output*/, int32_t /*top_k*/)
{
    if (n_experts < 0 || n_rows_per_expert < 0 || n_cols < 0) return 2;
    if (n_cols > 0 && (n_cols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q2_K_batched_bf16_can_implement(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * /*weights*/, const void * /*activations*/,
    const int32_t * /*sorted_token_ids*/, const int32_t * /*expert_offsets*/,
    const float * /*topk_weights*/, const void * /*output*/, int32_t /*top_k*/)
{
    if (n_experts < 0 || n_rows_per_expert < 0 || n_cols < 0) return 2;
    if (n_cols > 0 && (n_cols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q2_K_batched_f16_can_implement(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * /*weights*/, const void * /*activations*/,
    const int32_t * /*sorted_token_ids*/, const int32_t * /*expert_offsets*/,
    const float * /*topk_weights*/, const void * /*output*/, int32_t /*top_k*/)
{
    if (n_experts < 0 || n_rows_per_expert < 0 || n_cols < 0) return 2;
    if (n_cols > 0 && (n_cols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q3_K_batched_can_implement(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * /*weights*/, const void * /*activations*/,
    const int32_t * /*sorted_token_ids*/, const int32_t * /*expert_offsets*/,
    const float * /*topk_weights*/, const void * /*output*/, int32_t /*top_k*/)
{
    if (n_experts < 0 || n_rows_per_expert < 0 || n_cols < 0) return 2;
    if (n_cols > 0 && (n_cols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q3_K_batched_bf16_can_implement(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * /*weights*/, const void * /*activations*/,
    const int32_t * /*sorted_token_ids*/, const int32_t * /*expert_offsets*/,
    const float * /*topk_weights*/, const void * /*output*/, int32_t /*top_k*/)
{
    if (n_experts < 0 || n_rows_per_expert < 0 || n_cols < 0) return 2;
    if (n_cols > 0 && (n_cols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q3_K_batched_f16_can_implement(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * /*weights*/, const void * /*activations*/,
    const int32_t * /*sorted_token_ids*/, const int32_t * /*expert_offsets*/,
    const float * /*topk_weights*/, const void * /*output*/, int32_t /*top_k*/)
{
    if (n_experts < 0 || n_rows_per_expert < 0 || n_cols < 0) return 2;
    if (n_cols > 0 && (n_cols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q4_0_batched_can_implement(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * /*weights*/, const void * /*activations*/,
    const int32_t * /*sorted_token_ids*/, const int32_t * /*expert_offsets*/,
    const float * /*topk_weights*/, const void * /*output*/, int32_t /*top_k*/)
{
    if (n_experts < 0 || n_rows_per_expert < 0 || n_cols < 0) return 2;
    if (n_cols > 0 && n_cols < 64) return 2;
    if (n_cols > 0 && (n_cols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q4_0_batched_bf16_can_implement(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * /*weights*/, const void * /*activations*/,
    const int32_t * /*sorted_token_ids*/, const int32_t * /*expert_offsets*/,
    const float * /*topk_weights*/, const void * /*output*/, int32_t /*top_k*/)
{
    if (n_experts < 0 || n_rows_per_expert < 0 || n_cols < 0) return 2;
    if (n_cols > 0 && n_cols < 64) return 2;
    if (n_cols > 0 && (n_cols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q4_0_batched_f16_can_implement(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * /*weights*/, const void * /*activations*/,
    const int32_t * /*sorted_token_ids*/, const int32_t * /*expert_offsets*/,
    const float * /*topk_weights*/, const void * /*output*/, int32_t /*top_k*/)
{
    if (n_experts < 0 || n_rows_per_expert < 0 || n_cols < 0) return 2;
    if (n_cols > 0 && n_cols < 64) return 2;
    if (n_cols > 0 && (n_cols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q4_1_batched_can_implement(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * /*weights*/, const void * /*activations*/,
    const int32_t * /*sorted_token_ids*/, const int32_t * /*expert_offsets*/,
    const float * /*topk_weights*/, const void * /*output*/, int32_t /*top_k*/)
{
    if (n_experts < 0 || n_rows_per_expert < 0 || n_cols < 0) return 2;
    if (n_cols > 0 && n_cols < 64) return 2;
    if (n_cols > 0 && (n_cols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q4_1_batched_bf16_can_implement(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * /*weights*/, const void * /*activations*/,
    const int32_t * /*sorted_token_ids*/, const int32_t * /*expert_offsets*/,
    const float * /*topk_weights*/, const void * /*output*/, int32_t /*top_k*/)
{
    if (n_experts < 0 || n_rows_per_expert < 0 || n_cols < 0) return 2;
    if (n_cols > 0 && n_cols < 64) return 2;
    if (n_cols > 0 && (n_cols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q4_1_batched_f16_can_implement(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * /*weights*/, const void * /*activations*/,
    const int32_t * /*sorted_token_ids*/, const int32_t * /*expert_offsets*/,
    const float * /*topk_weights*/, const void * /*output*/, int32_t /*top_k*/)
{
    if (n_experts < 0 || n_rows_per_expert < 0 || n_cols < 0) return 2;
    if (n_cols > 0 && n_cols < 64) return 2;
    if (n_cols > 0 && (n_cols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q4_K_batched_can_implement(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * /*weights*/, const void * /*activations*/,
    const int32_t * /*sorted_token_ids*/, const int32_t * /*expert_offsets*/,
    const float * /*topk_weights*/, const void * /*output*/, int32_t /*top_k*/)
{
    if (n_experts < 0 || n_rows_per_expert < 0 || n_cols < 0) return 2;
    if (n_cols > 0 && (n_cols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q4_K_batched_bf16_can_implement(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * /*weights*/, const void * /*activations*/,
    const int32_t * /*sorted_token_ids*/, const int32_t * /*expert_offsets*/,
    const float * /*topk_weights*/, const void * /*output*/, int32_t /*top_k*/)
{
    if (n_experts < 0 || n_rows_per_expert < 0 || n_cols < 0) return 2;
    if (n_cols > 0 && (n_cols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q4_K_batched_f16_can_implement(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * /*weights*/, const void * /*activations*/,
    const int32_t * /*sorted_token_ids*/, const int32_t * /*expert_offsets*/,
    const float * /*topk_weights*/, const void * /*output*/, int32_t /*top_k*/)
{
    if (n_experts < 0 || n_rows_per_expert < 0 || n_cols < 0) return 2;
    if (n_cols > 0 && (n_cols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q5_0_batched_can_implement(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * /*weights*/, const void * /*activations*/,
    const int32_t * /*sorted_token_ids*/, const int32_t * /*expert_offsets*/,
    const float * /*topk_weights*/, const void * /*output*/, int32_t /*top_k*/)
{
    if (n_experts < 0 || n_rows_per_expert < 0 || n_cols < 0) return 2;
    if (n_cols > 0 && n_cols < 64) return 2;
    if (n_cols > 0 && (n_cols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q5_0_batched_bf16_can_implement(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * /*weights*/, const void * /*activations*/,
    const int32_t * /*sorted_token_ids*/, const int32_t * /*expert_offsets*/,
    const float * /*topk_weights*/, const void * /*output*/, int32_t /*top_k*/)
{
    if (n_experts < 0 || n_rows_per_expert < 0 || n_cols < 0) return 2;
    if (n_cols > 0 && n_cols < 64) return 2;
    if (n_cols > 0 && (n_cols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q5_0_batched_f16_can_implement(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * /*weights*/, const void * /*activations*/,
    const int32_t * /*sorted_token_ids*/, const int32_t * /*expert_offsets*/,
    const float * /*topk_weights*/, const void * /*output*/, int32_t /*top_k*/)
{
    if (n_experts < 0 || n_rows_per_expert < 0 || n_cols < 0) return 2;
    if (n_cols > 0 && n_cols < 64) return 2;
    if (n_cols > 0 && (n_cols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q5_1_batched_can_implement(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * /*weights*/, const void * /*activations*/,
    const int32_t * /*sorted_token_ids*/, const int32_t * /*expert_offsets*/,
    const float * /*topk_weights*/, const void * /*output*/, int32_t /*top_k*/)
{
    if (n_experts < 0 || n_rows_per_expert < 0 || n_cols < 0) return 2;
    if (n_cols > 0 && n_cols < 64) return 2;
    if (n_cols > 0 && (n_cols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q5_1_batched_bf16_can_implement(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * /*weights*/, const void * /*activations*/,
    const int32_t * /*sorted_token_ids*/, const int32_t * /*expert_offsets*/,
    const float * /*topk_weights*/, const void * /*output*/, int32_t /*top_k*/)
{
    if (n_experts < 0 || n_rows_per_expert < 0 || n_cols < 0) return 2;
    if (n_cols > 0 && n_cols < 64) return 2;
    if (n_cols > 0 && (n_cols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q5_1_batched_f16_can_implement(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * /*weights*/, const void * /*activations*/,
    const int32_t * /*sorted_token_ids*/, const int32_t * /*expert_offsets*/,
    const float * /*topk_weights*/, const void * /*output*/, int32_t /*top_k*/)
{
    if (n_experts < 0 || n_rows_per_expert < 0 || n_cols < 0) return 2;
    if (n_cols > 0 && n_cols < 64) return 2;
    if (n_cols > 0 && (n_cols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q5_K_batched_can_implement(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * /*weights*/, const void * /*activations*/,
    const int32_t * /*sorted_token_ids*/, const int32_t * /*expert_offsets*/,
    const float * /*topk_weights*/, const void * /*output*/, int32_t /*top_k*/)
{
    if (n_experts < 0 || n_rows_per_expert < 0 || n_cols < 0) return 2;
    if (n_cols > 0 && (n_cols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q5_K_batched_bf16_can_implement(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * /*weights*/, const void * /*activations*/,
    const int32_t * /*sorted_token_ids*/, const int32_t * /*expert_offsets*/,
    const float * /*topk_weights*/, const void * /*output*/, int32_t /*top_k*/)
{
    if (n_experts < 0 || n_rows_per_expert < 0 || n_cols < 0) return 2;
    if (n_cols > 0 && (n_cols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q5_K_batched_f16_can_implement(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * /*weights*/, const void * /*activations*/,
    const int32_t * /*sorted_token_ids*/, const int32_t * /*expert_offsets*/,
    const float * /*topk_weights*/, const void * /*output*/, int32_t /*top_k*/)
{
    if (n_experts < 0 || n_rows_per_expert < 0 || n_cols < 0) return 2;
    if (n_cols > 0 && (n_cols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q6_K_batched_can_implement(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * /*weights*/, const void * /*activations*/,
    const int32_t * /*sorted_token_ids*/, const int32_t * /*expert_offsets*/,
    const float * /*topk_weights*/, const void * /*output*/, int32_t /*top_k*/)
{
    if (n_experts < 0 || n_rows_per_expert < 0 || n_cols < 0) return 2;
    if (n_cols > 0 && (n_cols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q6_K_batched_bf16_can_implement(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * /*weights*/, const void * /*activations*/,
    const int32_t * /*sorted_token_ids*/, const int32_t * /*expert_offsets*/,
    const float * /*topk_weights*/, const void * /*output*/, int32_t /*top_k*/)
{
    if (n_experts < 0 || n_rows_per_expert < 0 || n_cols < 0) return 2;
    if (n_cols > 0 && (n_cols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q6_K_batched_f16_can_implement(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * /*weights*/, const void * /*activations*/,
    const int32_t * /*sorted_token_ids*/, const int32_t * /*expert_offsets*/,
    const float * /*topk_weights*/, const void * /*output*/, int32_t /*top_k*/)
{
    if (n_experts < 0 || n_rows_per_expert < 0 || n_cols < 0) return 2;
    if (n_cols > 0 && (n_cols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q8_0_batched_can_implement(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * /*weights*/, const void * /*activations*/,
    const int32_t * /*sorted_token_ids*/, const int32_t * /*expert_offsets*/,
    const float * /*topk_weights*/, const void * /*output*/, int32_t /*top_k*/)
{
    if (n_experts < 0 || n_rows_per_expert < 0 || n_cols < 0) return 2;
    if (n_cols > 0 && n_cols < 64) return 2;
    if (n_cols > 0 && (n_cols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q8_0_batched_bf16_can_implement(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * /*weights*/, const void * /*activations*/,
    const int32_t * /*sorted_token_ids*/, const int32_t * /*expert_offsets*/,
    const float * /*topk_weights*/, const void * /*output*/, int32_t /*top_k*/)
{
    if (n_experts < 0 || n_rows_per_expert < 0 || n_cols < 0) return 2;
    if (n_cols > 0 && n_cols < 64) return 2;
    if (n_cols > 0 && (n_cols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q8_0_batched_f16_can_implement(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * /*weights*/, const void * /*activations*/,
    const int32_t * /*sorted_token_ids*/, const int32_t * /*expert_offsets*/,
    const float * /*topk_weights*/, const void * /*output*/, int32_t /*top_k*/)
{
    if (n_experts < 0 || n_rows_per_expert < 0 || n_cols < 0) return 2;
    if (n_cols > 0 && n_cols < 64) return 2;
    if (n_cols > 0 && (n_cols % 32) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q8_K_batched_can_implement(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * /*weights*/, const void * /*activations*/,
    const int32_t * /*sorted_token_ids*/, const int32_t * /*expert_offsets*/,
    const float * /*topk_weights*/, const void * /*output*/, int32_t /*top_k*/)
{
    if (n_experts < 0 || n_rows_per_expert < 0 || n_cols < 0) return 2;
    if (n_cols > 0 && (n_cols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q8_K_batched_bf16_can_implement(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * /*weights*/, const void * /*activations*/,
    const int32_t * /*sorted_token_ids*/, const int32_t * /*expert_offsets*/,
    const float * /*topk_weights*/, const void * /*output*/, int32_t /*top_k*/)
{
    if (n_experts < 0 || n_rows_per_expert < 0 || n_cols < 0) return 2;
    if (n_cols > 0 && (n_cols % QK_K) != 0) return 2;
    return 0;
}

extern "C" int32_t baracuda_kernels_mmvq_q8_K_batched_f16_can_implement(
    int32_t n_experts, int32_t n_rows_per_expert, int32_t n_cols,
    const void * /*weights*/, const void * /*activations*/,
    const int32_t * /*sorted_token_ids*/, const int32_t * /*expert_offsets*/,
    const float * /*topk_weights*/, const void * /*output*/, int32_t /*top_k*/)
{
    if (n_experts < 0 || n_rows_per_expert < 0 || n_cols < 0) return 2;
    if (n_cols > 0 && (n_cols % QK_K) != 0) return 2;
    return 0;
}

