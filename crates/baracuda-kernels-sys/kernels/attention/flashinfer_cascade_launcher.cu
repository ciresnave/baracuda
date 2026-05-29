// baracuda-kernels Phase 46 — FlashInfer cascade attention launcher.
//
// Cascade attention is the building block for shared-prefix serving:
// the prefix tokens (system prompt, RAG context) are attended over once
// per request batch, then merged with each request's unique-suffix
// attention state via FlashInfer's `MergeStates` LSE-aware merge.
//
// The merge formula (numerically stable, base-2 LSE):
//
//   m'    = max(s_a, s_b)
//   w_a   = 2^(s_a - m')
//   w_b   = 2^(s_b - m')
//   v_out = (w_a * v_a + w_b * v_b) / (w_a + w_b)
//   s_out = m' + log2(w_a + w_b)
//
// Where `s_*` are log-sum-exp (base 2) per-row values and `v_*` are the
// partial attention outputs.
//
// Two variants exposed:
//
//   1. **MergeStateInPlace** — in-place pairwise merge:
//      `v <- merge(v, v_other);  s <- merge(s, s_other)`.
//      Fastest for the common case of "merge prefix into per-request
//      output".
//
//   2. **MergeStates** — many-way merge: given a stack of
//      `num_index_sets` partial states for each `(seq_pos, head)` cell,
//      collapse to a single merged state. Used when several prefix
//      caches overlap (cascade depth > 2).
//
// Scope (Phase 46 Tier 1):
//   - f16 + bf16 + f32 element types for `v`.
//   - LSE always f32 regardless of v dtype (FlashInfer convention).
//   - In-place pairwise merge (no mask pointer — pass nullptr).
//   - Many-way merge with `num_index_sets <= 128`. The path is
//     dispatched by FlashInfer based on `num_index_sets` vs `seq_len`.
//
// Caller contract (in-place merge):
//   - `v`         : `[seq_len, num_heads, head_dim]` element type T,
//                   merged in place.
//   - `s`         : `[seq_len, num_heads]` f32 log-sum-exp (base-2),
//                   merged in place.
//   - `v_other`   : `[seq_len, num_heads, head_dim]` element type T.
//   - `s_other`   : `[seq_len, num_heads]` f32 log-sum-exp (base-2).
//
// Status codes: 0 ok, 2 invalid_problem, 3 unsupported.

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "../../vendor/flashinfer/include/flashinfer/attention/cascade.cuh"

namespace {
constexpr int STATUS_OK          = 0;
constexpr int STATUS_INVALID_ARG = 2;
constexpr int STATUS_UNSUPPORTED = 3;

inline int translate(cudaError_t e) {
    return e == cudaSuccess ? STATUS_OK : STATUS_INVALID_ARG;
}

inline bool head_dim_supported(int32_t head_dim) {
    // FlashInfer's DISPATCH_HEAD_DIM macro supports 64 / 128 / 256
    // (the standard set for transformer head dims). Reject others
    // up front so callers get a clean unsupported, not a cuda
    // launch failure deep inside the dispatcher.
    return head_dim == 64 || head_dim == 128 || head_dim == 256;
}
}  // namespace

extern "C" {

// =====================================================================
// In-place pairwise merge
// =====================================================================

int baracuda_kernels_flashinfer_merge_state_in_place_f16_run(
    int32_t seq_len, int32_t num_heads, int32_t head_dim,
    void* v, void* s, const void* v_other, const void* s_other,
    void* stream)
{
    if (seq_len <= 0 || num_heads <= 0) return STATUS_INVALID_ARG;
    if (!head_dim_supported(head_dim)) return STATUS_UNSUPPORTED;
    if (!v || !s || !v_other || !s_other) return STATUS_INVALID_ARG;
    cudaError_t e = flashinfer::MergeStateInPlace<__half>(
        reinterpret_cast<__half*>(v),
        reinterpret_cast<float*>(s),
        reinterpret_cast<__half*>(const_cast<void*>(v_other)),
        reinterpret_cast<float*>(const_cast<void*>(s_other)),
        static_cast<uint32_t>(seq_len),
        static_cast<uint32_t>(num_heads),
        static_cast<uint32_t>(head_dim),
        /*mask=*/nullptr,
        reinterpret_cast<cudaStream_t>(stream));
    return translate(e);
}

int baracuda_kernels_flashinfer_merge_state_in_place_bf16_run(
    int32_t seq_len, int32_t num_heads, int32_t head_dim,
    void* v, void* s, const void* v_other, const void* s_other,
    void* stream)
{
    if (seq_len <= 0 || num_heads <= 0) return STATUS_INVALID_ARG;
    if (!head_dim_supported(head_dim)) return STATUS_UNSUPPORTED;
    if (!v || !s || !v_other || !s_other) return STATUS_INVALID_ARG;
    cudaError_t e = flashinfer::MergeStateInPlace<__nv_bfloat16>(
        reinterpret_cast<__nv_bfloat16*>(v),
        reinterpret_cast<float*>(s),
        reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(v_other)),
        reinterpret_cast<float*>(const_cast<void*>(s_other)),
        static_cast<uint32_t>(seq_len),
        static_cast<uint32_t>(num_heads),
        static_cast<uint32_t>(head_dim),
        /*mask=*/nullptr,
        reinterpret_cast<cudaStream_t>(stream));
    return translate(e);
}

int baracuda_kernels_flashinfer_merge_state_in_place_f32_run(
    int32_t seq_len, int32_t num_heads, int32_t head_dim,
    void* v, void* s, const void* v_other, const void* s_other,
    void* stream)
{
    if (seq_len <= 0 || num_heads <= 0) return STATUS_INVALID_ARG;
    if (!head_dim_supported(head_dim)) return STATUS_UNSUPPORTED;
    if (!v || !s || !v_other || !s_other) return STATUS_INVALID_ARG;
    cudaError_t e = flashinfer::MergeStateInPlace<float>(
        reinterpret_cast<float*>(v),
        reinterpret_cast<float*>(s),
        reinterpret_cast<float*>(const_cast<void*>(v_other)),
        reinterpret_cast<float*>(const_cast<void*>(s_other)),
        static_cast<uint32_t>(seq_len),
        static_cast<uint32_t>(num_heads),
        static_cast<uint32_t>(head_dim),
        /*mask=*/nullptr,
        reinterpret_cast<cudaStream_t>(stream));
    return translate(e);
}

int baracuda_kernels_flashinfer_merge_state_in_place_can_implement(
    int32_t seq_len, int32_t num_heads, int32_t head_dim)
{
    if (seq_len <= 0 || num_heads <= 0) return STATUS_INVALID_ARG;
    if (!head_dim_supported(head_dim)) return STATUS_UNSUPPORTED;
    return STATUS_OK;
}

// =====================================================================
// Many-way merge (cascade depth > 2)
// =====================================================================
//
// Layout:
//   v        : [seq_len, num_index_sets, num_heads, head_dim] T input
//   s        : [seq_len, num_index_sets, num_heads]           f32 input
//   v_merged : [seq_len, num_heads, head_dim]                 T output
//   s_merged : [seq_len, num_heads]                           f32 output

int baracuda_kernels_flashinfer_merge_states_f16_run(
    int32_t num_index_sets, int32_t seq_len, int32_t num_heads, int32_t head_dim,
    const void* v, const void* s, void* v_merged, void* s_merged,
    void* stream)
{
    if (num_index_sets <= 0 || seq_len <= 0 || num_heads <= 0) return STATUS_INVALID_ARG;
    if (!head_dim_supported(head_dim)) return STATUS_UNSUPPORTED;
    if (!v || !s || !v_merged || !s_merged) return STATUS_INVALID_ARG;
    cudaError_t e = flashinfer::MergeStates<__half, __half>(
        reinterpret_cast<__half*>(const_cast<void*>(v)),
        reinterpret_cast<float*>(const_cast<void*>(s)),
        reinterpret_cast<__half*>(v_merged),
        reinterpret_cast<float*>(s_merged),
        static_cast<uint32_t>(num_index_sets),
        static_cast<uint32_t>(seq_len),
        static_cast<uint32_t>(num_heads),
        static_cast<uint32_t>(head_dim),
        reinterpret_cast<cudaStream_t>(stream));
    return translate(e);
}

int baracuda_kernels_flashinfer_merge_states_bf16_run(
    int32_t num_index_sets, int32_t seq_len, int32_t num_heads, int32_t head_dim,
    const void* v, const void* s, void* v_merged, void* s_merged,
    void* stream)
{
    if (num_index_sets <= 0 || seq_len <= 0 || num_heads <= 0) return STATUS_INVALID_ARG;
    if (!head_dim_supported(head_dim)) return STATUS_UNSUPPORTED;
    if (!v || !s || !v_merged || !s_merged) return STATUS_INVALID_ARG;
    cudaError_t e = flashinfer::MergeStates<__nv_bfloat16, __nv_bfloat16>(
        reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(v)),
        reinterpret_cast<float*>(const_cast<void*>(s)),
        reinterpret_cast<__nv_bfloat16*>(v_merged),
        reinterpret_cast<float*>(s_merged),
        static_cast<uint32_t>(num_index_sets),
        static_cast<uint32_t>(seq_len),
        static_cast<uint32_t>(num_heads),
        static_cast<uint32_t>(head_dim),
        reinterpret_cast<cudaStream_t>(stream));
    return translate(e);
}

int baracuda_kernels_flashinfer_merge_states_f32_run(
    int32_t num_index_sets, int32_t seq_len, int32_t num_heads, int32_t head_dim,
    const void* v, const void* s, void* v_merged, void* s_merged,
    void* stream)
{
    if (num_index_sets <= 0 || seq_len <= 0 || num_heads <= 0) return STATUS_INVALID_ARG;
    if (!head_dim_supported(head_dim)) return STATUS_UNSUPPORTED;
    if (!v || !s || !v_merged || !s_merged) return STATUS_INVALID_ARG;
    cudaError_t e = flashinfer::MergeStates<float, float>(
        reinterpret_cast<float*>(const_cast<void*>(v)),
        reinterpret_cast<float*>(const_cast<void*>(s)),
        reinterpret_cast<float*>(v_merged),
        reinterpret_cast<float*>(s_merged),
        static_cast<uint32_t>(num_index_sets),
        static_cast<uint32_t>(seq_len),
        static_cast<uint32_t>(num_heads),
        static_cast<uint32_t>(head_dim),
        reinterpret_cast<cudaStream_t>(stream));
    return translate(e);
}

int baracuda_kernels_flashinfer_merge_states_can_implement(
    int32_t num_index_sets, int32_t seq_len, int32_t num_heads, int32_t head_dim)
{
    if (num_index_sets <= 0 || seq_len <= 0 || num_heads <= 0) return STATUS_INVALID_ARG;
    if (!head_dim_supported(head_dim)) return STATUS_UNSUPPORTED;
    return STATUS_OK;
}

}  // extern "C"
