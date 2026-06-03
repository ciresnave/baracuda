// baracuda-kernels Phase 46 — FlashInfer paged KV-cache append launcher.
//
// Companion to the existing baracuda `KvCacheAppendPlan` (which writes
// into a CONTIGUOUS `[B, H, L_max, D]` cache). This launcher writes
// new K/V slices into a PAGED store (`page_size` rows per page; per-
// request page table maps logical positions to physical page indices).
//
// FlashInfer's `AppendPagedKVCacheDecode` (single-token-per-request, the
// usual decode case) is wired here. The more general `AppendPagedKVCache`
// (ragged batch with `cu_seqlens`) is wired separately.
//
// Caller contract (single-token-per-request append):
//   - `paged_kv`         : opaque `paged_kv_t<DType, IdType>` fields
//                          passed individually through the FFI:
//       * `page_size`     : pages along the sequence axis (e.g. 16 or 32)
//       * `num_heads`     : KV heads (== num_qo_heads / gqa_group_size)
//       * `head_dim`      : per-head feature dim
//       * `batch_size`    : number of requests
//       * `k_data`        : `[max_num_pages, num_heads, page_size, head_dim]`
//                           contiguous (`kHND` layout)
//       * `v_data`        : same layout as `k_data`
//       * `indices`       : `[total_used_pages]` i32 page-id array
//       * `indptr`        : `[batch_size + 1]` i32 prefix sum into indices
//       * `last_page_len` : `[batch_size]` i32 in `[0, page_size]`
//   - `key`              : `[batch_size, num_heads, head_dim]` new K
//   - `value`            : same layout as `key`
//
// After the call, `last_page_len[b]` should be incremented by 1
// (caller's responsibility — this launcher does not touch it).
//
// Scope (Phase 46 Tier 1):
//   - f16 + bf16 + f32 K/V dtypes; i32 index dtype.
//   - kHND layout only (head-major within a page). NHD support is a
//     mechanical extension via the `paged_kv_t` ctor's layout flag.
//   - HEAD_DIM ∈ {64, 128, 256}.
//
// Status codes: 0 ok, 2 invalid_problem, 3 unsupported.

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "../../vendor/flashinfer/include/flashinfer/page.cuh"
#include "../../vendor/flashinfer/include/flashinfer/layout.cuh"

namespace {
constexpr int STATUS_OK          = 0;
constexpr int STATUS_INVALID_ARG = 2;
constexpr int STATUS_UNSUPPORTED = 3;

inline int translate(cudaError_t e) {
    return e == cudaSuccess ? STATUS_OK : STATUS_INVALID_ARG;
}

inline bool head_dim_supported(int32_t head_dim) {
    return head_dim == 64 || head_dim == 128 || head_dim == 256;
}

template <typename DType>
int run_append_decode_typed(
    int32_t batch_size, int32_t page_size, int32_t num_heads, int32_t head_dim,
    void* k_data, void* v_data, int32_t* indices, int32_t* indptr, int32_t* last_page_len,
    const void* key, const void* value, void* stream)
{
    flashinfer::paged_kv_t<DType, int32_t> paged_kv(
        static_cast<uint32_t>(num_heads),
        static_cast<uint32_t>(page_size),
        static_cast<uint32_t>(head_dim),
        static_cast<uint32_t>(batch_size),
        flashinfer::QKVLayout::kHND,
        reinterpret_cast<DType*>(k_data),
        reinterpret_cast<DType*>(v_data),
        indices,
        indptr,
        last_page_len,
        /*rope_pos_offset=*/nullptr);
    cudaError_t e = flashinfer::AppendPagedKVCacheDecode<DType, int32_t>(
        paged_kv,
        reinterpret_cast<DType*>(const_cast<void*>(key)),
        reinterpret_cast<DType*>(const_cast<void*>(value)),
        reinterpret_cast<cudaStream_t>(stream));
    return translate(e);
}
}  // namespace

extern "C" {

// Decode-time append (1 token per request).

int baracuda_kernels_flashinfer_paged_kv_append_decode_f16_run(
    int32_t batch_size, int32_t page_size, int32_t num_heads, int32_t head_dim,
    void* k_data, void* v_data, void* indices, void* indptr, void* last_page_len,
    const void* key, const void* value, void* stream)
{
    if (batch_size <= 0 || page_size <= 0 || num_heads <= 0) return STATUS_INVALID_ARG;
    if (!head_dim_supported(head_dim)) return STATUS_UNSUPPORTED;
    if (!k_data || !v_data || !indices || !indptr || !last_page_len) return STATUS_INVALID_ARG;
    if (!key || !value) return STATUS_INVALID_ARG;
    return run_append_decode_typed<__half>(
        batch_size, page_size, num_heads, head_dim,
        k_data, v_data,
        reinterpret_cast<int32_t*>(indices),
        reinterpret_cast<int32_t*>(indptr),
        reinterpret_cast<int32_t*>(last_page_len),
        key, value, stream);
}

int baracuda_kernels_flashinfer_paged_kv_append_decode_bf16_run(
    int32_t batch_size, int32_t page_size, int32_t num_heads, int32_t head_dim,
    void* k_data, void* v_data, void* indices, void* indptr, void* last_page_len,
    const void* key, const void* value, void* stream)
{
    if (batch_size <= 0 || page_size <= 0 || num_heads <= 0) return STATUS_INVALID_ARG;
    if (!head_dim_supported(head_dim)) return STATUS_UNSUPPORTED;
    if (!k_data || !v_data || !indices || !indptr || !last_page_len) return STATUS_INVALID_ARG;
    if (!key || !value) return STATUS_INVALID_ARG;
    return run_append_decode_typed<__nv_bfloat16>(
        batch_size, page_size, num_heads, head_dim,
        k_data, v_data,
        reinterpret_cast<int32_t*>(indices),
        reinterpret_cast<int32_t*>(indptr),
        reinterpret_cast<int32_t*>(last_page_len),
        key, value, stream);
}

int baracuda_kernels_flashinfer_paged_kv_append_decode_f32_run(
    int32_t batch_size, int32_t page_size, int32_t num_heads, int32_t head_dim,
    void* k_data, void* v_data, void* indices, void* indptr, void* last_page_len,
    const void* key, const void* value, void* stream)
{
    if (batch_size <= 0 || page_size <= 0 || num_heads <= 0) return STATUS_INVALID_ARG;
    if (!head_dim_supported(head_dim)) return STATUS_UNSUPPORTED;
    if (!k_data || !v_data || !indices || !indptr || !last_page_len) return STATUS_INVALID_ARG;
    if (!key || !value) return STATUS_INVALID_ARG;
    return run_append_decode_typed<float>(
        batch_size, page_size, num_heads, head_dim,
        k_data, v_data,
        reinterpret_cast<int32_t*>(indices),
        reinterpret_cast<int32_t*>(indptr),
        reinterpret_cast<int32_t*>(last_page_len),
        key, value, stream);
}

int baracuda_kernels_flashinfer_paged_kv_append_decode_can_implement(
    int32_t batch_size, int32_t page_size, int32_t num_heads, int32_t head_dim)
{
    if (batch_size <= 0 || page_size <= 0 || num_heads <= 0) return STATUS_INVALID_ARG;
    if (!head_dim_supported(head_dim)) return STATUS_UNSUPPORTED;
    return STATUS_OK;
}


// Per-symbol _can_implement companions. Delegate to family-level validator.
extern "C" int32_t baracuda_kernels_flashinfer_paged_kv_append_decode_f16_can_implement(
    int32_t batch_size, int32_t page_size, int32_t num_heads, int32_t head_dim)
{
    return baracuda_kernels_flashinfer_paged_kv_append_decode_can_implement(
        batch_size, page_size, num_heads, head_dim);
}

extern "C" int32_t baracuda_kernels_flashinfer_paged_kv_append_decode_bf16_can_implement(
    int32_t batch_size, int32_t page_size, int32_t num_heads, int32_t head_dim)
{
    return baracuda_kernels_flashinfer_paged_kv_append_decode_can_implement(
        batch_size, page_size, num_heads, head_dim);
}

extern "C" int32_t baracuda_kernels_flashinfer_paged_kv_append_decode_f32_can_implement(
    int32_t batch_size, int32_t page_size, int32_t num_heads, int32_t head_dim)
{
    return baracuda_kernels_flashinfer_paged_kv_append_decode_can_implement(
        batch_size, page_size, num_heads, head_dim);
}

}  // extern "C"
