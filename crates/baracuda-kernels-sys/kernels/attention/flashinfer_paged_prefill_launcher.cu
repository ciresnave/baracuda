// baracuda-kernels Phase 66 Tier 2 — FlashInfer batched paged-KV PREFILL launcher.
//
// Bridges baracuda's `extern "C"` FFI to FlashInfer's
// `BatchPrefillWithPagedKVCacheDispatched<CTA_TILE_Q, HEAD_DIM_QK,
// HEAD_DIM_VO, PosEncMode, USE_FP16_QK_REDUCTION, MaskMode, Variant, Params>`.
//
// This is the prefill counterpart to the Phase 46 paged DECODE launcher:
// multiple query rows per request (ragged, concatenated across the batch
// via `q_indptr`), each attending over its paged K/V history. The core
// serving primitive for the prompt-ingestion phase of vLLM-style stacks.
//
// Layout contract:
//   - `q`            : `[total_num_rows, num_qo_heads, head_dim]` T (ragged;
//                      request b's rows are `q_indptr[b] .. q_indptr[b+1]`)
//   - `q_indptr`     : `[batch + 1]` i32 (device), row prefix-sum into `q`
//   - `k_data`/`v_data` : `[max_pages, num_kv_heads, page_size, head_dim]` T,
//                      contiguous, kHND layout
//   - `kv_indices`   : `[total_used_pages]` i32, physical page IDs
//   - `kv_indptr`    : `[batch + 1]` i32 (device), PAGE prefix-sum
//   - `last_page_len`: `[batch]` i32 in `[1, page_size]`
//   - `o`            : `[total_num_rows, num_qo_heads, head_dim]` T
//   - `lse`          : `[total_num_rows, num_qo_heads]` f32
//
// Scope (Phase 66 Tier 2, v1):
//   - HEAD_DIM ∈ {64, 128, 256}; f16 + bf16 (prefill is tensor-core /
//     mma based — f32 Q/K is not supported by the prefill kernel).
//   - PosEncodingMode::kNone — apply RoPE before populating the cache.
//   - MaskMode ∈ {kNone, kCausal} (causal = standard autoregressive
//     prompt prefill). No custom mask, sliding window, soft-cap, ALiBi.
//   - `disable_split_kv = true`: no KV-split / merge path, so NO float
//     workspace and NO merge step are needed. (Long-context, few-request
//     split-KV parallelism is a future optimization.)
//   - No CUDA Graph capture.
//
// Workspace: allocated internally per call (a device int buffer for the
// scheduler's index tensors + a pinned host mirror). The launcher is
// therefore SYNCHRONOUS in v1 — the host-side `PrefillPlan` needs host
// copies of the indptr arrays, which forces a sync anyway. Prefill is not
// the per-token hot path, so this is acceptable.
//
// Status codes: 0 ok, 2 invalid_problem, 3 unsupported.

#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// --- MSVC nvcc cudaLaunchKernel workaround (identical to the Phase 46
// decode launcher; prefill.cuh also calls cudaLaunchKernel internally). --
namespace baracuda_paged_prefill_msvc_shim {
    static inline cudaError_t launch_kernel_explicit(
        const void* func, ::dim3 grid, ::dim3 block, void** args,
        std::size_t smem, cudaStream_t stream)
    {
        #ifdef __CUDART_API_PER_THREAD_DEFAULT_STREAM
        return ::cudaLaunchKernel_ptsz(func, grid, block, args, smem, stream);
        #else
        return ::cudaLaunchKernel(func, grid, block, args, smem, stream);
        #endif
    }
}
#undef cudaLaunchKernel
#define cudaLaunchKernel(func, grid, block, args, smem, stream) \
    ::baracuda_paged_prefill_msvc_shim::launch_kernel_explicit( \
        (const void*)(func), (grid), (block), (args), (smem), (stream))

#include "../../vendor/flashinfer/include/flashinfer/attention/prefill.cuh"
#include "../../vendor/flashinfer/include/flashinfer/attention/default_prefill_params.cuh"
#include "../../vendor/flashinfer/include/flashinfer/attention/scheduler.cuh"
#include "../../vendor/flashinfer/include/flashinfer/attention/variants.cuh"
#include "../../vendor/flashinfer/include/flashinfer/attention/mask.cuh"
#include "../../vendor/flashinfer/include/flashinfer/layout.cuh"
#include "../../vendor/flashinfer/include/flashinfer/page.cuh"
#include "../../vendor/flashinfer/include/flashinfer/utils.cuh"

#undef cudaLaunchKernel
#ifdef __CUDART_API_PER_THREAD_DEFAULT_STREAM
#define cudaLaunchKernel __CUDART_API_PTSZ(cudaLaunchKernel)
#endif

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

// Generous fixed int-workspace: the scheduler packs a handful of i32
// index arrays sized ~O(padded_batch_size). 16 MiB covers any realistic
// prefill batch (e.g. >1M CTA tiles) by a wide margin.
constexpr std::size_t kIntWorkspaceBytes = 16ull * 1024ull * 1024ull;

template <typename DType, int HEAD_DIM>
int run_paged_prefill_typed(
    int32_t batch_size, int32_t total_num_rows, int32_t page_size,
    int32_t num_qo_heads, int32_t num_kv_heads, float sm_scale, int32_t causal,
    void* k_data, void* v_data, int32_t* kv_indices, int32_t* kv_indptr_d,
    int32_t* last_page_len_d, const void* q, int32_t* q_indptr_d,
    void* o, void* lse, void* stream)
{
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    cudaError_t e = cudaSuccess;

    // --- 1. Pull the (small) index arrays to host for the scheduler. ---
    std::vector<int32_t> q_indptr_h(batch_size + 1);
    std::vector<int32_t> page_indptr_h(batch_size + 1);
    std::vector<int32_t> last_page_len_h(batch_size);
    e = cudaMemcpyAsync(q_indptr_h.data(), q_indptr_d, sizeof(int32_t) * (batch_size + 1),
                        cudaMemcpyDeviceToHost, s);
    if (e != cudaSuccess) return translate(e);
    e = cudaMemcpyAsync(page_indptr_h.data(), kv_indptr_d, sizeof(int32_t) * (batch_size + 1),
                        cudaMemcpyDeviceToHost, s);
    if (e != cudaSuccess) return translate(e);
    e = cudaMemcpyAsync(last_page_len_h.data(), last_page_len_d, sizeof(int32_t) * batch_size,
                        cudaMemcpyDeviceToHost, s);
    if (e != cudaSuccess) return translate(e);
    e = cudaStreamSynchronize(s);
    if (e != cudaSuccess) return translate(e);

    // --- 2. Token-level kv_indptr (the scheduler costs in tokens; the
    //        kernel itself reads page tables via paged_kv_t). ---
    std::vector<int32_t> kv_indptr_h(batch_size + 1);
    kv_indptr_h[0] = 0;
    for (int32_t b = 0; b < batch_size; ++b) {
        int32_t pages = page_indptr_h[b + 1] - page_indptr_h[b];
        int32_t kv_len = pages > 0 ? (pages - 1) * page_size + last_page_len_h[b] : 0;
        kv_indptr_h[b + 1] = kv_indptr_h[b] + kv_len;
    }

    // --- 3. Internal scheduler workspaces (device + pinned host mirror). ---
    void* int_buf = nullptr;
    void* pinned = nullptr;
    e = cudaMallocAsync(&int_buf, kIntWorkspaceBytes, s);
    if (e != cudaSuccess) return translate(e);
    e = cudaMallocHost(&pinned, kIntWorkspaceBytes);
    if (e != cudaSuccess) { cudaFreeAsync(int_buf, s); return translate(e); }

    // --- 4. Host-side plan (work partitioning). ---
    flashinfer::PrefillPlanInfo plan_info;
    e = flashinfer::PrefillPlan<int32_t>(
        /*float_buffer=*/nullptr, /*float_workspace_size_in_bytes=*/0,
        int_buf, pinned, kIntWorkspaceBytes, plan_info,
        q_indptr_h.data(), kv_indptr_h.data(),
        static_cast<uint32_t>(total_num_rows), static_cast<uint32_t>(batch_size),
        static_cast<uint32_t>(num_qo_heads), static_cast<uint32_t>(num_kv_heads),
        static_cast<uint32_t>(HEAD_DIM), static_cast<uint32_t>(HEAD_DIM),
        static_cast<uint32_t>(page_size),
        /*enable_cuda_graph=*/false, static_cast<uint32_t>(sizeof(DType)),
        /*window_left=*/-1, /*fixed_split_size=*/-1, /*disable_split_kv=*/true,
        /*num_colocated_ctas=*/0, s);
    if (e != cudaSuccess) { cudaFreeHost(pinned); cudaFreeAsync(int_buf, s); return translate(e); }

    // --- 5. Paged KV descriptor + params. ---
    flashinfer::paged_kv_t<DType, int32_t> paged_kv(
        static_cast<uint32_t>(num_kv_heads), static_cast<uint32_t>(page_size),
        static_cast<uint32_t>(HEAD_DIM), static_cast<uint32_t>(batch_size),
        flashinfer::QKVLayout::kHND,
        reinterpret_cast<DType*>(k_data), reinterpret_cast<DType*>(v_data),
        kv_indices, kv_indptr_d, last_page_len_d, /*rope_pos_offset=*/nullptr);

    using Params = flashinfer::BatchPrefillPagedParams<DType, DType, DType, int32_t>;
    Params params(
        reinterpret_cast<DType*>(const_cast<void*>(q)), paged_kv,
        /*maybe_custom_mask=*/nullptr, q_indptr_d, /*maybe_mask_indptr=*/nullptr,
        /*maybe_q_rope_offset=*/nullptr, reinterpret_cast<DType*>(o),
        reinterpret_cast<float*>(lse), /*maybe_alibi_slopes=*/nullptr,
        static_cast<uint32_t>(num_qo_heads),
        /*q_stride_n=*/static_cast<int32_t>(num_qo_heads) * HEAD_DIM,
        /*q_stride_h=*/HEAD_DIM, /*window_left=*/-1, /*logits_soft_cap=*/0.0f,
        sm_scale, /*rope_scale=*/1.0f, /*rope_theta=*/10000.0f);

    // Plan-derived index tensors (carved from the device int buffer).
    params.request_indices =
        flashinfer::GetPtrFromBaseOffset<int32_t>(int_buf, plan_info.request_indices_offset);
    params.qo_tile_indices =
        flashinfer::GetPtrFromBaseOffset<int32_t>(int_buf, plan_info.qo_tile_indices_offset);
    params.kv_tile_indices =
        flashinfer::GetPtrFromBaseOffset<int32_t>(int_buf, plan_info.kv_tile_indices_offset);
    params.o_indptr =
        flashinfer::GetPtrFromBaseOffset<int32_t>(int_buf, plan_info.o_indptr_offset);
    params.kv_chunk_size_ptr =
        flashinfer::GetPtrFromBaseOffset<int32_t>(int_buf, plan_info.kv_chunk_size_ptr_offset);
    params.padded_batch_size = static_cast<uint32_t>(plan_info.padded_batch_size);
    params.max_total_num_rows = static_cast<uint32_t>(total_num_rows);
    params.total_num_rows = nullptr;       // device counter only used in CUDA-graph mode
    params.partition_kv = plan_info.split_kv;  // false (split disabled)

    // --- 6. Dispatch. Two mask arms only (avoid instantiating the
    //        custom / multi-item-scoring kernels). ---
    using Variant = flashinfer::DefaultAttention<
        /*use_custom_mask=*/false, /*use_sliding_window=*/false,
        /*use_logits_soft_cap=*/false, /*use_alibi=*/false>;
    constexpr auto POS = flashinfer::PosEncodingMode::kNone;

    if (causal != 0) {
        DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q, CTA_TILE_Q, {
            e = flashinfer::BatchPrefillWithPagedKVCacheDispatched<
                CTA_TILE_Q, HEAD_DIM, HEAD_DIM, POS, /*USE_FP16_QK_REDUCTION=*/false,
                flashinfer::MaskMode::kCausal, Variant, Params>(
                    params, /*tmp_v=*/nullptr, /*tmp_s=*/nullptr, /*enable_pdl=*/false, s);
        });
    } else {
        DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q, CTA_TILE_Q, {
            e = flashinfer::BatchPrefillWithPagedKVCacheDispatched<
                CTA_TILE_Q, HEAD_DIM, HEAD_DIM, POS, /*USE_FP16_QK_REDUCTION=*/false,
                flashinfer::MaskMode::kNone, Variant, Params>(
                    params, /*tmp_v=*/nullptr, /*tmp_s=*/nullptr, /*enable_pdl=*/false, s);
        });
    }

    // --- 7. Sync (so the pinned mirror + int buffer outlive the launch),
    //        then release the transient workspaces. ---
    cudaError_t sync_e = cudaStreamSynchronize(s);
    cudaFreeHost(pinned);
    cudaFreeAsync(int_buf, s);
    if (e != cudaSuccess) return translate(e);
    return translate(sync_e);
}

template <typename DType>
int dispatch_head_dim(
    int32_t batch_size, int32_t total_num_rows, int32_t page_size, int32_t head_dim,
    int32_t num_qo_heads, int32_t num_kv_heads, float sm_scale, int32_t causal,
    void* k_data, void* v_data, int32_t* kv_indices, int32_t* kv_indptr_d,
    int32_t* last_page_len_d, const void* q, int32_t* q_indptr_d,
    void* o, void* lse, void* stream)
{
    if (head_dim == 64) {
        return run_paged_prefill_typed<DType, 64>(
            batch_size, total_num_rows, page_size, num_qo_heads, num_kv_heads, sm_scale, causal,
            k_data, v_data, kv_indices, kv_indptr_d, last_page_len_d, q, q_indptr_d, o, lse, stream);
    } else if (head_dim == 128) {
        return run_paged_prefill_typed<DType, 128>(
            batch_size, total_num_rows, page_size, num_qo_heads, num_kv_heads, sm_scale, causal,
            k_data, v_data, kv_indices, kv_indptr_d, last_page_len_d, q, q_indptr_d, o, lse, stream);
    } else if (head_dim == 256) {
        return run_paged_prefill_typed<DType, 256>(
            batch_size, total_num_rows, page_size, num_qo_heads, num_kv_heads, sm_scale, causal,
            k_data, v_data, kv_indices, kv_indptr_d, last_page_len_d, q, q_indptr_d, o, lse, stream);
    }
    return STATUS_UNSUPPORTED;
}

inline int validate(int32_t batch_size, int32_t total_num_rows, int32_t page_size,
                    int32_t head_dim, int32_t num_qo_heads, int32_t num_kv_heads,
                    const void* q, const void* o, const void* lse, const void* k_data,
                    const void* v_data, const void* kv_indices, const void* kv_indptr,
                    const void* last_page_len, const void* q_indptr) {
    if (batch_size <= 0 || total_num_rows <= 0 || page_size <= 0 || num_qo_heads <= 0 ||
        num_kv_heads <= 0)
        return STATUS_INVALID_ARG;
    if (num_qo_heads % num_kv_heads != 0) return STATUS_INVALID_ARG;
    if (!head_dim_supported(head_dim)) return STATUS_UNSUPPORTED;
    if (!q || !o || !lse || !k_data || !v_data) return STATUS_INVALID_ARG;
    if (!kv_indices || !kv_indptr || !last_page_len || !q_indptr) return STATUS_INVALID_ARG;
    return STATUS_OK;
}
}  // namespace

extern "C" {

int baracuda_kernels_flashinfer_paged_prefill_f16_run(
    int32_t batch_size, int32_t total_num_rows, int32_t page_size, int32_t head_dim,
    int32_t num_qo_heads, int32_t num_kv_heads, float sm_scale, int32_t causal,
    void* k_data, void* v_data, void* kv_indices, void* kv_indptr, void* last_page_len,
    const void* q, void* q_indptr, void* o, void* lse, void* stream)
{
    int v = validate(batch_size, total_num_rows, page_size, head_dim, num_qo_heads, num_kv_heads,
                     q, o, lse, k_data, v_data, kv_indices, kv_indptr, last_page_len, q_indptr);
    if (v != STATUS_OK) return v;
    return dispatch_head_dim<__half>(
        batch_size, total_num_rows, page_size, head_dim, num_qo_heads, num_kv_heads, sm_scale,
        causal, k_data, v_data, reinterpret_cast<int32_t*>(kv_indices),
        reinterpret_cast<int32_t*>(kv_indptr), reinterpret_cast<int32_t*>(last_page_len),
        q, reinterpret_cast<int32_t*>(q_indptr), o, lse, stream);
}

int baracuda_kernels_flashinfer_paged_prefill_bf16_run(
    int32_t batch_size, int32_t total_num_rows, int32_t page_size, int32_t head_dim,
    int32_t num_qo_heads, int32_t num_kv_heads, float sm_scale, int32_t causal,
    void* k_data, void* v_data, void* kv_indices, void* kv_indptr, void* last_page_len,
    const void* q, void* q_indptr, void* o, void* lse, void* stream)
{
    int v = validate(batch_size, total_num_rows, page_size, head_dim, num_qo_heads, num_kv_heads,
                     q, o, lse, k_data, v_data, kv_indices, kv_indptr, last_page_len, q_indptr);
    if (v != STATUS_OK) return v;
    return dispatch_head_dim<__nv_bfloat16>(
        batch_size, total_num_rows, page_size, head_dim, num_qo_heads, num_kv_heads, sm_scale,
        causal, k_data, v_data, reinterpret_cast<int32_t*>(kv_indices),
        reinterpret_cast<int32_t*>(kv_indptr), reinterpret_cast<int32_t*>(last_page_len),
        q, reinterpret_cast<int32_t*>(q_indptr), o, lse, stream);
}

int baracuda_kernels_flashinfer_paged_prefill_can_implement(
    int32_t batch_size, int32_t total_num_rows, int32_t page_size, int32_t head_dim,
    int32_t num_qo_heads, int32_t num_kv_heads)
{
    if (batch_size <= 0 || total_num_rows <= 0 || page_size <= 0 || num_qo_heads <= 0 ||
        num_kv_heads <= 0)
        return STATUS_INVALID_ARG;
    if (num_qo_heads % num_kv_heads != 0) return STATUS_INVALID_ARG;
    if (!head_dim_supported(head_dim)) return STATUS_UNSUPPORTED;
    return STATUS_OK;
}

}  // extern "C"
