// baracuda-kernels Phase 66 Tier 2 — shared FlashInfer prefill plumbing.
//
// Included by both the paged and ragged prefill launchers. Carries the
// MSVC `cudaLaunchKernel` shim (must wrap the prefill header include), the
// vendored includes, and the shared host-side scheduler / workspace logic
// (PrefillPlan + optional KV-split float workspace + plan-field wiring).
//
// All free functions are `inline` so the header can be included in two
// translation units without ODR violations.

#ifndef BARACUDA_FLASHINFER_PREFILL_COMMON_CUH
#define BARACUDA_FLASHINFER_PREFILL_COMMON_CUH

#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// --- MSVC nvcc cudaLaunchKernel workaround (see the decode launcher). ---
namespace baracuda_prefill_msvc_shim {
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
    ::baracuda_prefill_msvc_shim::launch_kernel_explicit( \
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

namespace baracuda_prefill {

constexpr int STATUS_OK          = 0;
constexpr int STATUS_INVALID_ARG = 2;
constexpr int STATUS_UNSUPPORTED = 3;

inline int translate(cudaError_t e) {
    return e == cudaSuccess ? STATUS_OK : STATUS_INVALID_ARG;
}

inline bool head_dim_supported(int32_t head_dim) {
    return head_dim == 64 || head_dim == 128 || head_dim == 256;
}

// Generous fixed int-workspace for the scheduler's index arrays.
constexpr std::size_t kIntWorkspaceBytes = 16ull * 1024ull * 1024ull;

// Upper bound on the float (split-KV) workspace. When the plan splits KV,
// padded_batch_size is bounded by the grid cap (~2*num_sm); tmp_v is
// num_qo_heads * padded * cta_tile_q(<=128) * head_dim floats, tmp_s adds
// num_qo_heads * padded * cta_tile_q. When the plan does NOT split, the
// buffer is untouched, so this bound only needs to cover the split case.
inline std::size_t float_workspace_bytes(int32_t num_qo_heads, int32_t head_dim) {
    int dev = 0, num_sm = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev);
    std::size_t padded = static_cast<std::size_t>(2 * num_sm) + 64;
    std::size_t cta = 128;
    std::size_t qh = static_cast<std::size_t>(num_qo_heads);
    std::size_t hd = static_cast<std::size_t>(head_dim);
    return qh * padded * cta * hd * sizeof(float)   // tmp_v
         + qh * padded * cta * sizeof(float)        // tmp_s
         + 4096;                                    // alignment slack
}

// Result of the host-side prefill plan + transient workspace allocation.
struct PlanResult {
    flashinfer::PrefillPlanInfo info;
    void* int_buf = nullptr;
    void* pinned = nullptr;
    void* float_buf = nullptr;  // null unless KV-split was enabled
    bool ok = false;
};

inline void plan_cleanup(PlanResult& r, cudaStream_t s) {
    if (r.pinned) cudaFreeHost(r.pinned);
    if (r.int_buf) cudaFreeAsync(r.int_buf, s);
    if (r.float_buf) cudaFreeAsync(r.float_buf, s);
    r.pinned = r.int_buf = r.float_buf = nullptr;
}

// Allocate scheduler workspaces + run PrefillPlan. `disable_split_kv`
// false → also allocate the float workspace so the plan may split KV.
// Wraps the throwing PrefillPlan in try/catch (workspace overflow throws).
inline PlanResult run_plan(
    const std::vector<int32_t>& qo_indptr_h, const std::vector<int32_t>& kv_indptr_h,
    int32_t total_num_rows, int32_t batch_size, int32_t num_qo_heads, int32_t num_kv_heads,
    int32_t head_dim, int32_t page_size, std::size_t sizeof_o, bool disable_split_kv,
    cudaStream_t s)
{
    PlanResult r;
    if (cudaMallocAsync(&r.int_buf, kIntWorkspaceBytes, s) != cudaSuccess) return r;
    if (cudaMallocHost(&r.pinned, kIntWorkspaceBytes) != cudaSuccess) { plan_cleanup(r, s); return r; }
    std::size_t float_bytes = 0;
    if (!disable_split_kv) {
        float_bytes = float_workspace_bytes(num_qo_heads, head_dim);
        if (cudaMallocAsync(&r.float_buf, float_bytes, s) != cudaSuccess) { plan_cleanup(r, s); return r; }
    }
    cudaError_t e = cudaErrorUnknown;
    try {
        e = flashinfer::PrefillPlan<int32_t>(
            r.float_buf, float_bytes, r.int_buf, r.pinned, kIntWorkspaceBytes, r.info,
            const_cast<int32_t*>(qo_indptr_h.data()), const_cast<int32_t*>(kv_indptr_h.data()),
            static_cast<uint32_t>(total_num_rows), static_cast<uint32_t>(batch_size),
            static_cast<uint32_t>(num_qo_heads), static_cast<uint32_t>(num_kv_heads),
            static_cast<uint32_t>(head_dim), static_cast<uint32_t>(head_dim),
            static_cast<uint32_t>(page_size), /*enable_cuda_graph=*/false, sizeof_o,
            /*window_left=*/-1, /*fixed_split_size=*/-1, disable_split_kv,
            /*num_colocated_ctas=*/0, s);
    } catch (...) {
        plan_cleanup(r, s);
        return r;  // ok stays false
    }
    if (e != cudaSuccess) { plan_cleanup(r, s); return r; }
    r.ok = true;
    return r;
}

// Wire the plan-derived index tensors into `params` (works for both the
// paged and ragged params structs — identical field names). Returns the
// tmp_v / tmp_s pointers for the dispatched call (null when not split).
template <typename Params, typename DTypeO>
inline void set_plan_fields(Params& params, const flashinfer::PrefillPlanInfo& info,
                            void* int_buf, void* float_buf, int32_t total_num_rows,
                            DTypeO** tmp_v_out, float** tmp_s_out) {
    params.request_indices =
        flashinfer::GetPtrFromBaseOffset<int32_t>(int_buf, info.request_indices_offset);
    params.qo_tile_indices =
        flashinfer::GetPtrFromBaseOffset<int32_t>(int_buf, info.qo_tile_indices_offset);
    params.kv_tile_indices =
        flashinfer::GetPtrFromBaseOffset<int32_t>(int_buf, info.kv_tile_indices_offset);
    params.o_indptr = flashinfer::GetPtrFromBaseOffset<int32_t>(int_buf, info.o_indptr_offset);
    params.kv_chunk_size_ptr =
        flashinfer::GetPtrFromBaseOffset<int32_t>(int_buf, info.kv_chunk_size_ptr_offset);
    params.padded_batch_size = static_cast<uint32_t>(info.padded_batch_size);
    params.max_total_num_rows = static_cast<uint32_t>(total_num_rows);
    params.total_num_rows = nullptr;  // non-graph: merge falls back to max_total_num_rows
    params.partition_kv = info.split_kv;

    DTypeO* tmp_v = nullptr;
    float* tmp_s = nullptr;
    if (info.split_kv) {
        params.merge_indptr =
            flashinfer::GetPtrFromBaseOffset<int32_t>(int_buf, info.merge_indptr_offset);
        tmp_v = flashinfer::GetPtrFromBaseOffset<DTypeO>(float_buf, info.v_offset);
        tmp_s = flashinfer::GetPtrFromBaseOffset<float>(float_buf, info.s_offset);
    }
    *tmp_v_out = tmp_v;
    *tmp_s_out = tmp_s;
}

// Token-level kv lengths from a PAGE indptr + last_page_len (for the
// scheduler's cost model; the kernel itself reads the page table).
inline std::vector<int32_t> kv_indptr_from_pages(
    const std::vector<int32_t>& page_indptr_h, const std::vector<int32_t>& last_page_len_h,
    int32_t batch_size, int32_t page_size)
{
    std::vector<int32_t> kv(batch_size + 1);
    kv[0] = 0;
    for (int32_t b = 0; b < batch_size; ++b) {
        int32_t pages = page_indptr_h[b + 1] - page_indptr_h[b];
        int32_t len = pages > 0 ? (pages - 1) * page_size + last_page_len_h[b] : 0;
        kv[b + 1] = kv[b] + len;
    }
    return kv;
}

}  // namespace baracuda_prefill

#endif  // BARACUDA_FLASHINFER_PREFILL_COMMON_CUH
