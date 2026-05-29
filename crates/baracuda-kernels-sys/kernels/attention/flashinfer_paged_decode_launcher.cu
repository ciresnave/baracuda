// baracuda-kernels Phase 46 — FlashInfer batched paged-KV decode launcher.
//
// Bridges baracuda's `extern "C"` FFI to FlashInfer's
// `BatchDecodeWithPagedKVCacheDispatched<HEAD_DIM, PosEncMode, Variant, Params>`.
//
// The key missing primitive in baracuda before Phase 46: batched
// attention DECODE (one Q row per request) against a PAGED K/V cache
// (vLLM-style block allocator). Each request's K/V history lives in a
// scattered set of fixed-size pages; the page table maps logical
// positions to physical page IDs.
//
// Layout contract:
//   - `q`            : `[batch, num_qo_heads, head_dim]` element type T_q
//   - `k_data`       : `[max_num_pages, num_kv_heads, page_size, head_dim]`
//                      element type T_kv, contiguous, kHND layout
//   - `v_data`       : same shape and layout as `k_data`
//   - `indices`      : `[total_used_pages]` i32, physical page IDs
//   - `indptr`       : `[batch + 1]` i32, prefix-sum into `indices`
//   - `last_page_len`: `[batch]` i32 in `[0, page_size]`
//   - `o`            : `[batch, num_qo_heads, head_dim]` element type T_o
//   - `lse`          : `[batch, num_qo_heads]` f32 (always f32 — FlashInfer
//                      convention)
//   - `workspace`    : caller-supplied i32-aligned device buffer (see
//                      `*_workspace_size`).
//
// Scope (Phase 46 Tier 1):
//   - HEAD_DIM ∈ {64, 128, 256}
//   - f16 + bf16 + f32 (Q=KV=O all same dtype)
//   - PosEncodingMode::kNone — caller is expected to apply RoPE BEFORE
//     populating the cache (baracuda's existing `RopePlan` covers this).
//     Note: FlashInfer's `kRoPELlama` mode requires the on-the-fly Q
//     rotation, which is a future tier (driver complexity, head-dim
//     constraints).
//   - Standard `DefaultAttention<false, false, false, false>` variant:
//     no custom mask, no sliding window, no logits soft-cap, no alibi.
//   - No CUDA Graph mode (the scheduler-driven mode that pads batch
//     size for graph capture is deferred).
//   - No PDL (programmatic stream serialization).
//
// Workspace layout (single contiguous device buffer):
//   - Offset 0                     : request_indices[batch]            i32
//   - Offset 4 * batch             : kv_tile_indices[batch]            i32
//   - Offset 8 * batch             : o_indptr[batch + 1]               i32
//   - Offset 8 * batch + 4 * (b+1) : kv_chunk_size_ptr[1]              i32
//
// Total workspace = (3 * batch + 2) * sizeof(int32_t) bytes.
//
// The init kernel writes:
//   request_indices[b]  = b
//   kv_tile_indices[b]  = 0
//   o_indptr[b]         = b   (with o_indptr[batch] = batch)
//   kv_chunk_size_ptr[] = INT32_MAX  (means: don't split; one chunk per request)
//
// Status codes: 0 ok, 2 invalid_problem, 3 unsupported.

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// --- MSVC nvcc per-thread-default-stream workaround --------------------
// FlashInfer's `decode.cuh` calls `cudaLaunchKernel((void*)kernel, ...)`.
// Under `-default-stream per-thread` (Windows nvcc), CUDA's headers
// `#define cudaLaunchKernel __CUDART_API_PTSZ(cudaLaunchKernel)`, which
// under MSVC interacts badly with the template overload at the FlashInfer
// call site (the template's name itself gets rewritten, then the
// non-template static-inline wrapper at `cuda_runtime_api.h:13929` and
// the templated one at `cuda_runtime.h:208` collide on overload
// resolution). We side-step the rewrite by introducing a TU-local
// explicit-signature wrapper and `#define`-aliasing `cudaLaunchKernel`
// to it BEFORE the FlashInfer header is parsed. This keeps the per-
// thread-default-stream behavior intact for our other calls while
// making the FlashInfer call site unambiguous.
namespace baracuda_paged_decode_msvc_shim {
    static inline cudaError_t launch_kernel_explicit(
        const void*  func,
        ::dim3       grid,
        ::dim3       block,
        void**       args,
        std::size_t  smem,
        cudaStream_t stream)
    {
        // Bypass the macro by calling the qualified `_ptsz` variant
        // directly when present (per-thread default stream); fall back
        // to the legacy `cudaLaunchKernel` otherwise. Both have the
        // same C signature, so this is ABI-safe.
        #ifdef __CUDART_API_PER_THREAD_DEFAULT_STREAM
        return ::cudaLaunchKernel_ptsz(func, grid, block, args, smem, stream);
        #else
        // Note: the macro is locally redefined just below, so we have
        // to undef-then-re-call through the underlying entry point.
        return ::cudaLaunchKernel(func, grid, block, args, smem, stream);
        #endif
    }
}

// Force FlashInfer's `cudaLaunchKernel((void*)kernel, ...)` to bind to
// our explicit-signature wrapper. The `void*` -> `const void*` is
// implicit; the rest of the args match exactly. This MUST come before
// the FlashInfer headers are included.
#undef cudaLaunchKernel
#define cudaLaunchKernel(func, grid, block, args, smem, stream) \
    ::baracuda_paged_decode_msvc_shim::launch_kernel_explicit( \
        (const void*)(func), (grid), (block), (args), (smem), (stream))

#include "../../vendor/flashinfer/include/flashinfer/attention/decode.cuh"
#include "../../vendor/flashinfer/include/flashinfer/attention/default_decode_params.cuh"
#include "../../vendor/flashinfer/include/flashinfer/attention/variants.cuh"
#include "../../vendor/flashinfer/include/flashinfer/layout.cuh"
#include "../../vendor/flashinfer/include/flashinfer/page.cuh"

// Restore the macro for any code below (we re-use our own init kernel
// launch via `<<<>>>` syntax, so we don't actually need it, but restoring
// keeps the file resilient to future additions).
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

// One-block init kernel: fills the auxiliary index tensors so we don't
// have to round-trip through host memory on every decode step. Launched
// with a single block of `batch + 1` threads.
__global__ void init_decode_workspace_kernel(
    int32_t* request_indices,
    int32_t* kv_tile_indices,
    int32_t* o_indptr,
    int32_t* kv_chunk_size_ptr,
    int32_t batch_size)
{
    int32_t tid = static_cast<int32_t>(threadIdx.x);
    if (tid < batch_size) {
        request_indices[tid] = tid;
        kv_tile_indices[tid] = 0;
        o_indptr[tid] = tid;
    }
    if (tid == batch_size) {
        o_indptr[batch_size] = batch_size;
        kv_chunk_size_ptr[0] = INT32_MAX;
    }
}

template <typename DType, int HEAD_DIM>
int run_paged_decode_typed(
    int32_t batch_size, int32_t page_size,
    int32_t num_qo_heads, int32_t num_kv_heads,
    float sm_scale,
    void* k_data, void* v_data, int32_t* indices, int32_t* indptr, int32_t* last_page_len,
    const void* q, void* o, void* lse,
    int32_t* workspace_i32, void* stream)
{
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    // Carve the workspace into the four index buffers.
    int32_t* request_indices  = workspace_i32;
    int32_t* kv_tile_indices  = workspace_i32 + batch_size;
    int32_t* o_indptr         = workspace_i32 + 2 * batch_size;
    int32_t* kv_chunk_size_p  = workspace_i32 + 3 * batch_size + 1;

    // Launch the one-block init kernel.
    int32_t init_threads = batch_size + 1;
    init_decode_workspace_kernel<<<1, init_threads, 0, s>>>(
        request_indices, kv_tile_indices, o_indptr, kv_chunk_size_p, batch_size);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return translate(e);

    // Build the paged_kv_t descriptor.
    flashinfer::paged_kv_t<DType, int32_t> paged_kv(
        static_cast<uint32_t>(num_kv_heads),
        static_cast<uint32_t>(page_size),
        static_cast<uint32_t>(HEAD_DIM),
        static_cast<uint32_t>(batch_size),
        flashinfer::QKVLayout::kHND,
        reinterpret_cast<DType*>(k_data),
        reinterpret_cast<DType*>(v_data),
        indices,
        indptr,
        last_page_len,
        /*rope_pos_offset=*/nullptr);

    // Build the BatchDecodeParams struct.
    using Params = flashinfer::BatchDecodeParams<DType, DType, DType, int32_t>;
    Params params;
    params.q                = reinterpret_cast<DType*>(const_cast<void*>(q));
    params.q_rope_offset    = nullptr;
    params.paged_kv         = paged_kv;
    params.o                = reinterpret_cast<DType*>(o);
    params.lse              = reinterpret_cast<float*>(lse);
    params.maybe_alibi_slopes = nullptr;
    params.padded_batch_size = static_cast<uint32_t>(batch_size);
    params.num_qo_heads     = static_cast<uint32_t>(num_qo_heads);
    params.q_stride_n       = static_cast<int32_t>(num_qo_heads) * HEAD_DIM;
    params.q_stride_h       = HEAD_DIM;
    params.window_left      = -1;        // no sliding-window
    params.logits_soft_cap  = 0.0f;      // no soft-cap
    params.sm_scale         = sm_scale;
    params.rope_rcp_scale   = 1.0f;
    params.rope_rcp_theta   = 1.0f / 10000.0f;
    params.request_indices  = request_indices;
    params.kv_tile_indices  = kv_tile_indices;
    params.o_indptr         = o_indptr;
    params.kv_chunk_size_ptr = kv_chunk_size_p;
    params.block_valid_mask = nullptr;
    params.partition_kv     = false;

    using Variant = flashinfer::DefaultAttention<
        /*use_custom_mask=*/false,
        /*use_sliding_window=*/false,
        /*use_logits_soft_cap=*/false,
        /*use_alibi=*/false>;

    e = flashinfer::BatchDecodeWithPagedKVCacheDispatched<
        HEAD_DIM, flashinfer::PosEncodingMode::kNone, Variant, Params>(
            params, /*tmp_v=*/nullptr, /*tmp_s=*/nullptr, /*enable_pdl=*/false, s);
    return translate(e);
}

template <typename DType>
int dispatch_head_dim(
    int32_t batch_size, int32_t page_size, int32_t head_dim,
    int32_t num_qo_heads, int32_t num_kv_heads,
    float sm_scale,
    void* k_data, void* v_data, int32_t* indices, int32_t* indptr, int32_t* last_page_len,
    const void* q, void* o, void* lse,
    int32_t* workspace_i32, void* stream)
{
    if (head_dim == 64) {
        return run_paged_decode_typed<DType, 64>(
            batch_size, page_size, num_qo_heads, num_kv_heads, sm_scale,
            k_data, v_data, indices, indptr, last_page_len, q, o, lse,
            workspace_i32, stream);
    } else if (head_dim == 128) {
        return run_paged_decode_typed<DType, 128>(
            batch_size, page_size, num_qo_heads, num_kv_heads, sm_scale,
            k_data, v_data, indices, indptr, last_page_len, q, o, lse,
            workspace_i32, stream);
    } else if (head_dim == 256) {
        return run_paged_decode_typed<DType, 256>(
            batch_size, page_size, num_qo_heads, num_kv_heads, sm_scale,
            k_data, v_data, indices, indptr, last_page_len, q, o, lse,
            workspace_i32, stream);
    }
    return STATUS_UNSUPPORTED;
}

inline std::size_t compute_workspace_bytes(int32_t batch_size) {
    // request_indices[batch] + kv_tile_indices[batch] + o_indptr[batch+1]
    // + kv_chunk_size_ptr[1]
    return static_cast<std::size_t>(3 * batch_size + 2) * sizeof(int32_t);
}
}  // namespace

extern "C" {

// Host-side workspace-size query. Returns bytes needed for the
// auxiliary index buffers; caller allocates and passes through `_run`.
std::size_t baracuda_kernels_flashinfer_paged_decode_workspace_size(int32_t batch_size)
{
    if (batch_size <= 0) return 0;
    return compute_workspace_bytes(batch_size);
}

int baracuda_kernels_flashinfer_paged_decode_f16_run(
    int32_t batch_size, int32_t page_size, int32_t head_dim,
    int32_t num_qo_heads, int32_t num_kv_heads,
    float sm_scale,
    void* k_data, void* v_data, void* indices, void* indptr, void* last_page_len,
    const void* q, void* o, void* lse,
    void* workspace, std::size_t workspace_bytes, void* stream)
{
    if (batch_size <= 0 || page_size <= 0 || num_qo_heads <= 0 || num_kv_heads <= 0)
        return STATUS_INVALID_ARG;
    if (num_qo_heads % num_kv_heads != 0) return STATUS_INVALID_ARG;
    if (!head_dim_supported(head_dim)) return STATUS_UNSUPPORTED;
    if (!q || !o || !lse || !k_data || !v_data) return STATUS_INVALID_ARG;
    if (!indices || !indptr || !last_page_len) return STATUS_INVALID_ARG;
    if (!workspace) return STATUS_INVALID_ARG;
    if (workspace_bytes < compute_workspace_bytes(batch_size)) return STATUS_INVALID_ARG;
    return dispatch_head_dim<__half>(
        batch_size, page_size, head_dim, num_qo_heads, num_kv_heads, sm_scale,
        k_data, v_data,
        reinterpret_cast<int32_t*>(indices),
        reinterpret_cast<int32_t*>(indptr),
        reinterpret_cast<int32_t*>(last_page_len),
        q, o, lse,
        reinterpret_cast<int32_t*>(workspace), stream);
}

int baracuda_kernels_flashinfer_paged_decode_bf16_run(
    int32_t batch_size, int32_t page_size, int32_t head_dim,
    int32_t num_qo_heads, int32_t num_kv_heads,
    float sm_scale,
    void* k_data, void* v_data, void* indices, void* indptr, void* last_page_len,
    const void* q, void* o, void* lse,
    void* workspace, std::size_t workspace_bytes, void* stream)
{
    if (batch_size <= 0 || page_size <= 0 || num_qo_heads <= 0 || num_kv_heads <= 0)
        return STATUS_INVALID_ARG;
    if (num_qo_heads % num_kv_heads != 0) return STATUS_INVALID_ARG;
    if (!head_dim_supported(head_dim)) return STATUS_UNSUPPORTED;
    if (!q || !o || !lse || !k_data || !v_data) return STATUS_INVALID_ARG;
    if (!indices || !indptr || !last_page_len) return STATUS_INVALID_ARG;
    if (!workspace) return STATUS_INVALID_ARG;
    if (workspace_bytes < compute_workspace_bytes(batch_size)) return STATUS_INVALID_ARG;
    return dispatch_head_dim<__nv_bfloat16>(
        batch_size, page_size, head_dim, num_qo_heads, num_kv_heads, sm_scale,
        k_data, v_data,
        reinterpret_cast<int32_t*>(indices),
        reinterpret_cast<int32_t*>(indptr),
        reinterpret_cast<int32_t*>(last_page_len),
        q, o, lse,
        reinterpret_cast<int32_t*>(workspace), stream);
}

int baracuda_kernels_flashinfer_paged_decode_f32_run(
    int32_t batch_size, int32_t page_size, int32_t head_dim,
    int32_t num_qo_heads, int32_t num_kv_heads,
    float sm_scale,
    void* k_data, void* v_data, void* indices, void* indptr, void* last_page_len,
    const void* q, void* o, void* lse,
    void* workspace, std::size_t workspace_bytes, void* stream)
{
    if (batch_size <= 0 || page_size <= 0 || num_qo_heads <= 0 || num_kv_heads <= 0)
        return STATUS_INVALID_ARG;
    if (num_qo_heads % num_kv_heads != 0) return STATUS_INVALID_ARG;
    if (!head_dim_supported(head_dim)) return STATUS_UNSUPPORTED;
    if (!q || !o || !lse || !k_data || !v_data) return STATUS_INVALID_ARG;
    if (!indices || !indptr || !last_page_len) return STATUS_INVALID_ARG;
    if (!workspace) return STATUS_INVALID_ARG;
    if (workspace_bytes < compute_workspace_bytes(batch_size)) return STATUS_INVALID_ARG;
    return dispatch_head_dim<float>(
        batch_size, page_size, head_dim, num_qo_heads, num_kv_heads, sm_scale,
        k_data, v_data,
        reinterpret_cast<int32_t*>(indices),
        reinterpret_cast<int32_t*>(indptr),
        reinterpret_cast<int32_t*>(last_page_len),
        q, o, lse,
        reinterpret_cast<int32_t*>(workspace), stream);
}

int baracuda_kernels_flashinfer_paged_decode_can_implement(
    int32_t batch_size, int32_t page_size, int32_t head_dim,
    int32_t num_qo_heads, int32_t num_kv_heads)
{
    if (batch_size <= 0 || page_size <= 0 || num_qo_heads <= 0 || num_kv_heads <= 0)
        return STATUS_INVALID_ARG;
    if (num_qo_heads % num_kv_heads != 0) return STATUS_INVALID_ARG;
    if (!head_dim_supported(head_dim)) return STATUS_UNSUPPORTED;
    return STATUS_OK;
}

}  // extern "C"
