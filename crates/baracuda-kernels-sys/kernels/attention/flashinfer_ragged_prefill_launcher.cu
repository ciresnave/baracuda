// baracuda-kernels Phase 66 Tier 2 — FlashInfer batched RAGGED-KV PREFILL
// launcher. Routes to `BatchPrefillWithRaggedKVCacheDispatched`.
//
// Like the paged prefill but the K/V history is stored CONTIGUOUSLY (no
// page table): K/V are `[total_kv_rows, num_kv_heads, head_dim]`, ragged
// across the batch via `kv_indptr`. Useful when the KV is not paged (e.g.
// the initial full-prompt prefill before paging, or non-paged serving).
//
// Layout:
//   - q        : [total_num_rows, num_qo_heads, head_dim]; q_indptr [batch+1]
//   - k / v    : [total_kv_rows, num_kv_heads, head_dim];  kv_indptr [batch+1]
//   - o        : [total_num_rows, num_qo_heads, head_dim]
//   - lse      : [total_num_rows, num_qo_heads] f32
//
// Scope mirrors paged prefill: HEAD_DIM ∈ {64,128,256}; f16+bf16; PosEnc
// kNone; MaskMode {kNone, kCausal}; `enable_split` opts into KV-split.
// Synchronous. Status codes: 0 ok, 2 invalid_problem, 3 unsupported.

#include "flashinfer_prefill_common.cuh"

namespace bp = baracuda_prefill;

namespace {
using bp::STATUS_OK;
using bp::STATUS_INVALID_ARG;
using bp::STATUS_UNSUPPORTED;

template <typename DType, int HEAD_DIM>
int run_ragged_prefill_typed(
    int32_t batch_size, int32_t total_num_rows, int32_t total_kv_rows,
    int32_t num_qo_heads, int32_t num_kv_heads, float sm_scale, int32_t causal, int32_t enable_split,
    const void* k_data, const void* v_data, int32_t* kv_indptr_d, const void* q,
    int32_t* q_indptr_d, void* o, void* lse, void* stream)
{
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    cudaError_t e = cudaSuccess;

    std::vector<int32_t> q_indptr_h(batch_size + 1), kv_indptr_h(batch_size + 1);
    e = cudaMemcpyAsync(q_indptr_h.data(), q_indptr_d, sizeof(int32_t) * (batch_size + 1),
                        cudaMemcpyDeviceToHost, s);
    if (e != cudaSuccess) return bp::translate(e);
    e = cudaMemcpyAsync(kv_indptr_h.data(), kv_indptr_d, sizeof(int32_t) * (batch_size + 1),
                        cudaMemcpyDeviceToHost, s);
    if (e != cudaSuccess) return bp::translate(e);
    if ((e = cudaStreamSynchronize(s)) != cudaSuccess) return bp::translate(e);
    (void)total_kv_rows;

    // Ragged has no pages — page_size = 1 for the scheduler's kv chunking.
    bp::PlanResult plan = bp::run_plan(
        q_indptr_h, kv_indptr_h, total_num_rows, batch_size, num_qo_heads, num_kv_heads, HEAD_DIM,
        /*page_size=*/1, sizeof(DType), /*disable_split_kv=*/enable_split == 0, s);
    if (!plan.ok) return STATUS_INVALID_ARG;

    using Params = flashinfer::BatchPrefillRaggedParams<DType, DType, DType, int32_t>;
    Params params(
        reinterpret_cast<DType*>(const_cast<void*>(q)), reinterpret_cast<DType*>(const_cast<void*>(k_data)),
        reinterpret_cast<DType*>(const_cast<void*>(v_data)), /*maybe_custom_mask=*/nullptr,
        q_indptr_d, kv_indptr_d, /*maybe_mask_indptr=*/nullptr, /*maybe_q_rope_offset=*/nullptr,
        /*maybe_k_rope_offset=*/nullptr, reinterpret_cast<DType*>(o), reinterpret_cast<float*>(lse),
        /*maybe_alibi_slopes=*/nullptr, static_cast<uint32_t>(num_qo_heads),
        static_cast<uint32_t>(num_kv_heads),
        /*q_stride_n=*/static_cast<uint32_t>(num_qo_heads) * HEAD_DIM, /*q_stride_h=*/HEAD_DIM,
        /*kv_stride_n=*/static_cast<uint32_t>(num_kv_heads) * HEAD_DIM, /*kv_stride_h=*/HEAD_DIM,
        /*window_left=*/-1, /*logits_soft_cap=*/0.0f, sm_scale, /*rope_scale=*/1.0f,
        /*rope_theta=*/10000.0f);

    DType* tmp_v = nullptr;
    float* tmp_s = nullptr;
    bp::set_plan_fields<Params, DType>(params, plan.info, plan.int_buf, plan.float_buf,
                                       total_num_rows, &tmp_v, &tmp_s);

    using Variant = flashinfer::DefaultAttention<false, false, false, false>;
    constexpr auto POS = flashinfer::PosEncodingMode::kNone;
    if (causal != 0) {
        DISPATCH_CTA_TILE_Q(plan.info.cta_tile_q, CTA_TILE_Q, {
            e = flashinfer::BatchPrefillWithRaggedKVCacheDispatched<
                CTA_TILE_Q, HEAD_DIM, HEAD_DIM, POS, false, flashinfer::MaskMode::kCausal, Variant,
                Params>(params, tmp_v, tmp_s, /*enable_pdl=*/false, s);
        });
    } else {
        DISPATCH_CTA_TILE_Q(plan.info.cta_tile_q, CTA_TILE_Q, {
            e = flashinfer::BatchPrefillWithRaggedKVCacheDispatched<
                CTA_TILE_Q, HEAD_DIM, HEAD_DIM, POS, false, flashinfer::MaskMode::kNone, Variant,
                Params>(params, tmp_v, tmp_s, /*enable_pdl=*/false, s);
        });
    }

    cudaError_t sync_e = cudaStreamSynchronize(s);
    bp::plan_cleanup(plan, s);
    if (e != cudaSuccess) return bp::translate(e);
    return bp::translate(sync_e);
}

template <typename DType>
int dispatch_head_dim(
    int32_t batch_size, int32_t total_num_rows, int32_t total_kv_rows, int32_t head_dim,
    int32_t num_qo_heads, int32_t num_kv_heads, float sm_scale, int32_t causal, int32_t enable_split,
    const void* k_data, const void* v_data, int32_t* kv_indptr_d, const void* q, int32_t* q_indptr_d,
    void* o, void* lse, void* stream)
{
    if (head_dim == 64)
        return run_ragged_prefill_typed<DType, 64>(batch_size, total_num_rows, total_kv_rows,
            num_qo_heads, num_kv_heads, sm_scale, causal, enable_split, k_data, v_data, kv_indptr_d,
            q, q_indptr_d, o, lse, stream);
    if (head_dim == 128)
        return run_ragged_prefill_typed<DType, 128>(batch_size, total_num_rows, total_kv_rows,
            num_qo_heads, num_kv_heads, sm_scale, causal, enable_split, k_data, v_data, kv_indptr_d,
            q, q_indptr_d, o, lse, stream);
    if (head_dim == 256)
        return run_ragged_prefill_typed<DType, 256>(batch_size, total_num_rows, total_kv_rows,
            num_qo_heads, num_kv_heads, sm_scale, causal, enable_split, k_data, v_data, kv_indptr_d,
            q, q_indptr_d, o, lse, stream);
    return STATUS_UNSUPPORTED;
}

inline int validate(int32_t batch_size, int32_t total_num_rows, int32_t total_kv_rows,
                    int32_t head_dim, int32_t num_qo_heads, int32_t num_kv_heads, const void* q,
                    const void* o, const void* lse, const void* k, const void* v, const void* kiptr,
                    const void* qiptr) {
    if (batch_size <= 0 || total_num_rows <= 0 || total_kv_rows <= 0 || num_qo_heads <= 0 ||
        num_kv_heads <= 0)
        return STATUS_INVALID_ARG;
    if (num_qo_heads % num_kv_heads != 0) return STATUS_INVALID_ARG;
    if (!bp::head_dim_supported(head_dim)) return STATUS_UNSUPPORTED;
    if (!q || !o || !lse || !k || !v || !kiptr || !qiptr) return STATUS_INVALID_ARG;
    return STATUS_OK;
}
}  // namespace

extern "C" {

#define BARACUDA_FI_RAGGED_PREFILL(NAME, DTYPE)                                                  \
    int NAME(int32_t batch_size, int32_t total_num_rows, int32_t total_kv_rows, int32_t head_dim,\
             int32_t num_qo_heads, int32_t num_kv_heads, float sm_scale, int32_t causal,         \
             int32_t enable_split, const void* k_data, const void* v_data, void* kv_indptr,      \
             const void* q, void* q_indptr, void* o, void* lse, void* stream) {                  \
        int v = validate(batch_size, total_num_rows, total_kv_rows, head_dim, num_qo_heads,      \
                         num_kv_heads, q, o, lse, k_data, v_data, kv_indptr, q_indptr);          \
        if (v != STATUS_OK) return v;                                                            \
        return dispatch_head_dim<DTYPE>(                                                         \
            batch_size, total_num_rows, total_kv_rows, head_dim, num_qo_heads, num_kv_heads,     \
            sm_scale, causal, enable_split, k_data, v_data,                                      \
            reinterpret_cast<int32_t*>(kv_indptr), q, reinterpret_cast<int32_t*>(q_indptr),      \
            o, lse, stream);                                                                     \
    }

BARACUDA_FI_RAGGED_PREFILL(baracuda_kernels_flashinfer_ragged_prefill_f16_run, __half)
BARACUDA_FI_RAGGED_PREFILL(baracuda_kernels_flashinfer_ragged_prefill_bf16_run, __nv_bfloat16)

#undef BARACUDA_FI_RAGGED_PREFILL

int baracuda_kernels_flashinfer_ragged_prefill_can_implement(
    int32_t batch_size, int32_t total_num_rows, int32_t total_kv_rows, int32_t head_dim,
    int32_t num_qo_heads, int32_t num_kv_heads)
{
    if (batch_size <= 0 || total_num_rows <= 0 || total_kv_rows <= 0 || num_qo_heads <= 0 ||
        num_kv_heads <= 0)
        return STATUS_INVALID_ARG;
    if (num_qo_heads % num_kv_heads != 0) return STATUS_INVALID_ARG;
    if (!bp::head_dim_supported(head_dim)) return STATUS_UNSUPPORTED;
    return STATUS_OK;
}

}  // extern "C"
