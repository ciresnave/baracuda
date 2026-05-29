// baracuda-kernels Phase 59b — Dao-AILab FlashAttention v2 VARLEN FW launcher.
//
// FA2 v2.8.3's FW and BW kernels handle varlen via a runtime
// `params.cu_seqlens_q / cu_seqlens_k != nullptr` switch — there is NO
// separate "varlen" .cu file family. The same instantiations the dense
// FW/BW launchers call do double duty.
//
// This TU plumbs the varlen FW path through baracuda's C-ABI surface:
//   - Q layout : [total_q, h,   d]    packed across batch dim
//   - K layout : [total_k, h_k, d]    packed
//   - V layout : [total_k, h_k, d]    packed
//   - O layout : [total_q, h,   d]    packed
//   - lse      : [h, total_q + 128*b] f32, padded format (matches FA2's
//                "unpadded_lse" convention for the BW pipeline).
//   - cu_seqlens_q : i32[b+1]   cumulative; cu_seqlens_q[0]=0, [b]=total_q
//   - cu_seqlens_k : i32[b+1]   same convention for K.
//
// The varlen BW path is defined in `fa2_backward_launcher.cu` because
// it shares the bwd dispatch table forward-declared there.

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cutlass/numeric_types.h>

#include "../../vendor/flash-attention/src/namespace_config.h"
#include "../../vendor/flash-attention/src/flash.h"

namespace FLASH_NAMESPACE {
// Forward declarations of the same FW instantiations the dense launcher
// uses. Linkage-deduplicated by the linker (one definition wins).
template<> void run_mha_fwd_<cutlass::half_t,     32,  false>(Flash_fwd_params&, cudaStream_t);
template<> void run_mha_fwd_<cutlass::half_t,     32,  true >(Flash_fwd_params&, cudaStream_t);
template<> void run_mha_fwd_<cutlass::half_t,     64,  false>(Flash_fwd_params&, cudaStream_t);
template<> void run_mha_fwd_<cutlass::half_t,     64,  true >(Flash_fwd_params&, cudaStream_t);
template<> void run_mha_fwd_<cutlass::half_t,     96,  false>(Flash_fwd_params&, cudaStream_t);
template<> void run_mha_fwd_<cutlass::half_t,     96,  true >(Flash_fwd_params&, cudaStream_t);
template<> void run_mha_fwd_<cutlass::half_t,     128, false>(Flash_fwd_params&, cudaStream_t);
template<> void run_mha_fwd_<cutlass::half_t,     128, true >(Flash_fwd_params&, cudaStream_t);
template<> void run_mha_fwd_<cutlass::half_t,     192, false>(Flash_fwd_params&, cudaStream_t);
template<> void run_mha_fwd_<cutlass::half_t,     192, true >(Flash_fwd_params&, cudaStream_t);
template<> void run_mha_fwd_<cutlass::half_t,     256, false>(Flash_fwd_params&, cudaStream_t);
template<> void run_mha_fwd_<cutlass::half_t,     256, true >(Flash_fwd_params&, cudaStream_t);

template<> void run_mha_fwd_<cutlass::bfloat16_t, 32,  false>(Flash_fwd_params&, cudaStream_t);
template<> void run_mha_fwd_<cutlass::bfloat16_t, 32,  true >(Flash_fwd_params&, cudaStream_t);
template<> void run_mha_fwd_<cutlass::bfloat16_t, 64,  false>(Flash_fwd_params&, cudaStream_t);
template<> void run_mha_fwd_<cutlass::bfloat16_t, 64,  true >(Flash_fwd_params&, cudaStream_t);
template<> void run_mha_fwd_<cutlass::bfloat16_t, 96,  false>(Flash_fwd_params&, cudaStream_t);
template<> void run_mha_fwd_<cutlass::bfloat16_t, 96,  true >(Flash_fwd_params&, cudaStream_t);
template<> void run_mha_fwd_<cutlass::bfloat16_t, 128, false>(Flash_fwd_params&, cudaStream_t);
template<> void run_mha_fwd_<cutlass::bfloat16_t, 128, true >(Flash_fwd_params&, cudaStream_t);
template<> void run_mha_fwd_<cutlass::bfloat16_t, 192, false>(Flash_fwd_params&, cudaStream_t);
template<> void run_mha_fwd_<cutlass::bfloat16_t, 192, true >(Flash_fwd_params&, cudaStream_t);
template<> void run_mha_fwd_<cutlass::bfloat16_t, 256, false>(Flash_fwd_params&, cudaStream_t);
template<> void run_mha_fwd_<cutlass::bfloat16_t, 256, true >(Flash_fwd_params&, cudaStream_t);
}  // namespace FLASH_NAMESPACE

namespace {

constexpr int STATUS_OK          = 0;
constexpr int STATUS_INVALID_ARG = 2;
constexpr int STATUS_UNSUPPORTED = 3;

inline bool is_supported_hdim(int d) {
    return d == 32 || d == 64 || d == 96 || d == 128 || d == 192 || d == 256;
}

inline int seq_rounded_128(int s) {
    return ((s + 127) / 128) * 128;
}

inline int head_size_rounded(int d) {
    int m = (d <= 128) ? 32 : 64;
    return ((d + m - 1) / m) * m;
}

// Fill a Flash_fwd_params for the varlen case (packed Q/K/V/O).
//
// Layouts match upstream FA2 mha_varlen_fwd:
//   Q / O : [total_q, h, d]   row_stride=d*h, head_stride=d, batch_stride=0
//   K / V : [total_k, h_k, d] row_stride=d*h_k, head_stride=d, batch_stride=0
//   lse   : [h, total_q + 128*b] f32, "unpadded" format consumed by BW.
void fill_varlen_fwd_params(
    FLASH_NAMESPACE::Flash_fwd_params& p,
    int b, int h, int h_k,
    int max_seqlen_q, int max_seqlen_k,
    int total_q, int total_k,
    int d,
    float softmax_scale, bool is_causal, bool is_bf16,
    const void* q, const void* k, const void* v,
    void* o,
    void* softmax_lse,
    const int* cu_seqlens_q,
    const int* cu_seqlens_k,
    const void* alibi_slopes_ptr,
    int alibi_batch_stride,
    int window_size_left,
    int window_size_right,
    float softcap)
{
    std::memset(&p, 0, sizeof(p));

    (void)total_k;

    p.q_ptr = const_cast<void*>(q);
    p.k_ptr = const_cast<void*>(k);
    p.v_ptr = const_cast<void*>(v);

    // Packed (varlen) stride convention.
    p.q_batch_stride = 0;
    p.k_batch_stride = 0;
    p.v_batch_stride = 0;
    p.q_head_stride  = int64_t(d);
    p.k_head_stride  = int64_t(d);
    p.v_head_stride  = int64_t(d);
    p.q_row_stride   = int64_t(d) * h;
    p.k_row_stride   = int64_t(d) * h_k;
    p.v_row_stride   = int64_t(d) * h_k;

    p.h           = h;
    p.h_k         = h_k;
    p.h_h_k_ratio = h / h_k;

    p.o_ptr            = o;
    p.o_batch_stride   = 0;
    p.o_head_stride    = int64_t(d);
    p.o_row_stride     = int64_t(d) * h;

    p.b                = b;
    p.seqlen_q         = max_seqlen_q;
    p.seqlen_k         = max_seqlen_k;
    p.d                = d;
    p.d_rounded        = head_size_rounded(d);
    p.seqlen_q_rounded = seq_rounded_128(max_seqlen_q);
    p.seqlen_k_rounded = seq_rounded_128(max_seqlen_k);
    p.total_q          = total_q;

    p.scale_softmax       = softmax_scale;
    p.scale_softmax_log2  = softmax_scale * float(1.4426950408889634);

    // Varlen activation.
    p.cu_seqlens_q   = const_cast<int*>(cu_seqlens_q);
    p.cu_seqlens_k   = const_cast<int*>(cu_seqlens_k);
    p.leftpad_k      = nullptr;
    p.seqused_k      = nullptr;
    p.knew_ptr       = nullptr;
    p.vnew_ptr       = nullptr;
    p.rotary_cos_ptr = nullptr;
    p.rotary_sin_ptr = nullptr;
    p.cache_batch_idx= nullptr;
    p.block_table    = nullptr;

    p.alibi_slopes_ptr           = const_cast<void*>(alibi_slopes_ptr);
    p.alibi_slopes_batch_stride  = int64_t(alibi_batch_stride);

    p.p_dropout                 = 1.0f;
    p.p_dropout_in_uint8_t      = 255;
    p.rp_dropout                = 1.0f;
    p.scale_softmax_rp_dropout  = softmax_scale;

    p.softcap            = softcap;
    p.window_size_left   = window_size_left;
    p.window_size_right  = window_size_right;
    if (is_causal) {
        p.window_size_right = 0;
    }

    p.p_ptr = nullptr;
    p.softmax_lse_ptr = softmax_lse;
    p.softmax_lseaccum_ptr = nullptr;
    p.rng_state = nullptr;

    p.is_bf16                  = is_bf16;
    p.is_causal                = is_causal;
    p.is_seqlens_k_cumulative  = true;
    p.is_rotary_interleaved    = false;
    p.unpadded_lse             = true;  // varlen: LSE packed [h, total_q+128*b]
    p.seqlenq_ngroups_swapped  = false;
    p.num_splits               = 1;
}

template <typename T>
bool dispatch_fwd(int head_dim, bool is_causal,
                  FLASH_NAMESPACE::Flash_fwd_params& params, cudaStream_t s) {
    if (is_causal) {
        switch (head_dim) {
            case 32:  FLASH_NAMESPACE::run_mha_fwd_<T, 32,  true>(params, s); return true;
            case 64:  FLASH_NAMESPACE::run_mha_fwd_<T, 64,  true>(params, s); return true;
            case 96:  FLASH_NAMESPACE::run_mha_fwd_<T, 96,  true>(params, s); return true;
            case 128: FLASH_NAMESPACE::run_mha_fwd_<T, 128, true>(params, s); return true;
            case 192: FLASH_NAMESPACE::run_mha_fwd_<T, 192, true>(params, s); return true;
            case 256: FLASH_NAMESPACE::run_mha_fwd_<T, 256, true>(params, s); return true;
            default:  return false;
        }
    } else {
        switch (head_dim) {
            case 32:  FLASH_NAMESPACE::run_mha_fwd_<T, 32,  false>(params, s); return true;
            case 64:  FLASH_NAMESPACE::run_mha_fwd_<T, 64,  false>(params, s); return true;
            case 96:  FLASH_NAMESPACE::run_mha_fwd_<T, 96,  false>(params, s); return true;
            case 128: FLASH_NAMESPACE::run_mha_fwd_<T, 128, false>(params, s); return true;
            case 192: FLASH_NAMESPACE::run_mha_fwd_<T, 192, false>(params, s); return true;
            case 256: FLASH_NAMESPACE::run_mha_fwd_<T, 256, false>(params, s); return true;
            default:  return false;
        }
    }
}

template <typename T>
int run_varlen_fwd_impl(
    int32_t batch, int32_t num_heads, int32_t num_heads_k,
    int32_t max_seqlen_q, int32_t max_seqlen_k,
    int32_t total_q, int32_t total_k,
    int32_t head_dim,
    float softmax_scale, int32_t is_causal,
    const void* alibi_slopes_ptr, int32_t alibi_batch_stride,
    int32_t window_size_left, int32_t window_size_right, float softcap,
    const void* q, const void* k, const void* v,
    const int* cu_seqlens_q, const int* cu_seqlens_k,
    void* out, void* softmax_lse,
    bool is_bf16, void* stream)
{
    if (!is_supported_hdim(head_dim)) return STATUS_UNSUPPORTED;
    if (batch <= 0 || num_heads <= 0 || num_heads_k <= 0 ||
        max_seqlen_q <= 0 || max_seqlen_k <= 0 ||
        total_q <= 0 || total_k <= 0) return STATUS_INVALID_ARG;
    if (num_heads_k > num_heads) return STATUS_INVALID_ARG;
    if (num_heads % num_heads_k != 0) return STATUS_INVALID_ARG;
    if (!q || !k || !v || !out || !softmax_lse ||
        !cu_seqlens_q || !cu_seqlens_k) return STATUS_INVALID_ARG;
    if (softcap < 0.0f) return STATUS_INVALID_ARG;

    FLASH_NAMESPACE::Flash_fwd_params params{};
    fill_varlen_fwd_params(
        params, batch, num_heads, num_heads_k,
        max_seqlen_q, max_seqlen_k, total_q, total_k, head_dim,
        softmax_scale, is_causal != 0, is_bf16,
        q, k, v, out, softmax_lse,
        cu_seqlens_q, cu_seqlens_k,
        alibi_slopes_ptr, alibi_batch_stride,
        window_size_left, window_size_right, softcap);

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    if (!dispatch_fwd<T>(head_dim, is_causal != 0, params, s)) {
        return STATUS_UNSUPPORTED;
    }
    cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? STATUS_OK : STATUS_INVALID_ARG;
}

}  // namespace

extern "C" {

int baracuda_kernels_fa2_sdpa_varlen_f16_run(
    int32_t batch, int32_t num_heads, int32_t num_heads_k,
    int32_t max_seqlen_q, int32_t max_seqlen_k,
    int32_t total_q, int32_t total_k,
    int32_t head_dim,
    float softmax_scale, int32_t is_causal,
    const void* alibi_slopes_ptr, int32_t alibi_batch_stride,
    int32_t window_size_left, int32_t window_size_right, float softcap,
    const void* q, const void* k, const void* v,
    const int* cu_seqlens_q, const int* cu_seqlens_k,
    void* out, void* softmax_lse,
    void* /*workspace*/, std::size_t /*workspace_bytes*/, void* stream)
{
    return run_varlen_fwd_impl<cutlass::half_t>(
        batch, num_heads, num_heads_k,
        max_seqlen_q, max_seqlen_k, total_q, total_k, head_dim,
        softmax_scale, is_causal,
        alibi_slopes_ptr, alibi_batch_stride,
        window_size_left, window_size_right, softcap,
        q, k, v, cu_seqlens_q, cu_seqlens_k,
        out, softmax_lse,
        /*is_bf16=*/false, stream);
}

int baracuda_kernels_fa2_sdpa_varlen_bf16_run(
    int32_t batch, int32_t num_heads, int32_t num_heads_k,
    int32_t max_seqlen_q, int32_t max_seqlen_k,
    int32_t total_q, int32_t total_k,
    int32_t head_dim,
    float softmax_scale, int32_t is_causal,
    const void* alibi_slopes_ptr, int32_t alibi_batch_stride,
    int32_t window_size_left, int32_t window_size_right, float softcap,
    const void* q, const void* k, const void* v,
    const int* cu_seqlens_q, const int* cu_seqlens_k,
    void* out, void* softmax_lse,
    void* /*workspace*/, std::size_t /*workspace_bytes*/, void* stream)
{
    return run_varlen_fwd_impl<cutlass::bfloat16_t>(
        batch, num_heads, num_heads_k,
        max_seqlen_q, max_seqlen_k, total_q, total_k, head_dim,
        softmax_scale, is_causal,
        alibi_slopes_ptr, alibi_batch_stride,
        window_size_left, window_size_right, softcap,
        q, k, v, cu_seqlens_q, cu_seqlens_k,
        out, softmax_lse,
        /*is_bf16=*/true, stream);
}

int baracuda_kernels_fa2_sdpa_varlen_f16_can_implement(
    int32_t batch, int32_t num_heads, int32_t num_heads_k,
    int32_t max_seqlen_q, int32_t max_seqlen_k,
    int32_t total_q, int32_t total_k,
    int32_t head_dim, int32_t /*is_causal*/,
    int32_t /*window_size_left*/, int32_t /*window_size_right*/,
    float softcap)
{
    if (!is_supported_hdim(head_dim)) return STATUS_UNSUPPORTED;
    if (batch <= 0 || num_heads <= 0 || num_heads_k <= 0 ||
        max_seqlen_q <= 0 || max_seqlen_k <= 0 ||
        total_q <= 0 || total_k <= 0) return STATUS_INVALID_ARG;
    if (num_heads_k > num_heads) return STATUS_INVALID_ARG;
    if (num_heads % num_heads_k != 0) return STATUS_INVALID_ARG;
    if (softcap < 0.0f) return STATUS_INVALID_ARG;
    return STATUS_OK;
}

int baracuda_kernels_fa2_sdpa_varlen_bf16_can_implement(
    int32_t batch, int32_t num_heads, int32_t num_heads_k,
    int32_t max_seqlen_q, int32_t max_seqlen_k,
    int32_t total_q, int32_t total_k,
    int32_t head_dim, int32_t is_causal,
    int32_t window_size_left, int32_t window_size_right, float softcap)
{
    return baracuda_kernels_fa2_sdpa_varlen_f16_can_implement(
        batch, num_heads, num_heads_k, max_seqlen_q, max_seqlen_k,
        total_q, total_k, head_dim, is_causal,
        window_size_left, window_size_right, softcap);
}

// Varlen FW LSE size in f32 elements: h * (total_q + 128 * batch).
// Caller multiplies by 4 for bytes.
std::size_t baracuda_kernels_fa2_sdpa_varlen_lse_size(
    int32_t batch, int32_t num_heads, int32_t total_q)
{
    if (batch <= 0 || num_heads <= 0 || total_q <= 0) return 0;
    return (std::size_t)num_heads * ((std::size_t)total_q + 128u * (std::size_t)batch);
}

}  // extern "C"
