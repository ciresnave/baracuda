// baracuda-kernels Phase 59b — Dao-AILab FlashAttention v2 backward launcher.
//
// Bridges baracuda's `extern "C"` FFI surface to FA2 v2.8.3's
// `run_mha_bwd_<T, headdim, is_causal>` template instantiations shipped
// from the vendored .cu files under
// `vendor/flash-attention/src/flash_bwd_hdim*_sm80.cu`.
//
// Same dispatch pattern as `fa2_launcher.cu`:
//   - head_dim ∈ {32, 64, 96, 128, 192, 256}, runtime switch.
//   - dtype ∈ {fp16, bf16}, separate FFI entry per dtype.
//   - causal / non-causal switch.
//
// Phase 59a-compatible feature plumbing on top of base BW:
//   - ALiBi slopes
//   - sliding window
//   - softcap
//
// BW-specific scratch contract:
//   FA2's BW kernels need TWO f32 scratch buffers, both pre-zeroed by
//   the caller (or by the wrapper if `clear_scratch=true`):
//
//     dq_accum   — shape [B, seqlen_q_rounded, H, head_size_rounded], f32
//                  Total: B * Sq_rounded * H * Hd_rounded * 4 bytes.
//     dsoftmax_d — shape [B, H, seqlen_q_rounded], f32
//                  Total: B * H * Sq_rounded * 4 bytes.
//
//   Where:
//     seqlen_q_rounded  = round_up(seqlen_q, 128)
//     head_size_rounded = round_up(d, d <= 128 ? 32 : 64)
//
//   FA2 dispatches dq_accum atomics across blocks (deterministic=false)
//   then a final convert kernel writes to dq. The caller-visible dq
//   tensor receives the converted result in the operand dtype.
//
//   The `softmax_d` buffer accumulates D = rowsum(dy . y) values in f32.
//
// LSE contract:
//   The `lse` input MUST be the f32 LSE written by the FA2 FW pass
//   (`softmax_lse` arg of `fa2_sdpa_*_run_v2`). Reusing baracuda's
//   bespoke FlashSdpa LSE (typed T) is INVALID — FA2 always stores LSE
//   in f32 regardless of element type.
//
// Status codes (match baracuda-kernels::attention::map_status):
//   0 = ok, 2 = invalid_problem, 3 = unsupported.

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cutlass/numeric_types.h>

#include "../../vendor/flash-attention/src/namespace_config.h"
#include "../../vendor/flash-attention/src/flash.h"

namespace FLASH_NAMESPACE {
// Explicit template instantiations emitted by the vendored BW .cu files.
// Same forward-declare pattern as fa2_launcher.cu — avoids dragging
// flash_bwd_launch_template.h (and the heavy CUTLASS templates) into
// this translation unit.
template<> void run_mha_bwd_<cutlass::half_t,     32,  false>(Flash_bwd_params&, cudaStream_t);
template<> void run_mha_bwd_<cutlass::half_t,     32,  true >(Flash_bwd_params&, cudaStream_t);
template<> void run_mha_bwd_<cutlass::half_t,     64,  false>(Flash_bwd_params&, cudaStream_t);
template<> void run_mha_bwd_<cutlass::half_t,     64,  true >(Flash_bwd_params&, cudaStream_t);
template<> void run_mha_bwd_<cutlass::half_t,     96,  false>(Flash_bwd_params&, cudaStream_t);
template<> void run_mha_bwd_<cutlass::half_t,     96,  true >(Flash_bwd_params&, cudaStream_t);
template<> void run_mha_bwd_<cutlass::half_t,     128, false>(Flash_bwd_params&, cudaStream_t);
template<> void run_mha_bwd_<cutlass::half_t,     128, true >(Flash_bwd_params&, cudaStream_t);
template<> void run_mha_bwd_<cutlass::half_t,     192, false>(Flash_bwd_params&, cudaStream_t);
template<> void run_mha_bwd_<cutlass::half_t,     192, true >(Flash_bwd_params&, cudaStream_t);
template<> void run_mha_bwd_<cutlass::half_t,     256, false>(Flash_bwd_params&, cudaStream_t);
template<> void run_mha_bwd_<cutlass::half_t,     256, true >(Flash_bwd_params&, cudaStream_t);

template<> void run_mha_bwd_<cutlass::bfloat16_t, 32,  false>(Flash_bwd_params&, cudaStream_t);
template<> void run_mha_bwd_<cutlass::bfloat16_t, 32,  true >(Flash_bwd_params&, cudaStream_t);
template<> void run_mha_bwd_<cutlass::bfloat16_t, 64,  false>(Flash_bwd_params&, cudaStream_t);
template<> void run_mha_bwd_<cutlass::bfloat16_t, 64,  true >(Flash_bwd_params&, cudaStream_t);
template<> void run_mha_bwd_<cutlass::bfloat16_t, 96,  false>(Flash_bwd_params&, cudaStream_t);
template<> void run_mha_bwd_<cutlass::bfloat16_t, 96,  true >(Flash_bwd_params&, cudaStream_t);
template<> void run_mha_bwd_<cutlass::bfloat16_t, 128, false>(Flash_bwd_params&, cudaStream_t);
template<> void run_mha_bwd_<cutlass::bfloat16_t, 128, true >(Flash_bwd_params&, cudaStream_t);
template<> void run_mha_bwd_<cutlass::bfloat16_t, 192, false>(Flash_bwd_params&, cudaStream_t);
template<> void run_mha_bwd_<cutlass::bfloat16_t, 192, true >(Flash_bwd_params&, cudaStream_t);
template<> void run_mha_bwd_<cutlass::bfloat16_t, 256, false>(Flash_bwd_params&, cudaStream_t);
template<> void run_mha_bwd_<cutlass::bfloat16_t, 256, true >(Flash_bwd_params&, cudaStream_t);
}  // namespace FLASH_NAMESPACE

namespace {

constexpr int STATUS_OK          = 0;
constexpr int STATUS_INVALID_ARG = 2;
constexpr int STATUS_UNSUPPORTED = 3;

inline bool is_supported_hdim(int d) {
    return d == 32 || d == 64 || d == 96 || d == 128 || d == 192 || d == 256;
}

// FA2's BW preprocess writes dQaccum at stride `H * head_size_rounded`.
// Match upstream's `round_multiple(head_size, head_size <= 128 ? 32 : 64)`.
inline int head_size_rounded(int d) {
    int m = (d <= 128) ? 32 : 64;
    return ((d + m - 1) / m) * m;
}

inline int seq_rounded_128(int s) {
    return ((s + 127) / 128) * 128;
}

// Fill a Flash_bwd_params for the dense case. Mirrors fa2_launcher.cu's
// fill_dense_params but extends to BW-specific fields.
//
// dq_accum_ptr / dsoftmax_sum_ptr come from caller-supplied workspace;
// the wrapper is responsible for zeroing them before launch (FA2's
// preprocess pass is meant to overwrite, but the deterministic-path
// dq_accum convert kernel reads the previous value — we keep
// deterministic=false in Phase 59b, matching FA2's default).
void fill_bwd_dense_params(
    FLASH_NAMESPACE::Flash_bwd_params& p,
    int b, int h, int h_k, int sq, int sk, int d,
    float softmax_scale, bool is_causal, bool is_bf16,
    const void* q, const void* k, const void* v, const void* o,
    const void* dout,
    void* dq, void* dk, void* dv,
    void* lse,                     // f32
    void* dq_accum, void* dsoftmax_sum,
    const void* alibi_slopes_ptr,
    int alibi_batch_stride,
    int window_size_left,
    int window_size_right,
    float softcap)
{
    std::memset(&p, 0, sizeof(p));

    // ---- Fwd-shape carry-over ----
    // FW pointers — BW reads Q/K/V/O to recompute on the fly.
    p.q_ptr = const_cast<void*>(q);
    p.k_ptr = const_cast<void*>(k);
    p.v_ptr = const_cast<void*>(v);
    p.o_ptr = const_cast<void*>(o);

    p.q_batch_stride = int64_t(h)   * sq * d;
    p.k_batch_stride = int64_t(h_k) * sk * d;
    p.v_batch_stride = int64_t(h_k) * sk * d;
    p.o_batch_stride = int64_t(h)   * sq * d;
    p.q_head_stride  = int64_t(sq) * d;
    p.k_head_stride  = int64_t(sk) * d;
    p.v_head_stride  = int64_t(sk) * d;
    p.o_head_stride  = int64_t(sq) * d;
    p.q_row_stride   = int64_t(d);
    p.k_row_stride   = int64_t(d);
    p.v_row_stride   = int64_t(d);
    p.o_row_stride   = int64_t(d);

    p.h           = h;
    p.h_k         = h_k;
    p.h_h_k_ratio = h / h_k;

    p.b                = b;
    p.seqlen_q         = sq;
    p.seqlen_k         = sk;
    p.d                = d;
    p.d_rounded        = head_size_rounded(d);
    p.seqlen_q_rounded = seq_rounded_128(sq);
    p.seqlen_k_rounded = seq_rounded_128(sk);
    p.total_q          = sq;

    p.scale_softmax       = softmax_scale;
    p.scale_softmax_log2  = softmax_scale * float(1.4426950408889634);

    // No varlen, no leftpad, no KV append, no rotary, no paged.
    p.cu_seqlens_q   = nullptr;
    p.cu_seqlens_k   = nullptr;
    p.leftpad_k      = nullptr;
    p.seqused_k      = nullptr;
    p.knew_ptr       = nullptr;
    p.vnew_ptr       = nullptr;
    p.rotary_cos_ptr = nullptr;
    p.rotary_sin_ptr = nullptr;
    p.cache_batch_idx= nullptr;
    p.block_table    = nullptr;

    // ALiBi.
    p.alibi_slopes_ptr           = const_cast<void*>(alibi_slopes_ptr);
    p.alibi_slopes_batch_stride  = int64_t(alibi_batch_stride);

    // No dropout — FA2 treats p_dropout = 1.0 as "keep everything".
    p.p_dropout                 = 1.0f;
    p.p_dropout_in_uint8_t      = 255;
    p.rp_dropout                = 1.0f;
    p.scale_softmax_rp_dropout  = softmax_scale;

    // Softcap + sliding window.
    p.softcap            = softcap;
    p.window_size_left   = window_size_left;
    p.window_size_right  = window_size_right;
    if (is_causal) {
        p.window_size_right = 0;
    }

    // No saved softmax-probs buffer.
    p.p_ptr = nullptr;

    // LSE — f32, FW-saved.
    p.softmax_lse_ptr       = lse;
    p.softmax_lseaccum_ptr  = nullptr;

    p.rng_state = nullptr;

    p.is_bf16                  = is_bf16;
    p.is_causal                = is_causal;
    p.is_seqlens_k_cumulative  = true;
    p.is_rotary_interleaved    = false;
    p.unpadded_lse             = false;
    p.seqlenq_ngroups_swapped  = false;
    p.num_splits               = 1;

    // ---- Bwd-specific ----
    p.do_ptr = const_cast<void*>(dout);
    p.dq_ptr = dq;
    p.dk_ptr = dk;
    p.dv_ptr = dv;

    p.do_batch_stride = int64_t(h) * sq * d;
    p.do_head_stride  = int64_t(sq) * d;
    p.do_row_stride   = int64_t(d);
    p.dq_batch_stride = int64_t(h) * sq * d;
    p.dq_head_stride  = int64_t(sq) * d;
    p.dq_row_stride   = int64_t(d);
    p.dk_batch_stride = int64_t(h_k) * sk * d;
    p.dk_head_stride  = int64_t(sk) * d;
    p.dk_row_stride   = int64_t(d);
    p.dv_batch_stride = int64_t(h_k) * sk * d;
    p.dv_head_stride  = int64_t(sk) * d;
    p.dv_row_stride   = int64_t(d);

    p.dq_accum_ptr = dq_accum;
    p.dk_accum_ptr = nullptr;
    p.dv_accum_ptr = nullptr;
    p.dsoftmax_sum = dsoftmax_sum;

    p.deterministic = false;
    p.dq_accum_split_stride = 0;
}

// Fill BW params for varlen — total_q / total_k packed, cu_seqlens_*
// indexing into the pack.
//
// Stride contract for packed tensors (matches upstream FA2 mha_varlen_bwd):
//   Q / dQ / O / dO : batch_stride ignored; row_stride = d * h ;
//                     head_stride = d. Layout [total_q, h, d].
//   K / dK / V / dV : batch_stride ignored; row_stride = d * h_k ;
//                     head_stride = d. Layout [total_k, h_k, d].
void fill_bwd_varlen_params(
    FLASH_NAMESPACE::Flash_bwd_params& p,
    int b, int h, int h_k,
    int max_seqlen_q, int max_seqlen_k,
    int total_q, int total_k,
    int d,
    float softmax_scale, bool is_causal, bool is_bf16,
    const void* q, const void* k, const void* v, const void* o,
    const void* dout,
    void* dq, void* dk, void* dv,
    void* lse,                          // f32 — packed [h, total_q + 128*b]
    void* dq_accum, void* dsoftmax_sum, // f32 scratch
    const int* cu_seqlens_q,            // i32[b+1]
    const int* cu_seqlens_k,            // i32[b+1]
    const void* alibi_slopes_ptr,
    int alibi_batch_stride,
    int window_size_left,
    int window_size_right,
    float softcap)
{
    std::memset(&p, 0, sizeof(p));

    // FW-shape carry-over (packed layout).
    p.q_ptr = const_cast<void*>(q);
    p.k_ptr = const_cast<void*>(k);
    p.v_ptr = const_cast<void*>(v);
    p.o_ptr = const_cast<void*>(o);

    // For varlen, FA2 ignores batch_stride and uses row/head strides.
    // Row stride = elements per row in the packed [total, h, d] layout.
    p.q_batch_stride = 0;
    p.k_batch_stride = 0;
    p.v_batch_stride = 0;
    p.o_batch_stride = 0;
    p.q_head_stride  = int64_t(d);
    p.k_head_stride  = int64_t(d);
    p.v_head_stride  = int64_t(d);
    p.o_head_stride  = int64_t(d);
    p.q_row_stride   = int64_t(d) * h;
    p.k_row_stride   = int64_t(d) * h_k;
    p.v_row_stride   = int64_t(d) * h_k;
    p.o_row_stride   = int64_t(d) * h;

    p.h           = h;
    p.h_k         = h_k;
    p.h_h_k_ratio = h / h_k;

    p.b                = b;
    p.seqlen_q         = max_seqlen_q;
    p.seqlen_k         = max_seqlen_k;
    p.d                = d;
    p.d_rounded        = head_size_rounded(d);
    // Per upstream mha_varlen_bwd: softmax_d / dq_accum are sized off
    // (total_q + 128 * b) rows in the unpadded varlen format. We pass
    // seqlen_q_rounded = round128(max_seqlen_q) for the bookkeeping
    // fields the BW kernel uses for tile sizing on a per-sequence basis.
    p.seqlen_q_rounded = seq_rounded_128(max_seqlen_q);
    p.seqlen_k_rounded = seq_rounded_128(max_seqlen_k);
    p.total_q          = total_q;

    p.scale_softmax       = softmax_scale;
    p.scale_softmax_log2  = softmax_scale * float(1.4426950408889634);

    // Varlen activation: cu_seqlens_* must be non-null.
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
    p.softmax_lse_ptr       = lse;
    p.softmax_lseaccum_ptr  = nullptr;
    p.rng_state = nullptr;

    p.is_bf16                  = is_bf16;
    p.is_causal                = is_causal;
    p.is_seqlens_k_cumulative  = true;
    p.is_rotary_interleaved    = false;
    // Varlen LSE is packed in [h, total_q + 128*b] format, NOT [b, h, sq].
    p.unpadded_lse             = true;
    p.seqlenq_ngroups_swapped  = false;
    p.num_splits               = 1;

    // Bwd-specific (packed strides match FW above).
    p.do_ptr = const_cast<void*>(dout);
    p.dq_ptr = dq;
    p.dk_ptr = dk;
    p.dv_ptr = dv;

    p.do_batch_stride = 0;
    p.do_head_stride  = int64_t(d);
    p.do_row_stride   = int64_t(d) * h;
    p.dq_batch_stride = 0;
    p.dq_head_stride  = int64_t(d);
    p.dq_row_stride   = int64_t(d) * h;
    p.dk_batch_stride = 0;
    p.dk_head_stride  = int64_t(d);
    p.dk_row_stride   = int64_t(d) * h_k;
    p.dv_batch_stride = 0;
    p.dv_head_stride  = int64_t(d);
    p.dv_row_stride   = int64_t(d) * h_k;

    p.dq_accum_ptr = dq_accum;
    p.dk_accum_ptr = nullptr;
    p.dv_accum_ptr = nullptr;
    p.dsoftmax_sum = dsoftmax_sum;

    p.deterministic = false;
    p.dq_accum_split_stride = 0;
}

template <typename T>
bool dispatch_bwd(int head_dim, bool is_causal,
                  FLASH_NAMESPACE::Flash_bwd_params& params, cudaStream_t s) {
    if (is_causal) {
        switch (head_dim) {
            case 32:  FLASH_NAMESPACE::run_mha_bwd_<T, 32,  true>(params, s); return true;
            case 64:  FLASH_NAMESPACE::run_mha_bwd_<T, 64,  true>(params, s); return true;
            case 96:  FLASH_NAMESPACE::run_mha_bwd_<T, 96,  true>(params, s); return true;
            case 128: FLASH_NAMESPACE::run_mha_bwd_<T, 128, true>(params, s); return true;
            // hd160/224 BW NOT SUPPORTED (kBlockKSmem=32 path; caller falls back to bespoke)
            case 192: FLASH_NAMESPACE::run_mha_bwd_<T, 192, true>(params, s); return true;
            case 256: FLASH_NAMESPACE::run_mha_bwd_<T, 256, true>(params, s); return true;
            // hd512 BW NOT SUPPORTED (FA2 BW kernel requires kBlockM >= 64; caller falls back to bespoke)
            default:  return false;
        }
    } else {
        switch (head_dim) {
            case 32:  FLASH_NAMESPACE::run_mha_bwd_<T, 32,  false>(params, s); return true;
            case 64:  FLASH_NAMESPACE::run_mha_bwd_<T, 64,  false>(params, s); return true;
            case 96:  FLASH_NAMESPACE::run_mha_bwd_<T, 96,  false>(params, s); return true;
            case 128: FLASH_NAMESPACE::run_mha_bwd_<T, 128, false>(params, s); return true;
            // hd160/224 BW NOT SUPPORTED (kBlockKSmem=32 path; caller falls back to bespoke)
            case 192: FLASH_NAMESPACE::run_mha_bwd_<T, 192, false>(params, s); return true;
            case 256: FLASH_NAMESPACE::run_mha_bwd_<T, 256, false>(params, s); return true;
            // hd512 BW NOT SUPPORTED (FA2 BW kernel requires kBlockM >= 64)
            default:  return false;
        }
    }
}

// Workspace sizing — dense.
//   dq_accum     : B * Sq_rounded * H * Hd_rounded * 4 bytes
//   softmax_d    : B * H * Sq_rounded * 4 bytes
// Returned via out-pointers so the same helper serves the run + size FFI.
inline void compute_dense_ws_sizes(
    int b, int h, int sq, int d,
    std::size_t& dq_accum_bytes, std::size_t& softmax_d_bytes)
{
    std::size_t sq_r = (std::size_t)seq_rounded_128(sq);
    std::size_t hd_r = (std::size_t)head_size_rounded(d);
    dq_accum_bytes  = (std::size_t)b * sq_r * (std::size_t)h * hd_r * 4u;
    softmax_d_bytes = (std::size_t)b * (std::size_t)h * sq_r * 4u;
}

inline void compute_varlen_ws_sizes(
    int b, int h, int max_seqlen_q, int total_q, int d,
    std::size_t& dq_accum_bytes, std::size_t& softmax_d_bytes,
    std::size_t& lse_bytes)
{
    (void)max_seqlen_q;
    std::size_t hd_r = (std::size_t)head_size_rounded(d);
    std::size_t padded_rows = (std::size_t)total_q + 128u * (std::size_t)b;
    dq_accum_bytes  = padded_rows * (std::size_t)h * hd_r * 4u;
    softmax_d_bytes = (std::size_t)h * padded_rows * 4u;
    lse_bytes       = softmax_d_bytes;  // same shape
}

template <typename T>
int run_bwd_impl(
    int32_t batch, int32_t num_heads, int32_t num_heads_k,
    int32_t seq_q, int32_t seq_k, int32_t head_dim,
    float softmax_scale, int32_t is_causal,
    const void* alibi_slopes_ptr, int32_t alibi_batch_stride,
    int32_t window_size_left, int32_t window_size_right, float softcap,
    const void* q, const void* k, const void* v, const void* o,
    const void* dout, const void* lse,
    void* dq, void* dk, void* dv,
    void* workspace, std::size_t workspace_bytes,
    bool is_bf16, void* stream)
{
    if (!is_supported_hdim(head_dim)) return STATUS_UNSUPPORTED;
    if (batch <= 0 || num_heads <= 0 || num_heads_k <= 0 ||
        seq_q <= 0 || seq_k <= 0) return STATUS_INVALID_ARG;
    if (num_heads_k > num_heads) return STATUS_INVALID_ARG;
    if (num_heads % num_heads_k != 0) return STATUS_INVALID_ARG;
    if (!q || !k || !v || !o || !dout || !lse || !dq || !dk || !dv) return STATUS_INVALID_ARG;
    if (softcap < 0.0f) return STATUS_INVALID_ARG;

    std::size_t dq_accum_bytes = 0;
    std::size_t softmax_d_bytes = 0;
    compute_dense_ws_sizes(batch, num_heads, seq_q, head_dim,
                           dq_accum_bytes, softmax_d_bytes);
    std::size_t total_bytes = dq_accum_bytes + softmax_d_bytes;
    if (workspace_bytes < total_bytes) return STATUS_INVALID_ARG;
    if (!workspace) return STATUS_INVALID_ARG;

    // Zero both scratch buffers — FA2's BW reads dq_accum before writing
    // when deterministic=false (one pass writes via atomic add, the
    // convert kernel reads). softmax_d is read by the seqk-parallel BW.
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    cudaError_t e = cudaMemsetAsync(workspace, 0, total_bytes, s);
    if (e != cudaSuccess) return STATUS_INVALID_ARG;

    void* dq_accum   = workspace;
    void* dsoftmax_d = static_cast<char*>(workspace) + dq_accum_bytes;

    FLASH_NAMESPACE::Flash_bwd_params params{};
    fill_bwd_dense_params(
        params, batch, num_heads, num_heads_k, seq_q, seq_k, head_dim,
        softmax_scale, is_causal != 0, is_bf16,
        q, k, v, o, dout,
        dq, dk, dv,
        const_cast<void*>(lse),
        dq_accum, dsoftmax_d,
        alibi_slopes_ptr, alibi_batch_stride,
        window_size_left, window_size_right, softcap);

    if (!dispatch_bwd<T>(head_dim, is_causal != 0, params, s)) {
        return STATUS_UNSUPPORTED;
    }
    e = cudaGetLastError();
    return e == cudaSuccess ? STATUS_OK : STATUS_INVALID_ARG;
}

template <typename T>
int run_varlen_bwd_impl(
    int32_t batch, int32_t num_heads, int32_t num_heads_k,
    int32_t max_seqlen_q, int32_t max_seqlen_k,
    int32_t total_q, int32_t total_k,
    int32_t head_dim,
    float softmax_scale, int32_t is_causal,
    const void* alibi_slopes_ptr, int32_t alibi_batch_stride,
    int32_t window_size_left, int32_t window_size_right, float softcap,
    const void* q, const void* k, const void* v, const void* o,
    const void* dout, const void* lse,
    const int* cu_seqlens_q, const int* cu_seqlens_k,
    void* dq, void* dk, void* dv,
    void* workspace, std::size_t workspace_bytes,
    bool is_bf16, void* stream)
{
    if (!is_supported_hdim(head_dim)) return STATUS_UNSUPPORTED;
    if (batch <= 0 || num_heads <= 0 || num_heads_k <= 0 ||
        max_seqlen_q <= 0 || max_seqlen_k <= 0 ||
        total_q <= 0 || total_k <= 0) return STATUS_INVALID_ARG;
    if (num_heads_k > num_heads) return STATUS_INVALID_ARG;
    if (num_heads % num_heads_k != 0) return STATUS_INVALID_ARG;
    if (!q || !k || !v || !o || !dout || !lse ||
        !cu_seqlens_q || !cu_seqlens_k ||
        !dq || !dk || !dv) return STATUS_INVALID_ARG;
    if (softcap < 0.0f) return STATUS_INVALID_ARG;

    std::size_t dq_accum_bytes = 0;
    std::size_t softmax_d_bytes = 0;
    std::size_t lse_bytes = 0;
    compute_varlen_ws_sizes(batch, num_heads, max_seqlen_q, total_q, head_dim,
                            dq_accum_bytes, softmax_d_bytes, lse_bytes);
    std::size_t total_bytes = dq_accum_bytes + softmax_d_bytes;
    if (workspace_bytes < total_bytes) return STATUS_INVALID_ARG;
    if (!workspace) return STATUS_INVALID_ARG;

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    cudaError_t e = cudaMemsetAsync(workspace, 0, total_bytes, s);
    if (e != cudaSuccess) return STATUS_INVALID_ARG;

    void* dq_accum   = workspace;
    void* dsoftmax_d = static_cast<char*>(workspace) + dq_accum_bytes;

    FLASH_NAMESPACE::Flash_bwd_params params{};
    fill_bwd_varlen_params(
        params, batch, num_heads, num_heads_k,
        max_seqlen_q, max_seqlen_k, total_q, total_k, head_dim,
        softmax_scale, is_causal != 0, is_bf16,
        q, k, v, o, dout,
        dq, dk, dv,
        const_cast<void*>(lse),
        dq_accum, dsoftmax_d,
        cu_seqlens_q, cu_seqlens_k,
        alibi_slopes_ptr, alibi_batch_stride,
        window_size_left, window_size_right, softcap);

    if (!dispatch_bwd<T>(head_dim, is_causal != 0, params, s)) {
        return STATUS_UNSUPPORTED;
    }
    e = cudaGetLastError();
    return e == cudaSuccess ? STATUS_OK : STATUS_INVALID_ARG;
}

}  // namespace

extern "C" {

// ============================================================================
// Phase 59b BW entry points — dense path.
// ============================================================================

int baracuda_kernels_fa2_sdpa_backward_f16_run(
    int32_t batch, int32_t num_heads, int32_t num_heads_k,
    int32_t seq_q, int32_t seq_k, int32_t head_dim,
    float softmax_scale, int32_t is_causal,
    const void* alibi_slopes_ptr, int32_t alibi_batch_stride,
    int32_t window_size_left, int32_t window_size_right, float softcap,
    const void* q, const void* k, const void* v, const void* o,
    const void* dout, const void* lse,
    void* dq, void* dk, void* dv,
    void* workspace, std::size_t workspace_bytes, void* stream)
{
    return run_bwd_impl<cutlass::half_t>(
        batch, num_heads, num_heads_k, seq_q, seq_k, head_dim,
        softmax_scale, is_causal,
        alibi_slopes_ptr, alibi_batch_stride,
        window_size_left, window_size_right, softcap,
        q, k, v, o, dout, lse,
        dq, dk, dv,
        workspace, workspace_bytes,
        /*is_bf16=*/false, stream);
}

int baracuda_kernels_fa2_sdpa_backward_bf16_run(
    int32_t batch, int32_t num_heads, int32_t num_heads_k,
    int32_t seq_q, int32_t seq_k, int32_t head_dim,
    float softmax_scale, int32_t is_causal,
    const void* alibi_slopes_ptr, int32_t alibi_batch_stride,
    int32_t window_size_left, int32_t window_size_right, float softcap,
    const void* q, const void* k, const void* v, const void* o,
    const void* dout, const void* lse,
    void* dq, void* dk, void* dv,
    void* workspace, std::size_t workspace_bytes, void* stream)
{
    return run_bwd_impl<cutlass::bfloat16_t>(
        batch, num_heads, num_heads_k, seq_q, seq_k, head_dim,
        softmax_scale, is_causal,
        alibi_slopes_ptr, alibi_batch_stride,
        window_size_left, window_size_right, softcap,
        q, k, v, o, dout, lse,
        dq, dk, dv,
        workspace, workspace_bytes,
        /*is_bf16=*/true, stream);
}

int baracuda_kernels_fa2_sdpa_backward_f16_can_implement(
    int32_t batch, int32_t num_heads, int32_t num_heads_k,
    int32_t seq_q, int32_t seq_k, int32_t head_dim,
    int32_t /*is_causal*/,
    int32_t /*window_size_left*/, int32_t /*window_size_right*/,
    float softcap)
{
    if (!is_supported_hdim(head_dim)) return STATUS_UNSUPPORTED;
    if (batch <= 0 || num_heads <= 0 || num_heads_k <= 0 ||
        seq_q <= 0 || seq_k <= 0) return STATUS_INVALID_ARG;
    if (num_heads_k > num_heads) return STATUS_INVALID_ARG;
    if (num_heads % num_heads_k != 0) return STATUS_INVALID_ARG;
    if (softcap < 0.0f) return STATUS_INVALID_ARG;
    return STATUS_OK;
}

int baracuda_kernels_fa2_sdpa_backward_bf16_can_implement(
    int32_t batch, int32_t num_heads, int32_t num_heads_k,
    int32_t seq_q, int32_t seq_k, int32_t head_dim,
    int32_t is_causal,
    int32_t window_size_left, int32_t window_size_right, float softcap)
{
    return baracuda_kernels_fa2_sdpa_backward_f16_can_implement(
        batch, num_heads, num_heads_k, seq_q, seq_k, head_dim,
        is_causal, window_size_left, window_size_right, softcap);
}

// Dense BW workspace size (bytes). Caller passes this much memory in
// the `workspace` arg of `..._backward_<dt>_run`.
std::size_t baracuda_kernels_fa2_sdpa_backward_workspace_size(
    int32_t batch, int32_t num_heads, int32_t seq_q, int32_t head_dim)
{
    if (batch <= 0 || num_heads <= 0 || seq_q <= 0 || head_dim <= 0) return 0;
    if (!is_supported_hdim(head_dim)) return 0;
    std::size_t dq_accum_bytes = 0;
    std::size_t softmax_d_bytes = 0;
    compute_dense_ws_sizes(batch, num_heads, seq_q, head_dim,
                           dq_accum_bytes, softmax_d_bytes);
    return dq_accum_bytes + softmax_d_bytes;
}

// ============================================================================
// Phase 59b BW entry points — varlen path.
// ============================================================================

int baracuda_kernels_fa2_sdpa_varlen_backward_f16_run(
    int32_t batch, int32_t num_heads, int32_t num_heads_k,
    int32_t max_seqlen_q, int32_t max_seqlen_k,
    int32_t total_q, int32_t total_k,
    int32_t head_dim,
    float softmax_scale, int32_t is_causal,
    const void* alibi_slopes_ptr, int32_t alibi_batch_stride,
    int32_t window_size_left, int32_t window_size_right, float softcap,
    const void* q, const void* k, const void* v, const void* o,
    const void* dout, const void* lse,
    const int* cu_seqlens_q, const int* cu_seqlens_k,
    void* dq, void* dk, void* dv,
    void* workspace, std::size_t workspace_bytes, void* stream)
{
    return run_varlen_bwd_impl<cutlass::half_t>(
        batch, num_heads, num_heads_k, max_seqlen_q, max_seqlen_k,
        total_q, total_k, head_dim,
        softmax_scale, is_causal,
        alibi_slopes_ptr, alibi_batch_stride,
        window_size_left, window_size_right, softcap,
        q, k, v, o, dout, lse,
        cu_seqlens_q, cu_seqlens_k,
        dq, dk, dv,
        workspace, workspace_bytes,
        /*is_bf16=*/false, stream);
}

int baracuda_kernels_fa2_sdpa_varlen_backward_bf16_run(
    int32_t batch, int32_t num_heads, int32_t num_heads_k,
    int32_t max_seqlen_q, int32_t max_seqlen_k,
    int32_t total_q, int32_t total_k,
    int32_t head_dim,
    float softmax_scale, int32_t is_causal,
    const void* alibi_slopes_ptr, int32_t alibi_batch_stride,
    int32_t window_size_left, int32_t window_size_right, float softcap,
    const void* q, const void* k, const void* v, const void* o,
    const void* dout, const void* lse,
    const int* cu_seqlens_q, const int* cu_seqlens_k,
    void* dq, void* dk, void* dv,
    void* workspace, std::size_t workspace_bytes, void* stream)
{
    return run_varlen_bwd_impl<cutlass::bfloat16_t>(
        batch, num_heads, num_heads_k, max_seqlen_q, max_seqlen_k,
        total_q, total_k, head_dim,
        softmax_scale, is_causal,
        alibi_slopes_ptr, alibi_batch_stride,
        window_size_left, window_size_right, softcap,
        q, k, v, o, dout, lse,
        cu_seqlens_q, cu_seqlens_k,
        dq, dk, dv,
        workspace, workspace_bytes,
        /*is_bf16=*/true, stream);
}

int baracuda_kernels_fa2_sdpa_varlen_backward_f16_can_implement(
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

int baracuda_kernels_fa2_sdpa_varlen_backward_bf16_can_implement(
    int32_t batch, int32_t num_heads, int32_t num_heads_k,
    int32_t max_seqlen_q, int32_t max_seqlen_k,
    int32_t total_q, int32_t total_k,
    int32_t head_dim, int32_t is_causal,
    int32_t window_size_left, int32_t window_size_right, float softcap)
{
    return baracuda_kernels_fa2_sdpa_varlen_backward_f16_can_implement(
        batch, num_heads, num_heads_k, max_seqlen_q, max_seqlen_k,
        total_q, total_k, head_dim, is_causal,
        window_size_left, window_size_right, softcap);
}

std::size_t baracuda_kernels_fa2_sdpa_varlen_backward_workspace_size(
    int32_t batch, int32_t num_heads,
    int32_t max_seqlen_q, int32_t total_q, int32_t head_dim)
{
    if (batch <= 0 || num_heads <= 0 ||
        max_seqlen_q <= 0 || total_q <= 0 || head_dim <= 0) return 0;
    if (!is_supported_hdim(head_dim)) return 0;
    std::size_t dq_accum_bytes = 0;
    std::size_t softmax_d_bytes = 0;
    std::size_t lse_bytes = 0;
    compute_varlen_ws_sizes(batch, num_heads, max_seqlen_q, total_q, head_dim,
                            dq_accum_bytes, softmax_d_bytes, lse_bytes);
    return dq_accum_bytes + softmax_d_bytes;
}

}  // extern "C"
