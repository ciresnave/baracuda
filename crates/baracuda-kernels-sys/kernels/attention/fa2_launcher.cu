// baracuda-kernels Phase 42 + Phase 59a — Dao-AILab FlashAttention v2 launcher.
//
// Bridges baracuda's `extern "C"` FFI surface to FA2 v2.8.3's
// `run_mha_fwd_<T, headdim, is_causal>` template instantiations
// shipped from the vendored .cu files under
// `vendor/flash-attention/src/flash_fwd_hdim*_sm80.cu`.
//
// Phase 42 (Tier-1) shipped head_dim=128 only, no GQA, no extras.
// Phase 59a extends FW coverage to the full upstream feature surface:
//
//   * head_dim ∈ {32, 64, 96, 128, 192, 256}, dispatched at runtime.
//     Upstream FA2 v2.8.3 does NOT ship head_dims 160, 224, or 512 —
//     those are permanently Tier-3-deferred (no upstream sources).
//   * GQA — `num_heads_k != num_heads` accepted when
//     `num_heads % num_heads_k == 0`. FA2's kernel handles the
//     broadcast via its `h_h_k_ratio` mechanism.
//   * ALiBi — per-head or per-batch-per-head additive positional bias.
//   * Sliding window — left/right window bounds on the attention range.
//   * Softcap — tanh-based score capping (Gemma-2 style).
//
// Phase 42 v1 symbols (`..._run`, `..._can_implement`) are preserved
// for backwards compatibility — they internally forward to the same
// launcher with all Phase 59a extras set to their disabled defaults.
//
// Param-struct contract (`Flash_fwd_params`, defined in
// `vendor/flash-attention/src/flash.h`):
//   - Strides expressed in *elements* (not bytes), int64_t.
//   - baracuda layout: `[B, H, S, D]` contiguous. FA2 accepts arbitrary
//     stride layouts via per-axis stride fields, so no physical
//     transpose is needed.
//   - `softmax_lse_ptr` always **f32** regardless of element type.

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// `cutlass::half_t` / `cutlass::bfloat16_t` — used as the template
// arguments to FA2's `run_mha_fwd_<T, headdim, is_causal>` instantiations.
#include <cutlass/numeric_types.h>

#include "../../vendor/flash-attention/src/namespace_config.h"
#include "../../vendor/flash-attention/src/flash.h"

namespace FLASH_NAMESPACE {
// Forward declarations of the explicit template instantiations
// emitted by the vendored .cu files. Avoids pulling
// `flash_fwd_launch_template.h` into this translation unit (it would
// re-instantiate the heavy CUTLASS templates here for no gain).
// Phase 42 + 59a: full head_dim set ∈ {32, 64, 96, 128, 192, 256}.
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

// Status codes match `attention/map_status` in `baracuda-kernels/src/attention/mod.rs`:
//   0 = ok, 2 = invalid_problem, 3 = unsupported.
constexpr int STATUS_OK          = 0;
constexpr int STATUS_INVALID_ARG = 2;
constexpr int STATUS_UNSUPPORTED = 3;

// FA2 v2.8.3 supports exactly these forward head_dims. We do NOT ship
// 160/224/512 — no upstream sources exist for them.
inline bool is_supported_hdim(int d) {
    return d == 32 || d == 64 || d == 96 || d == 128 || d == 192 || d == 256;
}

// Fill a Flash_fwd_params for the dense case with full Phase 59a
// feature plumbing.
//
// baracuda contract:
//   Q layout : [B, H,    Sq, D]  row-major contiguous
//   K layout : [B, H_k,  Sk, D]  row-major contiguous
//   V layout : [B, H_k,  Sk, D]  row-major contiguous
//   Y layout : [B, H,    Sq, D]  row-major contiguous
//   lse      : [B, H,    Sq]     row-major contiguous, FLOAT32
//
// FA2 supports per-axis strides; we set them directly without
// physical transpose.
void fill_dense_params(
    FLASH_NAMESPACE::Flash_fwd_params& p,
    int b, int h, int h_k, int sq, int sk, int d,
    float softmax_scale, bool is_causal, bool is_bf16,
    void* q, void* k, void* v, void* o, void* softmax_lse,
    // Phase 59a additions:
    void* alibi_slopes_ptr,
    int alibi_batch_stride,
    int window_size_left,
    int window_size_right,
    float softcap)
{
    // Zero everything first — defensive against fields we don't set.
    std::memset(&p, 0, sizeof(p));

    // Qkv_params
    p.q_ptr = q;
    p.k_ptr = k;
    p.v_ptr = v;

    // Per-element strides for [B, H, S, D] row-major contiguous layout.
    // Q has H heads; K/V have H_k heads (h_k == h when not GQA).
    p.q_batch_stride = int64_t(h)   * sq * d;
    p.k_batch_stride = int64_t(h_k) * sk * d;
    p.v_batch_stride = int64_t(h_k) * sk * d;
    p.q_head_stride  = int64_t(sq) * d;
    p.k_head_stride  = int64_t(sk) * d;
    p.v_head_stride  = int64_t(sk) * d;
    p.q_row_stride   = int64_t(d);
    p.k_row_stride   = int64_t(d);
    p.v_row_stride   = int64_t(d);

    p.h            = h;
    p.h_k          = h_k;
    // FA2 reads `h_h_k_ratio` to decide which K/V head each Q head reads.
    // The kernel handles the broadcast via integer division when h_h_k_ratio > 1.
    p.h_h_k_ratio  = h / h_k;

    // Flash_fwd_params additions
    p.o_ptr            = o;
    p.o_batch_stride   = int64_t(h) * sq * d;
    p.o_head_stride    = int64_t(sq) * d;
    p.o_row_stride     = int64_t(d);

    p.b                = b;
    p.seqlen_q         = sq;
    p.seqlen_k         = sk;
    p.d                = d;
    p.d_rounded        = d;
    p.seqlen_q_rounded = ((sq + 127) / 128) * 128;
    p.seqlen_k_rounded = ((sk + 127) / 128) * 128;
    p.total_q          = sq;

    p.scale_softmax       = softmax_scale;
    p.scale_softmax_log2  = softmax_scale * float(1.4426950408889634);  // log2(e)

    // No varlen / no leftpad
    p.cu_seqlens_q = nullptr;
    p.cu_seqlens_k = nullptr;
    p.leftpad_k    = nullptr;
    p.seqused_k    = nullptr;

    // No K/V append, no paged KV cache, no rotary.
    p.knew_ptr            = nullptr;
    p.vnew_ptr            = nullptr;
    p.rotary_cos_ptr      = nullptr;
    p.rotary_sin_ptr      = nullptr;
    p.cache_batch_idx     = nullptr;
    p.block_table         = nullptr;

    // Phase 59a: ALiBi slopes.
    //   - `alibi_slopes_ptr == nullptr` → disabled.
    //   - When set, layout is selected by `alibi_batch_stride`:
    //     * 0  → `[num_heads]`           (per-head, broadcast over batch).
    //     * h  → `[batch, num_heads]`   (per-batch-per-head).
    //   FA2's `alibi.h` reads from `alibi_slopes_ptr +
    //   bidb * alibi_slopes_batch_stride + bidh`.
    p.alibi_slopes_ptr           = alibi_slopes_ptr;
    p.alibi_slopes_batch_stride  = int64_t(alibi_batch_stride);

    // No dropout — p_dropout = 1.0 means *probability of keeping*,
    // which FA2 treats as "no dropout" (see DROPOUT_SWITCH in
    // flash_fwd_launch_template.h).
    p.p_dropout                 = 1.0f;
    p.p_dropout_in_uint8_t      = 255;
    p.rp_dropout                = 1.0f;
    p.scale_softmax_rp_dropout  = softmax_scale;

    // Phase 59a: softcap + sliding window.
    p.softcap            = softcap;
    p.window_size_left   = window_size_left;
    p.window_size_right  = window_size_right;
    if (is_causal) {
        // Causal forces right-window of 0 (only past-and-current
        // tokens visible). This composes with a caller-supplied
        // window_size_left: causal + left=N == "look at last N
        // past tokens, no future". When caller passes window_size_left=-1
        // (unbounded), causal + left=-1 == standard causal mask.
        p.window_size_right = 0;
    }

    // No softmax-saving output (only meaningful with dropout enabled).
    p.p_ptr = nullptr;

    // LSE output (always written; consumed by BW pass in Tier 2 / Phase 59b).
    p.softmax_lse_ptr = softmax_lse;
    p.softmax_lseaccum_ptr = nullptr;

    p.rng_state = nullptr;

    p.is_bf16                  = is_bf16;
    p.is_causal                = is_causal;
    p.is_seqlens_k_cumulative  = true;
    p.is_rotary_interleaved    = false;
    p.unpadded_lse             = false;
    p.seqlenq_ngroups_swapped  = false;
    p.num_splits               = 1;
}

// Dispatch on head_dim + dtype + is_causal to the right template
// instantiation. Returns true on success, false if head_dim unsupported.
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

// Shared run-path used by both v1 and v2 entry points. v1 callers pass
// the Phase 59a extras at their disabled defaults.
template <typename T>
int run_fwd_impl(
    int32_t batch, int32_t num_heads, int32_t num_heads_k,
    int32_t seq_q, int32_t seq_k, int32_t head_dim,
    float softmax_scale, int32_t is_causal,
    const void* q, const void* k, const void* v,
    void* out, void* softmax_lse,
    const void* alibi_slopes_ptr, int32_t alibi_batch_stride,
    int32_t window_size_left, int32_t window_size_right, float softcap,
    bool is_bf16, void* stream)
{
    if (!is_supported_hdim(head_dim)) return STATUS_UNSUPPORTED;
    if (batch <= 0 || num_heads <= 0 || num_heads_k <= 0 ||
        seq_q <= 0 || seq_k <= 0) return STATUS_INVALID_ARG;
    if (num_heads_k > num_heads) return STATUS_INVALID_ARG;
    if (num_heads % num_heads_k != 0) {
        return STATUS_INVALID_ARG;  // GQA requires h % h_k == 0
    }
    if (!q || !k || !v || !out || !softmax_lse) return STATUS_INVALID_ARG;
    if (softcap < 0.0f) return STATUS_INVALID_ARG;

    FLASH_NAMESPACE::Flash_fwd_params params{};
    fill_dense_params(
        params, batch, num_heads, num_heads_k, seq_q, seq_k, head_dim,
        softmax_scale, is_causal != 0, is_bf16,
        const_cast<void*>(q), const_cast<void*>(k), const_cast<void*>(v),
        out, softmax_lse,
        const_cast<void*>(alibi_slopes_ptr), alibi_batch_stride,
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

// ============================================================================
// Phase 42 v1 entry points — preserved. All Phase 59a extras at defaults.
// ============================================================================

int baracuda_kernels_fa2_sdpa_f16_run(
    int32_t batch, int32_t num_heads, int32_t num_heads_k,
    int32_t seq_q, int32_t seq_k, int32_t head_dim,
    float softmax_scale, int32_t is_causal,
    const void* q, const void* k, const void* v,
    void* out, void* softmax_lse,
    void* /*workspace*/, std::size_t /*workspace_bytes*/, void* stream)
{
    return run_fwd_impl<cutlass::half_t>(
        batch, num_heads, num_heads_k, seq_q, seq_k, head_dim,
        softmax_scale, is_causal,
        q, k, v, out, softmax_lse,
        /*alibi_slopes_ptr=*/nullptr, /*alibi_batch_stride=*/0,
        /*window_size_left=*/-1, /*window_size_right=*/-1, /*softcap=*/0.0f,
        /*is_bf16=*/false, stream);
}

int baracuda_kernels_fa2_sdpa_bf16_run(
    int32_t batch, int32_t num_heads, int32_t num_heads_k,
    int32_t seq_q, int32_t seq_k, int32_t head_dim,
    float softmax_scale, int32_t is_causal,
    const void* q, const void* k, const void* v,
    void* out, void* softmax_lse,
    void* /*workspace*/, std::size_t /*workspace_bytes*/, void* stream)
{
    return run_fwd_impl<cutlass::bfloat16_t>(
        batch, num_heads, num_heads_k, seq_q, seq_k, head_dim,
        softmax_scale, is_causal,
        q, k, v, out, softmax_lse,
        /*alibi_slopes_ptr=*/nullptr, /*alibi_batch_stride=*/0,
        /*window_size_left=*/-1, /*window_size_right=*/-1, /*softcap=*/0.0f,
        /*is_bf16=*/true, stream);
}

// v1 can_implement — mirrors the v1 run gates so the safe-plan layer
// can validate up-front without a CUDA launch.
int baracuda_kernels_fa2_sdpa_f16_can_implement(
    int32_t batch, int32_t num_heads, int32_t num_heads_k,
    int32_t seq_q, int32_t seq_k, int32_t head_dim,
    int32_t /*is_causal*/)
{
    if (!is_supported_hdim(head_dim)) return STATUS_UNSUPPORTED;
    if (batch <= 0 || num_heads <= 0 || num_heads_k <= 0 ||
        seq_q <= 0 || seq_k <= 0) return STATUS_INVALID_ARG;
    if (num_heads_k > num_heads) return STATUS_INVALID_ARG;
    if (num_heads % num_heads_k != 0) return STATUS_INVALID_ARG;
    return STATUS_OK;
}

int baracuda_kernels_fa2_sdpa_bf16_can_implement(
    int32_t batch, int32_t num_heads, int32_t num_heads_k,
    int32_t seq_q, int32_t seq_k, int32_t head_dim,
    int32_t /*is_causal*/)
{
    if (!is_supported_hdim(head_dim)) return STATUS_UNSUPPORTED;
    if (batch <= 0 || num_heads <= 0 || num_heads_k <= 0 ||
        seq_q <= 0 || seq_k <= 0) return STATUS_INVALID_ARG;
    if (num_heads_k > num_heads) return STATUS_INVALID_ARG;
    if (num_heads % num_heads_k != 0) return STATUS_INVALID_ARG;
    return STATUS_OK;
}

// ============================================================================
// Phase 59a v2 entry points — full feature surface.
// ============================================================================

int baracuda_kernels_fa2_sdpa_f16_run_v2(
    int32_t batch, int32_t num_heads, int32_t num_heads_k,
    int32_t seq_q, int32_t seq_k, int32_t head_dim,
    float softmax_scale, int32_t is_causal,
    const void* alibi_slopes_ptr, int32_t alibi_batch_stride,
    int32_t window_size_left, int32_t window_size_right, float softcap,
    const void* q, const void* k, const void* v,
    void* out, void* softmax_lse,
    void* /*workspace*/, std::size_t /*workspace_bytes*/, void* stream)
{
    return run_fwd_impl<cutlass::half_t>(
        batch, num_heads, num_heads_k, seq_q, seq_k, head_dim,
        softmax_scale, is_causal,
        q, k, v, out, softmax_lse,
        alibi_slopes_ptr, alibi_batch_stride,
        window_size_left, window_size_right, softcap,
        /*is_bf16=*/false, stream);
}

int baracuda_kernels_fa2_sdpa_bf16_run_v2(
    int32_t batch, int32_t num_heads, int32_t num_heads_k,
    int32_t seq_q, int32_t seq_k, int32_t head_dim,
    float softmax_scale, int32_t is_causal,
    const void* alibi_slopes_ptr, int32_t alibi_batch_stride,
    int32_t window_size_left, int32_t window_size_right, float softcap,
    const void* q, const void* k, const void* v,
    void* out, void* softmax_lse,
    void* /*workspace*/, std::size_t /*workspace_bytes*/, void* stream)
{
    return run_fwd_impl<cutlass::bfloat16_t>(
        batch, num_heads, num_heads_k, seq_q, seq_k, head_dim,
        softmax_scale, is_causal,
        q, k, v, out, softmax_lse,
        alibi_slopes_ptr, alibi_batch_stride,
        window_size_left, window_size_right, softcap,
        /*is_bf16=*/true, stream);
}

// v2 can_implement — validates the Phase 59a-extended gate set.
int baracuda_kernels_fa2_sdpa_f16_can_implement_v2(
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

int baracuda_kernels_fa2_sdpa_bf16_can_implement_v2(
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

}  // extern "C"
