// baracuda-kernels Phase 42 — Dao-AILab FlashAttention v2 launcher.
//
// Bridges baracuda's `extern "C"` FFI surface to FA2 v2.8.3's
// `run_mha_fwd_<T, headdim, is_causal>` template instantiations
// shipped from the vendored .cu files under
// `vendor/flash-attention/src/flash_fwd_hdim128_*_sm80.cu`.
//
// Scope (Tier 1):
//   - Forward only (BW deferred to Tier 2).
//   - head_dim = 128 only (other dims deferred to Tier 3).
//   - f16 + bf16 only.
//   - Dense layout only (no varlen / cu_seqlens).
//   - No GQA (callers requiring nheads_k != nheads must use baracuda's
//     bespoke FlashSdpaPlan, which routes broadcast through stride=0).
//   - No dropout / softcap / alibi / rotary / paged KV cache.
//
// Param-struct contract (`Flash_fwd_params`, defined in
// `vendor/flash-attention/src/flash.h`):
//   - Strides expressed in *elements* (not bytes), int64_t.
//   - Row-major contract is `[B, S, H, D]` per upstream FA2 — baracuda's
//     own layout convention is `[B, H, S, D]`. The launcher reshapes
//     via stride calculation, NOT via a physical transpose.
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
template<> void run_mha_fwd_<cutlass::half_t,     128, false>(Flash_fwd_params&, cudaStream_t);
template<> void run_mha_fwd_<cutlass::half_t,     128, true >(Flash_fwd_params&, cudaStream_t);
template<> void run_mha_fwd_<cutlass::bfloat16_t, 128, false>(Flash_fwd_params&, cudaStream_t);
template<> void run_mha_fwd_<cutlass::bfloat16_t, 128, true >(Flash_fwd_params&, cudaStream_t);
}  // namespace FLASH_NAMESPACE

namespace {

// Status codes match `attention/map_status` in `baracuda-kernels/src/attention/mod.rs`:
//   0 = ok, 2 = invalid_problem, 3 = unsupported.
constexpr int STATUS_OK          = 0;
constexpr int STATUS_INVALID_ARG = 2;
constexpr int STATUS_UNSUPPORTED = 3;

// Fill a Flash_fwd_params for the dense, single-head-dim case.
//
// baracuda contract:
//   Q layout : [B, H, Sq, D]  row-major contiguous
//   K layout : [B, H, Sk, D]  row-major contiguous
//   V layout : [B, H, Sk, D]  row-major contiguous
//   Y layout : [B, H, Sq, D]  row-major contiguous
//   lse      : [B, H, Sq]     row-major contiguous, FLOAT32
//
// FA2 expects per-element strides in [B, S, H, D] order. We translate
// without a physical transpose by setting:
//   batch_stride = H * S * D
//   row_stride   = D     (S is the row axis in FA2's expectation, but
//                         since our QKV are [B, H, S, D] *contiguous*,
//                         "row" — i.e. the next-S element — sits D
//                         elements ahead; same as the row stride in
//                         [B, S, H, D] when nheads == 1. For nheads > 1
//                         we use head_stride = S * D, which differs
//                         from FA2's [B, S, H, D] head_stride of D.)
//
// In practice we *cannot* shoehorn baracuda's [B, H, S, D] layout into
// FA2's [B, S, H, D] expectations without a physical transpose,
// because the *row* axis (S) and the *head* axis (H) are interleaved
// differently. So we make this launcher require the upstream FA2
// layout: B-major, then S, then H, then D. We surface this in the
// docstring of the FFI symbol and document that the safe-plan layer
// must materialize a transpose if the caller hands it [B, H, S, D]
// data.
//
// HOWEVER: passing strides individually IS enough to support [B, H, S, D]
// natively! For batch stride, head stride, row stride independently
// settable, we use:
//   B-axis stride: H*S*D  (q_batch_stride)
//   H-axis stride: S*D    (q_head_stride)
//   S-axis stride: D      (q_row_stride)
// FA2 uses each stride from its struct directly; it doesn't assume the
// stride layout is sequential [B, S, H, D]. So we CAN serve baracuda's
// [B, H, S, D] by passing q_head_stride = S*D and q_row_stride = D.

void fill_dense_params(
    FLASH_NAMESPACE::Flash_fwd_params& p,
    int b, int h, int sq, int sk, int d,
    float softmax_scale, bool is_causal, bool is_bf16,
    void* q, void* k, void* v, void* o, void* softmax_lse)
{
    // Zero everything first — defensive against fields we don't set
    // (paged-KV, alibi, rotary, splitkv, varlen, ...).
    std::memset(&p, 0, sizeof(p));

    // Qkv_params
    p.q_ptr = q;
    p.k_ptr = k;
    p.v_ptr = v;

    // Per-element strides for [B, H, S, D] row-major contiguous layout.
    // Q,K,V,O all share `d` per the Tier-1 contract (d_k == d_v == 128).
    p.q_batch_stride = int64_t(h) * sq * d;
    p.k_batch_stride = int64_t(h) * sk * d;
    p.v_batch_stride = int64_t(h) * sk * d;
    p.q_head_stride  = int64_t(sq) * d;
    p.k_head_stride  = int64_t(sk) * d;
    p.v_head_stride  = int64_t(sk) * d;
    p.q_row_stride   = int64_t(d);
    p.k_row_stride   = int64_t(d);
    p.v_row_stride   = int64_t(d);

    p.h            = h;
    p.h_k          = h;          // no GQA in Tier 1
    p.h_h_k_ratio  = 1;

    // Flash_fwd_params additions
    p.o_ptr            = o;
    p.o_batch_stride   = int64_t(h) * sq * d;
    p.o_head_stride    = int64_t(sq) * d;
    p.o_row_stride     = int64_t(d);

    p.b                = b;
    p.seqlen_q         = sq;
    p.seqlen_k         = sk;
    p.d                = d;
    // FA2 reads rounded sizes when `d` != kHeadDim or sk %% kBlockN != 0.
    // For Tier 1 we pin headdim = kHeadDim = 128 so d_rounded = d.
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

    // No K/V append, no paged KV cache, no rotary, no alibi.
    p.knew_ptr            = nullptr;
    p.vnew_ptr            = nullptr;
    p.rotary_cos_ptr      = nullptr;
    p.rotary_sin_ptr      = nullptr;
    p.cache_batch_idx     = nullptr;
    p.block_table         = nullptr;
    p.alibi_slopes_ptr    = nullptr;

    // No dropout — p_dropout = 1.0 means *probability of keeping*,
    // which FA2 treats as "no dropout". Specifically
    // `flash_fwd_launch_template.h` does
    // `DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, ...)` so
    // setting p_dropout = 1.0 resolves the switch to Is_dropout=false.
    p.p_dropout                 = 1.0f;
    p.p_dropout_in_uint8_t      = 255;
    p.rp_dropout                = 1.0f;
    p.scale_softmax_rp_dropout  = softmax_scale;

    // No softcap, no local window (-1 = unbounded both sides).
    p.softcap            = 0.0f;
    p.window_size_left   = -1;
    p.window_size_right  = -1;
    if (is_causal) {
        // Causal forces a right-window of 0 (only past-and-current
        // tokens visible) at the FA2 kernel level.
        p.window_size_right = 0;
    }

    // No softmax-saving output (saved softmax is only meaningful with
    // dropout enabled).
    p.p_ptr = nullptr;

    // LSE output (always written; consumed by BW pass in Tier 2).
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

}  // namespace

extern "C" {

int baracuda_kernels_fa2_sdpa_f16_run(
    int32_t batch, int32_t num_heads, int32_t num_heads_k,
    int32_t seq_q, int32_t seq_k, int32_t head_dim,
    float softmax_scale, int32_t is_causal,
    const void* q, const void* k, const void* v,
    void* out, void* softmax_lse,
    void* /*workspace*/, std::size_t /*workspace_bytes*/, void* stream)
{
    if (head_dim != 128)         return STATUS_UNSUPPORTED;
    if (num_heads_k != num_heads) return STATUS_UNSUPPORTED;  // no GQA in Tier 1
    if (batch <= 0 || num_heads <= 0 || seq_q <= 0 || seq_k <= 0) return STATUS_INVALID_ARG;
    if (!q || !k || !v || !out || !softmax_lse) return STATUS_INVALID_ARG;
    FLASH_NAMESPACE::Flash_fwd_params params{};
    fill_dense_params(
        params, batch, num_heads, seq_q, seq_k, head_dim,
        softmax_scale, is_causal != 0, /*is_bf16=*/false,
        const_cast<void*>(q), const_cast<void*>(k), const_cast<void*>(v),
        out, softmax_lse);
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    if (is_causal) {
        FLASH_NAMESPACE::run_mha_fwd_<cutlass::half_t, 128, true>(params, s);
    } else {
        FLASH_NAMESPACE::run_mha_fwd_<cutlass::half_t, 128, false>(params, s);
    }
    cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? STATUS_OK : STATUS_INVALID_ARG;
}

int baracuda_kernels_fa2_sdpa_bf16_run(
    int32_t batch, int32_t num_heads, int32_t num_heads_k,
    int32_t seq_q, int32_t seq_k, int32_t head_dim,
    float softmax_scale, int32_t is_causal,
    const void* q, const void* k, const void* v,
    void* out, void* softmax_lse,
    void* /*workspace*/, std::size_t /*workspace_bytes*/, void* stream)
{
    if (head_dim != 128)         return STATUS_UNSUPPORTED;
    if (num_heads_k != num_heads) return STATUS_UNSUPPORTED;
    if (batch <= 0 || num_heads <= 0 || seq_q <= 0 || seq_k <= 0) return STATUS_INVALID_ARG;
    if (!q || !k || !v || !out || !softmax_lse) return STATUS_INVALID_ARG;
    FLASH_NAMESPACE::Flash_fwd_params params{};
    fill_dense_params(
        params, batch, num_heads, seq_q, seq_k, head_dim,
        softmax_scale, is_causal != 0, /*is_bf16=*/true,
        const_cast<void*>(q), const_cast<void*>(k), const_cast<void*>(v),
        out, softmax_lse);
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    if (is_causal) {
        FLASH_NAMESPACE::run_mha_fwd_<cutlass::bfloat16_t, 128, true>(params, s);
    } else {
        FLASH_NAMESPACE::run_mha_fwd_<cutlass::bfloat16_t, 128, false>(params, s);
    }
    cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? STATUS_OK : STATUS_INVALID_ARG;
}

// Host-side "can_implement" mirrors the .run gates so the safe-plan
// layer can validate up-front without a CUDA launch.
int baracuda_kernels_fa2_sdpa_f16_can_implement(
    int32_t batch, int32_t num_heads, int32_t num_heads_k,
    int32_t seq_q, int32_t seq_k, int32_t head_dim,
    int32_t /*is_causal*/)
{
    if (head_dim != 128)         return STATUS_UNSUPPORTED;
    if (num_heads_k != num_heads) return STATUS_UNSUPPORTED;
    if (batch <= 0 || num_heads <= 0 || seq_q <= 0 || seq_k <= 0) return STATUS_INVALID_ARG;
    return STATUS_OK;
}

int baracuda_kernels_fa2_sdpa_bf16_can_implement(
    int32_t batch, int32_t num_heads, int32_t num_heads_k,
    int32_t seq_q, int32_t seq_k, int32_t head_dim,
    int32_t /*is_causal*/)
{
    if (head_dim != 128)         return STATUS_UNSUPPORTED;
    if (num_heads_k != num_heads) return STATUS_UNSUPPORTED;
    if (batch <= 0 || num_heads <= 0 || seq_q <= 0 || seq_k <= 0) return STATUS_INVALID_ARG;
    return STATUS_OK;
}

}  // extern "C"
