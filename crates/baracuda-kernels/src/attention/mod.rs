//! Attention op family — Phase 6 Category K.
//!
//! Today's wiring (Milestone 6.1 — positional encodings, FW + BW × 4 FP
//! dtypes):
//!
//! - **RoPE** (Rotary Position Embedding) — rotates pairs `(2i, 2i+1)` of
//!   a `[B, H, S, D]` Q/K tensor by per-position angles
//!   `θ_i = pos · base^(-2i/D)`. `head_dim` must be even. `positions`
//!   is an optional `i64[S]` override; when absent, the kernel uses
//!   the sequence index `s` as `pos`. Dominant positional encoding in
//!   modern LLMs (Llama / Mistral / Gemma / Qwen / Phi).
//!
//! - **ALiBi** (Attention with Linear Biases) — adds the per-head
//!   linear bias `slope[h] · (j - i)` to attention-score cell
//!   `(b, h, i, j)` of a `[B, H, Q, K]` score tensor. The BW is a
//!   pass-through copy for `dA` plus a deterministic per-head
//!   warp-shuffle reduction for `dslope` (no atomicAdd).
//!
//! Milestone 6.2 adds:
//!
//! - **SDPA** (naive scaled dot-product attention) — the PyTorch
//!   `F.scaled_dot_product_attention` baseline that materializes the
//!   full `[B, H, Q, K]` attention matrix. Three-kernel FW pipeline
//!   (scores / row-softmax / out) and five-kernel BW pipeline (dV /
//!   dattn / dscores-via-softmax-bw / dQ / dK), all bundled behind a
//!   single launcher symbol per direction. Optional pre-softmax mask
//!   and optional causal upper-tri mask. The saved softmax output
//!   (`attn` tensor) is shared between FW and BW with zero copying.
//!
//! Milestone 6.5 adds:
//!
//! - **KvCacheAppend** (decoder-inference helper) — writes newly
//!   generated `K_new` / `V_new` slices into running `K_cache` /
//!   `V_cache` buffers at per-sample offsets supplied via
//!   `cache_offsets[b]`. Pure copy (bit-exact); ragged-batch insertion
//!   is supported because each sample carries its own offset. No BW
//!   (inference-time op).
//!
//! Milestone 6.6 adds:
//!
//! - **FlashAttention** (Tri Dao 2022) — tiled fused online-softmax
//!   SDPA that avoids materializing the `[B, H, Q, K]` attention
//!   matrix. Saves a small `lse: [B, H, Q]` log-sum-exp tensor for the
//!   BW pass. Three-kernel deterministic BW pipeline (D = rowsum(y ⊙
//!   dy), then dQ per q-block, then dK / dV per k-block — each output
//!   cell is written by exactly one block, no atomicAdd). Trailblazer
//!   constraints: Br = Bc = 64, d_k = d_v ≤ 128, optional causal mask.
//!
//! ## Deferred
//!
//! PagedAttention is reserved in
//! [`baracuda_kernels_types::AttentionKind`] but not wired here.
//! Dropout on attention probs is deferred — wire `dropout_p = 0`.
//!
//! ## Design notes
//!
//! - Rank fixed to 4 in the trailblazer (matches Q/K/score-tensor
//!   conventions in PyTorch / JAX / FlashAttention). Higher-rank
//!   support (e.g. for grouped-query layouts threading through extra
//!   axes) lands in fanout milestones.
//! - All operands are contiguous row-major in the trailblazer (no
//!   strided / broadcast operands). The kernel computes flat offsets
//!   from a single unravel.
//! - f16 / bf16 detour through f32 for trig; f64 uses native double
//!   throughout.

pub mod alibi;
pub mod alibi_backward;
pub mod flash_sdpa;
pub mod flash_sdpa_backward;
// Phase 59b — packed-batch (varlen) FlashAttention v2 plans (FW + BW).
// Gated on the `fa2` cargo feature at runtime (descriptor compiles
// unconditionally for API discoverability; the run path errors with
// Unsupported when the feature is off).
pub mod flash_sdpa_varlen;
#[cfg(feature = "sm89")]
pub mod flash_sdpa_sm89;
pub mod kv_cache;
pub mod hyper_connection;
pub mod rope;
pub mod rope_backward;
// Phase 45 — long-context position-interpolation helpers (host-side
// cos/sin table builders for YaRN + LongRoPE). Pure Rust; no CUDA.
pub mod rope_scaling;
pub mod sdpa;
pub mod sdpa_backward;

pub use alibi::{AlibiArgs, AlibiDescriptor, AlibiPlan};
pub use alibi_backward::{AlibiBackwardArgs, AlibiBackwardDescriptor, AlibiBackwardPlan};
pub use flash_sdpa::{FlashSdpaArgs, FlashSdpaDescriptor, FlashSdpaPlan, FLASH_SDPA_MAX_D};
pub use flash_sdpa_backward::{
    FlashSdpaBackwardArgs, FlashSdpaBackwardDescriptor, FlashSdpaBackwardPlan,
};
// Phase 59b — varlen FW + BW.
pub use flash_sdpa_varlen::{
    FlashSdpaVarlenArgs, FlashSdpaVarlenBackwardArgs, FlashSdpaVarlenBackwardPlan,
    FlashSdpaVarlenDescriptor, FlashSdpaVarlenPlan,
};
#[cfg(feature = "sm89")]
pub use flash_sdpa_sm89::{FlashSdpaSm89Args, FlashSdpaSm89Descriptor, FlashSdpaSm89Plan};
pub use hyper_connection::{
    HyperConnectionArgs, HyperConnectionDescriptor, HyperConnectionPlan,
};
pub use kv_cache::{KvCacheAppendArgs, KvCacheAppendDescriptor, KvCacheAppendPlan};
pub use rope::{RopeArgs, RopeDescriptor, RopePlan};
pub use rope_backward::{RopeBackwardArgs, RopeBackwardDescriptor, RopeBackwardPlan};
// Phase 45 — RoPE scaling helpers.
pub use rope_scaling::{RopeScaledTableBuilder, RopeScaling};
pub use sdpa::{SdpaArgs, SdpaDescriptor, SdpaPlan};
pub use sdpa_backward::{SdpaBackwardArgs, SdpaBackwardDescriptor, SdpaBackwardPlan};

use baracuda_cutlass::{Error, Result};

/// Status-code → Result translation, shared across the attention family.
pub(crate) fn map_status(code: i32) -> Result<()> {
    match code {
        0 => Ok(()),
        1 => Err(Error::MisalignedOperand),
        2 => Err(Error::InvalidProblem(
            "baracuda-kernels-sys reported invalid problem",
        )),
        3 => Err(Error::Unsupported(
            "baracuda-kernels-sys reported unsupported configuration",
        )),
        4 => Err(Error::WorkspaceTooSmall { needed: 0, got: 0 }),
        n => Err(Error::CutlassInternal(n)),
    }
}

/// Crate-wide re-export of [`map_status`] for the sibling op-family
/// modules (Phase 54 — `gemm::sparse24` etc.) that share the same
/// status-code convention with the attention family. Dead-code allowed
/// because the call sites are feature-gated (`xformers_sparse24`) and
/// the function ships unconditionally.
#[doc(hidden)]
#[allow(dead_code)]
pub(crate) fn map_status_pub(code: i32) -> Result<()> {
    map_status(code)
}

/// Default RoPE base — `10000.0`, matching the original Llama /
/// Mistral / Gemma conventions.
pub const ROPE_DEFAULT_BASE: f32 = 10000.0;

// =========================================================================
// Phase 46 — FlashInfer cherry-pick: batched paged-KV decode, paged-KV
// append, cascade LSE merge. Three new plan families behind the
// `flashinfer` cargo feature. Plan files always compile; the FFI calls
// inside `run()` are `#[cfg(feature = "flashinfer")]`-gated so the
// public API surface exists even without the feature.
// =========================================================================
pub mod batch_paged_decode;
pub mod batch_paged_decode_fp8;
pub mod batch_paged_prefill;
pub mod batch_ragged_prefill;
pub mod cascade_attn;
pub mod paged_kv_append;

// Phase 54 — xFormers BlockSparseAttention (clean-room hand-port).
// Behind the `xformers_blocksparse` cargo feature on baracuda-kernels.
// Plan file always compiles; FFI calls inside `run()` are
// `#[cfg(feature = "xformers_blocksparse")]`-gated so the public API
// surface exists even without the feature.
pub mod sdpa_block_sparse;

// Phase 50 — Mamba-2 State-Space Duality (SSD) chunk-scan. Bespoke
// kernel under `kernels/ssd/ssd_chunk_scan_*.cu`; gated behind the
// `mamba` cargo feature because it pulls in the corresponding FFI
// symbols compiled from `baracuda-kernels-sys`. Listed under
// `attention` because of the SSD-as-attention duality (Dao & Gu 2024).
#[cfg(feature = "mamba")]
pub mod ssd_chunk_scan;

// Phase 50b — Mamba-1 selective_scan. Sibling to SSD; powers the
// Mamba-1 family (Mamba-7B / Falcon-Mamba / Codestral-Mamba). Gated
// behind the same `mamba` cargo feature as SSD.
#[cfg(feature = "mamba")]
pub mod selective_scan;

// Phase 56 — Ring Attention (sequence-parallel attention). Pure Rust
// orchestrator over the per-step kernel in `baracuda-kernels-sys` and
// the NCCL `Communicator::send` / `recv` rotation. Module compiles
// unconditionally for API discoverability; the `run()` method that
// actually invokes the kernel + NCCL collective is gated on the
// `ring_attention` cargo feature. See the module-level docs for the
// algorithm narrative + Tier-1 scope (head_dim=128, f16+bf16, FW only).
pub mod ring_attention;

pub use batch_paged_decode::{
    BatchPagedDecodeArgs, BatchPagedDecodeDescriptor, BatchPagedDecodePlan,
    PagedKvCacheDescriptor,
};
pub use batch_paged_prefill::{
    BatchPagedPrefillArgs, BatchPagedPrefillDescriptor, BatchPagedPrefillPlan,
};
pub use batch_paged_decode_fp8::{
    BatchPagedDecodeFp8Args, BatchPagedDecodeFp8Descriptor, BatchPagedDecodeFp8Plan, Fp8KvDtype,
};
pub use batch_ragged_prefill::{
    BatchRaggedPrefillArgs, BatchRaggedPrefillDescriptor, BatchRaggedPrefillPlan,
};
pub use cascade_attn::{
    CascadeAttentionArgs, CascadeAttentionDescriptor, CascadeAttentionPlan,
    CascadeMergeStatesArgs, CascadeMergeStatesDescriptor, CascadeMergeStatesPlan,
};
pub use paged_kv_append::{
    PagedKvAppendArgs, PagedKvAppendDescriptor, PagedKvAppendPlan,
};

// Phase 54 — xFormers BlockSparseAttention.
pub use sdpa_block_sparse::{
    SdpaBlockSparseArgs, SdpaBlockSparseDescriptor, SdpaBlockSparsePlan,
    SDPA_BLOCK_SPARSE_MAX_BLOCK, SDPA_BLOCK_SPARSE_MAX_D,
};

// Phase 50 — Mamba-2 SSD chunk-scan re-exports.
#[cfg(feature = "mamba")]
pub use ssd_chunk_scan::{
    SsdChunkScanArgs, SsdChunkScanBackwardArgs, SsdChunkScanBackwardDescriptor,
    SsdChunkScanBackwardPlan, SsdChunkScanDescriptor, SsdChunkScanPlan,
};

// Phase 50b — Mamba-1 selective_scan re-exports.
#[cfg(feature = "mamba")]
pub use selective_scan::{
    SelectiveScanArgs, SelectiveScanBackwardArgs, SelectiveScanBackwardDescriptor,
    SelectiveScanBackwardPlan, SelectiveScanDescriptor, SelectiveScanPlan,
};

// Phase 56 — Ring Attention re-exports.
pub use ring_attention::{
    RingAttentionArgs, RingAttentionDescriptor, RingAttentionPlan, RING_ATTENTION_HEAD_DIM,
};
