//! Raw C-ABI FFI surface for the vendored FlashInfer inference kernels.
//!
//! FlashInfer (`flashinfer-ai/flashinfer`, Apache-2.0) is header-only /
//! template-heavy, so there is no shared library to dynamically load.
//! Instead, baracuda compiles thin C-ABI launcher shims around the
//! vendored FlashInfer headers inside [`baracuda-kernels-sys`]; this
//! crate re-exports those `extern "C"` symbols under a dedicated crate
//! name so downstream code can depend on the FlashInfer FFI surface
//! without pulling the whole kernels-sys symbol table into scope.
//!
//! Almost all callers should prefer the safe, typed wrappers in
//! [`baracuda-flashinfer`] (the sibling crate) over these raw symbols.
//!
//! # Feature gating
//!
//! Every symbol is behind the `flashinfer` cargo feature (OFF by
//! default), which transitively enables `baracuda-kernels-sys/flashinfer`
//! and compiles the vendored launcher `.cu` files. With the feature
//! off, this crate is empty.
//!
//! # Symbol families
//!
//! - `*_paged_decode_*` — batched paged-KV decode
//!   (`BatchDecodeWithPagedKVCacheDispatched`). f16 / bf16 / f32.
//! - `*_paged_kv_append_decode_*` — decode-time KV-cache append.
//! - `*_merge_state_in_place_*` / `*_merge_states_*` — cascade /
//!   prefix-cache LSE-aware attention-state merge.
//! - `*_top_k_sampling_*` / `*_top_p_sampling_*` /
//!   `*_min_p_sampling_*` / `*_top_k_top_p_sampling_*` — sort-free
//!   sampling from a row-normalized probability tensor.

#![no_std]

// The raw FlashInfer C-ABI, compiled + defined in `baracuda-kernels-sys`.
// Re-exported verbatim (raw names preserved) so this stays an honest
// `-sys` facade. Grouped by family for readability.
#[cfg(feature = "flashinfer")]
pub use baracuda_kernels_sys::{
    // Paged-KV decode.
    baracuda_kernels_flashinfer_paged_decode_workspace_size,
    baracuda_kernels_flashinfer_paged_decode_f16_run,
    baracuda_kernels_flashinfer_paged_decode_bf16_run,
    baracuda_kernels_flashinfer_paged_decode_f32_run,
    baracuda_kernels_flashinfer_paged_decode_can_implement,
    // Paged-KV append (decode-time, one token per request).
    baracuda_kernels_flashinfer_paged_kv_append_decode_f16_run,
    baracuda_kernels_flashinfer_paged_kv_append_decode_bf16_run,
    baracuda_kernels_flashinfer_paged_kv_append_decode_f32_run,
    baracuda_kernels_flashinfer_paged_kv_append_decode_can_implement,
    // Cascade / prefix-cache state merge.
    baracuda_kernels_flashinfer_merge_state_in_place_f16_run,
    baracuda_kernels_flashinfer_merge_state_in_place_bf16_run,
    baracuda_kernels_flashinfer_merge_state_in_place_f32_run,
    baracuda_kernels_flashinfer_merge_state_in_place_can_implement,
    baracuda_kernels_flashinfer_merge_states_f16_run,
    baracuda_kernels_flashinfer_merge_states_bf16_run,
    baracuda_kernels_flashinfer_merge_states_f32_run,
    baracuda_kernels_flashinfer_merge_states_can_implement,
    // Sort-free sampling.
    baracuda_kernels_flashinfer_top_k_sampling_f32_run,
    baracuda_kernels_flashinfer_top_k_sampling_f32_can_implement,
    baracuda_kernels_flashinfer_top_p_sampling_f32_run,
    baracuda_kernels_flashinfer_top_p_sampling_f32_can_implement,
    baracuda_kernels_flashinfer_min_p_sampling_f32_run,
    baracuda_kernels_flashinfer_min_p_sampling_f32_can_implement,
    baracuda_kernels_flashinfer_top_k_top_p_sampling_f32_run,
    baracuda_kernels_flashinfer_top_k_top_p_sampling_f32_can_implement,
};
