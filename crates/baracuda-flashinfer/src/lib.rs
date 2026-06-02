//! Safe, typed Rust wrappers for NVIDIA **FlashInfer**'s inference-
//! serving kernels — the vLLM-style serving surface for the baracuda
//! CUDA stack.
//!
//! FlashInfer (`flashinfer-ai/flashinfer`, Apache-2.0) is an
//! inference-focused attention / sampling kernel library. baracuda
//! vendors a cherry-picked subset of it (see
//! `crates/baracuda-kernels-sys/vendor/flashinfer/`) and exposes typed
//! `*Plan` wrappers — the same select / can_implement / run shape used
//! across the rest of the baracuda kernel surface.
//!
//! This crate is a thin, cohesive **safe facade**: the plan
//! implementations live in [`baracuda_kernels`] (so they integrate with
//! the shared SKU / autotuner / telemetry machinery), and the raw FFI
//! lives in [`baracuda_flashinfer_sys`]. This crate re-groups those into
//! a serving-oriented API and is the documented entry point.
//!
//! # Capabilities
//!
//! ## [`attention`] — paged-KV attention
//!
//! - [`BatchPagedDecodePlan`] — batched paged-KV **decode** (one query
//!   row per request, KV history in a paged store). The core vLLM
//!   serving primitive. f16 / bf16 / f32, head_dim ∈ {64, 128, 256}.
//! - [`BatchPagedPrefillPlan`] — batched paged-KV **prefill** (multiple
//!   query rows per request, ragged via `q_indptr`, attending over the
//!   paged history; causal or full). The prompt-ingestion primitive.
//!   f16 / bf16.
//! - [`PagedKvAppendPlan`] — decode-time KV-cache **append** (writes the
//!   freshly-computed K/V for the current token into the paged store).
//! - [`CascadeAttentionPlan`] — LSE-aware pairwise **merge** of partial
//!   attention states, the building block for prefix-cache / shared-prompt
//!   reuse.
//! - [`CascadeMergeStatesPlan`] — many-way (fan-in > 2) cascade merge, for
//!   multi-level shared-prefix trees / overlapping prefix caches.
//!
//! ## [`sampling`] — sort-free token sampling
//!
//! - [`TopKTopPSamplingPlan`] — top-K / top-P / min-P / combined
//!   top-K+top-P sampling directly from a probability tensor, with no
//!   global sort. Select the variant via [`SamplerKind`].
//!
//! # Feature gating
//!
//! All kernels are behind the `flashinfer` cargo feature (OFF by
//! default). With the feature off the plan types still exist and
//! `select` / `can_implement` still validate shapes, but `run` returns
//! `Error::Unsupported`. Enable the feature to compile the vendored
//! kernels and execute on device.
//!
//! # What is *not* here (yet)
//!
//! FlashInfer's speculative-decode verification and FP8 KV-cache paths
//! require additional vendored sources / instantiations beyond baracuda's
//! current cherry-pick. They are tracked as follow-up tiers.
//!
//! # Example shape (paged decode)
//!
//! ```no_run
//! # #[cfg(feature = "flashinfer")]
//! # fn demo(stream: &baracuda_driver::Stream) -> baracuda_flashinfer::Result<()> {
//! use baracuda_flashinfer::{
//!     BatchPagedDecodePlan, BatchPagedDecodeDescriptor, PagedKvCacheDescriptor,
//!     ElementKind, PlanPreference,
//! };
//! use half::f16;
//!
//! let desc = BatchPagedDecodeDescriptor {
//!     batch_size: 8,
//!     num_qo_heads: 32,
//!     sm_scale: 1.0 / (128.0_f32).sqrt(),
//!     paged_kv: PagedKvCacheDescriptor {
//!         page_size: 16,
//!         num_total_pages: 1024,
//!         num_kv_heads: 8, // GQA group size 4
//!         head_dim: 128,
//!         element: ElementKind::F16,
//!     },
//! };
//! let plan = BatchPagedDecodePlan::<f16>::select(stream, &desc, PlanPreference::default())?;
//! let _ws_bytes = plan.workspace_size();
//! // ... allocate workspace + page table, then plan.run(stream, ws, args)
//! # Ok(())
//! # }
//! ```

#![no_std]

pub mod attention;
pub mod sampling;

/// Shared error / result type (re-exported from the baracuda kernel
/// surface so callers don't need a separate import).
pub use baracuda_kernels::{Error, Result};

/// Support types used across the FlashInfer plan APIs.
pub use baracuda_kernels::{
    contiguous_stride, BackendKind, ElementKind, PlanPreference, PrecisionGuarantee, TensorMut,
    TensorRef, Workspace,
};

/// Raw C-ABI FFI surface, for callers that need to drop below the safe
/// layer. Prefer the typed plans above.
pub use baracuda_flashinfer_sys as sys;

/// Glob-importable common surface: `use baracuda_flashinfer::prelude::*;`.
pub mod prelude {
    pub use crate::attention::{
        BatchPagedDecodeArgs, BatchPagedDecodeDescriptor, BatchPagedDecodePlan,
        BatchPagedPrefillArgs, BatchPagedPrefillDescriptor, BatchPagedPrefillPlan,
        CascadeAttentionArgs, CascadeAttentionDescriptor, CascadeAttentionPlan,
        CascadeMergeStatesArgs, CascadeMergeStatesDescriptor, CascadeMergeStatesPlan,
        PagedKvAppendArgs, PagedKvAppendDescriptor, PagedKvAppendPlan, PagedKvCacheDescriptor,
    };
    pub use crate::sampling::{
        SamplerKind, TopKTopPSamplingArgs, TopKTopPSamplingDescriptor, TopKTopPSamplingPlan,
    };
    pub use crate::{
        contiguous_stride, BackendKind, ElementKind, Error, PlanPreference, Result, TensorMut,
        TensorRef, Workspace,
    };
}
