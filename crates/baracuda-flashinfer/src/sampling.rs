//! Sort-free token sampling plans.
//!
//! FlashInfer's sampling kernels draw one token per row directly from a
//! row-normalized probability tensor using a rejection-based scheme — no
//! global argsort of the vocabulary. This is the decode-time hot path
//! for autoregressive generation.
//!
//! Pick the filter via [`SamplerKind`]:
//!
//! - [`SamplerKind::TopK`] — keep the top-`K` cells.
//! - [`SamplerKind::TopP`] — nucleus: keep the smallest set of largest
//!   cells whose cumulative mass exceeds `top_p`.
//! - [`SamplerKind::MinP`] — keep cells with `prob >= min_p * row_max`.
//! - [`SamplerKind::TopKTopP`] — combined top-K then top-P (the
//!   canonical Llama / Mistral / Gemma decode sampler).
//!
//! The plan implementation lives in [`baracuda_kernels::random`]; this
//! module re-exports it under the `baracuda-flashinfer` namespace.
//!
//! Sampling twice with the same `(seed_val, offset_val)` and identical
//! `probs` is bit-stable. Set `deterministic` in the descriptor to make
//! the rare cumulative-boundary tiebreak reproducible as well.
//!
//! [`TopKTopPSamplingArgs::valid`] is optional (`None` to skip the
//! per-row "sample accepted" flags). Note: the vendored FlashInfer
//! sampling kernel (v0.6.12) dereferences its per-row success pointer
//! **unconditionally**, so passing a raw null straight to the kernel
//! would be an illegal access. Phase 66 fixed the launcher to hand the
//! kernel a stream-ordered scratch success buffer whenever the caller
//! passes `None`, so `valid: None` is now safe (verified on an RTX 4070
//! in `tests/sampling_smoke.rs`).
//!
//! # Not yet wired
//!
//! Repetition- / frequency- / presence-penalty logit transforms and
//! per-row sampler parameter arrays are FlashInfer features that require
//! additional vendored sources; they are tracked as a follow-up tier.
//! Apply penalties with baracuda's elementwise / scatter ops before
//! normalizing to `probs` in the meantime.

pub use baracuda_kernels::{
    SamplerKind, TopKTopPSamplingArgs, TopKTopPSamplingDescriptor, TopKTopPSamplingPlan,
};
