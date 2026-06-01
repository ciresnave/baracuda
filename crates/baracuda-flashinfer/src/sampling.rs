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
//! # IMPORTANT — `valid` is currently required
//!
//! Although [`TopKTopPSamplingArgs::valid`] is an `Option` and is
//! *documented* by the Phase 46 plan as skippable with `None`, the
//! vendored FlashInfer sampling kernel (v0.6.12) dereferences the
//! per-row success pointer **unconditionally**. Passing `None` therefore
//! triggers a CUDA illegal-memory-access (surfacing at the next stream
//! sync), not a clean error — confirmed on an RTX 4070. Until the
//! underlying Phase 46 launcher is fixed to allocate a scratch success
//! buffer when `valid` is null, **always pass `Some(buffer)`** of
//! `[batch]` `u8`. The `tests/sampling_smoke.rs` smoke tests pass a
//! `valid` buffer for exactly this reason.
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
