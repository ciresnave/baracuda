//! Sort-free top-K / top-P / min-P sampling — Phase 46 (FlashInfer
//! cherry-pick).
//!
//! Faster decode-time alternative to baracuda's existing
//! `topk + softmax + multinomial` pipeline. The op takes a row-
//! normalized probability tensor (one row per request) and produces
//! one sampled index per row.
//!
//! ## Variants
//!
//! Each variant maps to a dedicated FlashInfer launcher. Pick the
//! sampler that matches your filter:
//!
//! - [`SamplerKind::TopK`] — keep only the top-`K` cells.
//! - [`SamplerKind::TopP`] — keep the smallest set of largest cells
//!   whose cumulative mass exceeds `top_p`.
//! - [`SamplerKind::MinP`] — keep cells whose probability
//!   `>= min_p * max_prob_in_row`.
//! - [`SamplerKind::TopKTopP`] — combine top-K and top-P (the
//!   canonical decode hot path for Llama / Mistral / Gemma).
//!
//! ## Determinism
//!
//! The sampler is internally rejection-based, drawing a fresh uniform
//! `u ~ U(0, 1)` per row from a philox stream seeded with
//! `(seed_val, offset_val)`. With `deterministic == true`, FlashInfer
//! falls back to a sort-based tiebreaker on the rare ambiguous-cell
//! case (cells where the cumulative-sum boundary lands exactly on a
//! cell start).
//!
//! Calling the sampler twice with the same `(seed_val, offset_val)`
//! and identical `probs` is bit-stable.
//!
//! ## Caller contract
//!
//! - `probs` : `[batch, vocab]` row-major f32. Each row must be
//!   non-negative and sum to ~1 (you should typically chain after
//!   softmax + exp).
//! - `output` : `[batch]` i32, written.
//! - `valid` : `[batch]` u8 bool, written if non-null. 1 means the
//!   sample was accepted; 0 means rejection sampling timed out and
//!   the caller should re-draw with a fresh seed.

use core::ffi::c_void;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, ElementKind, KernelSku, MathPrecision, OpCategory, PlanPreference,
    PrecisionGuarantee, RandomKind, TensorMut, TensorRef, Workspace,
};

use crate::attention::map_status;

/// Which sort-free sampler to run.
///
/// Note: only `PartialEq` is derived (not `Eq`) because `top_p` /
/// `min_p` are `f32`, and `f32` doesn't satisfy `Eq` (NaN != NaN).
#[derive(Copy, Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum SamplerKind {
    /// Keep only the top-`K` cells (K = `top_k`).
    TopK {
        /// Cells kept per row. Must be in `[1, vocab_size]`.
        top_k: i32,
    },
    /// Keep the smallest top-prob set whose cumulative mass > `top_p`.
    TopP {
        /// Cumulative-mass cutoff in `(0, 1]`.
        top_p: f32,
    },
    /// Keep cells `prob >= min_p * row_max`.
    MinP {
        /// Multiplier of the per-row max prob in `(0, 1]`.
        min_p: f32,
    },
    /// Combined top-K then top-P filter. Canonical decode sampler.
    TopKTopP {
        /// Cells kept per row. Must be in `[1, vocab_size]`.
        top_k: i32,
        /// Cumulative-mass cutoff in `(0, 1]`.
        top_p: f32,
    },
}

/// Descriptor for a sort-free sampling op.
#[derive(Copy, Clone, Debug)]
pub struct TopKTopPSamplingDescriptor {
    /// Batch size (rows of `probs`).
    pub batch_size: i32,
    /// Vocabulary size (columns of `probs`).
    pub vocab_size: i32,
    /// Sampler family + filter parameters.
    pub sampler: SamplerKind,
    /// If true, FlashInfer falls back to a sort-based tiebreaker on
    /// ambiguous cells. Documented in the module docstring.
    pub deterministic: bool,
}

/// Args bundle for a sort-free sampling launch.
pub struct TopKTopPSamplingArgs<'a> {
    /// Row-normalized probabilities `[batch, vocab]` f32.
    pub probs: TensorRef<'a, f32, 2>,
    /// Sampled indices `[batch]` i32 (written).
    pub output: TensorMut<'a, i32, 1>,
    /// Optional per-row "sample accepted" flags `[batch]` u8 bool.
    /// `None` to skip emitting them.
    pub valid: Option<TensorMut<'a, u8, 1>>,
    /// RNG seed (shared across the batch).
    pub seed_val: u64,
    /// RNG philox offset.
    pub offset_val: u64,
}

/// Sort-free top-K / top-P / min-P sampling plan.
///
/// Routes to FlashInfer's `Top*FromProb` family. Requires the
/// `flashinfer` cargo feature.
pub struct TopKTopPSamplingPlan {
    desc: TopKTopPSamplingDescriptor,
    sku: KernelSku,
}

impl TopKTopPSamplingPlan {
    /// Pick a sampler kernel + validate filter parameters against the
    /// descriptor. Returns `Error::InvalidProblem` for out-of-range
    /// `top_k` / `top_p` / `min_p` values.
    pub fn select(
        _stream: &Stream,
        desc: &TopKTopPSamplingDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.batch_size <= 0 || desc.vocab_size <= 0 {
            return Err(Error::InvalidProblem(
                "TopKTopPSamplingPlan: batch_size / vocab_size must be positive",
            ));
        }
        match desc.sampler {
            SamplerKind::TopK { top_k } => {
                if top_k <= 0 || top_k > desc.vocab_size {
                    return Err(Error::InvalidProblem(
                        "TopKTopPSamplingPlan: top_k must be in [1, vocab_size]",
                    ));
                }
            }
            SamplerKind::TopP { top_p } => {
                if !(top_p > 0.0 && top_p <= 1.0) {
                    return Err(Error::InvalidProblem(
                        "TopKTopPSamplingPlan: top_p must be in (0, 1]",
                    ));
                }
            }
            SamplerKind::MinP { min_p } => {
                if !(min_p > 0.0 && min_p <= 1.0) {
                    return Err(Error::InvalidProblem(
                        "TopKTopPSamplingPlan: min_p must be in (0, 1]",
                    ));
                }
            }
            SamplerKind::TopKTopP { top_k, top_p } => {
                if top_k <= 0 || top_k > desc.vocab_size {
                    return Err(Error::InvalidProblem(
                        "TopKTopPSamplingPlan: top_k must be in [1, vocab_size]",
                    ));
                }
                if !(top_p > 0.0 && top_p <= 1.0) {
                    return Err(Error::InvalidProblem(
                        "TopKTopPSamplingPlan: top_p must be in (0, 1]",
                    ));
                }
            }
        }
        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            bit_stable_on_same_hardware: true,
            // Deterministic across same-(seed, offset) repeat. Cell-
            // selection determinism on tiebreaker cells is controlled
            // by `desc.deterministic` (FlashInfer's sort-fallback).
            deterministic: desc.deterministic,
        };
        let sku = KernelSku {
            category: OpCategory::Random,
            op: RandomKind::Multinomial as u16,
            element: ElementKind::F32,
            aux_element: Some(ElementKind::I32),
            layout: None,
            epilogue: None,
            arch: ArchSku::Sm80,
            backend: BackendKind::FlashInfer,
            precision_guarantee,
        };
        Ok(Self { desc: *desc, sku })
    }

    /// Validate args against the descriptor (shape + contiguity check).
    pub fn can_implement(&self, args: &TopKTopPSamplingArgs<'_>) -> Result<()> {
        if args.probs.shape != [self.desc.batch_size, self.desc.vocab_size] {
            return Err(Error::InvalidProblem(
                "TopKTopPSamplingPlan: probs shape must be [batch_size, vocab_size]",
            ));
        }
        if args.output.shape != [self.desc.batch_size] {
            return Err(Error::InvalidProblem(
                "TopKTopPSamplingPlan: output shape must be [batch_size]",
            ));
        }
        if let Some(v) = &args.valid {
            if v.shape != [self.desc.batch_size] {
                return Err(Error::InvalidProblem(
                    "TopKTopPSamplingPlan: valid shape must be [batch_size]",
                ));
            }
        }
        if !args.probs.is_contiguous() || !args.output.is_contiguous() {
            return Err(Error::Unsupported(
                "TopKTopPSamplingPlan: probs / output must be contiguous",
            ));
        }
        Ok(())
    }

    /// Required workspace bytes (always 0 — sampling is workspace-free).
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// SKU identity (telemetry / autotuner key).
    #[inline]
    pub fn sku(&self) -> KernelSku {
        self.sku
    }

    /// Numerical guarantees of this plan.
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee
    }

    /// Launch the selected sampler on the supplied stream.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: TopKTopPSamplingArgs<'_>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        #[cfg(not(feature = "flashinfer"))]
        {
            let _ = (stream, &args);
            Err(Error::Unsupported(
                "TopKTopPSamplingPlan: `flashinfer` cargo feature is not enabled",
            ))
        }
        #[cfg(feature = "flashinfer")]
        {
            let stream_ptr = stream.as_raw() as *mut c_void;
            let probs_ptr = args.probs.data.as_raw().0 as *const c_void;
            let output_ptr = args.output.data.as_raw().0 as *mut c_void;
            let valid_ptr = match &args.valid {
                Some(v) => v.data.as_raw().0 as *mut c_void,
                None => core::ptr::null_mut::<c_void>(),
            };
            let det_flag = if self.desc.deterministic { 1 } else { 0 };

            let status = match self.desc.sampler {
                SamplerKind::TopK { top_k } => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_flashinfer_top_k_sampling_f32_run(
                        self.desc.batch_size,
                        self.desc.vocab_size,
                        top_k,
                        det_flag,
                        args.seed_val,
                        args.offset_val,
                        probs_ptr,
                        output_ptr,
                        valid_ptr,
                        stream_ptr,
                    )
                },
                SamplerKind::TopP { top_p } => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_flashinfer_top_p_sampling_f32_run(
                        self.desc.batch_size,
                        self.desc.vocab_size,
                        top_p,
                        det_flag,
                        args.seed_val,
                        args.offset_val,
                        probs_ptr,
                        output_ptr,
                        valid_ptr,
                        stream_ptr,
                    )
                },
                SamplerKind::MinP { min_p } => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_flashinfer_min_p_sampling_f32_run(
                        self.desc.batch_size,
                        self.desc.vocab_size,
                        min_p,
                        det_flag,
                        args.seed_val,
                        args.offset_val,
                        probs_ptr,
                        output_ptr,
                        valid_ptr,
                        stream_ptr,
                    )
                },
                SamplerKind::TopKTopP { top_k, top_p } => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_flashinfer_top_k_top_p_sampling_f32_run(
                        self.desc.batch_size,
                        self.desc.vocab_size,
                        top_k,
                        top_p,
                        det_flag,
                        args.seed_val,
                        args.offset_val,
                        probs_ptr,
                        output_ptr,
                        valid_ptr,
                        stream_ptr,
                    )
                },
            };
            map_status(status)
        }
    }
}
