//! Per-row sampling + speculative-decode verification — Phase 66 Tier 2.
//!
//! Companions to [`TopKTopPSamplingPlan`](super::TopKTopPSamplingPlan):
//!
//! - [`PerRowSamplingPlan`] — the same sort-free samplers, but the filter
//!   threshold is a device array `[batch]` (one value per request), the
//!   canonical serving case. Routes to FlashInfer's `*_arr` entry points.
//! - [`SpeculativeSamplingPlan`] — speculative-decode accept/reject
//!   verification (`ChainSpeculativeSampling`).
//!
//! Both require the `flashinfer` cargo feature for `run`.

use core::ffi::c_void;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, ElementKind, KernelSku, MathPrecision, OpCategory, PlanPreference,
    PrecisionGuarantee, RandomKind, TensorMut, TensorRef, Workspace,
};

use crate::attention::map_status;

/// Which per-row sampler to run (thresholds supplied as device arrays in
/// the args, not the descriptor).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PerRowSampler {
    /// Per-row top-P (`top_p_arr` required).
    TopP,
    /// Per-row min-P (`min_p_arr` required).
    MinP,
    /// Per-row top-K + top-P (`top_k_arr` i32 + `top_p_arr` f32 required).
    /// For effectively top-K-only per row, pass `top_p_arr` of all `1.0`.
    /// (Standalone per-row top-K is omitted because FlashInfer types its
    /// `top_k_arr` inconsistently across the two samplers — see the
    /// launcher comment.)
    TopKTopP,
}

/// Descriptor for a per-row sort-free sampling op.
#[derive(Copy, Clone, Debug)]
pub struct PerRowSamplingDescriptor {
    /// Batch size (rows of `probs`).
    pub batch_size: i32,
    /// Vocabulary size (columns of `probs`).
    pub vocab_size: i32,
    /// Sampler family.
    pub sampler: PerRowSampler,
    /// Sort-based tiebreak on ambiguous cells.
    pub deterministic: bool,
}

/// Args for a per-row sampling launch. Supply the array(s) the chosen
/// [`PerRowSampler`] needs; leave the others `None`.
pub struct PerRowSamplingArgs<'a> {
    /// Row-normalized probabilities `[batch, vocab]` f32.
    pub probs: TensorRef<'a, f32, 2>,
    /// Per-row top-K cells `[batch]` i32.
    pub top_k_arr: Option<TensorRef<'a, i32, 1>>,
    /// Per-row top-P cutoff `[batch]` f32.
    pub top_p_arr: Option<TensorRef<'a, f32, 1>>,
    /// Per-row min-P multiplier `[batch]` f32.
    pub min_p_arr: Option<TensorRef<'a, f32, 1>>,
    /// Sampled indices `[batch]` i32 (written).
    pub output: TensorMut<'a, i32, 1>,
    /// Optional per-row "sample accepted" flags `[batch]` u8.
    pub valid: Option<TensorMut<'a, u8, 1>>,
    /// RNG seed.
    pub seed_val: u64,
    /// RNG philox offset.
    pub offset_val: u64,
}

/// Per-row sort-free sampling plan.
pub struct PerRowSamplingPlan {
    desc: PerRowSamplingDescriptor,
    sku: KernelSku,
}

impl PerRowSamplingPlan {
    /// Validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &PerRowSamplingDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.batch_size <= 0 || desc.vocab_size <= 0 {
            return Err(Error::InvalidProblem(
                "PerRowSamplingPlan: batch_size / vocab_size must be positive",
            ));
        }
        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            bit_stable_on_same_hardware: true,
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

    /// Validate args (shapes + presence of the required array).
    pub fn can_implement(&self, args: &PerRowSamplingArgs<'_>) -> Result<()> {
        let b = self.desc.batch_size;
        if args.probs.shape != [b, self.desc.vocab_size] {
            return Err(Error::InvalidProblem(
                "PerRowSamplingPlan: probs shape must be [batch, vocab]",
            ));
        }
        if args.output.shape != [b] {
            return Err(Error::InvalidProblem("PerRowSamplingPlan: output shape must be [batch]"));
        }
        let need_k = matches!(self.desc.sampler, PerRowSampler::TopKTopP);
        let need_p = matches!(self.desc.sampler, PerRowSampler::TopP | PerRowSampler::TopKTopP);
        let need_minp = matches!(self.desc.sampler, PerRowSampler::MinP);
        if need_k && args.top_k_arr.is_none() {
            return Err(Error::InvalidProblem("PerRowSamplingPlan: top_k_arr required"));
        }
        if need_p && args.top_p_arr.is_none() {
            return Err(Error::InvalidProblem("PerRowSamplingPlan: top_p_arr required"));
        }
        if need_minp && args.min_p_arr.is_none() {
            return Err(Error::InvalidProblem("PerRowSamplingPlan: min_p_arr required"));
        }
        if let Some(t) = &args.top_k_arr {
            if t.shape != [b] {
                return Err(Error::InvalidProblem("PerRowSamplingPlan: top_k_arr must be [batch]"));
            }
        }
        if let Some(t) = &args.top_p_arr {
            if t.shape != [b] {
                return Err(Error::InvalidProblem("PerRowSamplingPlan: top_p_arr must be [batch]"));
            }
        }
        if let Some(t) = &args.min_p_arr {
            if t.shape != [b] {
                return Err(Error::InvalidProblem("PerRowSamplingPlan: min_p_arr must be [batch]"));
            }
        }
        if let Some(v) = &args.valid {
            if v.shape != [b] {
                return Err(Error::InvalidProblem("PerRowSamplingPlan: valid must be [batch]"));
            }
        }
        if !args.probs.is_contiguous() || !args.output.is_contiguous() {
            return Err(Error::Unsupported(
                "PerRowSamplingPlan: probs / output must be contiguous",
            ));
        }
        Ok(())
    }

    /// Workspace bytes — always 0.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// SKU identity.
    #[inline]
    pub fn sku(&self) -> KernelSku {
        self.sku
    }

    /// Numerical guarantees.
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee
    }

    /// Launch the selected per-row sampler.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: PerRowSamplingArgs<'_>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        #[cfg(not(feature = "flashinfer"))]
        {
            let _ = (stream, &args);
            Err(Error::Unsupported(
                "PerRowSamplingPlan: `flashinfer` cargo feature is not enabled",
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
            let det = if self.desc.deterministic { 1 } else { 0 };
            let k_ptr = args
                .top_k_arr
                .as_ref()
                .map_or(core::ptr::null::<c_void>(), |t| t.data.as_raw().0 as *const c_void);
            let p_ptr = args
                .top_p_arr
                .as_ref()
                .map_or(core::ptr::null::<c_void>(), |t| t.data.as_raw().0 as *const c_void);
            let mp_ptr = args
                .min_p_arr
                .as_ref()
                .map_or(core::ptr::null::<c_void>(), |t| t.data.as_raw().0 as *const c_void);

            let status = match self.desc.sampler {
                PerRowSampler::TopP => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_flashinfer_top_p_sampling_f32_arr_run(
                        self.desc.batch_size, self.desc.vocab_size, p_ptr, det,
                        args.seed_val, args.offset_val, probs_ptr, output_ptr, valid_ptr, stream_ptr,
                    )
                },
                PerRowSampler::MinP => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_flashinfer_min_p_sampling_f32_arr_run(
                        self.desc.batch_size, self.desc.vocab_size, mp_ptr, det,
                        args.seed_val, args.offset_val, probs_ptr, output_ptr, valid_ptr, stream_ptr,
                    )
                },
                PerRowSampler::TopKTopP => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_flashinfer_top_k_top_p_sampling_f32_arr_run(
                        self.desc.batch_size, self.desc.vocab_size, k_ptr, p_ptr, det,
                        args.seed_val, args.offset_val, probs_ptr, output_ptr, valid_ptr, stream_ptr,
                    )
                },
            };
            map_status(status)
        }
    }
}

// =========================================================================
// Speculative-decode verification.
// =========================================================================

/// Descriptor for a speculative-decode verification op.
#[derive(Copy, Clone, Debug)]
pub struct SpeculativeSamplingDescriptor {
    /// Number of requests.
    pub batch_size: i32,
    /// Draft tokens proposed per request.
    pub num_speculative_tokens: i32,
    /// Vocabulary size.
    pub vocab_size: i32,
    /// Sort-based tiebreak on ambiguous cells.
    pub deterministic: bool,
}

/// Args bundle for speculative verification.
pub struct SpeculativeSamplingArgs<'a> {
    /// Draft probabilities `[batch, num_spec, vocab]` f32.
    pub draft_probs: TensorRef<'a, f32, 3>,
    /// Draft sampled token ids `[batch, num_spec]` i32.
    pub draft_token_ids: TensorRef<'a, i32, 2>,
    /// Target probabilities `[batch, num_spec + 1, vocab]` f32.
    pub target_probs: TensorRef<'a, f32, 3>,
    /// Accepted/corrected token ids `[batch, num_spec + 1]` i32 (written).
    pub output_token_ids: TensorMut<'a, i32, 2>,
    /// Per-request accepted-token count `[batch]` i32 (written).
    pub output_accepted_token_num: TensorMut<'a, i32, 1>,
    /// Per-request emitted-draft count `[batch]` i32 (written).
    pub output_emitted_draft_token_num: TensorMut<'a, i32, 1>,
    /// RNG seed.
    pub seed_val: u64,
    /// RNG philox offset.
    pub offset_val: u64,
}

/// Speculative-decode verification plan (FlashInfer `ChainSpeculativeSampling`).
pub struct SpeculativeSamplingPlan {
    desc: SpeculativeSamplingDescriptor,
    sku: KernelSku,
}

impl SpeculativeSamplingPlan {
    /// Validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &SpeculativeSamplingDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.batch_size <= 0 || desc.num_speculative_tokens <= 0 || desc.vocab_size <= 0 {
            return Err(Error::InvalidProblem(
                "SpeculativeSamplingPlan: extents must be positive",
            ));
        }
        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            bit_stable_on_same_hardware: true,
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

    /// Validate args against the descriptor.
    pub fn can_implement(&self, args: &SpeculativeSamplingArgs<'_>) -> Result<()> {
        let d = &self.desc;
        if args.draft_probs.shape != [d.batch_size, d.num_speculative_tokens, d.vocab_size] {
            return Err(Error::InvalidProblem("SpeculativeSamplingPlan: draft_probs shape"));
        }
        if args.draft_token_ids.shape != [d.batch_size, d.num_speculative_tokens] {
            return Err(Error::InvalidProblem("SpeculativeSamplingPlan: draft_token_ids shape"));
        }
        if args.target_probs.shape != [d.batch_size, d.num_speculative_tokens + 1, d.vocab_size] {
            return Err(Error::InvalidProblem("SpeculativeSamplingPlan: target_probs shape"));
        }
        if args.output_token_ids.shape != [d.batch_size, d.num_speculative_tokens + 1] {
            return Err(Error::InvalidProblem("SpeculativeSamplingPlan: output_token_ids shape"));
        }
        if args.output_accepted_token_num.shape != [d.batch_size]
            || args.output_emitted_draft_token_num.shape != [d.batch_size]
        {
            return Err(Error::InvalidProblem(
                "SpeculativeSamplingPlan: output count arrays must be [batch]",
            ));
        }
        if !args.draft_probs.is_contiguous()
            || !args.draft_token_ids.is_contiguous()
            || !args.target_probs.is_contiguous()
            || !args.output_token_ids.is_contiguous()
        {
            return Err(Error::Unsupported("SpeculativeSamplingPlan: tensors must be contiguous"));
        }
        Ok(())
    }

    /// Workspace bytes — always 0.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// SKU identity.
    #[inline]
    pub fn sku(&self) -> KernelSku {
        self.sku
    }

    /// Numerical guarantees.
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee
    }

    /// Run the accept/reject verification.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: SpeculativeSamplingArgs<'_>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        #[cfg(not(feature = "flashinfer"))]
        {
            let _ = (stream, &args);
            Err(Error::Unsupported(
                "SpeculativeSamplingPlan: `flashinfer` cargo feature is not enabled",
            ))
        }
        #[cfg(feature = "flashinfer")]
        {
            let stream_ptr = stream.as_raw() as *mut c_void;
            let status = unsafe {
                baracuda_kernels_sys::baracuda_kernels_flashinfer_chain_speculative_sampling_f32_run(
                    self.desc.batch_size,
                    self.desc.num_speculative_tokens,
                    self.desc.vocab_size,
                    if self.desc.deterministic { 1 } else { 0 },
                    args.seed_val,
                    args.offset_val,
                    args.draft_probs.data.as_raw().0 as *const c_void,
                    args.draft_token_ids.data.as_raw().0 as *const c_void,
                    args.target_probs.data.as_raw().0 as *const c_void,
                    args.output_token_ids.data.as_raw().0 as *mut c_void,
                    args.output_accepted_token_num.data.as_raw().0 as *mut c_void,
                    args.output_emitted_draft_token_num.data.as_raw().0 as *mut c_void,
                    stream_ptr,
                )
            };
            map_status(status)
        }
    }
}
