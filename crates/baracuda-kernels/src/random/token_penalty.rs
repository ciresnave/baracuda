//! Token-penalty logit transform — Phase 66 Tier 2 (bespoke).
//!
//! Applies the three standard autoregressive sampling penalties to a
//! logits tensor in place, given a per-`(request, token)` occurrence
//! count. Unlike the other plans in this module this is a NATIVE baracuda
//! kernel (FlashInfer ships no penalty op) — it is **not** behind the
//! `flashinfer` cargo feature and always runs.
//!
//! For each cell with prior count `c`:
//! - repetition penalty (HF, multiplicative): if `c > 0`,
//!   `logit = logit > 0 ? logit / rep : logit * rep` (`rep >= 1` penalizes)
//! - frequency penalty (OpenAI, additive): `logit -= freq * c`
//! - presence penalty (OpenAI, additive): `logit -= pres * (c > 0)`
//!
//! Disable a penalty by passing `rep = 1.0` / `freq = 0.0` / `pres = 0.0`.
//! Apply this BEFORE softmax + sampling.

use core::ffi::c_void;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, ElementKind, KernelSku, MathPrecision, OpCategory, PlanPreference,
    PrecisionGuarantee, RandomKind, TensorMut, TensorRef, Workspace,
};

use crate::attention::map_status;

/// Descriptor for a token-penalty logit transform.
#[derive(Copy, Clone, Debug)]
pub struct TokenPenaltyDescriptor {
    /// Batch size (rows of `logits` / `counts`).
    pub batch_size: i32,
    /// Vocabulary size (columns).
    pub vocab_size: i32,
    /// Repetition penalty (`>= 1.0` penalizes; `1.0` disables).
    pub rep_penalty: f32,
    /// Frequency penalty (subtracted `* count`; `0.0` disables).
    pub freq_penalty: f32,
    /// Presence penalty (subtracted once if seen; `0.0` disables).
    pub pres_penalty: f32,
}

/// Args bundle for a token-penalty launch.
pub struct TokenPenaltyArgs<'a> {
    /// Logits `[batch, vocab]` f32, modified in place.
    pub logits: TensorMut<'a, f32, 2>,
    /// Prior occurrence counts `[batch, vocab]` i32.
    pub counts: TensorRef<'a, i32, 2>,
}

/// Token-penalty plan (native baracuda; always available).
pub struct TokenPenaltyPlan {
    desc: TokenPenaltyDescriptor,
    sku: KernelSku,
}

impl TokenPenaltyPlan {
    /// Validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &TokenPenaltyDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.batch_size <= 0 || desc.vocab_size <= 0 {
            return Err(Error::InvalidProblem(
                "TokenPenaltyPlan: batch_size / vocab_size must be positive",
            ));
        }
        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Random,
            op: RandomKind::Multinomial as u16,
            element: ElementKind::F32,
            aux_element: Some(ElementKind::I32),
            layout: None,
            epilogue: None,
            arch: ArchSku::Sm80,
            backend: BackendKind::Bespoke,
            precision_guarantee,
        };
        Ok(Self { desc: *desc, sku })
    }

    /// Validate args against the descriptor.
    pub fn can_implement(&self, args: &TokenPenaltyArgs<'_>) -> Result<()> {
        let shape = [self.desc.batch_size, self.desc.vocab_size];
        if args.logits.shape != shape || args.counts.shape != shape {
            return Err(Error::InvalidProblem(
                "TokenPenaltyPlan: logits / counts shape must be [batch, vocab]",
            ));
        }
        if !args.logits.is_contiguous() || !args.counts.is_contiguous() {
            return Err(Error::Unsupported(
                "TokenPenaltyPlan: logits / counts must be contiguous",
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

    /// Apply the penalties in place.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: TokenPenaltyArgs<'_>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let logits_ptr = args.logits.data.as_raw().0 as *mut c_void;
        let counts_ptr = args.counts.data.as_raw().0 as *const c_void;
        let status = unsafe {
            baracuda_kernels_sys::baracuda_kernels_apply_token_penalty_f32_run(
                self.desc.batch_size,
                self.desc.vocab_size,
                self.desc.rep_penalty,
                self.desc.freq_penalty,
                self.desc.pres_penalty,
                logits_ptr,
                counts_ptr,
                stream_ptr,
            )
        };
        map_status(status)
    }
}
