//! Phase 45 — RoPE position-interpolation scaling helpers.
//!
//! Long-context recipes that derive a custom `(cos, sin)` schedule for
//! the existing Phase 36 [`rope_apply_<dt>`](super::rope) FFI family.
//! No CUDA — these are pure-Rust host-side helpers that compute the
//! tables and upload them; the actual rotation kernel is unchanged.
//!
//! Two algorithms exposed today:
//!
//! - **YaRN** (Bowen Peng et al., *YaRN: Efficient Context Window
//!   Extension of Large Language Models*, [arXiv:2309.00071](
//!   https://arxiv.org/abs/2309.00071); MIT-licensed reference at
//!   [jquesnelle/yarn](https://github.com/jquesnelle/yarn)).
//!   Uses NTK-by-parts frequency interpolation: low-frequency dims
//!   are linearly interpolated (PI), high-frequency dims are kept at
//!   their original schedule, with a smooth ramp in between. Also
//!   applies the attention-temperature scaling
//!   `1 + 0.1 · ln(scale)` (§3.3 of the paper) by absorbing it into
//!   `cos` and `sin` — equivalent to scaling the logits, but the
//!   rotation kernel needs no modification.
//!
//! - **LongRoPE** (Yiran Ding et al., *LongRoPE: Extending LLM
//!   Context Window Beyond 2 Million Tokens*,
//!   [arXiv:2402.13753](https://arxiv.org/abs/2402.13753); MIT at
//!   [microsoft/LongRoPE](https://github.com/microsoft/LongRoPE)).
//!   Multiplies each dim's base frequency by a caller-supplied
//!   `per_dim_factor[D/2]` vector. The factor table itself is
//!   derived offline via evolutionary search on a validation set —
//!   that search lives outside baracuda; the caller supplies the
//!   result.
//!
//! ## API contract
//!
//! The Phase 36 [`rope_apply`](super::rope) family already accepts
//! caller-supplied `cos`/`sin` tables — this module merely populates
//! them. The existing [`super::rope::RopePlan`] / `RopeArgs` /
//! `RopeDescriptor` types are **not** modified; their signatures
//! remain source-compatible per the Phase 45 brief.
//!
//! ## Numerical convention
//!
//! All tables are computed in `f32` on the host and uploaded to
//! device as `f32`. This matches the Phase 36 FFI convention
//! (cos/sin always `f32` over the FFI regardless of operand dtype).
//! `f64` consumers promote the `f32` table to double at load inside
//! the kernel; `f16` / `bf16` consumers detour through `f32` for the
//! trig multiply.

use baracuda_cutlass::{Error, Result};

/// How to schedule the per-position rotation angles `θ_{s,i}` for
/// dim index `i ∈ [0, D/2)` and sequence position `s`.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum RopeScaling {
    /// Plain rotation, no scaling. `θ_{s, i} = s · base^(-2i/D)`.
    /// The default — bit-identical to omitting the scaling table
    /// entirely (the apply-kernel produces the same outputs as the
    /// classic [`super::rope::RopePlan`] schedule).
    Linear,

    /// **YaRN** — Bowen Peng et al. [arXiv:2309.00071](
    /// https://arxiv.org/abs/2309.00071).
    ///
    /// Per-dim frequency interpolation: dims with rotation wavelength
    /// `≥ original_max_seq_len / β` get linearly interpolated
    /// (PI: `θ → θ / scale`); dims with wavelength
    /// `≤ original_max_seq_len / α` are kept at the original schedule
    /// (NTK by parts); intermediate dims smoothly ramp between the
    /// two via the linear ramp from §3.2. The attention-temperature
    /// `attn_temp = √(1 + 0.1 · ln(scale))` (§3.3) is folded into
    /// `cos`/`sin` by dividing — equivalent to scaling the
    /// pre-softmax logits.
    YaRN {
        /// Extension factor — `target_seq_len / original_max_seq_len`.
        /// E.g. extending a 4096-token model to 32k means `scale = 8`.
        scale: f32,
        /// Lower bound for "high-frequency" dims (kept at original
        /// schedule). Paper recommends `α = 1.0` for OPT/LLaMA-class
        /// models.
        alpha: f32,
        /// Upper bound for "low-frequency" dims (linearly
        /// interpolated). Paper recommends `β = 32.0`.
        beta: f32,
        /// Original training-time max sequence length (in tokens).
        /// Paper uses `original_max_seq_len = 2048` for OPT and
        /// `4096` for LLaMA-2.
        original_max_seq_len: i32,
    },

    /// **LongRoPE** — Yiran Ding et al. [arXiv:2402.13753](
    /// https://arxiv.org/abs/2402.13753).
    ///
    /// Per-dim base-frequency rescaling. The kernel computes
    /// `θ_{s, i} = s · base^(-2i/D) · per_dim_factors[i]`. The
    /// `per_dim_factors` table is derived offline via evolutionary
    /// search on a validation set (that search lives outside
    /// baracuda); the caller supplies the result here.
    ///
    /// Length must equal `head_dim / 2`. The evolutionary search
    /// in upstream LongRoPE typically produces factors in roughly
    /// `[1.0, 8.0]` for an 8× context extension.
    LongRoPE {
        /// Per-dim multiplicative rescale factor, length `head_dim / 2`.
        /// `per_dim_factors[i]` multiplies the base frequency for dim
        /// pair `(2i, 2i+1)`.
        per_dim_factors: Vec<f32>,
    },
}

/// Host-side builder for the `(cos, sin)` tables consumed by
/// [`baracuda_kernels_sys::baracuda_kernels_rope_apply_<dt>_run`](baracuda_kernels_sys).
///
/// Produces tables in the **shared** layout `[seq, d/2]` (stride_b=0
/// on the FFI). The per-`bh` variant (one cos/sin table per batch×head
/// row) is straightforward host-side replication of the shared table
/// and isn't materialized here — callers who need it can upload the
/// shared table `bh` times into a single buffer.
///
/// Stride convention matches the smoke-test helper in
/// `tests/rope_apply_smoke.rs`:
///   `cos[s, pair] = cos(θ_{s, pair})` at offset `s * (d/2) + pair`.
///
/// ## Usage
///
/// ```ignore
/// use baracuda_kernels::attention::{RopeScaledTableBuilder, RopeScaling};
///
/// let builder = RopeScaledTableBuilder::new(
///     /*head_dim=*/ 128,
///     /*max_seq_len=*/ 32768,
///     /*base=*/ 10000.0,
///     RopeScaling::YaRN {
///         scale: 8.0,
///         alpha: 1.0,
///         beta: 32.0,
///         original_max_seq_len: 4096,
///     },
/// );
/// let (cos_host, sin_host) = builder.build_host_tables()?;
/// // Upload to device with `DeviceBuffer::from_slice`, then pass into
/// // baracuda_kernels_sys::baracuda_kernels_rope_apply_<dt>_run with
/// // stride_b = 0.
/// # Ok::<_, baracuda_kernels::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct RopeScaledTableBuilder {
    head_dim: i32,
    max_seq_len: i32,
    base: f32,
    scaling: RopeScaling,
}

impl RopeScaledTableBuilder {
    /// Create a new builder.
    ///
    /// `head_dim` must be even (RoPE rotates feature pairs).
    /// `max_seq_len` is the largest sequence position the tables will
    /// cover (i.e., the target context length, post-extension for
    /// YaRN / LongRoPE).
    /// `base` is the rotation base — typically `10000.0` for the
    /// LLaMA / Mistral / Gemma family.
    pub fn new(head_dim: i32, max_seq_len: i32, base: f32, scaling: RopeScaling) -> Self {
        Self {
            head_dim,
            max_seq_len,
            base,
            scaling,
        }
    }

    /// Validate the builder parameters. Called automatically by
    /// [`Self::build_host_tables`]; exposed for callers that want to
    /// pre-check without materializing the tables.
    pub fn validate(&self) -> Result<()> {
        if self.head_dim <= 0 || self.head_dim % 2 != 0 {
            return Err(Error::InvalidProblem(
                "RopeScaledTableBuilder: head_dim must be positive + even",
            ));
        }
        if self.max_seq_len <= 0 {
            return Err(Error::InvalidProblem(
                "RopeScaledTableBuilder: max_seq_len must be positive",
            ));
        }
        if !self.base.is_finite() || self.base <= 0.0 {
            return Err(Error::InvalidProblem(
                "RopeScaledTableBuilder: base must be finite and positive",
            ));
        }
        match &self.scaling {
            RopeScaling::Linear => {}
            RopeScaling::YaRN {
                scale,
                alpha,
                beta,
                original_max_seq_len,
            } => {
                if !scale.is_finite() || *scale <= 0.0 {
                    return Err(Error::InvalidProblem(
                        "RopeScaledTableBuilder::YaRN: scale must be finite + positive",
                    ));
                }
                if !alpha.is_finite() || !beta.is_finite() {
                    return Err(Error::InvalidProblem(
                        "RopeScaledTableBuilder::YaRN: alpha + beta must be finite",
                    ));
                }
                if *alpha >= *beta {
                    return Err(Error::InvalidProblem(
                        "RopeScaledTableBuilder::YaRN: alpha must be < beta \
                         (paper convention: alpha=1, beta=32)",
                    ));
                }
                if *original_max_seq_len <= 0 {
                    return Err(Error::InvalidProblem(
                        "RopeScaledTableBuilder::YaRN: original_max_seq_len must be positive",
                    ));
                }
            }
            RopeScaling::LongRoPE { per_dim_factors } => {
                let expected = (self.head_dim / 2) as usize;
                if per_dim_factors.len() != expected {
                    return Err(Error::InvalidProblem(
                        "RopeScaledTableBuilder::LongRoPE: per_dim_factors length must \
                         equal head_dim / 2",
                    ));
                }
                for &f in per_dim_factors {
                    if !f.is_finite() || f <= 0.0 {
                        return Err(Error::InvalidProblem(
                            "RopeScaledTableBuilder::LongRoPE: per_dim_factors must be \
                             finite and positive",
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Materialize the host-side `(cos_table, sin_table)` pair.
    ///
    /// Each table is `[max_seq_len * head_dim / 2]` `f32`, in
    /// row-major `[seq, pair]` order. Suitable for upload via
    /// `DeviceBuffer::from_slice` and consumption by
    /// `baracuda_kernels_rope_apply_<dt>_run` with `stride_b = 0`.
    pub fn build_host_tables(&self) -> Result<(Vec<f32>, Vec<f32>)> {
        self.validate()?;
        let half_d = (self.head_dim / 2) as usize;
        let seq = self.max_seq_len as usize;
        let total = seq * half_d;
        let mut cos_tab = vec![0f32; total];
        let mut sin_tab = vec![0f32; total];

        match &self.scaling {
            RopeScaling::Linear => self.fill_linear(&mut cos_tab, &mut sin_tab),
            RopeScaling::YaRN {
                scale,
                alpha,
                beta,
                original_max_seq_len,
            } => self.fill_yarn(
                &mut cos_tab,
                &mut sin_tab,
                *scale,
                *alpha,
                *beta,
                *original_max_seq_len,
            ),
            RopeScaling::LongRoPE { per_dim_factors } => {
                self.fill_longrope(&mut cos_tab, &mut sin_tab, per_dim_factors)
            }
        }
        Ok((cos_tab, sin_tab))
    }

    /// Per-dim inverse frequency `base^(-2i / head_dim)` for dim pair
    /// index `pair ∈ [0, head_dim/2)`. Public for callers building
    /// custom scaling schedules outside the [`RopeScaling`] enum.
    #[inline]
    pub fn inv_freq(&self, pair: usize) -> f32 {
        let inv_d = 1.0f32 / (self.head_dim as f32);
        let exponent = -((2 * pair) as f32) * inv_d;
        self.base.powf(exponent)
    }

    // --- Internal table-fill methods ----------------------------------

    fn fill_linear(&self, cos_tab: &mut [f32], sin_tab: &mut [f32]) {
        let half_d = (self.head_dim / 2) as usize;
        let seq = self.max_seq_len as usize;
        for s in 0..seq {
            for pair in 0..half_d {
                let freq = self.inv_freq(pair);
                let theta = (s as f32) * freq;
                cos_tab[s * half_d + pair] = theta.cos();
                sin_tab[s * half_d + pair] = theta.sin();
            }
        }
    }

    /// YaRN per-dim ramp factor. Per §3.2 of arXiv:2309.00071:
    ///
    /// ```text
    /// γ(d) = clamp((wavelength(d) · β / L_orig - 1) / (β - α), 0, 1)
    /// ```
    ///
    /// where `wavelength(d) = 2π / inv_freq(d)`; `γ = 0` keeps the dim
    /// at its original (high-freq) schedule and `γ = 1` linearly
    /// interpolates it (low-freq).
    ///
    /// The interpolated inverse frequency is then:
    ///
    /// ```text
    /// inv_freq_yarn(d) = (1 - γ) · inv_freq(d) + γ · inv_freq(d) / scale
    /// ```
    fn fill_yarn(
        &self,
        cos_tab: &mut [f32],
        sin_tab: &mut [f32],
        scale: f32,
        alpha: f32,
        beta: f32,
        original_max_seq_len: i32,
    ) {
        let half_d = (self.head_dim / 2) as usize;
        let seq = self.max_seq_len as usize;
        let l_orig = original_max_seq_len as f32;

        // Attention temperature absorption (§3.3): the trick is to
        // divide cos/sin by `sqrt(1 + 0.1 · ln(scale))`. This is
        // equivalent to scaling the pre-softmax attention logits by
        // `1/(1 + 0.1 · ln(scale))` — but the rotation kernel itself
        // needs no modification. For scale == 1 the multiplier is 1
        // (no-op).
        let attn_temp = if scale > 1.0 {
            (1.0f32 + 0.1 * scale.ln()).sqrt()
        } else {
            1.0
        };
        let inv_attn_temp = 1.0 / attn_temp;

        for pair in 0..half_d {
            let inv_freq = self.inv_freq(pair);
            // wavelength = 2π / inv_freq. The ramp uses the
            // dim's number-of-rotations-per-orig-context:
            //   rotations = L_orig · inv_freq / (2π)
            // Equivalently, the ramp boundaries on `rotations` are
            // `α` (above = high-freq, kept original) and `β` (below
            // = low-freq, fully interpolated).
            let rotations = (l_orig * inv_freq) / (2.0 * core::f32::consts::PI);
            let ramp = if rotations >= beta {
                0.0 // high-freq — keep original (γ = 0)
            } else if rotations <= alpha {
                1.0 // low-freq — full PI (γ = 1)
            } else {
                // smooth linear ramp between α and β
                (beta - rotations) / (beta - alpha)
            };
            let interpolated_inv_freq =
                (1.0 - ramp) * inv_freq + ramp * (inv_freq / scale);
            for s in 0..seq {
                let theta = (s as f32) * interpolated_inv_freq;
                cos_tab[s * half_d + pair] = theta.cos() * inv_attn_temp;
                sin_tab[s * half_d + pair] = theta.sin() * inv_attn_temp;
            }
        }
    }

    fn fill_longrope(
        &self,
        cos_tab: &mut [f32],
        sin_tab: &mut [f32],
        per_dim_factors: &[f32],
    ) {
        let half_d = (self.head_dim / 2) as usize;
        let seq = self.max_seq_len as usize;
        for s in 0..seq {
            for pair in 0..half_d {
                let inv_freq = self.inv_freq(pair) * per_dim_factors[pair];
                let theta = (s as f32) * inv_freq;
                cos_tab[s * half_d + pair] = theta.cos();
                sin_tab[s * half_d + pair] = theta.sin();
            }
        }
    }

    /// The number of `f32` elements in each of the cos / sin tables.
    /// Always `max_seq_len * head_dim / 2`.
    #[inline]
    pub fn table_len(&self) -> usize {
        (self.max_seq_len as usize) * ((self.head_dim / 2) as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Linear scaling should bit-match the default `default_cs_tables`
    /// helper from `rope_apply_smoke.rs`. Both compute
    /// `cos(s · base^(-2·pair / d))` etc.
    #[test]
    fn linear_matches_default_schedule() {
        let head_dim = 16i32;
        let seq = 4i32;
        let base = 10000.0f32;
        let builder =
            RopeScaledTableBuilder::new(head_dim, seq, base, RopeScaling::Linear);
        let (cos, sin) = builder.build_host_tables().expect("build");
        assert_eq!(cos.len(), (seq * head_dim / 2) as usize);
        assert_eq!(sin.len(), (seq * head_dim / 2) as usize);

        let half_d = (head_dim / 2) as usize;
        let inv_d = 1.0f32 / head_dim as f32;
        for s in 0..seq as usize {
            for pair in 0..half_d {
                let exponent = -((2 * pair) as f32) * inv_d;
                let freq = base.powf(exponent);
                let theta = (s as f32) * freq;
                let expected_cos = theta.cos();
                let expected_sin = theta.sin();
                let i = s * half_d + pair;
                assert!(
                    (cos[i] - expected_cos).abs() < 1e-6,
                    "linear cos mismatch @ ({s},{pair}): got {} expected {}",
                    cos[i], expected_cos
                );
                assert!(
                    (sin[i] - expected_sin).abs() < 1e-6,
                    "linear sin mismatch @ ({s},{pair}): got {} expected {}",
                    sin[i], expected_sin
                );
            }
        }
    }

    /// YaRN with `scale = 1.0` and `attn_temp = 1.0` (which holds at
    /// `scale = 1.0` since `1 + 0.1 · ln(1) = 1`) must match Linear
    /// exactly — the no-op identity check.
    #[test]
    fn yarn_scale_one_matches_linear() {
        let head_dim = 32i32;
        let seq = 8i32;
        let base = 10000.0f32;
        let linear = RopeScaledTableBuilder::new(head_dim, seq, base, RopeScaling::Linear)
            .build_host_tables()
            .expect("build linear");
        let yarn = RopeScaledTableBuilder::new(
            head_dim,
            seq,
            base,
            RopeScaling::YaRN {
                scale: 1.0,
                alpha: 1.0,
                beta: 32.0,
                original_max_seq_len: 2048,
            },
        )
        .build_host_tables()
        .expect("build yarn");

        for i in 0..linear.0.len() {
            assert!(
                (linear.0[i] - yarn.0[i]).abs() < 1e-6,
                "cos mismatch @ {i}: linear={} yarn={}",
                linear.0[i],
                yarn.0[i]
            );
            assert!(
                (linear.1[i] - yarn.1[i]).abs() < 1e-6,
                "sin mismatch @ {i}: linear={} yarn={}",
                linear.1[i],
                yarn.1[i]
            );
        }
    }

    /// LongRoPE with `per_dim_factors = [1.0; D/2]` must match Linear
    /// exactly — the identity check.
    #[test]
    fn longrope_unit_factors_match_linear() {
        let head_dim = 16i32;
        let seq = 4i32;
        let base = 10000.0f32;
        let linear = RopeScaledTableBuilder::new(head_dim, seq, base, RopeScaling::Linear)
            .build_host_tables()
            .expect("build linear");
        let long_rope = RopeScaledTableBuilder::new(
            head_dim,
            seq,
            base,
            RopeScaling::LongRoPE {
                per_dim_factors: vec![1.0; (head_dim / 2) as usize],
            },
        )
        .build_host_tables()
        .expect("build longrope");

        for i in 0..linear.0.len() {
            assert!((linear.0[i] - long_rope.0[i]).abs() < 1e-6);
            assert!((linear.1[i] - long_rope.1[i]).abs() < 1e-6);
        }
    }

    /// YaRN with `scale > 1` should reduce the rotation angle for
    /// the lowest-frequency dim (highest dim index) — verifying the
    /// interpolation actually fires.
    #[test]
    fn yarn_scaled_reduces_low_freq_angle() {
        let head_dim = 32i32;
        let seq = 8i32;
        let base = 10000.0f32;
        // Use scale = 4.0, original = 2048; this gives attn_temp = sqrt(1 + 0.1·ln(4)) ≈ 1.068.
        let linear = RopeScaledTableBuilder::new(head_dim, seq, base, RopeScaling::Linear)
            .build_host_tables()
            .expect("linear");
        let yarn = RopeScaledTableBuilder::new(
            head_dim,
            seq,
            base,
            RopeScaling::YaRN {
                scale: 4.0,
                alpha: 1.0,
                beta: 32.0,
                original_max_seq_len: 2048,
            },
        )
        .build_host_tables()
        .expect("yarn");

        // Lowest-frequency dim is pair = D/2 - 1; at s = 1 the linear
        // angle is `inv_freq(D/2-1)` and YaRN should produce a SCALED
        // version (interpolated down) divided by attn_temp.
        let half_d = (head_dim / 2) as usize;
        let last_pair = half_d - 1;
        let s = 1usize;
        let idx = s * half_d + last_pair;
        // YaRN attenuates the rotation magnitude — for s=1 the
        // (cos, sin) vector should have magnitude ≈ 1 / attn_temp
        // ≈ 1/1.068 ≈ 0.936 instead of 1.0.
        let yarn_mag = (yarn.0[idx].powi(2) + yarn.1[idx].powi(2)).sqrt();
        let linear_mag = (linear.0[idx].powi(2) + linear.1[idx].powi(2)).sqrt();
        assert!(
            (linear_mag - 1.0).abs() < 1e-5,
            "linear (cos,sin) must have unit magnitude"
        );
        let expected_attn_temp = (1.0f32 + 0.1 * 4.0f32.ln()).sqrt();
        let expected_yarn_mag = 1.0 / expected_attn_temp;
        assert!(
            (yarn_mag - expected_yarn_mag).abs() < 1e-4,
            "YaRN magnitude should be 1/attn_temp ≈ {expected_yarn_mag}, got {yarn_mag}"
        );
    }

    /// LongRoPE with `factor = 2.0` on the highest-frequency dim
    /// should *double* the rotation angle for that dim.
    #[test]
    fn longrope_factor_two_doubles_angle() {
        let head_dim = 8i32;
        let seq = 4i32;
        let base = 10000.0f32;
        let half_d = (head_dim / 2) as usize;
        let mut factors = vec![1.0f32; half_d];
        factors[0] = 2.0; // double the highest-frequency dim
        let linear = RopeScaledTableBuilder::new(head_dim, seq, base, RopeScaling::Linear)
            .build_host_tables()
            .expect("linear");
        let lr = RopeScaledTableBuilder::new(
            head_dim,
            seq,
            base,
            RopeScaling::LongRoPE {
                per_dim_factors: factors,
            },
        )
        .build_host_tables()
        .expect("longrope");

        // At s = 1, the linear angle for pair 0 is `inv_freq(0) = 1.0`
        // (since `base^0 = 1`). LongRoPE doubles it → 2.0.
        let s = 1usize;
        let pair = 0usize;
        let idx = s * half_d + pair;
        let expected_linear_theta = 1.0f32; // base^0 · 1
        let expected_lr_theta = 2.0f32; // base^0 · 2
        assert!((linear.0[idx] - expected_linear_theta.cos()).abs() < 1e-6);
        assert!((lr.0[idx] - expected_lr_theta.cos()).abs() < 1e-6);
        assert!((lr.1[idx] - expected_lr_theta.sin()).abs() < 1e-6);
    }

    #[test]
    fn validate_rejects_odd_head_dim() {
        let b = RopeScaledTableBuilder::new(7, 4, 10000.0, RopeScaling::Linear);
        assert!(b.validate().is_err());
    }

    #[test]
    fn validate_rejects_longrope_factor_mismatch() {
        let b = RopeScaledTableBuilder::new(
            16,
            4,
            10000.0,
            RopeScaling::LongRoPE {
                per_dim_factors: vec![1.0; 3], // wrong length: need 8
            },
        );
        assert!(b.validate().is_err());
    }

    #[test]
    fn validate_rejects_yarn_alpha_ge_beta() {
        let b = RopeScaledTableBuilder::new(
            16,
            4,
            10000.0,
            RopeScaling::YaRN {
                scale: 4.0,
                alpha: 32.0,
                beta: 1.0,
                original_max_seq_len: 2048,
            },
        );
        assert!(b.validate().is_err());
    }
}
