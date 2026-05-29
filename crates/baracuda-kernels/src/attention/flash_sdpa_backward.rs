//! Flash Attention backward plan (Milestone 6.6).
//!
//! Three-kernel deterministic BW pipeline that operates on the
//! FW-saved `lse` (`[B, H, Q]` log-sum-exp) without ever materializing
//! the `[B, H, Q, K]` attention matrix:
//!
//! ```text
//! K1: D = rowsum(dy ⊙ y)                              shape [B, H, Q]
//! K2 (one block per (b, h, q_block)):
//!     for each key block K_j, V_j:
//!         S_ij  = Q_i · K_j^T · scale (+ causal mask)
//!         P_ij  = exp(S_ij − lse_i[:, None])
//!         dP_ij = dy_i · V_j^T
//!         dS_ij = P_ij ⊙ (dP_ij − D_i[:, None])
//!         dQ_i += dS_ij · K_j · scale
//! K3 (one block per (b, h, k_block)):
//!     for each query block Q_i:
//!         (re-derive P_ij and dS_ij from saved lse)
//!         dV_j += P_ij^T · dy_i
//!         dK_j += dS_ij^T · Q_i · scale
//! ```
//!
//! Each output cell is written by exactly one CUDA block (the one that
//! "owns" its q-block for dQ, k-block for dK / dV), so the launcher is
//! deterministic and uses no `atomicAdd`. Bespoke constraints match
//! the bespoke FW plan: `Br = Bc = 64`, `d_k = d_v ≤ 128`.
//!
//! ## Phase 59b — FA2 BW backend
//!
//! `FlashSdpaBackwardPlan` mirrors [`super::FlashSdpaPlan`]'s
//! [`BackendChoice`] machinery (Phase 42 + 59a):
//!
//! - **Bespoke** (default) — the three-kernel deterministic pipeline
//!   described above. f16 / bf16 / f32 / f64; `d_k ≤ 128`.
//! - **FlashAttentionV2** (Phase 59b, requires `fa2` cargo feature) —
//!   vendored Dao-AILab Flash Attention v2.8.3 BW kernels. Supports
//!   head_dim ∈ {32, 64, 96, 128, 192, 256}, GQA, ALiBi, sliding
//!   window, softcap. f16 / bf16 only. Non-deterministic
//!   (`atomicAdd` into `dq_accum`); FA2's BW kernels are not
//!   bit-stable run-to-run.
//!
//! The default routing heuristic ([`should_use_fa2_bw`]) is more
//! permissive than the FW heuristic — BW is launch-dominated by the
//! work-per-cell ratio rather than launch overhead, so we route
//! through FA2 whenever it's *eligible* (f16/bf16 + supported head_dim
//! + GQA divisibility). Override via [`PlanPreference::prefer_backend`].
//!
//! ## LSE dtype contract for the FA2 backend
//!
//! FA2 always stores LSE in f32, regardless of the operand element
//! type. To dispatch through the FA2 backend the caller MUST supply
//! `lse_f32: Some(_)` on [`FlashSdpaBackwardArgs`]. The bespoke `lse`
//! arg (typed `T`) is left for bespoke-path callers and IGNORED on
//! the FA2 path. Mismatching this (e.g. bespoke `lse` + FA2 backend)
//! returns `Error::InvalidProblem` from `can_implement`.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, AttentionKind, BackendKind, Element, ElementKind, KernelSku, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::flash_sdpa::FLASH_SDPA_MAX_D;
use super::map_status;

/// Internal backend tag for `FlashSdpaBackwardPlan`. Mirrors the FW
/// plan's [`super::flash_sdpa::BackendChoice`] (Phase 42 + 59a) and
/// extends it to the BW pipeline (Phase 59b).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum BackendChoice {
    Bespoke,
    #[cfg(feature = "fa2")]
    FlashAttentionV2,
}

impl BackendChoice {
    fn as_public(self) -> BackendKind {
        match self {
            BackendChoice::Bespoke => BackendKind::Bespoke,
            #[cfg(feature = "fa2")]
            BackendChoice::FlashAttentionV2 => BackendKind::FlashAttentionV2,
        }
    }
}

/// Upstream FA2 v2.8.3 supports exactly these BW head_dims (same as FW).
#[cfg(feature = "fa2")]
const FA2_BW_SUPPORTED_HEAD_DIMS: &[i32] = &[32, 64, 96, 128, 192, 256];

#[cfg(feature = "fa2")]
#[inline]
fn fa2_bw_supports_head_dim(d: i32) -> bool {
    FA2_BW_SUPPORTED_HEAD_DIMS.iter().any(|&v| v == d)
}

/// FA2 BW routing heuristic. More permissive than the FW heuristic:
/// BW is work-per-cell-bound rather than launch-overhead-bound, so we
/// route to FA2 whenever it's eligible (supported head_dim, fp16/bf16,
/// GQA divisibility) without a size threshold.
#[cfg(feature = "fa2")]
fn should_use_fa2_bw(desc: &FlashSdpaBackwardDescriptor, num_heads_k: i32) -> bool {
    if !fa2_bw_supports_head_dim(desc.d_k) || desc.d_k != desc.d_v {
        return false;
    }
    if !matches!(desc.element, ElementKind::F16 | ElementKind::Bf16) {
        return false;
    }
    if num_heads_k <= 0 || num_heads_k > desc.num_heads || desc.num_heads % num_heads_k != 0 {
        return false;
    }
    true
}

/// Descriptor for a Flash Attention backward op.
///
/// `#[non_exhaustive]` (Phase 59b) — new optional fields (sliding window,
/// softcap, future plumbing) may land in later phases. Downstream
/// callers MUST construct via [`Self::new`] + chainable `with_*`
/// setters. Follows the same convention as
/// [`super::FlashSdpaDescriptor`] (Phase 59a).
#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
pub struct FlashSdpaBackwardDescriptor {
    /// Batch size (`B`).
    pub batch_size: i32,
    /// Number of attention heads (`H`).
    pub num_heads: i32,
    /// Query sequence length (`Q`).
    pub query_len: i32,
    /// Key / value sequence length (`K`).
    pub key_len: i32,
    /// Head dimension of Q and K (`D_k`).
    pub d_k: i32,
    /// Head dimension of V (`D_v`). The bespoke backend requires `d_v == d_k`.
    pub d_v: i32,
    /// Score scaling factor — must match the FW scale.
    pub scale: f32,
    /// Apply upper-triangular causal mask — must match the FW
    /// descriptor's value.
    pub is_causal: bool,
    /// Element type.
    pub element: ElementKind,
    /// Phase 59b — sliding-window left bound (FA2-only).
    pub window_size_left: Option<i32>,
    /// Phase 59b — sliding-window right bound (FA2-only).
    pub window_size_right: Option<i32>,
    /// Phase 59b — softcap (Gemma-2 style; FA2-only).
    pub softcap: f32,
}

impl FlashSdpaBackwardDescriptor {
    /// Build a `FlashSdpaBackwardDescriptor`. Defaults the Phase 59b
    /// extras to disabled (`window_size_*=None`, `softcap=0.0`).
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn new(
        batch_size: i32,
        num_heads: i32,
        query_len: i32,
        key_len: i32,
        d_k: i32,
        d_v: i32,
        scale: f32,
        is_causal: bool,
        element: ElementKind,
    ) -> Self {
        Self {
            batch_size,
            num_heads,
            query_len,
            key_len,
            d_k,
            d_v,
            scale,
            is_causal,
            element,
            window_size_left: None,
            window_size_right: None,
            softcap: 0.0,
        }
    }

    /// Builder: sliding-window left bound. `None` = unbounded.
    #[inline]
    pub fn with_window_size_left(mut self, n: Option<i32>) -> Self {
        self.window_size_left = n;
        self
    }

    /// Builder: sliding-window right bound. `None` = unbounded.
    #[inline]
    pub fn with_window_size_right(mut self, n: Option<i32>) -> Self {
        self.window_size_right = n;
        self
    }

    /// Builder: softcap value (`0.0` = disabled).
    #[inline]
    pub fn with_softcap(mut self, cap: f32) -> Self {
        self.softcap = cap;
        self
    }
}

/// Args bundle for a Flash Attention backward launch.
///
/// On the bespoke backend, Q/K/V/y/lse/dy/dq/dk/dv/d_ws are all consumed
/// as documented. On the FA2 backend the `lse` arg is IGNORED (FA2
/// stores LSE in f32; the caller must instead supply `lse_f32`). The
/// `d_ws` arg is IGNORED on the FA2 backend — FA2 uses a single
/// caller-supplied `workspace` (sized via [`FlashSdpaBackwardPlan::workspace_size`])
/// rather than a typed scratch slot.
pub struct FlashSdpaBackwardArgs<'a, T: Element> {
    /// Query tensor used in FW — shape `[B, H, Q, D_k]`.
    pub q: TensorRef<'a, T, 4>,
    /// Key tensor used in FW — shape `[B, H_k, K, D_k]`. For the bespoke
    /// backend `H_k` must equal `H` (no GQA). For the FA2 backend
    /// `H_k` may differ as long as `H % H_k == 0`.
    pub k: TensorRef<'a, T, 4>,
    /// Value tensor used in FW — shape `[B, H_k, K, D_v]`.
    pub v: TensorRef<'a, T, 4>,
    /// Saved FW output `y` — shape `[B, H, Q, D_v]`. Consumed by the
    /// bespoke `D = rowsum(y ⊙ dy)` reduction kernel **and** the FA2
    /// preprocess kernel that recomputes `dot(do, o)`.
    pub y: TensorRef<'a, T, 4>,
    /// Saved FW log-sum-exp **typed T** — shape `[B, H, Q]`. The
    /// bespoke backend consumes this directly. The FA2 backend IGNORES
    /// this field (it requires `lse_f32` instead — FA2 stores LSE in
    /// f32 regardless of operand dtype).
    pub lse: TensorRef<'a, T, 3>,
    /// Upstream gradient on the FW output — shape `[B, H, Q, D_v]`.
    pub dy: TensorRef<'a, T, 4>,
    /// Scratch buffer for the bespoke `D = rowsum(y ⊙ dy)` —
    /// shape `[B, H, Q]`, element type `T`. IGNORED on the FA2 backend
    /// (FA2 uses workspace for its own dq_accum + dsoftmax_d scratch).
    pub d_ws: TensorMut<'a, T, 3>,
    /// Output gradient `dQ` — shape `[B, H, Q, D_k]`.
    pub dq: TensorMut<'a, T, 4>,
    /// Output gradient `dK` — shape `[B, H_k, K, D_k]`.
    pub dk: TensorMut<'a, T, 4>,
    /// Output gradient `dV` — shape `[B, H_k, K, D_v]`.
    pub dv: TensorMut<'a, T, 4>,
    /// Phase 59b — f32 LSE (FA2 backend). REQUIRED on the FA2 backend
    /// (FA2 stores LSE in f32 regardless of operand dtype; reuse the
    /// f32 LSE written by [`super::FlashSdpaPlan::run`] via the
    /// workspace pointer). IGNORED on the bespoke backend.
    pub lse_f32: Option<TensorRef<'a, f32, 3>>,
    /// Phase 59b — ALiBi slopes (FA2-only). Shape `[1, H]` (per-head
    /// broadcast) or `[B, H]` (per-batch-per-head). Setting this on
    /// the bespoke backend is an error.
    pub alibi_slopes: Option<TensorRef<'a, f32, 2>>,
}

/// Flash Attention backward plan.
///
/// Three-kernel deterministic BW pipeline on the bespoke backend.
/// On the FA2 backend it routes to vendored FA2 v2.8.3 BW kernels.
///
/// **When to use**: autograd partner for [`super::FlashSdpaPlan`].
///
/// **Dtypes**: bespoke = `f32`, `f64`, `f16`, `bf16`; FA2 = `f16`, `bf16`.
///
/// **Shape limits**: bespoke is `d_k == d_v ≤ 128`. FA2 supports
/// `d_k ∈ {32, 64, 96, 128, 192, 256}` with `d_k == d_v`.
///
/// **Workspace**: bespoke = 0. FA2 = `dq_accum + dsoftmax_d` (see
/// [`Self::workspace_size`]).
///
/// **Precision guarantee**: bespoke is deterministic + bit-stable;
/// FA2 uses `atomicAdd` into `dq_accum` and is NOT bit-stable
/// run-to-run.
pub struct FlashSdpaBackwardPlan<T: Element> {
    desc: FlashSdpaBackwardDescriptor,
    sku: KernelSku,
    backend: BackendChoice,
    _marker: PhantomData<T>,
}

impl<T: Element> FlashSdpaBackwardPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &FlashSdpaBackwardDescriptor,
        pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::FlashSdpaBackwardPlan: descriptor element != T",
            ));
        }
        if desc.batch_size < 0
            || desc.num_heads < 0
            || desc.query_len < 0
            || desc.key_len < 0
            || desc.d_k < 0
            || desc.d_v < 0
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FlashSdpaBackwardPlan: extents must be non-negative",
            ));
        }
        if !desc.scale.is_finite() {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FlashSdpaBackwardPlan: scale must be finite",
            ));
        }
        if desc.d_k != desc.d_v {
            return Err(Error::Unsupported(
                "baracuda-kernels::FlashSdpaBackwardPlan: requires d_k == d_v",
            ));
        }
        let dtype_in_scope = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !dtype_in_scope {
            return Err(Error::Unsupported(
                "baracuda-kernels::FlashSdpaBackwardPlan: wired today: `{f32, f16, bf16, f64}`",
            ));
        }
        if desc.softcap < 0.0 || !desc.softcap.is_finite() {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FlashSdpaBackwardPlan: softcap must be finite and non-negative",
            ));
        }

        let backend = pick_backend::<T>(desc, pref);

        // Bespoke backend keeps the d_k ≤ 128 cap. FA2 lifts it.
        if matches!(backend, BackendChoice::Bespoke) && desc.d_k > FLASH_SDPA_MAX_D {
            return Err(Error::Unsupported(
                "baracuda-kernels::FlashSdpaBackwardPlan: bespoke kernel requires d_k ≤ 128 \
                 (enable `fa2` feature for d_k > 128)",
            ));
        }
        // Bespoke kernel doesn't honour Phase 59b extras.
        #[cfg(feature = "fa2")]
        let is_fa2 = matches!(backend, BackendChoice::FlashAttentionV2);
        #[cfg(not(feature = "fa2"))]
        let is_fa2 = false;
        if !is_fa2 {
            if desc.window_size_left.is_some() || desc.window_size_right.is_some() {
                return Err(Error::Unsupported(
                    "baracuda-kernels::FlashSdpaBackwardPlan: sliding window requires the FA2 backend",
                ));
            }
            if desc.softcap != 0.0 {
                return Err(Error::Unsupported(
                    "baracuda-kernels::FlashSdpaBackwardPlan: softcap requires the FA2 backend",
                ));
            }
        }

        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            // Bespoke: deterministic. FA2: atomicAdd into dq_accum, not
            // bit-stable. We tag the SKU honestly per-backend.
            bit_stable_on_same_hardware: matches!(backend, BackendChoice::Bespoke),
            deterministic: matches!(backend, BackendChoice::Bespoke),
        };
        let sku = KernelSku {
            category: OpCategory::Attention,
            op: AttentionKind::FlashAttention as u16,
            element: T::KIND,
            aux_element: None,
            layout: None,
            epilogue: None,
            arch: ArchSku::Sm80,
            backend: backend.as_public(),
            precision_guarantee,
        };
        Ok(Self {
            desc: *desc,
            sku,
            backend,
            _marker: PhantomData,
        })
    }

    /// Which backend the plan picked.
    #[inline]
    pub fn backend(&self) -> BackendKind {
        self.backend.as_public()
    }

    /// Validate args.
    pub fn can_implement(&self, args: &FlashSdpaBackwardArgs<'_, T>) -> Result<()> {
        // Bespoke uses H_k = H; FA2 derives H_k from args.k.shape[1].
        let num_heads_k = args.k.shape[1];
        if num_heads_k <= 0
            || num_heads_k > self.desc.num_heads
            || self.desc.num_heads % num_heads_k != 0
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FlashSdpaBackwardPlan: K shape[1] (num_heads_k) must divide num_heads",
            ));
        }
        let is_gqa = num_heads_k != self.desc.num_heads;
        #[cfg(feature = "fa2")]
        let backend_is_fa2 = matches!(self.backend, BackendChoice::FlashAttentionV2);
        #[cfg(not(feature = "fa2"))]
        let backend_is_fa2 = false;
        if is_gqa && !backend_is_fa2 {
            return Err(Error::Unsupported(
                "baracuda-kernels::FlashSdpaBackwardPlan: GQA on the bespoke backend is unsupported",
            ));
        }

        let shape_q = [
            self.desc.batch_size,
            self.desc.num_heads,
            self.desc.query_len,
            self.desc.d_k,
        ];
        let shape_k = [
            self.desc.batch_size,
            num_heads_k,
            self.desc.key_len,
            self.desc.d_k,
        ];
        let shape_v = [
            self.desc.batch_size,
            num_heads_k,
            self.desc.key_len,
            self.desc.d_v,
        ];
        let shape_y = [
            self.desc.batch_size,
            self.desc.num_heads,
            self.desc.query_len,
            self.desc.d_v,
        ];
        let shape_lse = [
            self.desc.batch_size,
            self.desc.num_heads,
            self.desc.query_len,
        ];
        if args.q.shape != shape_q {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FlashSdpaBackwardPlan: Q shape mismatch",
            ));
        }
        if args.k.shape != shape_k {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FlashSdpaBackwardPlan: K shape mismatch",
            ));
        }
        if args.v.shape != shape_v {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FlashSdpaBackwardPlan: V shape mismatch",
            ));
        }
        if args.y.shape != shape_y {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FlashSdpaBackwardPlan: y shape mismatch",
            ));
        }
        if args.lse.shape != shape_lse {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FlashSdpaBackwardPlan: lse shape must be [B, H, Q]",
            ));
        }
        if args.dy.shape != shape_y {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FlashSdpaBackwardPlan: dy shape mismatch",
            ));
        }
        if args.d_ws.shape != shape_lse {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FlashSdpaBackwardPlan: d_ws shape must be [B, H, Q]",
            ));
        }
        if args.dq.shape != shape_q {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FlashSdpaBackwardPlan: dQ shape mismatch with Q",
            ));
        }
        if args.dk.shape != shape_k {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FlashSdpaBackwardPlan: dK shape mismatch with K",
            ));
        }
        if args.dv.shape != shape_v {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FlashSdpaBackwardPlan: dV shape mismatch with V",
            ));
        }
        if !args.q.is_contiguous()
            || !args.k.is_contiguous()
            || !args.v.is_contiguous()
            || !args.y.is_contiguous()
            || !args.lse.is_contiguous()
            || !args.dy.is_contiguous()
            || !args.d_ws.is_contiguous()
            || !args.dq.is_contiguous()
            || !args.dk.is_contiguous()
            || !args.dv.is_contiguous()
        {
            return Err(Error::Unsupported(
                "baracuda-kernels::FlashSdpaBackwardPlan: requires contiguous tensors",
            ));
        }

        // Phase 59b — FA2 backend extra checks.
        if backend_is_fa2 {
            if args.lse_f32.is_none() {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::FlashSdpaBackwardPlan: FA2 backend requires lse_f32 \
                     (FA2 stores LSE in f32 regardless of operand dtype)",
                ));
            }
            let lse_f32 = args.lse_f32.as_ref().unwrap();
            if lse_f32.shape != shape_lse {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::FlashSdpaBackwardPlan: lse_f32 shape must be [B, H, Q]",
                ));
            }
            if !lse_f32.is_contiguous() {
                return Err(Error::Unsupported(
                    "baracuda-kernels::FlashSdpaBackwardPlan: lse_f32 must be contiguous",
                ));
            }
            if let Some(slopes) = args.alibi_slopes.as_ref() {
                if slopes.shape[1] != self.desc.num_heads {
                    return Err(Error::InvalidProblem(
                        "baracuda-kernels::FlashSdpaBackwardPlan: alibi_slopes shape[1] must equal num_heads",
                    ));
                }
                if slopes.shape[0] != 1 && slopes.shape[0] != self.desc.batch_size {
                    return Err(Error::InvalidProblem(
                        "baracuda-kernels::FlashSdpaBackwardPlan: alibi_slopes shape[0] must be 1 or batch_size",
                    ));
                }
            }
        } else if args.alibi_slopes.is_some() {
            return Err(Error::Unsupported(
                "baracuda-kernels::FlashSdpaBackwardPlan: ALiBi requires the FA2 backend",
            ));
        }
        Ok(())
    }

    /// Workspace size in bytes.
    ///
    /// Bespoke = 0 (the `d_ws` arg carries the only scratch).
    ///
    /// FA2 = `dq_accum + dsoftmax_d` packed back-to-back:
    /// - `dq_accum`   : `B * seqlen_q_rounded * H * head_size_rounded * 4`
    /// - `dsoftmax_d` : `B * H * seqlen_q_rounded * 4`
    ///
    /// where `seqlen_q_rounded = round_up(Q, 128)` and
    /// `head_size_rounded = round_up(d, d <= 128 ? 32 : 64)`.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        match self.backend {
            BackendChoice::Bespoke => 0,
            #[cfg(feature = "fa2")]
            BackendChoice::FlashAttentionV2 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_fa2_sdpa_backward_workspace_size(
                    self.desc.batch_size,
                    self.desc.num_heads,
                    self.desc.query_len,
                    self.desc.d_k,
                )
            },
        }
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

    /// Launch the BW pipeline on the supplied stream.
    pub fn run(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: FlashSdpaBackwardArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if args.q.numel() == 0 || args.k.numel() == 0 {
            return Ok(());
        }

        // Phase 59b — FA2 dispatch path.
        #[cfg(feature = "fa2")]
        if matches!(self.backend, BackendChoice::FlashAttentionV2) {
            let capturing = stream.is_capturing().unwrap_or(false);
            if !capturing {
                return self.run_fa2_bw(stream, workspace, &args);
            }
            // capture-mode → fall through to bespoke
        }
        let _ = workspace;

        // Bespoke path (preserved from pre-Phase-59b).
        let stream_ptr = stream.as_raw() as *mut c_void;
        let q_ptr = args.q.data.as_raw().0 as *const c_void;
        let k_ptr = args.k.data.as_raw().0 as *const c_void;
        let v_ptr = args.v.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *const c_void;
        let lse_ptr = args.lse.data.as_raw().0 as *const c_void;
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let d_ws_ptr = args.d_ws.data.as_raw().0 as *mut c_void;
        let dq_ptr = args.dq.data.as_raw().0 as *mut c_void;
        let dk_ptr = args.dk.data.as_raw().0 as *mut c_void;
        let dv_ptr = args.dv.data.as_raw().0 as *mut c_void;
        let is_causal_flag = if self.desc.is_causal { 1 } else { 0 };

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_flash_sdpa_backward_f32_run(
                    self.desc.batch_size, self.desc.num_heads,
                    self.desc.query_len, self.desc.key_len,
                    self.desc.d_k, self.desc.d_v, self.desc.scale, is_causal_flag,
                    q_ptr, k_ptr, v_ptr, y_ptr, lse_ptr, dy_ptr, d_ws_ptr,
                    dq_ptr, dk_ptr, dv_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_flash_sdpa_backward_f16_run(
                    self.desc.batch_size, self.desc.num_heads,
                    self.desc.query_len, self.desc.key_len,
                    self.desc.d_k, self.desc.d_v, self.desc.scale, is_causal_flag,
                    q_ptr, k_ptr, v_ptr, y_ptr, lse_ptr, dy_ptr, d_ws_ptr,
                    dq_ptr, dk_ptr, dv_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_flash_sdpa_backward_bf16_run(
                    self.desc.batch_size, self.desc.num_heads,
                    self.desc.query_len, self.desc.key_len,
                    self.desc.d_k, self.desc.d_v, self.desc.scale, is_causal_flag,
                    q_ptr, k_ptr, v_ptr, y_ptr, lse_ptr, dy_ptr, d_ws_ptr,
                    dq_ptr, dk_ptr, dv_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_flash_sdpa_backward_f64_run(
                    self.desc.batch_size, self.desc.num_heads,
                    self.desc.query_len, self.desc.key_len,
                    self.desc.d_k, self.desc.d_v, self.desc.scale, is_causal_flag,
                    q_ptr, k_ptr, v_ptr, y_ptr, lse_ptr, dy_ptr, d_ws_ptr,
                    dq_ptr, dk_ptr, dv_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::FlashSdpaBackwardPlan::run reached an unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }

    /// FA2 BW launch path (Phase 59b).
    #[cfg(feature = "fa2")]
    fn run_fa2_bw(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: &FlashSdpaBackwardArgs<'_, T>,
    ) -> Result<()> {
        let stream_ptr = stream.as_raw() as *mut c_void;
        let q_ptr = args.q.data.as_raw().0 as *const c_void;
        let k_ptr = args.k.data.as_raw().0 as *const c_void;
        let v_ptr = args.v.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *const c_void;
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let dq_ptr = args.dq.data.as_raw().0 as *mut c_void;
        let dk_ptr = args.dk.data.as_raw().0 as *mut c_void;
        let dv_ptr = args.dv.data.as_raw().0 as *mut c_void;
        let lse_ptr = args.lse_f32.as_ref().unwrap().data.as_raw().0 as *const c_void;
        let is_causal_flag = if self.desc.is_causal { 1 } else { 0 };

        let num_heads_k = args.k.shape[1];

        let (alibi_ptr, alibi_batch_stride) = match args.alibi_slopes.as_ref() {
            None => (core::ptr::null::<c_void>(), 0i32),
            Some(slopes) => {
                let ptr = slopes.data.as_raw().0 as *const c_void;
                let batch_stride = if slopes.shape[0] == 1 {
                    0_i32
                } else {
                    self.desc.num_heads
                };
                (ptr, batch_stride)
            }
        };

        let window_left = self.desc.window_size_left.unwrap_or(-1);
        let window_right = self.desc.window_size_right.unwrap_or(-1);
        let softcap = self.desc.softcap;

        let need = self.workspace_size();
        let (ws_ptr, ws_bytes) = match workspace {
            Workspace::None => {
                if need > 0 {
                    return Err(Error::WorkspaceTooSmall { needed: need, got: 0 });
                }
                (core::ptr::null_mut::<c_void>(), 0usize)
            }
            Workspace::Borrowed(slice) => {
                if slice.len() < need {
                    return Err(Error::WorkspaceTooSmall {
                        needed: need,
                        got: slice.len(),
                    });
                }
                (slice.as_raw().0 as *mut c_void, slice.len())
            }
        };

        let status = match T::KIND {
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_fa2_sdpa_backward_f16_run(
                    self.desc.batch_size, self.desc.num_heads, num_heads_k,
                    self.desc.query_len, self.desc.key_len, self.desc.d_k,
                    self.desc.scale, is_causal_flag,
                    alibi_ptr, alibi_batch_stride,
                    window_left, window_right, softcap,
                    q_ptr, k_ptr, v_ptr, y_ptr, dy_ptr, lse_ptr,
                    dq_ptr, dk_ptr, dv_ptr,
                    ws_ptr, ws_bytes, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_fa2_sdpa_backward_bf16_run(
                    self.desc.batch_size, self.desc.num_heads, num_heads_k,
                    self.desc.query_len, self.desc.key_len, self.desc.d_k,
                    self.desc.scale, is_causal_flag,
                    alibi_ptr, alibi_batch_stride,
                    window_left, window_right, softcap,
                    q_ptr, k_ptr, v_ptr, y_ptr, dy_ptr, lse_ptr,
                    dq_ptr, dk_ptr, dv_ptr,
                    ws_ptr, ws_bytes, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::FlashSdpaBackwardPlan::run_fa2_bw: FA2 BW supports only f16 / bf16",
                ));
            }
        };
        map_status(status)
    }
}

/// Internal: pick backend for the BW descriptor + preference.
fn pick_backend<T: Element>(
    #[cfg_attr(not(feature = "fa2"), allow(unused_variables))] desc: &FlashSdpaBackwardDescriptor,
    pref: PlanPreference,
) -> BackendChoice {
    match pref.prefer_backend {
        Some(BackendKind::Bespoke) => BackendChoice::Bespoke,
        #[cfg(feature = "fa2")]
        Some(BackendKind::FlashAttentionV2) => {
            if fa2_bw_is_eligible::<T>(desc) {
                BackendChoice::FlashAttentionV2
            } else {
                BackendChoice::Bespoke
            }
        }
        _ => {
            #[cfg(feature = "fa2")]
            {
                if should_use_fa2_bw(desc, desc.num_heads) {
                    return BackendChoice::FlashAttentionV2;
                }
            }
            BackendChoice::Bespoke
        }
    }
}

/// Hard eligibility check for FA2 BW.
#[cfg(feature = "fa2")]
fn fa2_bw_is_eligible<T: Element>(desc: &FlashSdpaBackwardDescriptor) -> bool {
    fa2_bw_supports_head_dim(desc.d_k)
        && desc.d_k == desc.d_v
        && matches!(T::KIND, ElementKind::F16 | ElementKind::Bf16)
}
