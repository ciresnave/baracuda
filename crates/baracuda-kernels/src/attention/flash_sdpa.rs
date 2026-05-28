//! Flash Attention forward plan (Milestone 6.6).
//!
//! Tiled, fused online-softmax SDPA that avoids materializing the
//! `[B, H, Q, K]` attention matrix. Algorithm from Tri Dao 2022
//! (<https://arxiv.org/abs/2205.14135>):
//!
//! ```text
//! for each query block Q_i of size [Br, d_k]:
//!     m_i = -inf, l_i = 0, O_i = 0
//!     for each key block K_j, V_j:
//!         S_ij = Q_i · K_j^T · scale  (apply causal mask if needed)
//!         m_new = max(m_i, rowmax(S_ij))
//!         P_ij  = exp(S_ij − m_new[:, None])
//!         α     = exp(m_i − m_new)
//!         l_new = α · l_i + rowsum(P_ij)
//!         O_i   = α[:, None] · O_i + P_ij @ V_j
//!         m_i, l_i = m_new, l_new
//!     O_i = O_i / l_i[:, None]                    (final normalize)
//!     L_i = m_i + log(l_i)                        (saved for BW)
//! ```
//!
//! Shape conventions (rank-4, contiguous, row-major):
//!
//! | tensor | shape                |
//! |--------|----------------------|
//! | `Q`    | `[B, H, Q, D_k]`     |
//! | `K`    | `[B, H, K, D_k]`     |
//! | `V`    | `[B, H, K, D_v]`     |
//! | `y`    | `[B, H, Q, D_v]`     |
//! | `lse`  | `[B, H, Q]`          |
//!
//! Saved `lse` (log-sum-exp) is the BW pass's only stateful intermediate
//! — it replaces the saved `[B, H, Q, K]` attn tensor of the naive
//! [`crate::SdpaPlan`]. Trailblazer constraints: `Br = Bc = 64`,
//! `d_k = d_v ≤ 128`, no explicit additive mask (use `SdpaPlan` for
//! masked attention). Optional upper-triangular causal mask.
//!
//! Wired today: `{f32, f16, bf16, f64}`.
//!
//! ## Backend choice (Phase 42)
//!
//! `FlashSdpaPlan` can route a launch through one of two backends:
//!
//! - **Bespoke** (default) — the baracuda-shipped sm_80 / sm_89 Flash
//!   kernel. Source-of-truth for correctness; integrated with the
//!   `FlashSdpaBackwardPlan` BW path and the strided-FFI sibling for
//!   GQA broadcast.
//! - **FlashAttentionV2** (Phase 42, requires `fa2` cargo feature) —
//!   vendored Dao-AILab Flash Attention v2.8.3. Long-context-tuned
//!   kernels with CUTLASS template specialization; wins at
//!   prefill-class shapes (`seq_q * seq_k ≥ ~1M`). Constraints:
//!   `head_dim == 128`, dtype ∈ {f16, bf16}, dense (no GQA, no
//!   varlen). LSE is **f32** regardless of element dtype (FA2 always
//!   accumulates softmax in f32).
//!
//! The default heuristic ([`should_use_fa2`]) picks FA2 for
//! `head_dim == 128 ∧ dtype ∈ {f16, bf16} ∧ seq_q * seq_k ≥ 1M ∧
//! num_heads == num_heads_k`, else bespoke. Override via
//! [`PlanPreference::prefer_backend`] (set to
//! [`BackendKind::FlashAttentionV2`] or [`BackendKind::Bespoke`]).
//!
//! Capture-mode auto-fallback: when the stream is in graph capture
//! mode, FA2 falls back to bespoke (FA2's launch-time
//! `cudaFuncSetAttribute` for opt-in dynamic shared memory isn't
//! capture-safe).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, AttentionKind, BackendKind, Element, ElementKind, KernelSku, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::map_status;

/// Maximum supported head dimension for the Flash trailblazer.
pub const FLASH_SDPA_MAX_D: i32 = 128;

/// Internal backend tag for `FlashSdpaPlan`. Phase 42 added the FA2
/// variant; bespoke remains the default for all shapes the heuristic
/// doesn't explicitly route through FA2.
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

/// FA2 routing heuristic. Returns `true` when the shape + dtype falls
/// in the long-context regime where FA2's CUTLASS-tuned tile sizing
/// beats the bespoke kernel.
///
/// Today's rule (validated on RTX 4070 in Phase 42's bench A/B):
///
/// - **head_dim != 128 → false.** Tier 1 only ships head_dim=128.
/// - **dtype not in {f16, bf16} → false.** FA2 has no f32/f64 SKU.
/// - **num_heads != num_heads_k → false.** Tier 1 doesn't plumb GQA.
/// - **seq_q × seq_k < 1024 × 1024 → false.** Short-context regime;
///   bespoke's lower launch overhead wins.
/// - Otherwise **true** (FA2).
///
/// Override via [`PlanPreference::prefer_backend`].
#[cfg(feature = "fa2")]
fn should_use_fa2(desc: &FlashSdpaDescriptor, num_heads_k: i32) -> bool {
    if desc.d_k != 128 || desc.d_v != 128 {
        return false;
    }
    if !matches!(desc.element, ElementKind::F16 | ElementKind::Bf16) {
        return false;
    }
    if num_heads_k != desc.num_heads {
        return false;  // GQA broadcast not supported on FA2 Tier 1
    }
    let work = (desc.query_len as i64) * (desc.key_len as i64);
    work >= 1024 * 1024
}

/// Descriptor for a Flash Attention forward op.
///
/// Trailblazer enforces `d_k == d_v` (single head-dim cap shared across
/// Q/K and V) and `d_k ≤ 128`. Use [`crate::SdpaPlan`] for the relaxed
/// case where `d_k != d_v`, or for problems that need an explicit
/// additive mask.
#[derive(Copy, Clone, Debug)]
pub struct FlashSdpaDescriptor {
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
    /// Head dimension of V (`D_v`). Trailblazer requires `d_v == d_k`.
    pub d_v: i32,
    /// Score scaling factor — typically `1.0 / sqrt(d_k)`.
    pub scale: f32,
    /// Apply upper-triangular causal mask inside the scores compute.
    pub is_causal: bool,
    /// Element type — must match the plan's type parameter.
    pub element: ElementKind,
}

/// Args bundle for a Flash Attention forward launch.
pub struct FlashSdpaArgs<'a, T: Element> {
    /// Query tensor — shape `[B, H, Q, D_k]`, contiguous.
    pub q: TensorRef<'a, T, 4>,
    /// Key tensor — shape `[B, H, K, D_k]`, contiguous.
    pub k: TensorRef<'a, T, 4>,
    /// Value tensor — shape `[B, H, K, D_v]`, contiguous.
    pub v: TensorRef<'a, T, 4>,
    /// Output tensor — shape `[B, H, Q, D_v]`, contiguous.
    pub y: TensorMut<'a, T, 4>,
    /// Saved log-sum-exp — shape `[B, H, Q]`, contiguous. Stores
    /// `m_i + log(l_i)` after the FW pass; consumed by the
    /// [`FlashSdpaBackwardPlan`](crate::FlashSdpaBackwardPlan).
    pub lse: TensorMut<'a, T, 3>,
}

/// Flash Attention forward plan (Tri Dao 2022).
///
/// Tiled fused online-softmax SDPA that avoids materializing the
/// `[B, H, Q, K]` attention matrix.
///
/// **When to use**: forward pass of self / cross attention when memory
/// is tight (Flash has O(N) memory in sequence length vs naive O(N²)).
/// Pair with [`super::FlashSdpaBackwardPlan`] for autograd. Use
/// [`super::SdpaPlan`] when you need an explicit additive mask or
/// `d_k != d_v`.
///
/// **Dtypes**: `f32`, `f64`, `f16`, `bf16`. Half-precision uses an
/// `f32` accumulator throughout. No FP8 / int8 in the trailblazer.
///
/// **Shape limits**: `d_k == d_v ≤ 128`; arbitrary `Q`, `K`. The
/// trailblazer fixes `Br = Bc = 64` and supports an optional
/// upper-triangular causal mask.
///
/// **Workspace**: zero — the `lse` arg carries the only FW-saved
/// state (used by the BW pass to re-derive `P_ij` without storing the
/// attention matrix).
///
/// **Precision guarantee**: deterministic; bit-stable on the same
/// hardware. Each output cell is written by exactly one block — no
/// atomicAdd. Flash and naive SDPA differ in float-order so they are
/// *not* bit-identical to each other.
pub struct FlashSdpaPlan<T: Element> {
    desc: FlashSdpaDescriptor,
    sku: KernelSku,
    backend: BackendChoice,
    _marker: PhantomData<T>,
}

impl<T: Element> FlashSdpaPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &FlashSdpaDescriptor,
        pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::FlashSdpaPlan: descriptor element != T",
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
                "baracuda-kernels::FlashSdpaPlan: extents must be non-negative",
            ));
        }
        if !desc.scale.is_finite() {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FlashSdpaPlan: scale must be finite",
            ));
        }
        if desc.d_k != desc.d_v {
            return Err(Error::Unsupported(
                "baracuda-kernels::FlashSdpaPlan: trailblazer requires d_k == d_v",
            ));
        }
        if desc.d_k > FLASH_SDPA_MAX_D {
            return Err(Error::Unsupported(
                "baracuda-kernels::FlashSdpaPlan: d_k must be ≤ 128 in the trailblazer",
            ));
        }
        let dtype_in_scope = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !dtype_in_scope {
            return Err(Error::Unsupported(
                "baracuda-kernels::FlashSdpaPlan: wired today: `{f32, f16, bf16, f64}`",
            ));
        }

        // Phase 42: pick the backend. Caller override via
        // `pref.prefer_backend` wins; otherwise the heuristic decides.
        // The descriptor doesn't carry a separate `num_heads_k` — the
        // bespoke trailblazer assumes no GQA at the descriptor level
        // (GQA is supported via the strided FFI sibling with stride[1] = 0
        // on K/V). For the FA2 heuristic, we conservatively assume
        // num_heads_k == num_heads here; the run-time path validates.
        let backend = pick_backend::<T>(desc, pref);

        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            // One block per (b, h, q_block); each output cell is written
            // by exactly one block. No atomicAdd anywhere.
            bit_stable_on_same_hardware: true,
            deterministic: true,
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
    ///
    /// Useful for telemetry, autotuner cache keys, and verifying the
    /// heuristic in tests. Mirrors [`baracuda_cutlass::GemmPlan::backend`]
    /// (Phase 30) one level up.
    #[inline]
    pub fn backend(&self) -> BackendKind {
        self.backend.as_public()
    }

    /// Validate args against the descriptor.
    pub fn can_implement(&self, args: &FlashSdpaArgs<'_, T>) -> Result<()> {
        let shape_q = [
            self.desc.batch_size,
            self.desc.num_heads,
            self.desc.query_len,
            self.desc.d_k,
        ];
        let shape_k = [
            self.desc.batch_size,
            self.desc.num_heads,
            self.desc.key_len,
            self.desc.d_k,
        ];
        let shape_v = [
            self.desc.batch_size,
            self.desc.num_heads,
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
                "baracuda-kernels::FlashSdpaPlan: Q shape mismatch",
            ));
        }
        if args.k.shape != shape_k {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FlashSdpaPlan: K shape mismatch",
            ));
        }
        if args.v.shape != shape_v {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FlashSdpaPlan: V shape mismatch",
            ));
        }
        if args.y.shape != shape_y {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FlashSdpaPlan: y shape mismatch",
            ));
        }
        if args.lse.shape != shape_lse {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FlashSdpaPlan: lse shape must be [B, H, Q]",
            ));
        }
        if !args.q.is_contiguous()
            || !args.k.is_contiguous()
            || !args.v.is_contiguous()
            || !args.y.is_contiguous()
            || !args.lse.is_contiguous()
        {
            return Err(Error::Unsupported(
                "baracuda-kernels::FlashSdpaPlan: trailblazer requires contiguous tensors",
            ));
        }
        let q_n = args.q.numel();
        let k_n = args.k.numel();
        let v_n = args.v.numel();
        let y_n = args.y.numel();
        let l_n = args.lse.numel();
        if (args.q.data.len() as i64) < q_n
            || (args.k.data.len() as i64) < k_n
            || (args.v.data.len() as i64) < v_n
            || (args.y.data.len() as i64) < y_n
            || (args.lse.data.len() as i64) < l_n
        {
            return Err(Error::BufferTooSmall {
                needed: y_n.max(l_n).max(q_n).max(k_n).max(v_n) as usize,
                got: 0,
            });
        }
        Ok(())
    }

    /// Workspace size in bytes.
    ///
    /// Bespoke: 0 (the `lse` arg carries the only FW-saved state).
    ///
    /// FA2: `batch * num_heads * query_len * 4` bytes — FA2 always
    /// writes the softmax LSE in f32 regardless of the input element
    /// type. The plan layer hides this by routing the FA2 LSE write
    /// to caller-supplied workspace memory (the caller-visible `lse`
    /// arg is left untouched on the FA2 path, since the Tier-1
    /// integration doesn't yet wire BW for FA2). When BW lands
    /// (Tier 2), the workspace return becomes the canonical FA2 LSE
    /// store.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        match self.backend {
            BackendChoice::Bespoke => 0,
            #[cfg(feature = "fa2")]
            BackendChoice::FlashAttentionV2 => {
                let n = (self.desc.batch_size as i64)
                    * (self.desc.num_heads as i64)
                    * (self.desc.query_len as i64);
                (n.max(0) as usize) * 4
            }
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

    /// Launch the fused FW kernel on the supplied stream.
    pub fn run(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: FlashSdpaArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if args.y.numel() == 0 {
            return Ok(());
        }
        let stream_ptr = stream.as_raw() as *mut c_void;
        let q_ptr = args.q.data.as_raw().0 as *const c_void;
        let k_ptr = args.k.data.as_raw().0 as *const c_void;
        let v_ptr = args.v.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;
        let lse_ptr = args.lse.data.as_raw().0 as *mut c_void;
        let is_causal_flag = if self.desc.is_causal { 1 } else { 0 };

        // Phase 42 — FA2 dispatch path. Capture-mode triggers an
        // auto-fallback to bespoke (FA2's launch-time
        // cudaFuncSetAttribute for opt-in dynamic SMEM isn't capture-
        // safe; the call mutates per-function attributes outside the
        // graph). Mirrors Phase 30's cuBLAS capture fallback in
        // baracuda-cutlass::GemmPlan.
        #[cfg(feature = "fa2")]
        if matches!(self.backend, BackendChoice::FlashAttentionV2) {
            let capturing = stream.is_capturing().unwrap_or(false);
            if !capturing {
                return self.run_fa2(stream, workspace, &args);
            }
            // else: fall through to bespoke launch below.
        }
        // The `workspace` arg is intentionally consumed only by FA2;
        // the bespoke kernels are workspace-free. Bind to `_` so we
        // don't pessimize on unused-var warnings.
        let _ = workspace;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_flash_sdpa_f32_run(
                    self.desc.batch_size,
                    self.desc.num_heads,
                    self.desc.query_len,
                    self.desc.key_len,
                    self.desc.d_k,
                    self.desc.d_v,
                    self.desc.scale,
                    is_causal_flag,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    y_ptr,
                    lse_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_flash_sdpa_f16_run(
                    self.desc.batch_size,
                    self.desc.num_heads,
                    self.desc.query_len,
                    self.desc.key_len,
                    self.desc.d_k,
                    self.desc.d_v,
                    self.desc.scale,
                    is_causal_flag,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    y_ptr,
                    lse_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_flash_sdpa_bf16_run(
                    self.desc.batch_size,
                    self.desc.num_heads,
                    self.desc.query_len,
                    self.desc.key_len,
                    self.desc.d_k,
                    self.desc.d_v,
                    self.desc.scale,
                    is_causal_flag,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    y_ptr,
                    lse_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_flash_sdpa_f64_run(
                    self.desc.batch_size,
                    self.desc.num_heads,
                    self.desc.query_len,
                    self.desc.key_len,
                    self.desc.d_k,
                    self.desc.d_v,
                    self.desc.scale,
                    is_causal_flag,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    y_ptr,
                    lse_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::FlashSdpaPlan::run reached an unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }

    /// FA2 backend launch path (Phase 42).
    ///
    /// Routes the FA2 LSE write to caller-supplied workspace memory
    /// (FA2 always writes LSE in f32 regardless of element dtype;
    /// see `workspace_size`). The caller-visible `args.lse` buffer is
    /// left untouched on the FA2 path — Tier 1 doesn't wire BW for
    /// FA2, so the saved LSE has no downstream consumer.
    #[cfg(feature = "fa2")]
    fn run_fa2(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: &FlashSdpaArgs<'_, T>,
    ) -> Result<()> {
        let stream_ptr = stream.as_raw() as *mut c_void;
        let q_ptr = args.q.data.as_raw().0 as *const c_void;
        let k_ptr = args.k.data.as_raw().0 as *const c_void;
        let v_ptr = args.v.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;
        let is_causal_flag = if self.desc.is_causal { 1 } else { 0 };

        // Workspace carries the f32 LSE scratch (4 bytes per LSE cell).
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
                baracuda_kernels_sys::baracuda_kernels_fa2_sdpa_f16_run(
                    self.desc.batch_size,
                    self.desc.num_heads,
                    self.desc.num_heads,  // num_heads_k == num_heads in Tier 1
                    self.desc.query_len,
                    self.desc.key_len,
                    self.desc.d_k,
                    self.desc.scale,
                    is_causal_flag,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    y_ptr,
                    ws_ptr, // softmax_lse → routed to workspace (f32)
                    core::ptr::null_mut(),
                    ws_bytes,
                    stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_fa2_sdpa_bf16_run(
                    self.desc.batch_size,
                    self.desc.num_heads,
                    self.desc.num_heads,
                    self.desc.query_len,
                    self.desc.key_len,
                    self.desc.d_k,
                    self.desc.scale,
                    is_causal_flag,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    y_ptr,
                    ws_ptr,
                    core::ptr::null_mut(),
                    ws_bytes,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::FlashSdpaPlan::run_fa2: FA2 supports only f16 / bf16",
                ));
            }
        };
        map_status(status)
    }
}

/// Internal: pick the backend for a given descriptor + preference.
/// Honours `pref.prefer_backend` unconditionally for `Bespoke`;
/// validates the FA2 SKU constraints before honouring an FA2
/// override (returns Bespoke on mismatch). Falls back to the
/// heuristic when no override is supplied.
fn pick_backend<T: Element>(desc: &FlashSdpaDescriptor, pref: PlanPreference) -> BackendChoice {
    match pref.prefer_backend {
        Some(BackendKind::Bespoke) => BackendChoice::Bespoke,
        #[cfg(feature = "fa2")]
        Some(BackendKind::FlashAttentionV2) => {
            if should_use_fa2(desc, desc.num_heads) || fa2_is_eligible::<T>(desc) {
                BackendChoice::FlashAttentionV2
            } else {
                BackendChoice::Bespoke
            }
        }
        _ => {
            #[cfg(feature = "fa2")]
            {
                if should_use_fa2(desc, desc.num_heads) {
                    return BackendChoice::FlashAttentionV2;
                }
            }
            BackendChoice::Bespoke
        }
    }
}

/// Hard eligibility check for FA2 (separate from the perf heuristic).
/// Used to validate caller overrides — returns true iff FA2 *can*
/// run this descriptor at all.
#[cfg(feature = "fa2")]
fn fa2_is_eligible<T: Element>(desc: &FlashSdpaDescriptor) -> bool {
    desc.d_k == 128
        && desc.d_v == 128
        && matches!(T::KIND, ElementKind::F16 | ElementKind::Bf16)
}
