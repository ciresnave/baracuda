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
    _marker: PhantomData<T>,
}

impl<T: Element> FlashSdpaPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &FlashSdpaDescriptor,
        _pref: PlanPreference,
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
            backend: BackendKind::Bespoke,
            precision_guarantee,
        };
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
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

    /// Workspace size in bytes — zero (the `lse` arg carries the only
    /// FW-saved state).
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

    /// Launch the fused FW kernel on the supplied stream.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
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
}
