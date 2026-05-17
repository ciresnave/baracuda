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
//! deterministic and uses no `atomicAdd`. Trailblazer constraints match
//! the FW plan: `Br = Bc = 64`, `d_k = d_v ≤ 128`.

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

/// Descriptor for a Flash Attention backward op.
#[derive(Copy, Clone, Debug)]
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
    /// Head dimension of V (`D_v`). Trailblazer requires `d_v == d_k`.
    pub d_v: i32,
    /// Score scaling factor — must match the FW scale.
    pub scale: f32,
    /// Apply upper-triangular causal mask — must match the FW
    /// descriptor's value.
    pub is_causal: bool,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a Flash Attention backward launch.
pub struct FlashSdpaBackwardArgs<'a, T: Element> {
    /// Query tensor used in FW — shape `[B, H, Q, D_k]`.
    pub q: TensorRef<'a, T, 4>,
    /// Key tensor used in FW — shape `[B, H, K, D_k]`.
    pub k: TensorRef<'a, T, 4>,
    /// Value tensor used in FW — shape `[B, H, K, D_v]`.
    pub v: TensorRef<'a, T, 4>,
    /// Saved FW output `y` — shape `[B, H, Q, D_v]`. Consumed by the
    /// `D = rowsum(y ⊙ dy)` reduction kernel.
    pub y: TensorRef<'a, T, 4>,
    /// Saved FW log-sum-exp — shape `[B, H, Q]`. Stores
    /// `m_i + log(l_i)` from the FW pass; used to re-derive `P_ij`
    /// without materializing the full attention matrix.
    pub lse: TensorRef<'a, T, 3>,
    /// Upstream gradient on the FW output — shape `[B, H, Q, D_v]`.
    pub dy: TensorRef<'a, T, 4>,
    /// Scratch buffer for `D = rowsum(y ⊙ dy)` — shape `[B, H, Q]`,
    /// element type `T`. Contents on entry are ignored.
    pub d_ws: TensorMut<'a, T, 3>,
    /// Output gradient `dQ` — shape `[B, H, Q, D_k]`.
    pub dq: TensorMut<'a, T, 4>,
    /// Output gradient `dK` — shape `[B, H, K, D_k]`.
    pub dk: TensorMut<'a, T, 4>,
    /// Output gradient `dV` — shape `[B, H, K, D_v]`.
    pub dv: TensorMut<'a, T, 4>,
}

/// Flash Attention backward plan.
pub struct FlashSdpaBackwardPlan<T: Element> {
    desc: FlashSdpaBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> FlashSdpaBackwardPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &FlashSdpaBackwardDescriptor,
        _pref: PlanPreference,
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
                "baracuda-kernels::FlashSdpaBackwardPlan: trailblazer requires d_k == d_v",
            ));
        }
        if desc.d_k > FLASH_SDPA_MAX_D {
            return Err(Error::Unsupported(
                "baracuda-kernels::FlashSdpaBackwardPlan: d_k must be ≤ 128 in the trailblazer",
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

        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            // dQ is written by the q-block owner; dK / dV by the k-block
            // owner. No atomicAdd.
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

    /// Validate args.
    pub fn can_implement(&self, args: &FlashSdpaBackwardArgs<'_, T>) -> Result<()> {
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
                "baracuda-kernels::FlashSdpaBackwardPlan: trailblazer requires contiguous tensors",
            ));
        }
        Ok(())
    }

    /// Workspace size in bytes — zero. The caller provides the per-row
    /// `D` scratch explicitly via the `d_ws` arg.
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

    /// Launch the BW pipeline (D / dQ / dK-dV kernels) on the supplied
    /// stream.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: FlashSdpaBackwardArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if args.q.numel() == 0 || args.k.numel() == 0 {
            return Ok(());
        }
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
                    dy_ptr,
                    d_ws_ptr,
                    dq_ptr,
                    dk_ptr,
                    dv_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_flash_sdpa_backward_f16_run(
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
                    dy_ptr,
                    d_ws_ptr,
                    dq_ptr,
                    dk_ptr,
                    dv_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_flash_sdpa_backward_bf16_run(
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
                    dy_ptr,
                    d_ws_ptr,
                    dq_ptr,
                    dk_ptr,
                    dv_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_flash_sdpa_backward_f64_run(
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
                    dy_ptr,
                    d_ws_ptr,
                    dq_ptr,
                    dk_ptr,
                    dv_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
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
}
