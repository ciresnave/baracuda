//! Naive Scaled Dot-Product Attention (SDPA) backward plan.
//!
//! Given the FW formula
//! `y = softmax(Q @ K^T / sqrt(d_k) + mask) @ V`, and upstream
//! `dy: [B, H, Q, D_v]`, computes `dQ`, `dK`, `dV`:
//!
//! ```text
//! dV       = attn^T @ dy                           shape [B, H, K, D_v]
//! dattn    = dy @ V^T                              shape [B, H, Q, K]
//! dscores  = softmax_bw(attn, dattn)               shape [B, H, Q, K]
//! dQ       = dscores @ K * scale                   shape [B, H, Q, D_k]
//! dK       = dscores^T @ Q * scale                 shape [B, H, K, D_k]
//! ```
//!
//! The launcher fires five sub-kernels under one symbol. A caller-
//! allocated `[B, H, Q, K]` scratch buffer (passed as `dscores_ws` in
//! the args) is reused: the dattn kernel writes into it, the dscores
//! kernel then overwrites it in place, and dQ / dK both read from it.
//! No additional workspace is needed beyond the saved FW `attn`
//! tensor and this single scratch.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, AttentionKind, BackendKind, Element, ElementKind, KernelSku, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::map_status;

/// Descriptor for a SDPA backward op.
#[derive(Copy, Clone, Debug)]
pub struct SdpaBackwardDescriptor {
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
    /// Head dimension of V (`D_v`).
    pub d_v: i32,
    /// Score scaling factor — must match the FW scale.
    pub scale: f32,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a SDPA backward launch.
pub struct SdpaBackwardArgs<'a, T: Element> {
    /// Query tensor used in FW — shape `[B, H, Q, D_k]`.
    pub q: TensorRef<'a, T, 4>,
    /// Key tensor used in FW — shape `[B, H, K, D_k]`.
    pub k: TensorRef<'a, T, 4>,
    /// Value tensor used in FW — shape `[B, H, K, D_v]`.
    pub v: TensorRef<'a, T, 4>,
    /// Saved softmax output from FW — shape `[B, H, Q, K]`. This is
    /// the `attn` field returned by [`SdpaPlan::run`].
    pub attn: TensorRef<'a, T, 4>,
    /// Upstream gradient on the FW output — shape `[B, H, Q, D_v]`.
    pub dy: TensorRef<'a, T, 4>,
    /// Scratch workspace for dattn / dscores — shape `[B, H, Q, K]`.
    /// Contents on entry are ignored; overwritten by the BW pipeline.
    pub dscores_ws: TensorMut<'a, T, 4>,
    /// Output gradient `dQ` — shape `[B, H, Q, D_k]`.
    pub dq: TensorMut<'a, T, 4>,
    /// Output gradient `dK` — shape `[B, H, K, D_k]`.
    pub dk: TensorMut<'a, T, 4>,
    /// Output gradient `dV` — shape `[B, H, K, D_v]`.
    pub dv: TensorMut<'a, T, 4>,
}

/// Naive SDPA backward plan.
///
/// Computes `dQ`, `dK`, `dV` from upstream `dy` and the saved softmax
/// output `attn` from the FW pass. Launches five sub-kernels behind a
/// single symbol per dtype (dV / dattn / dscores-via-softmax-bw / dQ /
/// dK), with a caller-supplied `dscores_ws` scratch reused as both the
/// `dattn` and `dscores` buffer.
///
/// **When to use**: autograd partner for [`super::SdpaPlan`]. The
/// `dscores_ws` arg is the only extra allocation the BW needs beyond
/// the saved-FW `attn`.
///
/// **Dtypes**: `f32`, `f64`, `f16`, `bf16` — matching the FW plan.
///
/// **Workspace**: zero. `dscores_ws` is passed explicitly via the args
/// rather than via the workspace channel because it is op-shaped (`[B,
/// H, Q, K]`) rather than a flat byte scratch.
///
/// **Precision guarantee**: deterministic; bit-stable on the same
/// hardware. No atomicAdd.
pub struct SdpaBackwardPlan<T: Element> {
    desc: SdpaBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> SdpaBackwardPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &SdpaBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::SdpaBackwardPlan: descriptor element != T",
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
                "baracuda-kernels::SdpaBackwardPlan: extents must be non-negative",
            ));
        }
        if !desc.scale.is_finite() {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SdpaBackwardPlan: scale must be finite",
            ));
        }
        let dtype_in_scope = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !dtype_in_scope {
            return Err(Error::Unsupported(
                "baracuda-kernels::SdpaBackwardPlan: wired today: `{f32, f16, bf16, f64}`",
            ));
        }

        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            // Per-cell deterministic — no atomic ops.
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Attention,
            op: AttentionKind::Sdpa as u16,
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
    pub fn can_implement(&self, args: &SdpaBackwardArgs<'_, T>) -> Result<()> {
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
        let shape_attn = [
            self.desc.batch_size,
            self.desc.num_heads,
            self.desc.query_len,
            self.desc.key_len,
        ];
        let shape_dy = [
            self.desc.batch_size,
            self.desc.num_heads,
            self.desc.query_len,
            self.desc.d_v,
        ];
        if args.q.shape != shape_q {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SdpaBackwardPlan: Q shape mismatch",
            ));
        }
        if args.k.shape != shape_k {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SdpaBackwardPlan: K shape mismatch",
            ));
        }
        if args.v.shape != shape_v {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SdpaBackwardPlan: V shape mismatch",
            ));
        }
        if args.attn.shape != shape_attn {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SdpaBackwardPlan: attn shape mismatch",
            ));
        }
        if args.dy.shape != shape_dy {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SdpaBackwardPlan: dy shape mismatch",
            ));
        }
        if args.dscores_ws.shape != shape_attn {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SdpaBackwardPlan: dscores_ws shape must match attn [B, H, Q, K]",
            ));
        }
        if args.dq.shape != shape_q {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SdpaBackwardPlan: dQ shape mismatch with Q",
            ));
        }
        if args.dk.shape != shape_k {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SdpaBackwardPlan: dK shape mismatch with K",
            ));
        }
        if args.dv.shape != shape_v {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SdpaBackwardPlan: dV shape mismatch with V",
            ));
        }
        // Phase 14.4: attn + dscores_ws must remain contig (algorithm
        // requires linear sweeps over the [B, H, Q, K] dimension). The
        // outer (B, H, S) strides on Q/K/V/dy/dQ/dK/dV may be
        // arbitrary; the innermost head_dim axis must remain stride=1.
        if !args.attn.is_contiguous() {
            return Err(Error::Unsupported(
                "baracuda-kernels::SdpaBackwardPlan: attn must be contiguous",
            ));
        }
        if !args.dscores_ws.is_contiguous() {
            return Err(Error::Unsupported(
                "baracuda-kernels::SdpaBackwardPlan: dscores_ws must be contiguous",
            ));
        }
        if args.q.stride[3] != 1
            || args.k.stride[3] != 1
            || args.v.stride[3] != 1
            || args.dy.stride[3] != 1
            || args.dq.stride[3] != 1
            || args.dk.stride[3] != 1
            || args.dv.stride[3] != 1
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SdpaBackwardPlan: head_dim axis stride must be 1 \
                 for Q / K / V / dy / dQ / dK / dV",
            ));
        }
        // BW does not support GQA broadcast (would require atomicAdd
        // over Q-head groups). Reject zero strides on K / V heads.
        if args.k.stride[1] == 0 || args.v.stride[1] == 0 {
            return Err(Error::Unsupported(
                "baracuda-kernels::SdpaBackwardPlan: BW does not support zero-stride \
                 (GQA broadcast) on K or V — caller must expand before BW",
            ));
        }
        Ok(())
    }

    /// Workspace size in bytes — zero. The caller already provides the
    /// `dscores_ws` tensor explicitly via the args.
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

    /// Launch all five sub-kernels in pipeline.
    ///
    /// Phase 14.4: dispatches between the contig fast path and the
    /// strided sibling FFI. BW does NOT support GQA broadcast (zero
    /// strides on K/V); the strided path's `can_implement` rejects.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: SdpaBackwardArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if args.attn.numel() == 0 {
            return Ok(());
        }
        let stream_ptr = stream.as_raw() as *mut c_void;
        let q_ptr = args.q.data.as_raw().0 as *const c_void;
        let k_ptr = args.k.data.as_raw().0 as *const c_void;
        let v_ptr = args.v.data.as_raw().0 as *const c_void;
        let attn_ptr = args.attn.data.as_raw().0 as *const c_void;
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let ws_ptr = args.dscores_ws.data.as_raw().0 as *mut c_void;
        let dq_ptr = args.dq.data.as_raw().0 as *mut c_void;
        let dk_ptr = args.dk.data.as_raw().0 as *mut c_void;
        let dv_ptr = args.dv.data.as_raw().0 as *mut c_void;

        let contig = args.q.is_contiguous()
            && args.k.is_contiguous()
            && args.v.is_contiguous()
            && args.dy.is_contiguous()
            && args.dq.is_contiguous()
            && args.dk.is_contiguous()
            && args.dv.is_contiguous();

        let status = unsafe {
            if contig {
                match T::KIND {
                    ElementKind::F32 => baracuda_kernels_sys::baracuda_kernels_sdpa_backward_f32_run(
                        self.desc.batch_size,
                        self.desc.num_heads,
                        self.desc.query_len,
                        self.desc.key_len,
                        self.desc.d_k,
                        self.desc.d_v,
                        self.desc.scale,
                        q_ptr, k_ptr, v_ptr, attn_ptr, dy_ptr,
                        ws_ptr, dq_ptr, dk_ptr, dv_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    ),
                    ElementKind::F16 => baracuda_kernels_sys::baracuda_kernels_sdpa_backward_f16_run(
                        self.desc.batch_size,
                        self.desc.num_heads,
                        self.desc.query_len,
                        self.desc.key_len,
                        self.desc.d_k,
                        self.desc.d_v,
                        self.desc.scale,
                        q_ptr, k_ptr, v_ptr, attn_ptr, dy_ptr,
                        ws_ptr, dq_ptr, dk_ptr, dv_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    ),
                    ElementKind::Bf16 => baracuda_kernels_sys::baracuda_kernels_sdpa_backward_bf16_run(
                        self.desc.batch_size,
                        self.desc.num_heads,
                        self.desc.query_len,
                        self.desc.key_len,
                        self.desc.d_k,
                        self.desc.d_v,
                        self.desc.scale,
                        q_ptr, k_ptr, v_ptr, attn_ptr, dy_ptr,
                        ws_ptr, dq_ptr, dk_ptr, dv_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    ),
                    ElementKind::F64 => baracuda_kernels_sys::baracuda_kernels_sdpa_backward_f64_run(
                        self.desc.batch_size,
                        self.desc.num_heads,
                        self.desc.query_len,
                        self.desc.key_len,
                        self.desc.d_k,
                        self.desc.d_v,
                        self.desc.scale,
                        q_ptr, k_ptr, v_ptr, attn_ptr, dy_ptr,
                        ws_ptr, dq_ptr, dk_ptr, dv_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    ),
                    _ => {
                        return Err(Error::Unsupported(
                            "baracuda-kernels::SdpaBackwardPlan::run reached an unimplemented dtype",
                        ));
                    }
                }
            } else {
                let stride_q: [i64; 3] = [args.q.stride[0], args.q.stride[1], args.q.stride[2]];
                let stride_k: [i64; 3] = [args.k.stride[0], args.k.stride[1], args.k.stride[2]];
                let stride_v: [i64; 3] = [args.v.stride[0], args.v.stride[1], args.v.stride[2]];
                let stride_dy: [i64; 3] = [args.dy.stride[0], args.dy.stride[1], args.dy.stride[2]];
                let stride_dq: [i64; 3] = [args.dq.stride[0], args.dq.stride[1], args.dq.stride[2]];
                let stride_dk: [i64; 3] = [args.dk.stride[0], args.dk.stride[1], args.dk.stride[2]];
                let stride_dv: [i64; 3] = [args.dv.stride[0], args.dv.stride[1], args.dv.stride[2]];
                match T::KIND {
                    ElementKind::F32 => baracuda_kernels_sys::baracuda_kernels_sdpa_backward_f32_strided_run(
                        self.desc.batch_size,
                        self.desc.num_heads,
                        self.desc.query_len,
                        self.desc.key_len,
                        self.desc.d_k,
                        self.desc.d_v,
                        stride_q.as_ptr(), stride_k.as_ptr(), stride_v.as_ptr(),
                        stride_dy.as_ptr(),
                        stride_dq.as_ptr(), stride_dk.as_ptr(), stride_dv.as_ptr(),
                        self.desc.scale,
                        q_ptr, k_ptr, v_ptr, attn_ptr, dy_ptr,
                        ws_ptr, dq_ptr, dk_ptr, dv_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    ),
                    ElementKind::F16 => baracuda_kernels_sys::baracuda_kernels_sdpa_backward_f16_strided_run(
                        self.desc.batch_size,
                        self.desc.num_heads,
                        self.desc.query_len,
                        self.desc.key_len,
                        self.desc.d_k,
                        self.desc.d_v,
                        stride_q.as_ptr(), stride_k.as_ptr(), stride_v.as_ptr(),
                        stride_dy.as_ptr(),
                        stride_dq.as_ptr(), stride_dk.as_ptr(), stride_dv.as_ptr(),
                        self.desc.scale,
                        q_ptr, k_ptr, v_ptr, attn_ptr, dy_ptr,
                        ws_ptr, dq_ptr, dk_ptr, dv_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    ),
                    ElementKind::Bf16 => baracuda_kernels_sys::baracuda_kernels_sdpa_backward_bf16_strided_run(
                        self.desc.batch_size,
                        self.desc.num_heads,
                        self.desc.query_len,
                        self.desc.key_len,
                        self.desc.d_k,
                        self.desc.d_v,
                        stride_q.as_ptr(), stride_k.as_ptr(), stride_v.as_ptr(),
                        stride_dy.as_ptr(),
                        stride_dq.as_ptr(), stride_dk.as_ptr(), stride_dv.as_ptr(),
                        self.desc.scale,
                        q_ptr, k_ptr, v_ptr, attn_ptr, dy_ptr,
                        ws_ptr, dq_ptr, dk_ptr, dv_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    ),
                    ElementKind::F64 => baracuda_kernels_sys::baracuda_kernels_sdpa_backward_f64_strided_run(
                        self.desc.batch_size,
                        self.desc.num_heads,
                        self.desc.query_len,
                        self.desc.key_len,
                        self.desc.d_k,
                        self.desc.d_v,
                        stride_q.as_ptr(), stride_k.as_ptr(), stride_v.as_ptr(),
                        stride_dy.as_ptr(),
                        stride_dq.as_ptr(), stride_dk.as_ptr(), stride_dv.as_ptr(),
                        self.desc.scale,
                        q_ptr, k_ptr, v_ptr, attn_ptr, dy_ptr,
                        ws_ptr, dq_ptr, dk_ptr, dv_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    ),
                    _ => {
                        return Err(Error::Unsupported(
                            "baracuda-kernels::SdpaBackwardPlan::run reached an unimplemented dtype",
                        ));
                    }
                }
            }
        };
        map_status(status)
    }
}
