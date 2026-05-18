//! Flash Attention forward plan, sm_89 (Ada Lovelace) specialization
//! (Phase 10 Milestone 10.3).
//!
//! Sibling of [`FlashSdpaPlan`](super::FlashSdpaPlan) (sm_80 baseline /
//! Milestone 6.6). Same algorithm (Tri Dao 2022 tiled fused online-softmax
//! SDPA), same descriptor + args shape — callers swap by changing the
//! plan type. The math is bit-for-bit equivalent to the baseline (modulo
//! identical float-order in the FMA inner loops); what's different is
//! the *data-movement strategy*:
//!
//! - **`cp.async.cg.shared.global` double-buffered K/V loads.** While the
//!   current K-tile's GEMM runs, the next K/V tile is prefetched into a
//!   second SMEM stage via `__pipeline_memcpy_async`. Hides global-memory
//!   latency behind compute instead of stalling the warp on each
//!   `__syncthreads` after a load.
//! - **256 threads/block** (vs sm_80 baseline's 128). Ada's larger
//!   per-SM register file lets us run wider blocks without losing
//!   occupancy, which means fewer grid-stride iterations per tile pass.
//!
//! Dtype scope: **f16 + bf16 only**. f32 / f64 stay on the sm_80 baseline
//! plan — Ada's tensor cores don't help f32, and f64 is rare in
//! transformer inference.
//!
//! What this plan does *not* yet do (documented follow-ups in
//! [`baracuda_flash_sdpa_sm89.cuh`](../../../baracuda-kernels-sys/kernels/include/baracuda_flash_sdpa_sm89.cuh)):
//!
//! - `ldmatrix.sync.aligned.x4` warp-cooperative fragment loads.
//! - `nvcuda::wmma` m16n8k16 matmul fragments (would change float-order
//!   and need a separate tolerance budget).
//! - FP8 (E4M3 / E5M2) attention.
//!
//! Available only when the `sm89` cargo feature is enabled.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, AttentionKind, BackendKind, Element, ElementKind, KernelSku, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::map_status;
use super::flash_sdpa::FLASH_SDPA_MAX_D;

/// Descriptor for an sm_89-specialized Flash Attention forward op.
///
/// Same shape as [`FlashSdpaDescriptor`](super::FlashSdpaDescriptor) so
/// callers can swap plans by changing the type parameter only.
#[derive(Copy, Clone, Debug)]
pub struct FlashSdpaSm89Descriptor {
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
    /// Element type — must match the plan's type parameter and must be
    /// one of [`ElementKind::F16`] / [`ElementKind::Bf16`].
    pub element: ElementKind,
}

/// Args bundle. Identical layout to [`FlashSdpaArgs`](super::FlashSdpaArgs).
pub struct FlashSdpaSm89Args<'a, T: Element> {
    /// Query tensor — shape `[B, H, Q, D_k]`, contiguous.
    pub q: TensorRef<'a, T, 4>,
    /// Key tensor — shape `[B, H, K, D_k]`, contiguous.
    pub k: TensorRef<'a, T, 4>,
    /// Value tensor — shape `[B, H, K, D_v]`, contiguous.
    pub v: TensorRef<'a, T, 4>,
    /// Output tensor — shape `[B, H, Q, D_v]`, contiguous.
    pub y: TensorMut<'a, T, 4>,
    /// Saved log-sum-exp — shape `[B, H, Q]`, contiguous.
    pub lse: TensorMut<'a, T, 3>,
}

/// Flash Attention forward plan, sm_89 specialization.
pub struct FlashSdpaSm89Plan<T: Element> {
    desc: FlashSdpaSm89Descriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> FlashSdpaSm89Plan<T> {
    /// Pick a kernel. Rejects anything other than f16 / bf16 — callers
    /// wanting f32 / f64 should use the baseline
    /// [`FlashSdpaPlan`](super::FlashSdpaPlan) instead.
    pub fn select(
        _stream: &Stream,
        desc: &FlashSdpaSm89Descriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::FlashSdpaSm89Plan: descriptor element != T",
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
                "baracuda-kernels::FlashSdpaSm89Plan: extents must be non-negative",
            ));
        }
        if !desc.scale.is_finite() {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FlashSdpaSm89Plan: scale must be finite",
            ));
        }
        if desc.d_k != desc.d_v {
            return Err(Error::Unsupported(
                "baracuda-kernels::FlashSdpaSm89Plan: trailblazer requires d_k == d_v",
            ));
        }
        if desc.d_k > FLASH_SDPA_MAX_D {
            return Err(Error::Unsupported(
                "baracuda-kernels::FlashSdpaSm89Plan: d_k must be ≤ 128",
            ));
        }
        let dtype_in_scope = matches!(T::KIND, ElementKind::F16 | ElementKind::Bf16);
        if !dtype_in_scope {
            return Err(Error::Unsupported(
                "baracuda-kernels::FlashSdpaSm89Plan: f16 / bf16 only — use FlashSdpaPlan for f32 / f64",
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
            arch: ArchSku::Sm89,
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
    pub fn can_implement(&self, args: &FlashSdpaSm89Args<'_, T>) -> Result<()> {
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
                "baracuda-kernels::FlashSdpaSm89Plan: Q shape mismatch",
            ));
        }
        if args.k.shape != shape_k {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FlashSdpaSm89Plan: K shape mismatch",
            ));
        }
        if args.v.shape != shape_v {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FlashSdpaSm89Plan: V shape mismatch",
            ));
        }
        if args.y.shape != shape_y {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FlashSdpaSm89Plan: y shape mismatch",
            ));
        }
        if args.lse.shape != shape_lse {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FlashSdpaSm89Plan: lse shape must be [B, H, Q]",
            ));
        }
        if !args.q.is_contiguous()
            || !args.k.is_contiguous()
            || !args.v.is_contiguous()
            || !args.y.is_contiguous()
            || !args.lse.is_contiguous()
        {
            return Err(Error::Unsupported(
                "baracuda-kernels::FlashSdpaSm89Plan: trailblazer requires contiguous tensors",
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
    /// FW-saved state, same as the sm_80 baseline).
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// SKU identity (`arch = Sm89`).
    #[inline]
    pub fn sku(&self) -> KernelSku {
        self.sku
    }

    /// Numerical guarantees.
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee
    }

    /// Launch the sm_89 fused FW kernel on the supplied stream.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: FlashSdpaSm89Args<'_, T>,
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
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_flash_sdpa_sm89_f16_run(
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
                baracuda_kernels_sys::baracuda_kernels_flash_sdpa_sm89_bf16_run(
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
                    "baracuda-kernels::FlashSdpaSm89Plan::run reached an unsupported dtype \
                     (f16 / bf16 only — use FlashSdpaPlan for f32 / f64)",
                ));
            }
        };
        map_status(status)
    }
}
