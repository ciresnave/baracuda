//! Flash Attention varlen plan family (Phase 59b).
//!
//! Varlen (variable-length) attention runs a batch of sequences with
//! **different lengths** through a single attention launch, packed into
//! contiguous tensors with auxiliary `cu_seqlens_*` index tensors that
//! record where each sequence starts.
//!
//! ## Why varlen matters
//!
//! Two classic workloads are wasteful under dense (padded) attention:
//!
//! 1. **Packed-batch training**: instead of padding each sample to
//!    `max_seqlen` (and wasting FLOPs on padding tokens), concatenate
//!    them. A batch of `[7, 13, 5, 11]`-token sequences becomes a single
//!    packed `[36]`-token sequence with `cu_seqlens = [0, 7, 20, 25, 36]`.
//!    Attention runs per-sequence, with no wasted compute.
//! 2. **Paged-attention prefill** (servers like vLLM, SGLang): different
//!    requests have different prompt lengths; varlen lets the
//!    multi-request prefill run in one call rather than launching one
//!    attention call per request.
//!
//! ## Layout contract
//!
//! - **Q / dQ / output / dO**: shape `[total_q, num_heads, d_k]`,
//!   row-major contiguous. `total_q = sum(seqlens_q)`.
//! - **K / dK / V / dV**: shape `[total_k, num_heads_k, d_k]`. `total_k`
//!   is independent of `total_q` (cross-attention) or equal to it
//!   (self-attention).
//! - **`cu_seqlens_q`**: `i32[batch + 1]`, cumulative offsets. Convention:
//!   `cu_seqlens_q[0] = 0`, `cu_seqlens_q[batch] = total_q`,
//!   `cu_seqlens_q[i+1] - cu_seqlens_q[i] = length of sample i`.
//! - **`cu_seqlens_k`**: same convention, `total_k`.
//! - **LSE** (FW output / BW input): **always f32**, shape
//!   `[num_heads, total_q + 128 * batch]` (FA2's "unpadded LSE" format
//!   with sentinel padding rows for the BW pipeline's row scheduling).
//!   Size in bytes:
//!   `baracuda_kernels_fa2_sdpa_varlen_lse_size(...) * 4`.
//!
//! ## Constraints
//!
//! Same as the dense FA2 path:
//! - dtype ∈ {f16, bf16}
//! - head_dim ∈ {32, 64, 96, 128, 192, 256}
//! - `num_heads % num_heads_k == 0` (GQA)
//! - sliding window / softcap / ALiBi all optional, FA2-native
//!
//! ## Workspace (BW only)
//!
//! FW workspace is 0 (FA2 varlen FW writes LSE directly to caller-supplied
//! buffer; no internal scratch needed for the kernels we ship).
//!
//! BW workspace is `dq_accum + dsoftmax_d`, sized off
//! `total_q + 128 * batch` rows. Use [`FlashSdpaVarlenBackwardPlan::workspace_size`].

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, AttentionKind, BackendKind, Element, ElementKind, KernelSku, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};


// ---------------------------------------------------------------------------
// Descriptor (shared between FW + BW)
// ---------------------------------------------------------------------------

/// Descriptor for a packed-batch (varlen) Flash Attention op.
///
/// `#[non_exhaustive]` — construct via [`Self::new`] + chainable
/// `with_*` setters.
#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
pub struct FlashSdpaVarlenDescriptor {
    /// Number of sequences in the pack.
    pub batch_size: i32,
    /// Number of attention heads (`H`).
    pub num_heads: i32,
    /// Number of K/V heads (`H_k`). Must divide `num_heads`.
    pub num_heads_k: i32,
    /// Maximum query length across sequences (`max(seqlens_q)`). Used
    /// for tile-sizing bookkeeping.
    pub max_seqlen_q: i32,
    /// Maximum key length across sequences (`max(seqlens_k)`).
    pub max_seqlen_k: i32,
    /// Head dim of Q and K (`D_k`).
    pub d_k: i32,
    /// Head dim of V (`D_v`). Must equal `d_k`.
    pub d_v: i32,
    /// Score scale (typically `1 / sqrt(d_k)`).
    pub scale: f32,
    /// Apply upper-triangular causal mask per sequence.
    pub is_causal: bool,
    /// Element type. Must be `F16` or `Bf16` for FA2.
    pub element: ElementKind,
    /// Sliding-window left bound (FA2).
    pub window_size_left: Option<i32>,
    /// Sliding-window right bound (FA2).
    pub window_size_right: Option<i32>,
    /// Softcap (Gemma-2). `0.0` = disabled.
    pub softcap: f32,
}

impl FlashSdpaVarlenDescriptor {
    /// Build a `FlashSdpaVarlenDescriptor` with the required fields.
    /// Sliding window and softcap default to disabled.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn new(
        batch_size: i32,
        num_heads: i32,
        num_heads_k: i32,
        max_seqlen_q: i32,
        max_seqlen_k: i32,
        d_k: i32,
        d_v: i32,
        scale: f32,
        is_causal: bool,
        element: ElementKind,
    ) -> Self {
        Self {
            batch_size,
            num_heads,
            num_heads_k,
            max_seqlen_q,
            max_seqlen_k,
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

    /// Builder: sliding-window left bound.
    #[inline]
    pub fn with_window_size_left(mut self, n: Option<i32>) -> Self {
        self.window_size_left = n;
        self
    }

    /// Builder: sliding-window right bound.
    #[inline]
    pub fn with_window_size_right(mut self, n: Option<i32>) -> Self {
        self.window_size_right = n;
        self
    }

    /// Builder: softcap value.
    #[inline]
    pub fn with_softcap(mut self, cap: f32) -> Self {
        self.softcap = cap;
        self
    }
}

// ---------------------------------------------------------------------------
// FW
// ---------------------------------------------------------------------------

/// Args bundle for a packed-batch Flash Attention forward launch.
pub struct FlashSdpaVarlenArgs<'a, T: Element> {
    /// Query — packed `[total_q, H, D_k]` row-major contiguous.
    pub q: TensorRef<'a, T, 3>,
    /// Key — packed `[total_k, H_k, D_k]`.
    pub k: TensorRef<'a, T, 3>,
    /// Value — packed `[total_k, H_k, D_v]`.
    pub v: TensorRef<'a, T, 3>,
    /// Output — packed `[total_q, H, D_v]` (caller-allocated).
    pub y: TensorMut<'a, T, 3>,
    /// Log-sum-exp output — **f32**, packed `[H, total_q + 128 * batch]`.
    /// Caller allocates; use [`FlashSdpaVarlenPlan::lse_size`] to size.
    pub lse: TensorMut<'a, f32, 2>,
    /// Per-sequence start offsets for Q — `i32[batch + 1]`.
    pub cu_seqlens_q: TensorRef<'a, i32, 1>,
    /// Per-sequence start offsets for K/V — `i32[batch + 1]`.
    pub cu_seqlens_k: TensorRef<'a, i32, 1>,
    /// Optional ALiBi slopes — f32 `[1, H]` or `[B, H]`.
    pub alibi_slopes: Option<TensorRef<'a, f32, 2>>,
}

/// Flash Attention varlen FW plan (Phase 59b).
///
/// Routes through the FA2 varlen FW path (`fa2_varlen_launcher.cu`).
/// `f16` / `bf16` only.
pub struct FlashSdpaVarlenPlan<T: Element> {
    desc: FlashSdpaVarlenDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> FlashSdpaVarlenPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &FlashSdpaVarlenDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_desc::<T>(desc)?;
        let sku = KernelSku {
            category: OpCategory::Attention,
            op: AttentionKind::FlashAttention as u16,
            element: T::KIND,
            aux_element: None,
            layout: None,
            epilogue: None,
            arch: ArchSku::Sm80,
            backend: BackendKind::FlashAttentionV2,
            precision_guarantee: PrecisionGuarantee {
                math_precision: MathPrecision::F32,
                accumulator: ElementKind::F32,
                // FA2 varlen FW is deterministic (one block per
                // (q_block, b, h) tile); matches the dense FW.
                bit_stable_on_same_hardware: true,
                deterministic: true,
            },
        };
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Required f32-element count for the `lse` output tensor.
    /// `H * (total_q + 128 * batch)`. Caller multiplies by 4 for bytes.
    #[inline]
    pub fn lse_size(&self, total_q: i32) -> usize {
        #[cfg(feature = "fa2")]
        unsafe {
            baracuda_kernels_sys::baracuda_kernels_fa2_sdpa_varlen_lse_size(
                self.desc.batch_size,
                self.desc.num_heads,
                total_q,
            )
        }
        #[cfg(not(feature = "fa2"))]
        {
            let _ = total_q;
            0
        }
    }

    /// Workspace size — 0 for FA2 varlen FW.
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

    /// Validate args.
    pub fn can_implement(&self, args: &FlashSdpaVarlenArgs<'_, T>) -> Result<()> {
        let total_q = args.q.shape[0];
        let total_k = args.k.shape[0];

        if args.q.shape[1] != self.desc.num_heads
            || args.q.shape[2] != self.desc.d_k
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FlashSdpaVarlenPlan: Q shape must be [total_q, H, D_k]",
            ));
        }
        if args.k.shape[1] != self.desc.num_heads_k
            || args.k.shape[2] != self.desc.d_k
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FlashSdpaVarlenPlan: K shape must be [total_k, H_k, D_k]",
            ));
        }
        if args.v.shape != [total_k, self.desc.num_heads_k, self.desc.d_v] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FlashSdpaVarlenPlan: V shape must be [total_k, H_k, D_v]",
            ));
        }
        if args.y.shape != [total_q, self.desc.num_heads, self.desc.d_v] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FlashSdpaVarlenPlan: y shape must be [total_q, H, D_v]",
            ));
        }
        if args.cu_seqlens_q.shape != [self.desc.batch_size + 1] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FlashSdpaVarlenPlan: cu_seqlens_q shape must be [batch + 1]",
            ));
        }
        if args.cu_seqlens_k.shape != [self.desc.batch_size + 1] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FlashSdpaVarlenPlan: cu_seqlens_k shape must be [batch + 1]",
            ));
        }
        let needed_lse = self.lse_size(total_q);
        let lse_n = (args.lse.numel() as usize).max(0);
        if lse_n < needed_lse {
            return Err(Error::BufferTooSmall {
                needed: needed_lse,
                got: lse_n,
            });
        }
        if !args.q.is_contiguous()
            || !args.k.is_contiguous()
            || !args.v.is_contiguous()
            || !args.y.is_contiguous()
            || !args.cu_seqlens_q.is_contiguous()
            || !args.cu_seqlens_k.is_contiguous()
        {
            return Err(Error::Unsupported(
                "baracuda-kernels::FlashSdpaVarlenPlan: all tensors must be contiguous",
            ));
        }
        if let Some(slopes) = args.alibi_slopes.as_ref() {
            if slopes.shape[1] != self.desc.num_heads {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::FlashSdpaVarlenPlan: alibi_slopes shape[1] must equal num_heads",
                ));
            }
            if slopes.shape[0] != 1 && slopes.shape[0] != self.desc.batch_size {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::FlashSdpaVarlenPlan: alibi_slopes shape[0] must be 1 or batch_size",
                ));
            }
        }
        Ok(())
    }

    /// Launch the FA2 varlen FW kernel.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: FlashSdpaVarlenArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if args.q.numel() == 0 {
            return Ok(());
        }
        #[cfg(not(feature = "fa2"))]
        {
            let _ = stream;
            return Err(Error::Unsupported(
                "baracuda-kernels::FlashSdpaVarlenPlan: varlen requires the `fa2` cargo feature",
            ));
        }
        #[cfg(feature = "fa2")]
        {
            let total_q = args.q.shape[0];
            let total_k = args.k.shape[0];
            let stream_ptr = stream.as_raw() as *mut c_void;
            let q_ptr = args.q.data.as_raw().0 as *const c_void;
            let k_ptr = args.k.data.as_raw().0 as *const c_void;
            let v_ptr = args.v.data.as_raw().0 as *const c_void;
            let y_ptr = args.y.data.as_raw().0 as *mut c_void;
            let lse_ptr = args.lse.data.as_raw().0 as *mut c_void;
            let cu_q_ptr = args.cu_seqlens_q.data.as_raw().0 as *const i32;
            let cu_k_ptr = args.cu_seqlens_k.data.as_raw().0 as *const i32;
            let is_causal_flag = if self.desc.is_causal { 1 } else { 0 };
            let (alibi_ptr, alibi_batch_stride) = alibi_dispatch(&self.desc, &args.alibi_slopes);

            let window_left = self.desc.window_size_left.unwrap_or(-1);
            let window_right = self.desc.window_size_right.unwrap_or(-1);
            let softcap = self.desc.softcap;

            let status = match T::KIND {
                ElementKind::F16 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_fa2_sdpa_varlen_f16_run(
                        self.desc.batch_size, self.desc.num_heads, self.desc.num_heads_k,
                        self.desc.max_seqlen_q, self.desc.max_seqlen_k,
                        total_q, total_k, self.desc.d_k,
                        self.desc.scale, is_causal_flag,
                        alibi_ptr, alibi_batch_stride,
                        window_left, window_right, softcap,
                        q_ptr, k_ptr, v_ptr,
                        cu_q_ptr, cu_k_ptr,
                        y_ptr, lse_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                ElementKind::Bf16 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_fa2_sdpa_varlen_bf16_run(
                        self.desc.batch_size, self.desc.num_heads, self.desc.num_heads_k,
                        self.desc.max_seqlen_q, self.desc.max_seqlen_k,
                        total_q, total_k, self.desc.d_k,
                        self.desc.scale, is_causal_flag,
                        alibi_ptr, alibi_batch_stride,
                        window_left, window_right, softcap,
                        q_ptr, k_ptr, v_ptr,
                        cu_q_ptr, cu_k_ptr,
                        y_ptr, lse_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                _ => {
                    return Err(Error::Unsupported(
                        "baracuda-kernels::FlashSdpaVarlenPlan: FA2 supports only f16 / bf16",
                    ));
                }
            };
            map_status(status)
        }
    }
}

// ---------------------------------------------------------------------------
// BW
// ---------------------------------------------------------------------------

/// Args bundle for a packed-batch Flash Attention backward launch.
pub struct FlashSdpaVarlenBackwardArgs<'a, T: Element> {
    /// Query (saved from FW) — packed `[total_q, H, D_k]`.
    pub q: TensorRef<'a, T, 3>,
    /// Key (saved from FW) — packed `[total_k, H_k, D_k]`.
    pub k: TensorRef<'a, T, 3>,
    /// Value (saved from FW) — packed `[total_k, H_k, D_v]`.
    pub v: TensorRef<'a, T, 3>,
    /// Saved FW output `y` — packed `[total_q, H, D_v]`.
    pub y: TensorRef<'a, T, 3>,
    /// Upstream gradient `dy` — packed `[total_q, H, D_v]`.
    pub dy: TensorRef<'a, T, 3>,
    /// FW-saved LSE — **f32**, packed `[H, total_q + 128 * batch]`.
    pub lse: TensorRef<'a, f32, 2>,
    /// `cu_seqlens_q` — `i32[batch + 1]`.
    pub cu_seqlens_q: TensorRef<'a, i32, 1>,
    /// `cu_seqlens_k` — `i32[batch + 1]`.
    pub cu_seqlens_k: TensorRef<'a, i32, 1>,
    /// dQ — packed `[total_q, H, D_k]` (caller-allocated).
    pub dq: TensorMut<'a, T, 3>,
    /// dK — packed `[total_k, H_k, D_k]`.
    pub dk: TensorMut<'a, T, 3>,
    /// dV — packed `[total_k, H_k, D_v]`.
    pub dv: TensorMut<'a, T, 3>,
    /// Optional ALiBi slopes.
    pub alibi_slopes: Option<TensorRef<'a, f32, 2>>,
}

/// Flash Attention varlen BW plan (Phase 59b).
pub struct FlashSdpaVarlenBackwardPlan<T: Element> {
    desc: FlashSdpaVarlenDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> FlashSdpaVarlenBackwardPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &FlashSdpaVarlenDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_desc::<T>(desc)?;
        let sku = KernelSku {
            category: OpCategory::Attention,
            op: AttentionKind::FlashAttention as u16,
            element: T::KIND,
            aux_element: None,
            layout: None,
            epilogue: None,
            arch: ArchSku::Sm80,
            backend: BackendKind::FlashAttentionV2,
            precision_guarantee: PrecisionGuarantee {
                math_precision: MathPrecision::F32,
                accumulator: ElementKind::F32,
                // FA2 BW uses atomicAdd into dq_accum; not bit-stable.
                bit_stable_on_same_hardware: false,
                deterministic: false,
            },
        };
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Required workspace size in bytes.
    pub fn workspace_size(&self, total_q: i32) -> usize {
        #[cfg(feature = "fa2")]
        unsafe {
            baracuda_kernels_sys::baracuda_kernels_fa2_sdpa_varlen_backward_workspace_size(
                self.desc.batch_size,
                self.desc.num_heads,
                self.desc.max_seqlen_q,
                total_q,
                self.desc.d_k,
            )
        }
        #[cfg(not(feature = "fa2"))]
        {
            let _ = total_q;
            0
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

    /// Validate args.
    pub fn can_implement(&self, args: &FlashSdpaVarlenBackwardArgs<'_, T>) -> Result<()> {
        let total_q = args.q.shape[0];
        let total_k = args.k.shape[0];
        let shape_q = [total_q, self.desc.num_heads, self.desc.d_k];
        let shape_k = [total_k, self.desc.num_heads_k, self.desc.d_k];
        let shape_v = [total_k, self.desc.num_heads_k, self.desc.d_v];
        let shape_y = [total_q, self.desc.num_heads, self.desc.d_v];
        if args.q.shape != shape_q {
            return Err(Error::InvalidProblem("FlashSdpaVarlenBackwardPlan: Q shape mismatch"));
        }
        if args.k.shape != shape_k {
            return Err(Error::InvalidProblem("FlashSdpaVarlenBackwardPlan: K shape mismatch"));
        }
        if args.v.shape != shape_v {
            return Err(Error::InvalidProblem("FlashSdpaVarlenBackwardPlan: V shape mismatch"));
        }
        if args.y.shape != shape_y {
            return Err(Error::InvalidProblem("FlashSdpaVarlenBackwardPlan: y shape mismatch"));
        }
        if args.dy.shape != shape_y {
            return Err(Error::InvalidProblem("FlashSdpaVarlenBackwardPlan: dy shape mismatch"));
        }
        if args.dq.shape != shape_q {
            return Err(Error::InvalidProblem("FlashSdpaVarlenBackwardPlan: dQ shape mismatch"));
        }
        if args.dk.shape != shape_k {
            return Err(Error::InvalidProblem("FlashSdpaVarlenBackwardPlan: dK shape mismatch"));
        }
        if args.dv.shape != shape_v {
            return Err(Error::InvalidProblem("FlashSdpaVarlenBackwardPlan: dV shape mismatch"));
        }
        if args.cu_seqlens_q.shape != [self.desc.batch_size + 1] {
            return Err(Error::InvalidProblem(
                "FlashSdpaVarlenBackwardPlan: cu_seqlens_q shape must be [batch + 1]",
            ));
        }
        if args.cu_seqlens_k.shape != [self.desc.batch_size + 1] {
            return Err(Error::InvalidProblem(
                "FlashSdpaVarlenBackwardPlan: cu_seqlens_k shape must be [batch + 1]",
            ));
        }
        // LSE: f32 [H, total_q + 128*B]
        let lse_cols = (total_q as usize).saturating_add(128usize * self.desc.batch_size as usize);
        if args.lse.shape[0] != self.desc.num_heads
            || (args.lse.shape[1] as usize) < lse_cols
        {
            return Err(Error::InvalidProblem(
                "FlashSdpaVarlenBackwardPlan: lse shape must be [H, total_q + 128*B] f32",
            ));
        }
        if !args.q.is_contiguous()
            || !args.k.is_contiguous()
            || !args.v.is_contiguous()
            || !args.y.is_contiguous()
            || !args.dy.is_contiguous()
            || !args.lse.is_contiguous()
            || !args.cu_seqlens_q.is_contiguous()
            || !args.cu_seqlens_k.is_contiguous()
            || !args.dq.is_contiguous()
            || !args.dk.is_contiguous()
            || !args.dv.is_contiguous()
        {
            return Err(Error::Unsupported(
                "FlashSdpaVarlenBackwardPlan: all tensors must be contiguous",
            ));
        }
        if let Some(slopes) = args.alibi_slopes.as_ref() {
            if slopes.shape[1] != self.desc.num_heads {
                return Err(Error::InvalidProblem(
                    "FlashSdpaVarlenBackwardPlan: alibi_slopes shape[1] must equal num_heads",
                ));
            }
            if slopes.shape[0] != 1 && slopes.shape[0] != self.desc.batch_size {
                return Err(Error::InvalidProblem(
                    "FlashSdpaVarlenBackwardPlan: alibi_slopes shape[0] must be 1 or batch_size",
                ));
            }
        }
        Ok(())
    }

    /// Launch the FA2 varlen BW kernel.
    pub fn run(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: FlashSdpaVarlenBackwardArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if args.q.numel() == 0 {
            return Ok(());
        }
        #[cfg(not(feature = "fa2"))]
        {
            let _ = (stream, workspace);
            return Err(Error::Unsupported(
                "baracuda-kernels::FlashSdpaVarlenBackwardPlan: requires the `fa2` cargo feature",
            ));
        }
        #[cfg(feature = "fa2")]
        {
            let total_q = args.q.shape[0];
            let total_k = args.k.shape[0];
            let need = self.workspace_size(total_q);
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

            let stream_ptr = stream.as_raw() as *mut c_void;
            let q_ptr = args.q.data.as_raw().0 as *const c_void;
            let k_ptr = args.k.data.as_raw().0 as *const c_void;
            let v_ptr = args.v.data.as_raw().0 as *const c_void;
            let y_ptr = args.y.data.as_raw().0 as *const c_void;
            let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
            let lse_ptr = args.lse.data.as_raw().0 as *const c_void;
            let cu_q_ptr = args.cu_seqlens_q.data.as_raw().0 as *const i32;
            let cu_k_ptr = args.cu_seqlens_k.data.as_raw().0 as *const i32;
            let dq_ptr = args.dq.data.as_raw().0 as *mut c_void;
            let dk_ptr = args.dk.data.as_raw().0 as *mut c_void;
            let dv_ptr = args.dv.data.as_raw().0 as *mut c_void;
            let is_causal_flag = if self.desc.is_causal { 1 } else { 0 };
            let (alibi_ptr, alibi_batch_stride) = alibi_dispatch(&self.desc, &args.alibi_slopes);

            let window_left = self.desc.window_size_left.unwrap_or(-1);
            let window_right = self.desc.window_size_right.unwrap_or(-1);
            let softcap = self.desc.softcap;

            let status = match T::KIND {
                ElementKind::F16 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_fa2_sdpa_varlen_backward_f16_run(
                        self.desc.batch_size, self.desc.num_heads, self.desc.num_heads_k,
                        self.desc.max_seqlen_q, self.desc.max_seqlen_k,
                        total_q, total_k, self.desc.d_k,
                        self.desc.scale, is_causal_flag,
                        alibi_ptr, alibi_batch_stride,
                        window_left, window_right, softcap,
                        q_ptr, k_ptr, v_ptr, y_ptr, dy_ptr, lse_ptr,
                        cu_q_ptr, cu_k_ptr,
                        dq_ptr, dk_ptr, dv_ptr,
                        ws_ptr, ws_bytes, stream_ptr,
                    )
                },
                ElementKind::Bf16 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_fa2_sdpa_varlen_backward_bf16_run(
                        self.desc.batch_size, self.desc.num_heads, self.desc.num_heads_k,
                        self.desc.max_seqlen_q, self.desc.max_seqlen_k,
                        total_q, total_k, self.desc.d_k,
                        self.desc.scale, is_causal_flag,
                        alibi_ptr, alibi_batch_stride,
                        window_left, window_right, softcap,
                        q_ptr, k_ptr, v_ptr, y_ptr, dy_ptr, lse_ptr,
                        cu_q_ptr, cu_k_ptr,
                        dq_ptr, dk_ptr, dv_ptr,
                        ws_ptr, ws_bytes, stream_ptr,
                    )
                },
                _ => {
                    return Err(Error::Unsupported(
                        "FlashSdpaVarlenBackwardPlan: FA2 supports only f16 / bf16",
                    ));
                }
            };
            map_status(status)
        }
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn validate_desc<T: Element>(desc: &FlashSdpaVarlenDescriptor) -> Result<()> {
    if desc.element != T::KIND {
        return Err(Error::Unsupported(
            "FlashSdpaVarlen{Backward}Plan: descriptor element != T",
        ));
    }
    if desc.batch_size < 0
        || desc.num_heads < 0
        || desc.num_heads_k < 0
        || desc.max_seqlen_q < 0
        || desc.max_seqlen_k < 0
        || desc.d_k < 0
        || desc.d_v < 0
    {
        return Err(Error::InvalidProblem(
            "FlashSdpaVarlen{Backward}Plan: extents must be non-negative",
        ));
    }
    if !desc.scale.is_finite() {
        return Err(Error::InvalidProblem(
            "FlashSdpaVarlen{Backward}Plan: scale must be finite",
        ));
    }
    if desc.d_k != desc.d_v {
        return Err(Error::Unsupported(
            "FlashSdpaVarlen{Backward}Plan: requires d_k == d_v",
        ));
    }
    if !matches!(T::KIND, ElementKind::F16 | ElementKind::Bf16) {
        return Err(Error::Unsupported(
            "FlashSdpaVarlen{Backward}Plan: FA2 varlen supports only f16 / bf16",
        ));
    }
    let supported = matches!(desc.d_k, 32 | 64 | 96 | 128 | 192 | 256);
    if !supported {
        return Err(Error::Unsupported(
            "FlashSdpaVarlen{Backward}Plan: head_dim must be in {32, 64, 96, 128, 192, 256}",
        ));
    }
    if desc.num_heads_k > desc.num_heads || desc.num_heads % desc.num_heads_k != 0 {
        return Err(Error::InvalidProblem(
            "FlashSdpaVarlen{Backward}Plan: num_heads_k must divide num_heads",
        ));
    }
    if desc.softcap < 0.0 || !desc.softcap.is_finite() {
        return Err(Error::InvalidProblem(
            "FlashSdpaVarlen{Backward}Plan: softcap must be finite and non-negative",
        ));
    }
    Ok(())
}

#[cfg(feature = "fa2")]
fn alibi_dispatch(
    desc: &FlashSdpaVarlenDescriptor,
    slopes: &Option<TensorRef<'_, f32, 2>>,
) -> (*const c_void, i32) {
    match slopes {
        None => (core::ptr::null::<c_void>(), 0i32),
        Some(s) => {
            let ptr = s.data.as_raw().0 as *const c_void;
            let batch_stride = if s.shape[0] == 1 {
                0_i32
            } else {
                desc.num_heads
            };
            (ptr, batch_stride)
        }
    }
}

// Fallback shape-matched stub for the `fa2`-OFF path. The varlen plan's
// `run()` returns `Error::Unsupported` without ever calling this when
// `fa2` is OFF, so the function is dead in that build; we keep it so the
// call-site type-checks under both feature configurations.
#[cfg(not(feature = "fa2"))]
#[allow(dead_code)]
fn alibi_dispatch(
    _desc: &FlashSdpaVarlenDescriptor,
    _slopes: &Option<TensorRef<'_, f32, 2>>,
) -> (*const c_void, i32) {
    (core::ptr::null::<c_void>(), 0)
}
