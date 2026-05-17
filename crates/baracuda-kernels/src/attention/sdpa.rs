//! Naive Scaled Dot-Product Attention (SDPA) forward plan.
//!
//! Computes the standard PyTorch
//! `F.scaled_dot_product_attention` baseline that materializes the full
//! attention matrix:
//!
//! ```text
//! scores = (Q @ K^T) * scale + mask (optional) + causal-mask (optional)
//! attn   = row-softmax(scores)
//! y      = attn @ V
//! ```
//!
//! Shape conventions (all rank-4, contiguous, row-major):
//!
//! | tensor | shape                         |
//! |--------|-------------------------------|
//! | `Q`    | `[B, H, Q, D_k]`              |
//! | `K`    | `[B, H, K, D_k]`              |
//! | `V`    | `[B, H, K, D_v]`              |
//! | `mask` | `[B, H, Q, K]` (optional)     |
//! | `attn` | `[B, H, Q, K]` (saved for BW) |
//! | `y`    | `[B, H, Q, D_v]`              |
//!
//! `attn` is both the FW workspace for the scores intermediate **and**
//! the saved softmax output the BW plan consumes — no separate scratch
//! buffer is needed.
//!
//! Flash attention / FA-2 / KV-cache / paged attention are deferred to
//! future milestones; this trailblazer materializes the full
//! `[B, H, Q, K]` attention matrix.
//!
//! Wired today: `{f32, f16, bf16, f64}`. Dropout is not wired in this
//! trailblazer.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, AttentionKind, BackendKind, Element, ElementKind, KernelSku, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::map_status;

/// Descriptor for a naive SDPA forward op.
///
/// `d_k` and `d_v` may differ — V's head dimension can be larger or
/// smaller than Q/K's. All extents are `i32` to match the kernel ABI.
#[derive(Copy, Clone, Debug)]
pub struct SdpaDescriptor {
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
    /// Head dimension of V (`D_v`). May differ from `d_k`.
    pub d_v: i32,
    /// Score scaling factor — typically `1.0 / sqrt(d_k)`.
    pub scale: f32,
    /// Apply upper-triangular causal mask inside the scores kernel.
    pub is_causal: bool,
    /// Whether `mask` will be supplied at launch time.
    pub has_mask: bool,
    /// Element type — must match the plan's type parameter.
    pub element: ElementKind,
}

/// Args bundle for a SDPA forward launch.
pub struct SdpaArgs<'a, T: Element> {
    /// Query tensor — shape `[B, H, Q, D_k]`, contiguous.
    pub q: TensorRef<'a, T, 4>,
    /// Key tensor — shape `[B, H, K, D_k]`, contiguous.
    pub k: TensorRef<'a, T, 4>,
    /// Value tensor — shape `[B, H, K, D_v]`, contiguous.
    pub v: TensorRef<'a, T, 4>,
    /// Optional pre-softmax mask — shape `[B, H, Q, K]`, contiguous.
    /// Must be `Some(_)` iff the descriptor's `has_mask == true`.
    pub mask: Option<TensorRef<'a, T, 4>>,
    /// Output tensor — shape `[B, H, Q, D_v]`, contiguous.
    pub y: TensorMut<'a, T, 4>,
    /// Saved softmax output — shape `[B, H, Q, K]`, contiguous. The
    /// FW launcher uses this buffer first as the raw-scores intermediate
    /// and then overwrites it in-place with the softmax output. BW
    /// consumes the same tensor.
    pub attn: TensorMut<'a, T, 4>,
}

/// Naive SDPA forward plan.
pub struct SdpaPlan<T: Element> {
    desc: SdpaDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> SdpaPlan<T> {
    /// Pick a kernel.
    pub fn select(_stream: &Stream, desc: &SdpaDescriptor, _pref: PlanPreference) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::SdpaPlan: descriptor element != T",
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
                "baracuda-kernels::SdpaPlan: extents must be non-negative",
            ));
        }
        if !desc.scale.is_finite() {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SdpaPlan: scale must be finite",
            ));
        }
        let dtype_in_scope = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !dtype_in_scope {
            return Err(Error::Unsupported(
                "baracuda-kernels::SdpaPlan: wired today: `{f32, f16, bf16, f64}`",
            ));
        }

        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            // No atomic ops anywhere; per-cell deterministic.
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

    /// Validate args against the descriptor.
    pub fn can_implement(&self, args: &SdpaArgs<'_, T>) -> Result<()> {
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
        let shape_y = [
            self.desc.batch_size,
            self.desc.num_heads,
            self.desc.query_len,
            self.desc.d_v,
        ];
        if args.q.shape != shape_q {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SdpaPlan: Q shape mismatch",
            ));
        }
        if args.k.shape != shape_k {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SdpaPlan: K shape mismatch",
            ));
        }
        if args.v.shape != shape_v {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SdpaPlan: V shape mismatch",
            ));
        }
        if args.attn.shape != shape_attn {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SdpaPlan: attn shape mismatch",
            ));
        }
        if args.y.shape != shape_y {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SdpaPlan: y shape mismatch",
            ));
        }
        if !args.q.is_contiguous()
            || !args.k.is_contiguous()
            || !args.v.is_contiguous()
            || !args.y.is_contiguous()
            || !args.attn.is_contiguous()
        {
            return Err(Error::Unsupported(
                "baracuda-kernels::SdpaPlan: trailblazer requires contiguous tensors",
            ));
        }
        match (&args.mask, self.desc.has_mask) {
            (Some(m), true) => {
                if m.shape != shape_attn {
                    return Err(Error::InvalidProblem(
                        "baracuda-kernels::SdpaPlan: mask shape must be [B, H, Q, K]",
                    ));
                }
                if !m.is_contiguous() {
                    return Err(Error::Unsupported(
                        "baracuda-kernels::SdpaPlan: trailblazer requires contiguous mask",
                    ));
                }
                let mn = m.numel();
                if (m.data.len() as i64) < mn {
                    return Err(Error::BufferTooSmall {
                        needed: mn as usize,
                        got: m.data.len(),
                    });
                }
            }
            (None, false) => {}
            (Some(_), false) | (None, true) => {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::SdpaPlan: mask presence must match descriptor's has_mask",
                ));
            }
        }
        let attn_n = args.attn.numel();
        let y_n = args.y.numel();
        let q_n = args.q.numel();
        let k_n = args.k.numel();
        let v_n = args.v.numel();
        if (args.q.data.len() as i64) < q_n
            || (args.k.data.len() as i64) < k_n
            || (args.v.data.len() as i64) < v_n
            || (args.attn.data.len() as i64) < attn_n
            || (args.y.data.len() as i64) < y_n
        {
            return Err(Error::BufferTooSmall {
                needed: attn_n.max(y_n).max(q_n).max(k_n).max(v_n) as usize,
                got: 0,
            });
        }
        Ok(())
    }

    /// Workspace size in bytes — zero (the `attn` arg doubles as
    /// scratch).
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

    /// Launch all three sub-kernels (scores / row-softmax / out) in
    /// pipeline on the supplied stream.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: SdpaArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if args.attn.numel() == 0 || args.y.numel() == 0 {
            return Ok(());
        }
        let stream_ptr = stream.as_raw() as *mut c_void;
        let q_ptr = args.q.data.as_raw().0 as *const c_void;
        let k_ptr = args.k.data.as_raw().0 as *const c_void;
        let v_ptr = args.v.data.as_raw().0 as *const c_void;
        let mask_ptr = match &args.mask {
            Some(m) => m.data.as_raw().0 as *const c_void,
            None => core::ptr::null::<c_void>(),
        };
        let has_mask_flag = if self.desc.has_mask { 1 } else { 0 };
        let is_causal_flag = if self.desc.is_causal { 1 } else { 0 };
        let attn_ptr = args.attn.data.as_raw().0 as *mut c_void;
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_sdpa_f32_run(
                    self.desc.batch_size,
                    self.desc.num_heads,
                    self.desc.query_len,
                    self.desc.key_len,
                    self.desc.d_k,
                    self.desc.d_v,
                    self.desc.scale,
                    is_causal_flag,
                    has_mask_flag,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    mask_ptr,
                    attn_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_sdpa_f16_run(
                    self.desc.batch_size,
                    self.desc.num_heads,
                    self.desc.query_len,
                    self.desc.key_len,
                    self.desc.d_k,
                    self.desc.d_v,
                    self.desc.scale,
                    is_causal_flag,
                    has_mask_flag,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    mask_ptr,
                    attn_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_sdpa_bf16_run(
                    self.desc.batch_size,
                    self.desc.num_heads,
                    self.desc.query_len,
                    self.desc.key_len,
                    self.desc.d_k,
                    self.desc.d_v,
                    self.desc.scale,
                    is_causal_flag,
                    has_mask_flag,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    mask_ptr,
                    attn_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_sdpa_f64_run(
                    self.desc.batch_size,
                    self.desc.num_heads,
                    self.desc.query_len,
                    self.desc.key_len,
                    self.desc.d_k,
                    self.desc.d_v,
                    self.desc.scale,
                    is_causal_flag,
                    has_mask_flag,
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    mask_ptr,
                    attn_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::SdpaPlan::run reached an unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
