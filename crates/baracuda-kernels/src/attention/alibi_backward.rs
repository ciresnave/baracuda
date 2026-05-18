//! Attention with Linear Biases (ALiBi) backward plan.
//!
//! For the FW `Y[b, h, i, j] = A[b, h, i, j] + slope[h] · (j - i)`:
//!
//! ```text
//! dA[b, h, i, j] = dY[b, h, i, j]                (pass-through copy)
//! dslope[h]      = Σ_{b, i, j} dY[b, h, i, j] · (j - i)
//! ```
//!
//! The kernel fires two device kernels when both grads are requested:
//! a pass-through copy for `dA` and a one-block-per-head warp-shuffle
//! reduction for `dslope` (fully deterministic, no atomicAdd). Either
//! output may be `None` to skip its kernel.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, AttentionKind, BackendKind, Element, ElementKind, KernelSku, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::map_status;

/// Descriptor for an ALiBi backward op.
#[derive(Copy, Clone, Debug)]
pub struct AlibiBackwardDescriptor {
    /// Batch size (`B`).
    pub batch_size: i32,
    /// Number of attention heads (`H`).
    pub num_heads: i32,
    /// Query sequence length (`Q`).
    pub query_len: i32,
    /// Key sequence length (`K`).
    pub key_len: i32,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for an ALiBi backward launch.
///
/// Either `dscores` or `dslopes` may be `None` to skip computing that
/// gradient (mirrors the optional-output convention of other BW plans
/// in the family). Both `None` is rejected.
pub struct AlibiBackwardArgs<'a, T: Element> {
    /// Upstream gradient `dY` — shape `[B, H, Q, K]`, contiguous.
    pub dy: TensorRef<'a, T, 4>,
    /// Output gradient for the scores tensor (pass-through copy of dY).
    /// `None` to skip.
    pub dscores: Option<TensorMut<'a, T, 4>>,
    /// Output gradient for the slopes vector (reduction of
    /// `dy · (j - i)` over `(b, i, j)` per head). `None` to skip.
    pub dslopes: Option<TensorMut<'a, T, 1>>,
}

/// Attention with Linear Biases (ALiBi) backward plan.
///
/// `dA` is a pass-through copy of `dY`; `dslope[h]` is the per-head
/// reduction `Σ_{b, i, j} dY[b, h, i, j] · (j - i)`.
///
/// **When to use**: autograd partner for [`super::AlibiPlan`]. Either
/// output is optional — pass `None` to skip that kernel.
///
/// **Dtypes**: `f32`, `f64`, `f16`, `bf16` — must match the FW plan.
///
/// **Workspace**: zero.
///
/// **Precision guarantee**: deterministic; bit-stable on the same
/// hardware. The reduction runs as one block per head with a
/// warp-shuffle tree reduction — no atomicAdd.
pub struct AlibiBackwardPlan<T: Element> {
    desc: AlibiBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> AlibiBackwardPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &AlibiBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::AlibiBackwardPlan: descriptor element != T",
            ));
        }
        if desc.batch_size < 0 || desc.num_heads < 0 || desc.query_len < 0 || desc.key_len < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::AlibiBackwardPlan: extents must be non-negative",
            ));
        }
        let dtype_in_scope = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !dtype_in_scope {
            return Err(Error::Unsupported(
                "baracuda-kernels::AlibiBackwardPlan: wired today: `{f32, f16, bf16, f64}`",
            ));
        }

        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            // Per-head tree reduction inside a single block is deterministic.
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Attention,
            op: AttentionKind::Alibi as u16,
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
    pub fn can_implement(&self, args: &AlibiBackwardArgs<'_, T>) -> Result<()> {
        let want_shape = [
            self.desc.batch_size,
            self.desc.num_heads,
            self.desc.query_len,
            self.desc.key_len,
        ];
        if args.dy.shape != want_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::AlibiBackwardPlan: dy shape mismatch",
            ));
        }
        if !args.dy.is_contiguous() {
            return Err(Error::Unsupported(
                "baracuda-kernels::AlibiBackwardPlan: trailblazer requires contiguous dy",
            ));
        }
        if args.dscores.is_none() && args.dslopes.is_none() {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::AlibiBackwardPlan: at least one of dscores / dslopes must be supplied",
            ));
        }
        if let Some(ref d) = args.dscores {
            if d.shape != want_shape {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::AlibiBackwardPlan: dscores shape mismatch",
                ));
            }
            if !d.is_contiguous() {
                return Err(Error::Unsupported(
                    "baracuda-kernels::AlibiBackwardPlan: trailblazer requires contiguous dscores",
                ));
            }
            let numel = args.dy.numel();
            if (d.data.len() as i64) < numel {
                return Err(Error::BufferTooSmall {
                    needed: numel as usize,
                    got: d.data.len(),
                });
            }
        }
        if let Some(ref s) = args.dslopes {
            if s.shape != [self.desc.num_heads] {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::AlibiBackwardPlan: dslopes shape must be [num_heads]",
                ));
            }
            if (s.data.len() as i64) < self.desc.num_heads as i64 {
                return Err(Error::BufferTooSmall {
                    needed: self.desc.num_heads as usize,
                    got: s.data.len(),
                });
            }
        }
        let numel = args.dy.numel();
        if (args.dy.data.len() as i64) < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: args.dy.data.len(),
            });
        }
        Ok(())
    }

    /// Workspace size in bytes.
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

    /// Launch.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: AlibiBackwardArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.dy.numel();
        if numel == 0 {
            return Ok(());
        }
        let stream_ptr = stream.as_raw() as *mut c_void;
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let da_ptr = match &args.dscores {
            Some(d) => d.data.as_raw().0 as *mut c_void,
            None => core::ptr::null_mut(),
        };
        let dslope_ptr = match &args.dslopes {
            Some(s) => s.data.as_raw().0 as *mut c_void,
            None => core::ptr::null_mut(),
        };

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_alibi_backward_f32_run(
                    self.desc.batch_size,
                    self.desc.num_heads,
                    self.desc.query_len,
                    self.desc.key_len,
                    dy_ptr,
                    da_ptr,
                    dslope_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_alibi_backward_f16_run(
                    self.desc.batch_size,
                    self.desc.num_heads,
                    self.desc.query_len,
                    self.desc.key_len,
                    dy_ptr,
                    da_ptr,
                    dslope_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_alibi_backward_bf16_run(
                    self.desc.batch_size,
                    self.desc.num_heads,
                    self.desc.query_len,
                    self.desc.key_len,
                    dy_ptr,
                    da_ptr,
                    dslope_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_alibi_backward_f64_run(
                    self.desc.batch_size,
                    self.desc.num_heads,
                    self.desc.query_len,
                    self.desc.key_len,
                    dy_ptr,
                    da_ptr,
                    dslope_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::AlibiBackwardPlan::run reached an unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
