//! Attention with Linear Biases (ALiBi) forward plan.
//!
//! Given attention scores `A: [B, H, Q, K]` and per-head slopes
//! `slope: [H]`:
//!
//! ```text
//! y[b, h, i, j] = A[b, h, i, j] + slope[h] · (j - i)
//! ```
//!
//! Used by MPT / BLOOM in lieu of positional embeddings. The trailblazer
//! implements the non-causal elementwise variant — pass any combination
//! of slopes you like.
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

/// Descriptor for an ALiBi forward op.
#[derive(Copy, Clone, Debug)]
pub struct AlibiDescriptor {
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

/// Args bundle for an ALiBi forward launch.
pub struct AlibiArgs<'a, T: Element> {
    /// Attention scores `A` — shape `[B, H, Q, K]`, contiguous.
    pub scores: TensorRef<'a, T, 4>,
    /// Per-head slopes — shape `[H]`.
    pub slopes: TensorRef<'a, T, 1>,
    /// Output biased scores — same shape as `scores`.
    pub out: TensorMut<'a, T, 4>,
}

/// ALiBi forward plan.
pub struct AlibiPlan<T: Element> {
    desc: AlibiDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> AlibiPlan<T> {
    /// Pick a kernel.
    pub fn select(_stream: &Stream, desc: &AlibiDescriptor, _pref: PlanPreference) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::AlibiPlan: descriptor element != T",
            ));
        }
        if desc.batch_size < 0 || desc.num_heads < 0 || desc.query_len < 0 || desc.key_len < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::AlibiPlan: extents must be non-negative",
            ));
        }
        let dtype_in_scope = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !dtype_in_scope {
            return Err(Error::Unsupported(
                "baracuda-kernels::AlibiPlan: wired today: `{f32, f16, bf16, f64}`",
            ));
        }

        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            // Single add per cell — fully deterministic.
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
    pub fn can_implement(&self, args: &AlibiArgs<'_, T>) -> Result<()> {
        let want_shape = [
            self.desc.batch_size,
            self.desc.num_heads,
            self.desc.query_len,
            self.desc.key_len,
        ];
        if args.scores.shape != want_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::AlibiPlan: scores shape mismatch",
            ));
        }
        if args.out.shape != want_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::AlibiPlan: out shape mismatch",
            ));
        }
        if args.slopes.shape != [self.desc.num_heads] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::AlibiPlan: slopes shape must be [num_heads]",
            ));
        }
        if !args.scores.is_contiguous() || !args.out.is_contiguous() {
            return Err(Error::Unsupported(
                "baracuda-kernels::AlibiPlan: trailblazer requires contiguous scores / out",
            ));
        }
        let numel = args.scores.numel();
        if (args.scores.data.len() as i64) < numel || (args.out.data.len() as i64) < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: args.scores.data.len().min(args.out.data.len()),
            });
        }
        if (args.slopes.data.len() as i64) < self.desc.num_heads as i64 {
            return Err(Error::BufferTooSmall {
                needed: self.desc.num_heads as usize,
                got: args.slopes.data.len(),
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
        args: AlibiArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.scores.numel();
        if numel == 0 {
            return Ok(());
        }
        let stream_ptr = stream.as_raw() as *mut c_void;
        let scores_ptr = args.scores.data.as_raw().0 as *const c_void;
        let slopes_ptr = args.slopes.data.as_raw().0 as *const c_void;
        let out_ptr = args.out.data.as_raw().0 as *mut c_void;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_alibi_f32_run(
                    self.desc.batch_size,
                    self.desc.num_heads,
                    self.desc.query_len,
                    self.desc.key_len,
                    scores_ptr,
                    slopes_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_alibi_f16_run(
                    self.desc.batch_size,
                    self.desc.num_heads,
                    self.desc.query_len,
                    self.desc.key_len,
                    scores_ptr,
                    slopes_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_alibi_bf16_run(
                    self.desc.batch_size,
                    self.desc.num_heads,
                    self.desc.query_len,
                    self.desc.key_len,
                    scores_ptr,
                    slopes_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_alibi_f64_run(
                    self.desc.batch_size,
                    self.desc.num_heads,
                    self.desc.query_len,
                    self.desc.key_len,
                    scores_ptr,
                    slopes_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::AlibiPlan::run reached an unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
