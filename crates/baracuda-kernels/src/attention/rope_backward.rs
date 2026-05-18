//! Rotary Position Embedding (RoPE) backward plan.
//!
//! Rotation matrices are orthogonal, so BW is rotation by `-θ`:
//!
//! ```text
//! dx[2i]   = dy[2i]   · cos(θ) + dy[2i+1] · sin(θ)
//! dx[2i+1] = dy[2i+1] · cos(θ) - dy[2i]   · sin(θ)
//! ```
//!
//! Shape, dtype scope, and `positions` semantics match [`super::rope`].

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, AttentionKind, BackendKind, Element, ElementKind, KernelSku, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::map_status;

/// Descriptor for a RoPE backward op.
#[derive(Copy, Clone, Debug)]
pub struct RopeBackwardDescriptor {
    /// Batch size (`B`).
    pub batch_size: i32,
    /// Number of attention heads (`H`).
    pub num_heads: i32,
    /// Sequence length (`S`).
    pub seq_len: i32,
    /// Head dimension (`D`). Must be even.
    pub head_dim: i32,
    /// Rotary base; must match the FW base.
    pub base: f32,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a RoPE backward launch.
pub struct RopeBackwardArgs<'a, T: Element> {
    /// Upstream gradient `dy` — shape `[B, H, S, D]`, contiguous.
    pub dy: TensorRef<'a, T, 4>,
    /// Optional `[S]` `i64` position indices — must match the FW call.
    pub positions: Option<TensorRef<'a, i64, 1>>,
    /// Output gradient `dx` — same shape as `dy`.
    pub dx: TensorMut<'a, T, 4>,
}

/// Rotary Position Embedding backward plan.
///
/// Rotation matrices are orthogonal — BW is rotation by `-θ`. No saved
/// FW state is needed; the kernel re-derives `θ` from `positions` (or
/// from `s` when `positions` is omitted) on the fly.
///
/// **When to use**: autograd partner for [`super::RopePlan`]. `base`,
/// `head_dim`, and the optional `positions` arg must match the FW
/// call.
///
/// **Dtypes / shape limits / workspace / precision**: identical to
/// [`super::RopePlan`].
pub struct RopeBackwardPlan<T: Element> {
    desc: RopeBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> RopeBackwardPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &RopeBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::RopeBackwardPlan: descriptor element != T",
            ));
        }
        if desc.batch_size < 0
            || desc.num_heads < 0
            || desc.seq_len < 0
            || desc.head_dim < 0
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RopeBackwardPlan: extents must be non-negative",
            ));
        }
        if desc.head_dim % 2 != 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RopeBackwardPlan: head_dim must be even",
            ));
        }
        if !desc.base.is_finite() || desc.base <= 0.0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RopeBackwardPlan: base must be finite and positive",
            ));
        }
        let dtype_in_scope = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !dtype_in_scope {
            return Err(Error::Unsupported(
                "baracuda-kernels::RopeBackwardPlan: wired today: `{f32, f16, bf16, f64}`",
            ));
        }

        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Attention,
            op: AttentionKind::Rope as u16,
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
    pub fn can_implement(&self, args: &RopeBackwardArgs<'_, T>) -> Result<()> {
        let want_shape = [
            self.desc.batch_size,
            self.desc.num_heads,
            self.desc.seq_len,
            self.desc.head_dim,
        ];
        if args.dy.shape != want_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RopeBackwardPlan: dy shape mismatch",
            ));
        }
        if args.dx.shape != want_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RopeBackwardPlan: dx shape mismatch",
            ));
        }
        if !args.dy.is_contiguous() || !args.dx.is_contiguous() {
            return Err(Error::Unsupported(
                "baracuda-kernels::RopeBackwardPlan: trailblazer requires contiguous dy / dx",
            ));
        }
        if let Some(ref p) = args.positions {
            if p.shape != [self.desc.seq_len] {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::RopeBackwardPlan: positions shape must be [seq_len]",
                ));
            }
        }
        let numel = args.dy.numel();
        if (args.dy.data.len() as i64) < numel || (args.dx.data.len() as i64) < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: args.dy.data.len().min(args.dx.data.len()),
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
        args: RopeBackwardArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.dy.numel();
        if numel == 0 {
            return Ok(());
        }
        let stream_ptr = stream.as_raw() as *mut c_void;
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let dx_ptr = args.dx.data.as_raw().0 as *mut c_void;
        let (pos_ptr, pos_default_flag) = match &args.positions {
            Some(p) => (p.data.as_raw().0 as *const c_void, 0i32),
            None => (core::ptr::null::<c_void>(), 1i32),
        };

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_rope_backward_f32_run(
                    self.desc.batch_size,
                    self.desc.num_heads,
                    self.desc.seq_len,
                    self.desc.head_dim,
                    self.desc.base,
                    pos_default_flag,
                    dy_ptr,
                    pos_ptr,
                    dx_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_rope_backward_f16_run(
                    self.desc.batch_size,
                    self.desc.num_heads,
                    self.desc.seq_len,
                    self.desc.head_dim,
                    self.desc.base,
                    pos_default_flag,
                    dy_ptr,
                    pos_ptr,
                    dx_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_rope_backward_bf16_run(
                    self.desc.batch_size,
                    self.desc.num_heads,
                    self.desc.seq_len,
                    self.desc.head_dim,
                    self.desc.base,
                    pos_default_flag,
                    dy_ptr,
                    pos_ptr,
                    dx_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_rope_backward_f64_run(
                    self.desc.batch_size,
                    self.desc.num_heads,
                    self.desc.seq_len,
                    self.desc.head_dim,
                    self.desc.base,
                    pos_default_flag,
                    dy_ptr,
                    pos_ptr,
                    dx_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::RopeBackwardPlan::run reached an unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
