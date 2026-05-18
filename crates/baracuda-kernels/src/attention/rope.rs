//! Rotary Position Embedding (RoPE) forward plan.
//!
//! For a Q/K tensor `x` of shape `[batch, num_heads, seq_len, head_dim]`,
//! RoPE rotates consecutive feature pairs `(2i, 2i+1)` by position-
//! dependent angles `θ_i = pos · base^(-2i/head_dim)`:
//!
//! ```text
//! y[2i]   = x[2i]   · cos(θ_i) - x[2i+1] · sin(θ_i)
//! y[2i+1] = x[2i+1] · cos(θ_i) + x[2i]   · sin(θ_i)
//! ```
//!
//! `head_dim` must be even. `base` defaults to `10000.0`. `positions` is
//! an optional `[seq_len]` `i64` tensor; when absent, the kernel uses
//! the sequence index as the position (canonical "no prefix-cache" case).
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

/// Descriptor for a RoPE forward op.
///
/// All sizes are `i32` to match the kernel ABI (which uses int32 for
/// per-axis extents to keep param-block size bounded).
#[derive(Copy, Clone, Debug)]
pub struct RopeDescriptor {
    /// Batch size (`B`).
    pub batch_size: i32,
    /// Number of attention heads (`H`).
    pub num_heads: i32,
    /// Sequence length (`S`).
    pub seq_len: i32,
    /// Head dimension (`D`). **Must be even** — RoPE rotates feature
    /// pairs.
    pub head_dim: i32,
    /// Rotary base; default `10000.0`. Pass a different value for
    /// extended-context recipes (e.g. NTK-aware scaling uses
    /// `base * scaling_factor`).
    pub base: f32,
    /// Element type — must match the plan's type parameter.
    pub element: ElementKind,
}

/// Args bundle for a RoPE forward launch.
pub struct RopeArgs<'a, T: Element> {
    /// Input Q or K tensor — shape `[B, H, S, D]`, contiguous row-major.
    pub x: TensorRef<'a, T, 4>,
    /// Optional explicit `[S]` `i64` position indices. `None` means the
    /// kernel uses `positions[s] = s` (canonical case).
    pub positions: Option<TensorRef<'a, i64, 1>>,
    /// Output tensor — same shape as `x`.
    pub y: TensorMut<'a, T, 4>,
}

/// Rotary Position Embedding forward plan.
///
/// Rotates consecutive feature pairs `(2i, 2i+1)` of a `[B, H, S, D]`
/// Q or K tensor by per-position angles `θ_i = pos · base^(-2i/D)`.
///
/// **When to use**: dominant positional encoding in modern LLMs (Llama,
/// Mistral, Gemma, Qwen, Phi). Apply once per Q and once per K before
/// the attention matmul. Pair with [`super::RopeBackwardPlan`] for
/// autograd.
///
/// **Dtypes**: `f32`, `f64`, `f16`, `bf16`. `f16` / `bf16` detour
/// through `f32` for the trig + multiply; `f64` uses native double.
///
/// **Shape limits**: rank-4, contiguous, row-major. `head_dim` must
/// be even (RoPE rotates pairs). Optional `positions: i64[S]` override
/// — when absent the kernel uses `positions[s] = s`.
///
/// **Workspace**: zero.
///
/// **Precision guarantee**: deterministic; bit-stable on the same
/// hardware. Per-cell write, no atomics.
pub struct RopePlan<T: Element> {
    desc: RopeDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> RopePlan<T> {
    /// Pick a kernel.
    pub fn select(_stream: &Stream, desc: &RopeDescriptor, _pref: PlanPreference) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::RopePlan: descriptor element != T",
            ));
        }
        if desc.batch_size < 0
            || desc.num_heads < 0
            || desc.seq_len < 0
            || desc.head_dim < 0
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RopePlan: extents must be non-negative",
            ));
        }
        if desc.head_dim % 2 != 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RopePlan: head_dim must be even (RoPE rotates pairs)",
            ));
        }
        if !desc.base.is_finite() || desc.base <= 0.0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RopePlan: base must be finite and positive",
            ));
        }
        let dtype_in_scope = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !dtype_in_scope {
            return Err(Error::Unsupported(
                "baracuda-kernels::RopePlan: wired today: `{f32, f16, bf16, f64}`",
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
    pub fn can_implement(&self, args: &RopeArgs<'_, T>) -> Result<()> {
        let want_shape = [
            self.desc.batch_size,
            self.desc.num_heads,
            self.desc.seq_len,
            self.desc.head_dim,
        ];
        if args.x.shape != want_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RopePlan: x shape mismatch with descriptor",
            ));
        }
        if args.y.shape != want_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RopePlan: y shape mismatch with descriptor",
            ));
        }
        if !args.x.is_contiguous() || !args.y.is_contiguous() {
            return Err(Error::Unsupported(
                "baracuda-kernels::RopePlan: trailblazer requires contiguous x / y",
            ));
        }
        if let Some(ref p) = args.positions {
            if p.shape != [self.desc.seq_len] {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::RopePlan: positions shape must be [seq_len]",
                ));
            }
            if (p.data.len() as i64) < self.desc.seq_len as i64 {
                return Err(Error::BufferTooSmall {
                    needed: self.desc.seq_len as usize,
                    got: p.data.len(),
                });
            }
        }
        let numel = args.x.numel();
        if (args.x.data.len() as i64) < numel || (args.y.data.len() as i64) < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: args.x.data.len().min(args.y.data.len()),
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
        args: RopeArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.x.numel();
        if numel == 0 {
            return Ok(());
        }
        let stream_ptr = stream.as_raw() as *mut c_void;
        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;
        let (pos_ptr, pos_default_flag) = match &args.positions {
            Some(p) => (p.data.as_raw().0 as *const c_void, 0i32),
            None => (core::ptr::null::<c_void>(), 1i32),
        };

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_rope_f32_run(
                    self.desc.batch_size,
                    self.desc.num_heads,
                    self.desc.seq_len,
                    self.desc.head_dim,
                    self.desc.base,
                    pos_default_flag,
                    x_ptr,
                    pos_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_rope_f16_run(
                    self.desc.batch_size,
                    self.desc.num_heads,
                    self.desc.seq_len,
                    self.desc.head_dim,
                    self.desc.base,
                    pos_default_flag,
                    x_ptr,
                    pos_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_rope_bf16_run(
                    self.desc.batch_size,
                    self.desc.num_heads,
                    self.desc.seq_len,
                    self.desc.head_dim,
                    self.desc.base,
                    pos_default_flag,
                    x_ptr,
                    pos_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_rope_f64_run(
                    self.desc.batch_size,
                    self.desc.num_heads,
                    self.desc.seq_len,
                    self.desc.head_dim,
                    self.desc.base,
                    pos_default_flag,
                    x_ptr,
                    pos_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::RopePlan::run reached an unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
