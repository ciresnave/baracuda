//! Causal-Conv1d — Phase 50 (gated behind the `mamba` feature).
//!
//! Hand-port of Tri Dao's depthwise causal 1-D convolution primitive.
//! Used by Mamba / Mamba-2 between the input projection and the SSM
//! block; also useful on its own for any causal-conv sequence model
//! (TCN, WaveNet, simple n-gram LMs).
//!
//! ## Op semantics
//!
//! Depthwise (per-channel) causal cross-correlation:
//!
//! ```text
//!     y[b, c, t] = act( sum_{k=0..W-1} weight[c, k] * x_padded[b, c, t - (W-1-k)]
//!                       + bias[c] )
//! ```
//!
//! where `x_padded` is `x` with `(W-1)` zeros prepended along the
//! length axis. The "causal" property: `y[t]` depends only on
//! `x[≤ t]`.
//!
//! ## Shape contract
//!
//! | tensor   | shape           |
//! |----------|-----------------|
//! | `x`      | `[B, C, L]`     |
//! | `weight` | `[C, W]`        |
//! | `bias`   | `[C]` or None   |
//! | `y`      | `[B, C, L]`     |
//!
//! ## Trailblazer scope
//!
//! - **Widths**: `W ∈ {2, 3, 4}`. Wider widths require a different
//!   register-resident kernel and are deferred.
//! - **Activation**: SiLU or none. SiLU is Mamba's default.
//! - **Dtypes**: `f32`, `f16`, `bf16`, `f64`.
//! - **Layout**: NCL (channels-second, like cuDNN's NCHW). Contiguous.
//! - **No variable-length / no decode-step `causal_conv1d_update`**.
//!
//! ## Numerical guarantees
//!
//! - FW: deterministic, bit-stable on same hardware. Each `y[b, c, t]`
//!   cell is written by exactly one thread; no atomicAdd.
//! - BW: `dx` is deterministic. `dw` and `db` are accumulated across
//!   `(b, t)` via `atomicAdd` — order-dependent across batch / time
//!   samples. f16 / bf16 atomicAdd is provided natively on sm_80+.
//!
//! See `vendor/causal-conv1d/VENDOR.md` in `baracuda-kernels-sys` for
//! upstream attribution + scope notes.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, ConvKind, Element, ElementKind, KernelSku, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

/// Status-code → Result translation, local to this module so it stays
/// usable when the `cudnn` feature is off (the regular conv family
/// imports `map_status` from `conv/mod.rs` which is cudnn-gated).
fn map_status(code: i32) -> Result<()> {
    match code {
        0 => Ok(()),
        1 => Err(Error::MisalignedOperand),
        2 => Err(Error::InvalidProblem(
            "baracuda-kernels-sys reported invalid problem",
        )),
        3 => Err(Error::Unsupported(
            "baracuda-kernels-sys reported unsupported configuration",
        )),
        4 => Err(Error::WorkspaceTooSmall { needed: 0, got: 0 }),
        n => Err(Error::CutlassInternal(n)),
    }
}

/// Descriptor for a causal-conv1d FW op.
#[derive(Copy, Clone, Debug)]
pub struct CausalConv1dDescriptor {
    /// Batch size (`B`).
    pub batch_size: i32,
    /// Channel count (`C`).
    pub channels: i32,
    /// Sequence length (`L`).
    pub seq_len: i32,
    /// Filter width (`W`). Must be 2, 3, or 4.
    pub width: i32,
    /// Apply SiLU activation to the output.
    pub use_silu: bool,
    /// Element dtype — must match the plan's type parameter.
    pub element: ElementKind,
}

/// Args bundle for a causal-conv1d FW launch.
pub struct CausalConv1dArgs<'a, T: Element> {
    /// Input — shape `[B, C, L]`, contiguous.
    pub x: TensorRef<'a, T, 3>,
    /// Filter — shape `[C, W]`, contiguous.
    pub weight: TensorRef<'a, T, 2>,
    /// Optional per-channel bias — shape `[C]`, contiguous.
    pub bias: Option<TensorRef<'a, T, 1>>,
    /// Output — shape `[B, C, L]`, contiguous.
    pub y: TensorMut<'a, T, 3>,
}

/// Causal-conv1d forward plan.
pub struct CausalConv1dPlan<T: Element> {
    desc: CausalConv1dDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> CausalConv1dPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &CausalConv1dDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::CausalConv1dPlan: descriptor element != T",
            ));
        }
        if desc.batch_size < 0 || desc.channels < 0 || desc.seq_len < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CausalConv1dPlan: extents must be non-negative",
            ));
        }
        if desc.width < 2 || desc.width > 4 {
            return Err(Error::Unsupported(
                "baracuda-kernels::CausalConv1dPlan: width must be in {2, 3, 4}",
            ));
        }
        let dtype_in_scope = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !dtype_in_scope {
            return Err(Error::Unsupported(
                "baracuda-kernels::CausalConv1dPlan: wired today: `{f32, f16, bf16, f64}`",
            ));
        }
        let precision_guarantee = PrecisionGuarantee {
            math_precision: if matches!(T::KIND, ElementKind::F64) {
                MathPrecision::F64
            } else {
                MathPrecision::F32
            },
            accumulator: if matches!(T::KIND, ElementKind::F64) {
                ElementKind::F64
            } else {
                ElementKind::F32
            },
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Convolution,
            op: ConvKind::Conv1d as u16,
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
    pub fn can_implement(&self, args: &CausalConv1dArgs<'_, T>) -> Result<()> {
        let shape_x = [self.desc.batch_size, self.desc.channels, self.desc.seq_len];
        let shape_w = [self.desc.channels, self.desc.width];
        if args.x.shape != shape_x || args.y.shape != shape_x {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CausalConv1dPlan: x / y shape must be [B, C, L]",
            ));
        }
        if args.weight.shape != shape_w {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CausalConv1dPlan: weight shape must be [C, W]",
            ));
        }
        if let Some(b) = &args.bias {
            if b.shape != [self.desc.channels] {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::CausalConv1dPlan: bias shape must be [C]",
                ));
            }
            if !b.is_contiguous() {
                return Err(Error::Unsupported(
                    "baracuda-kernels::CausalConv1dPlan: bias must be contiguous",
                ));
            }
        }
        if !args.x.is_contiguous() || !args.weight.is_contiguous() || !args.y.is_contiguous() {
            return Err(Error::Unsupported(
                "baracuda-kernels::CausalConv1dPlan: trailblazer requires contiguous tensors",
            ));
        }
        Ok(())
    }

    /// Workspace size in bytes — zero.
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

    /// Launch the FW kernel on the supplied stream.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: CausalConv1dArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if self.desc.batch_size == 0 || self.desc.channels == 0 || self.desc.seq_len == 0 {
            return Ok(());
        }
        let stream_ptr = stream.as_raw() as *mut c_void;
        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let w_ptr = args.weight.data.as_raw().0 as *const c_void;
        let b_ptr = args.bias.as_ref()
            .map(|b| b.data.as_raw().0 as *const c_void)
            .unwrap_or(core::ptr::null());
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;
        let use_silu = if self.desc.use_silu { 1 } else { 0 };

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_causal_conv1d_f32_run(
                    self.desc.batch_size, self.desc.channels, self.desc.seq_len,
                    self.desc.width, use_silu,
                    x_ptr, w_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_causal_conv1d_f16_run(
                    self.desc.batch_size, self.desc.channels, self.desc.seq_len,
                    self.desc.width, use_silu,
                    x_ptr, w_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_causal_conv1d_bf16_run(
                    self.desc.batch_size, self.desc.channels, self.desc.seq_len,
                    self.desc.width, use_silu,
                    x_ptr, w_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_causal_conv1d_f64_run(
                    self.desc.batch_size, self.desc.channels, self.desc.seq_len,
                    self.desc.width, use_silu,
                    x_ptr, w_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => return Err(Error::Unsupported(
                "baracuda-kernels::CausalConv1dPlan: dtype not wired",
            )),
        };
        map_status(status)
    }
}

// ============================================================================
// BACKWARD
// ============================================================================

/// Descriptor for a causal-conv1d BW op.
#[derive(Copy, Clone, Debug)]
pub struct CausalConv1dBackwardDescriptor {
    /// Batch size (`B`).
    pub batch_size: i32,
    /// Channel count (`C`).
    pub channels: i32,
    /// Sequence length (`L`).
    pub seq_len: i32,
    /// Filter width (`W`). Must be 2, 3, or 4.
    pub width: i32,
    /// Whether the FW used SiLU activation (needed to compute the
    /// `act'(pre)` factor in the BW).
    pub use_silu: bool,
    /// Element dtype.
    pub element: ElementKind,
}

/// Args bundle for a causal-conv1d BW launch.
///
/// `dw` and `db` are atomic-accumulated; caller must zero them
/// before launch.
pub struct CausalConv1dBackwardArgs<'a, T: Element> {
    /// Input (saved from FW) — shape `[B, C, L]`.
    pub x: TensorRef<'a, T, 3>,
    /// Filter (saved from FW) — shape `[C, W]`.
    pub weight: TensorRef<'a, T, 2>,
    /// Optional bias (saved from FW) — shape `[C]`.
    pub bias: Option<TensorRef<'a, T, 1>>,
    /// Output gradient — shape `[B, C, L]`.
    pub dy: TensorRef<'a, T, 3>,
    /// `dx` — shape `[B, C, L]`.
    pub dx: TensorMut<'a, T, 3>,
    /// `dw` — shape `[C, W]`. **Must be zero-initialized.**
    pub dw: TensorMut<'a, T, 2>,
    /// `db` — shape `[C]`, optional (only when `bias` is present).
    /// **Must be zero-initialized.**
    pub db: Option<TensorMut<'a, T, 1>>,
}

/// Causal-conv1d backward plan.
pub struct CausalConv1dBackwardPlan<T: Element> {
    desc: CausalConv1dBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> CausalConv1dBackwardPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &CausalConv1dBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::CausalConv1dBackwardPlan: descriptor element != T",
            ));
        }
        if desc.width < 2 || desc.width > 4 {
            return Err(Error::Unsupported(
                "baracuda-kernels::CausalConv1dBackwardPlan: width must be in {2, 3, 4}",
            ));
        }
        let dtype_in_scope = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !dtype_in_scope {
            return Err(Error::Unsupported(
                "baracuda-kernels::CausalConv1dBackwardPlan: wired today: `{f32, f16, bf16, f64}`",
            ));
        }
        let precision_guarantee = PrecisionGuarantee {
            math_precision: if matches!(T::KIND, ElementKind::F64) {
                MathPrecision::F64
            } else {
                MathPrecision::F32
            },
            accumulator: if matches!(T::KIND, ElementKind::F64) {
                ElementKind::F64
            } else {
                ElementKind::F32
            },
            // dw/db atomicAdd → not deterministic / not bit-stable.
            bit_stable_on_same_hardware: false,
            deterministic: false,
        };
        let sku = KernelSku {
            category: OpCategory::Convolution,
            op: ConvKind::Conv1d as u16,
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

    /// Workspace size in bytes — zero.
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

    /// Launch the BW kernel on the supplied stream.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: CausalConv1dBackwardArgs<'_, T>,
    ) -> Result<()> {
        if self.desc.batch_size == 0 || self.desc.channels == 0 || self.desc.seq_len == 0 {
            return Ok(());
        }
        let stream_ptr = stream.as_raw() as *mut c_void;
        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let w_ptr = args.weight.data.as_raw().0 as *const c_void;
        let b_ptr = args.bias.as_ref()
            .map(|b| b.data.as_raw().0 as *const c_void)
            .unwrap_or(core::ptr::null());
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let dx_ptr = args.dx.data.as_raw().0 as *mut c_void;
        let dw_ptr = args.dw.data.as_raw().0 as *mut c_void;
        let db_ptr = args.db.as_ref()
            .map(|b| b.data.as_raw().0 as *mut c_void)
            .unwrap_or(core::ptr::null_mut());
        let use_silu = if self.desc.use_silu { 1 } else { 0 };

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_causal_conv1d_f32_backward_run(
                    self.desc.batch_size, self.desc.channels, self.desc.seq_len,
                    self.desc.width, use_silu,
                    x_ptr, w_ptr, b_ptr, dy_ptr,
                    dx_ptr, dw_ptr, db_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_causal_conv1d_f16_backward_run(
                    self.desc.batch_size, self.desc.channels, self.desc.seq_len,
                    self.desc.width, use_silu,
                    x_ptr, w_ptr, b_ptr, dy_ptr,
                    dx_ptr, dw_ptr, db_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_causal_conv1d_bf16_backward_run(
                    self.desc.batch_size, self.desc.channels, self.desc.seq_len,
                    self.desc.width, use_silu,
                    x_ptr, w_ptr, b_ptr, dy_ptr,
                    dx_ptr, dw_ptr, db_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_causal_conv1d_f64_backward_run(
                    self.desc.batch_size, self.desc.channels, self.desc.seq_len,
                    self.desc.width, use_silu,
                    x_ptr, w_ptr, b_ptr, dy_ptr,
                    dx_ptr, dw_ptr, db_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => return Err(Error::Unsupported(
                "baracuda-kernels::CausalConv1dBackwardPlan: dtype not wired",
            )),
        };
        map_status(status)
    }
}
