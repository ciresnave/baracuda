//! Backward plan for the unary elementwise family.
//!
//! Sibling of [`crate::UnaryPlan`] for gradient computation:
//! `dx = backward(dy, [one saved tensor per op])`.
//!
//! Unary backward formulas group into two save-shapes:
//! - **Saved-x**: gradient references the forward input. Sin BW:
//!   `dx = dy * cos(x)`. Cos BW: `dx = -dy * sin(x)`. Log BW: `dx = dy / x`.
//! - **Saved-y**: gradient references the forward output. Exp BW:
//!   `dx = dy * y`. Sigmoid BW: `dx = dy * y * (1 - y)`. Tanh BW:
//!   `dx = dy * (1 - y²)`. Sqrt BW: `dx = dy / (2y)`.
//!
//! The kernel ABI is uniform `(dy, saved, dx)` — the save-shape choice
//! lives entirely in the dispatcher (`select` rejects ops not yet wired;
//! `can_implement` checks the right `Option` field is present; `run`
//! passes the corresponding pointer).
//!
//! Trailblazer scope: `Sin BW × f32` (saved-x) and `Exp BW × f32`
//! (saved-y). Other unary BW ops + dtypes land in fanout sessions
//! following this same template.
//!
//! Trailblazer constraints: contig-only (no strided / broadcast support);
//! `dy.shape == saved.shape == dx.shape == desc.shape`.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, UnaryKind, Workspace,
};

/// Descriptor for a unary backward op.
#[derive(Copy, Clone, Debug)]
pub struct UnaryBackwardDescriptor<const N: usize> {
    /// Which forward unary op this is the backward of.
    pub kind: UnaryKind,
    /// Tensor shape (shared by dy / saved / dx).
    pub shape: [i32; N],
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a unary backward launch.
///
/// Exactly one of `x` / `y` must be supplied, matching the op's
/// requirement:
/// - Saved-x ops (Sin, Cos, Log, ...): pass `x`; leave `y = None`.
/// - Saved-y ops (Exp, Sigmoid, Tanh, Sqrt, ...): pass `y`; leave `x = None`.
///
/// The dispatcher validates the match against `desc.kind`.
pub struct UnaryBackwardArgs<'a, T: Element, const N: usize> {
    /// Upstream gradient (input to backward).
    pub dy: TensorRef<'a, T, N>,
    /// Saved forward input. Required by saved-x ops; ignored otherwise.
    pub x: Option<TensorRef<'a, T, N>>,
    /// Saved forward output. Required by saved-y ops; ignored otherwise.
    pub y: Option<TensorRef<'a, T, N>>,
    /// Gradient w.r.t. the input.
    pub dx: TensorMut<'a, T, N>,
}

/// Which forward tensor the op's BW formula references.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum SaveShape {
    /// Backward references the forward input `x`.
    X,
    /// Backward references the forward output `y`.
    Y,
}

#[inline]
fn save_shape_for(kind: UnaryKind) -> Option<SaveShape> {
    // Only ops with wired backward kernels return Some. Unwired ops
    // (Neg, Abs, ...) return None so `select` can reject cleanly.
    match kind {
        // Saved-x: gradient references the forward input.
        UnaryKind::Sin
        | UnaryKind::Log
        | UnaryKind::Log1p
        | UnaryKind::Log2
        | UnaryKind::Log10
        | UnaryKind::Atan
        | UnaryKind::Cos
        | UnaryKind::Tan
        | UnaryKind::Sinh
        | UnaryKind::Cosh
        | UnaryKind::Asin
        | UnaryKind::Acos
        | UnaryKind::Asinh
        | UnaryKind::Acosh
        | UnaryKind::Atanh
        | UnaryKind::Square
        | UnaryKind::Cube
        | UnaryKind::Tanhshrink
        | UnaryKind::Logit
        | UnaryKind::Reciprocal
        | UnaryKind::Erf
        | UnaryKind::Erfc
        | UnaryKind::Relu
        | UnaryKind::Hardtanh
        | UnaryKind::Relu6
        | UnaryKind::Hardsigmoid
        | UnaryKind::Hardswish
        | UnaryKind::Softplus
        | UnaryKind::Silu
        | UnaryKind::Mish
        | UnaryKind::Gelu
        | UnaryKind::GeluTanh
        | UnaryKind::Selu
        | UnaryKind::LeakyRelu
        | UnaryKind::Elu
        | UnaryKind::Hardshrink
        | UnaryKind::Softshrink => Some(SaveShape::X),
        // Saved-y: gradient references the forward output.
        UnaryKind::Exp
        | UnaryKind::Expm1
        | UnaryKind::Exp2
        | UnaryKind::Tanh
        | UnaryKind::Sigmoid
        | UnaryKind::Sqrt
        | UnaryKind::Rsqrt => Some(SaveShape::Y),
        _ => None,
    }
}

/// Unary backward plan.
pub struct UnaryBackwardPlan<T: Element, const N: usize> {
    desc: UnaryBackwardDescriptor<N>,
    sku: KernelSku,
    save_shape: SaveShape,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> UnaryBackwardPlan<T, N> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &UnaryBackwardDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::UnaryBackwardPlan: descriptor element != T",
            ));
        }
        for &d in desc.shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::UnaryBackwardPlan: shape dims must be non-negative",
                ));
            }
        }

        // Wired today across all four FP dtypes:
        //   - Saved-x: Sin, Cos, Tan, Sinh, Cosh, Asin, Acos, Asinh,
        //              Acosh, Atan, Atanh, Log, Log1p, Log2, Log10,
        //              Square, Cube, Tanhshrink, Logit, Reciprocal,
        //              Erf, Erfc
        //   - Saved-y: Exp, Expm1, Exp2, Tanh, Sigmoid, Sqrt, Rsqrt
        // Remaining unary BW ops (Sign, Abs, Neg, ...) land in later
        // waves.
        let is_saved_x_op = matches!(
            desc.kind,
            UnaryKind::Sin
                | UnaryKind::Log
                | UnaryKind::Log1p
                | UnaryKind::Log2
                | UnaryKind::Log10
                | UnaryKind::Atan
                | UnaryKind::Cos
                | UnaryKind::Tan
                | UnaryKind::Sinh
                | UnaryKind::Cosh
                | UnaryKind::Asin
                | UnaryKind::Acos
                | UnaryKind::Asinh
                | UnaryKind::Acosh
                | UnaryKind::Atanh
                | UnaryKind::Square
                | UnaryKind::Cube
                | UnaryKind::Tanhshrink
                | UnaryKind::Logit
                | UnaryKind::Reciprocal
                | UnaryKind::Erf
                | UnaryKind::Erfc
                | UnaryKind::Relu
                | UnaryKind::Hardtanh
                | UnaryKind::Relu6
                | UnaryKind::Hardsigmoid
                | UnaryKind::Hardswish
                | UnaryKind::Softplus
                | UnaryKind::Silu
                | UnaryKind::Mish
                | UnaryKind::Gelu
                | UnaryKind::GeluTanh
                | UnaryKind::Selu
                | UnaryKind::LeakyRelu
                | UnaryKind::Elu
                | UnaryKind::Hardshrink
                | UnaryKind::Softshrink
        );
        let is_saved_y_op = matches!(
            desc.kind,
            UnaryKind::Exp
                | UnaryKind::Expm1
                | UnaryKind::Exp2
                | UnaryKind::Tanh
                | UnaryKind::Sigmoid
                | UnaryKind::Sqrt
                | UnaryKind::Rsqrt
        );
        let dtype_in_fp_family = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        let supported = (is_saved_x_op || is_saved_y_op) && dtype_in_fp_family;
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::UnaryBackwardPlan: wired today: \
                 saved-x `{Sin, Cos, Tan, Sinh, Cosh, Asin, Acos, Asinh, Acosh, \
                 Atan, Atanh, Log, Log1p, Log2, Log10, Square, Cube, \
                 Tanhshrink, Logit, Reciprocal, Erf, Erfc, Relu, Hardtanh, \
                 Relu6, Hardsigmoid, Hardswish, Softplus, Silu, Mish, Gelu, \
                 GeluTanh, Selu, LeakyRelu, Elu, Hardshrink, Softshrink}` and saved-y \
                 `{Exp, Expm1, Exp2, Tanh, Sigmoid, Sqrt, Rsqrt}` × \
                 `{f32, f16, bf16, f64}`; other (kind, dtype) pairs land in later fanout",
            ));
        }
        let save_shape = save_shape_for(desc.kind).expect("supported op must have a save shape");

        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::UnaryElementwise,
            op: desc.kind as u16,
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
            save_shape,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &UnaryBackwardArgs<'_, T, N>) -> Result<()> {
        if args.dy.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::UnaryBackwardPlan: dy shape mismatch",
            ));
        }
        if args.dx.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::UnaryBackwardPlan: dx shape mismatch",
            ));
        }
        if !args.dy.is_contiguous() || !args.dx.is_contiguous() {
            return Err(Error::Unsupported(
                "baracuda-kernels::UnaryBackwardPlan: trailblazer requires contiguous \
                 dy / dx; strided fanout lands later",
            ));
        }
        let saved = match self.save_shape {
            SaveShape::X => args.x.as_ref().ok_or(Error::InvalidProblem(
                "baracuda-kernels::UnaryBackwardPlan: this op needs saved input `x` \
                 (a saved-x backward); pass it in `args.x`",
            ))?,
            SaveShape::Y => args.y.as_ref().ok_or(Error::InvalidProblem(
                "baracuda-kernels::UnaryBackwardPlan: this op needs saved output `y` \
                 (a saved-y backward); pass it in `args.y`",
            ))?,
        };
        if saved.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::UnaryBackwardPlan: saved tensor shape mismatch",
            ));
        }
        if !saved.is_contiguous() {
            return Err(Error::Unsupported(
                "baracuda-kernels::UnaryBackwardPlan: saved tensor must be contiguous \
                 (strided fanout lands later)",
            ));
        }
        let numel = args.dy.numel();
        let dy_len = args.dy.data.len() as i64;
        let dx_len = args.dx.data.len() as i64;
        let saved_len = saved.data.len() as i64;
        if dy_len < numel || dx_len < numel || saved_len < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: dy_len.min(dx_len).min(saved_len) as usize,
            });
        }
        Ok(())
    }

    /// Workspace size in bytes.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Kernel SKU identity.
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
        args: UnaryBackwardArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.dy.numel();
        if numel == 0 {
            return Ok(());
        }
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let dx_ptr = args.dx.data.as_raw().0 as *mut c_void;
        let saved_ptr = match self.save_shape {
            SaveShape::X => args
                .x
                .as_ref()
                .expect("can_implement guarantees x is present for saved-x ops")
                .data
                .as_raw()
                .0 as *const c_void,
            SaveShape::Y => args
                .y
                .as_ref()
                .expect("can_implement guarantees y is present for saved-y ops")
                .data
                .as_raw()
                .0 as *const c_void,
        };
        let stream_ptr = stream.as_raw() as *mut c_void;

        let status = match (self.desc.kind, T::KIND) {
            // -------- Sin (saved-x, transcendental) --------
            (UnaryKind::Sin, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sin_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sin, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sin_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sin, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sin_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sin, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sin_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Log (saved-x, no transcendental) --------
            (UnaryKind::Log, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Log, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Log, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Log, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Log1p (saved-x) --------
            (UnaryKind::Log1p, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log1p_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Log1p, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log1p_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Log1p, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log1p_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Log1p, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log1p_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Log2 (saved-x) --------
            (UnaryKind::Log2, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log2_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Log2, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log2_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Log2, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log2_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Log2, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log2_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Log10 (saved-x) --------
            (UnaryKind::Log10, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log10_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Log10, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log10_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Log10, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log10_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Log10, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log10_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Atan (saved-x, no transcendental) --------
            (UnaryKind::Atan, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_atan_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Atan, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_atan_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Atan, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_atan_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Atan, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_atan_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Cos (saved-x, transcendental) --------
            (UnaryKind::Cos, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cos_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Cos, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cos_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Cos, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cos_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Cos, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cos_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Tan (saved-x, transcendental) --------
            (UnaryKind::Tan, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_tan_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Tan, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_tan_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Tan, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_tan_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Tan, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_tan_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Sinh (saved-x, transcendental) --------
            (UnaryKind::Sinh, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sinh_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sinh, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sinh_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sinh, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sinh_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sinh, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sinh_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Cosh (saved-x, transcendental) --------
            (UnaryKind::Cosh, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cosh_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Cosh, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cosh_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Cosh, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cosh_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Cosh, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cosh_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Asin (saved-x, sqrt) --------
            (UnaryKind::Asin, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_asin_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Asin, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_asin_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Asin, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_asin_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Asin, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_asin_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Acos (saved-x, sqrt) --------
            (UnaryKind::Acos, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_acos_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Acos, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_acos_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Acos, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_acos_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Acos, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_acos_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Asinh (saved-x, sqrt) --------
            (UnaryKind::Asinh, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_asinh_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Asinh, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_asinh_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Asinh, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_asinh_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Asinh, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_asinh_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Acosh (saved-x, sqrt) --------
            (UnaryKind::Acosh, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_acosh_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Acosh, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_acosh_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Acosh, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_acosh_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Acosh, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_acosh_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Atanh (saved-x, no transcendental) --------
            (UnaryKind::Atanh, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_atanh_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Atanh, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_atanh_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Atanh, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_atanh_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Atanh, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_atanh_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Square (saved-x) --------
            (UnaryKind::Square, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_square_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Square, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_square_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Square, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_square_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Square, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_square_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Cube (saved-x) --------
            (UnaryKind::Cube, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cube_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Cube, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cube_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Cube, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cube_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Cube, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cube_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Exp2 (saved-y) --------
            (UnaryKind::Exp2, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_exp2_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Exp2, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_exp2_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Exp2, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_exp2_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Exp2, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_exp2_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Exp (saved-y) --------
            (UnaryKind::Exp, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_exp_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Exp, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_exp_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Exp, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_exp_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Exp, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_exp_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Expm1 (saved-y) --------
            (UnaryKind::Expm1, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_expm1_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Expm1, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_expm1_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Expm1, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_expm1_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Expm1, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_expm1_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Tanh (saved-y) --------
            (UnaryKind::Tanh, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_tanh_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Tanh, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_tanh_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Tanh, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_tanh_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Tanh, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_tanh_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Sigmoid (saved-y) --------
            (UnaryKind::Sigmoid, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sigmoid_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sigmoid, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sigmoid_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sigmoid, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sigmoid_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sigmoid, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sigmoid_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Sqrt (saved-y) --------
            (UnaryKind::Sqrt, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sqrt_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sqrt, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sqrt_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sqrt, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sqrt_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sqrt, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sqrt_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Rsqrt (saved-y) --------
            (UnaryKind::Rsqrt, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_rsqrt_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Rsqrt, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_rsqrt_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Rsqrt, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_rsqrt_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Rsqrt, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_rsqrt_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Tanhshrink (saved-x, transcendental) --------
            (UnaryKind::Tanhshrink, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_tanhshrink_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Tanhshrink, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_tanhshrink_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Tanhshrink, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_tanhshrink_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Tanhshrink, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_tanhshrink_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Logit (saved-x, no transcendental) --------
            (UnaryKind::Logit, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_logit_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Logit, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_logit_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Logit, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_logit_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Logit, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_logit_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Reciprocal (saved-x, no transcendental) --------
            (UnaryKind::Reciprocal, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_reciprocal_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Reciprocal, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_reciprocal_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Reciprocal, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_reciprocal_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Reciprocal, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_reciprocal_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Erf (saved-x, transcendental) --------
            (UnaryKind::Erf, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_erf_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Erf, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_erf_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Erf, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_erf_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Erf, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_erf_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Erfc (saved-x, transcendental) --------
            (UnaryKind::Erfc, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_erfc_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Erfc, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_erfc_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Erfc, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_erfc_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Erfc, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_erfc_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- ReLU (saved-x, piecewise activation — Category B' trailblazer) --------
            (UnaryKind::Relu, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_relu_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Relu, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_relu_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Relu, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_relu_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Relu, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_relu_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Hardtanh (saved-x, piecewise activation) --------
            (UnaryKind::Hardtanh, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardtanh_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Hardtanh, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardtanh_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Hardtanh, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardtanh_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Hardtanh, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardtanh_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- ReLU6 (saved-x, piecewise activation) --------
            (UnaryKind::Relu6, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_relu6_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Relu6, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_relu6_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Relu6, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_relu6_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Relu6, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_relu6_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Hardsigmoid (saved-x, piecewise + scalar div) --------
            (UnaryKind::Hardsigmoid, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardsigmoid_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Hardsigmoid, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardsigmoid_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Hardsigmoid, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardsigmoid_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Hardsigmoid, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardsigmoid_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Hardswish (saved-x, three-region piecewise) --------
            (UnaryKind::Hardswish, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardswish_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Hardswish, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardswish_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Hardswish, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardswish_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Hardswish, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardswish_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Softplus (saved-x, smooth, one exp) --------
            (UnaryKind::Softplus, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_softplus_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Softplus, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_softplus_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Softplus, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_softplus_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Softplus, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_softplus_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- SiLU / Swish (saved-x, smooth) --------
            (UnaryKind::Silu, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_silu_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Silu, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_silu_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Silu, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_silu_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Silu, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_silu_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Mish (saved-x, smooth, chained transcendentals) --------
            (UnaryKind::Mish, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_mish_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Mish, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_mish_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Mish, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_mish_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Mish, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_mish_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- GELU (exact / erf-based) (saved-x) --------
            (UnaryKind::Gelu, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_gelu_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Gelu, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_gelu_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Gelu, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_gelu_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Gelu, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_gelu_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- GELU (tanh approximation) (saved-x) --------
            (UnaryKind::GeluTanh, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_gelu_tanh_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::GeluTanh, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_gelu_tanh_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::GeluTanh, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_gelu_tanh_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::GeluTanh, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_gelu_tanh_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- SELU (saved-x, piecewise + exp on neg branch) --------
            (UnaryKind::Selu, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_selu_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Selu, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_selu_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Selu, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_selu_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Selu, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_selu_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- LeakyReLU (α=0.01, saved-x, piecewise) --------
            (UnaryKind::LeakyRelu, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_leaky_relu_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::LeakyRelu, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_leaky_relu_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::LeakyRelu, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_leaky_relu_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::LeakyRelu, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_leaky_relu_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- ELU (α=1.0, saved-x, piecewise + exp on neg branch) --------
            (UnaryKind::Elu, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_elu_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Elu, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_elu_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Elu, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_elu_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Elu, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_elu_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Hardshrink (λ=0.5, saved-x, piecewise mask) --------
            (UnaryKind::Hardshrink, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardshrink_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Hardshrink, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardshrink_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Hardshrink, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardshrink_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Hardshrink, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardshrink_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Softshrink (λ=0.5, saved-x, piecewise mask — same shape as Hardshrink BW) --------
            (UnaryKind::Softshrink, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_softshrink_backward_f32_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Softshrink, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_softshrink_backward_f16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Softshrink, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_softshrink_backward_bf16_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Softshrink, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_softshrink_backward_f64_run(
                    numel, dy_ptr, saved_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::UnaryBackwardPlan::run reached an unimplemented \
                     (kind, dtype) pair — select() should have caught this",
                ));
            }
        };
        map_status(status)
    }
}

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
