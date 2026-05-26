//! Unary elementwise plan.
//!
//! 1→1 sibling of [`crate::BinaryPlan`]. Mirrors the same shape
//! (descriptor + args + select/can_implement/run/sku/precision_guarantee)
//! but for ops with a single input.
//!
//! Today only [`UnaryKind::Neg`] on `f32` is wired — the Phase 3 unary
//! trailblazer SKU. The other unary variants (abs / sqrt / exp / log /
//! sin / relu / gelu / silu / …) land in fanout sessions; the `Neg`
//! instantiation in `baracuda-kernels-sys` is the template pattern they
//! follow (one templated functor + one INSTANTIATE pair per dtype).
//!
//! Both the contig fast path and the strided path are wired from day
//! one — the dispatcher picks based on `is_contiguous()` of input and
//! output. Unary doesn't support broadcasting (input shape must equal
//! output shape) — a "broadcast" unary would be `f(x[0])` replicated,
//! which is a trivially host-side computation.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, UnaryKind, Workspace,
};

/// Descriptor for a unary elementwise op.
///
/// `shape` is both the input and the output shape (unary doesn't change
/// shape). `element` must match `T::KIND` at `select` time.
#[derive(Copy, Clone, Debug)]
pub struct UnaryDescriptor<const N: usize> {
    /// Which unary op to apply.
    pub kind: UnaryKind,
    /// Tensor shape — input and output share it.
    pub shape: [i32; N],
    /// Primary element type. Must match the type parameter `T` of the
    /// containing plan.
    pub element: ElementKind,
}

/// Args bundle for a unary elementwise launch.
///
/// Aliasing `y` with `x` is allowed (in-place) — the kernel reads
/// `x[i]` before writing `y[i]` per thread, no inter-thread race.
pub struct UnaryArgs<'a, T: Element, const N: usize> {
    /// Input.
    pub x: TensorRef<'a, T, N>,
    /// Output.
    pub y: TensorMut<'a, T, N>,
}

/// Unary elementwise plan.
///
/// `T: Element` is the kernel's element type (today: must be `f32`).
/// `const N: usize` is the tensor rank.
pub struct UnaryPlan<T: Element, const N: usize> {
    desc: UnaryDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> UnaryPlan<T, N> {
    /// Pick a kernel for `desc`. Returns [`Error::Unsupported`] if the
    /// `(kind, T::KIND)` pair isn't wired today.
    pub fn select(
        _stream: &Stream,
        desc: &UnaryDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::UnaryPlan: descriptor element != type parameter T",
            ));
        }
        for &d in desc.shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::UnaryPlan: shape dims must be non-negative",
                ));
            }
        }

        // Supported matrix: each fanout session adds its ops to
        // `kind_in_scope`. After the Phase 3 math / rounding fanout
        // every `kind_in_scope` member is wired across the full FP
        // `dtype_in_scope`, so the supported check reduces to the
        // straight cross product `kind_in_scope && dtype_in_scope`.
        // The match arms in `run` / `run_strided` remain the
        // authoritative dispatch table; the unreachable `_ =>` arm
        // catches any future drift.
        let kind_in_scope = matches!(
            desc.kind,
            UnaryKind::Neg
                | UnaryKind::Abs
                | UnaryKind::Sign
                | UnaryKind::Reciprocal
                | UnaryKind::Square
                | UnaryKind::Cube
                | UnaryKind::Sqrt
                | UnaryKind::Rsqrt
                | UnaryKind::Cbrt
                | UnaryKind::Exp
                | UnaryKind::Exp2
                | UnaryKind::Expm1
                | UnaryKind::Log
                | UnaryKind::Log2
                | UnaryKind::Log10
                | UnaryKind::Log1p
                | UnaryKind::Sin
                | UnaryKind::Cos
                | UnaryKind::Tan
                | UnaryKind::Asin
                | UnaryKind::Acos
                | UnaryKind::Atan
                | UnaryKind::Sinh
                | UnaryKind::Cosh
                | UnaryKind::Tanh
                | UnaryKind::Asinh
                | UnaryKind::Acosh
                | UnaryKind::Atanh
                | UnaryKind::Floor
                | UnaryKind::Ceil
                | UnaryKind::Round
                | UnaryKind::Trunc
                | UnaryKind::Frac
                | UnaryKind::Relu
                | UnaryKind::Gelu
                | UnaryKind::GeluTanh
                | UnaryKind::Silu
                | UnaryKind::Mish
                | UnaryKind::Sigmoid
                | UnaryKind::Softplus
                | UnaryKind::Hardswish
                | UnaryKind::Hardsigmoid
                | UnaryKind::Hardtanh
                | UnaryKind::Erf
                | UnaryKind::Erfc
                | UnaryKind::Lgamma
                | UnaryKind::Logit
                | UnaryKind::Softsign
                | UnaryKind::Tanhshrink
                | UnaryKind::Relu6
                | UnaryKind::Selu
                | UnaryKind::LeakyRelu
                | UnaryKind::Elu
                | UnaryKind::Hardshrink
                | UnaryKind::Softshrink
        );
        let dtype_in_scope = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        let supported = kind_in_scope && dtype_in_scope;
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::UnaryPlan: this (kind, dtype) cell is not yet \
                 wired; see the dispatcher's kind / dtype scope for the supported set",
            ));
        }

        // Arch-agnostic SIMT path; same precision guarantee as the
        // binary plan's f32 path.
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
            _marker: PhantomData,
        })
    }

    /// Validate that this plan can launch with `args`.
    ///
    /// Unary doesn't broadcast: `x.shape` must equal `y.shape` must
    /// equal `desc.shape`. Both can be non-contiguous (transposed /
    /// sliced views); the dispatcher routes to the strided kernel in
    /// that case.
    pub fn can_implement(&self, args: &UnaryArgs<'_, T, N>) -> Result<()> {
        if args.x.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::UnaryPlan: X shape mismatch with descriptor",
            ));
        }
        if args.y.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::UnaryPlan: Y shape mismatch with descriptor",
            ));
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::UnaryPlan: tensor rank > 8 not supported \
                 (kernel param block fixes MAX_RANK = 8)",
            ));
        }
        let numel = args.y.numel();
        let x_len = args.x.data.len() as i64;
        let y_len = args.y.data.len() as i64;
        if x_len < numel || y_len < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: x_len.min(y_len) as usize,
            });
        }
        Ok(())
    }

    /// Workspace size in bytes. Always `0` for the trailblazer SKU.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Identity of the kernel this plan picked.
    #[inline]
    pub fn sku(&self) -> KernelSku {
        self.sku
    }

    /// Numerical guarantees for this plan's kernel.
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee
    }

    /// Launch.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: UnaryArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.y.numel();
        if numel == 0 {
            return Ok(());
        }
        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        // Contig fast path requires both operands to be fully contiguous.
        // Any other case (transposed / strided view) routes to the
        // strided kernel.
        let all_contig = args.x.is_contiguous() && args.y.is_contiguous();

        if !all_contig {
            return self.run_strided(stream_ptr, x_ptr, y_ptr, numel, &args);
        }

        let status = match (self.desc.kind, T::KIND) {
            (UnaryKind::Neg, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_neg_f32_run(
                    numel,
                    x_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            // Abs × {f32, f16, bf16, f64}
            (UnaryKind::Abs, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_abs_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Abs, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_abs_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Abs, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_abs_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Abs, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_abs_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Sign × {f32, f16, bf16, f64}
            (UnaryKind::Sign, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sign_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sign, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sign_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sign, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sign_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sign, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sign_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Reciprocal × {f32, f16, bf16, f64}
            (UnaryKind::Reciprocal, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_reciprocal_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Reciprocal, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_reciprocal_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Reciprocal, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_reciprocal_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Reciprocal, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_reciprocal_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Square × {f32, f16, bf16, f64}
            (UnaryKind::Square, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_square_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Square, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_square_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Square, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_square_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Square, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_square_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Cube × {f32, f16, bf16, f64}
            (UnaryKind::Cube, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cube_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Cube, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cube_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Cube, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cube_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Cube, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cube_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Sqrt × {f32, f16, bf16, f64}
            (UnaryKind::Sqrt, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sqrt_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sqrt, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sqrt_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sqrt, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sqrt_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sqrt, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sqrt_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Rsqrt × {f32, f16, bf16, f64}
            (UnaryKind::Rsqrt, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_rsqrt_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Rsqrt, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_rsqrt_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Rsqrt, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_rsqrt_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Rsqrt, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_rsqrt_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Exp × {f32, f16, bf16, f64}
            (UnaryKind::Exp, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_exp_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Exp, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_exp_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Exp, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_exp_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Exp, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_exp_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Expm1 × {f32, f16, bf16, f64}
            (UnaryKind::Expm1, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_expm1_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Expm1, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_expm1_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Expm1, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_expm1_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Expm1, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_expm1_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Log × {f32, f16, bf16, f64}
            (UnaryKind::Log, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Log, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Log, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Log, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Log1p × {f32, f16, bf16, f64}
            (UnaryKind::Log1p, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log1p_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Log1p, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log1p_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Log1p, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log1p_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Log1p, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log1p_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Sin × {f32, f16, bf16, f64}
            (UnaryKind::Sin, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sin_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sin, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sin_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sin, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sin_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sin, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sin_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Cos × {f32, f16, bf16, f64}
            (UnaryKind::Cos, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cos_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Cos, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cos_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Cos, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cos_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Cos, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cos_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Tan × {f32, f16, bf16, f64}
            (UnaryKind::Tan, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_tan_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Tan, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_tan_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Tan, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_tan_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Tan, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_tan_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Sinh × {f32, f16, bf16, f64}
            (UnaryKind::Sinh, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sinh_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sinh, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sinh_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sinh, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sinh_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sinh, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sinh_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Cosh × {f32, f16, bf16, f64}
            (UnaryKind::Cosh, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cosh_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Cosh, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cosh_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Cosh, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cosh_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Cosh, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cosh_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Tanh × {f32, f16, bf16, f64}
            (UnaryKind::Tanh, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_tanh_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Tanh, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_tanh_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Tanh, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_tanh_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Tanh, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_tanh_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Relu × {f32, f16, bf16, f64}
            (UnaryKind::Relu, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_relu_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Relu, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_relu_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Relu, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_relu_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Relu, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_relu_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Gelu × {f32, f16, bf16, f64}
            (UnaryKind::Gelu, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_gelu_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Gelu, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_gelu_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Gelu, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_gelu_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Gelu, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_gelu_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // GeluTanh × {f32, f16, bf16, f64}
            (UnaryKind::GeluTanh, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_gelu_tanh_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::GeluTanh, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_gelu_tanh_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::GeluTanh, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_gelu_tanh_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::GeluTanh, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_gelu_tanh_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Silu × {f32, f16, bf16, f64}
            (UnaryKind::Silu, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_silu_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Silu, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_silu_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Silu, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_silu_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Silu, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_silu_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Mish × {f32, f16, bf16, f64}
            (UnaryKind::Mish, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_mish_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Mish, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_mish_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Mish, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_mish_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Mish, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_mish_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Sigmoid × {f32, f16, bf16, f64}
            (UnaryKind::Sigmoid, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sigmoid_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sigmoid, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sigmoid_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sigmoid, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sigmoid_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sigmoid, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sigmoid_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Softplus × {f32, f16, bf16, f64}
            (UnaryKind::Softplus, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_softplus_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Softplus, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_softplus_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Softplus, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_softplus_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Softplus, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_softplus_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Hardswish × {f32, f16, bf16, f64}
            (UnaryKind::Hardswish, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardswish_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Hardswish, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardswish_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Hardswish, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardswish_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Hardswish, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardswish_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Hardsigmoid × {f32, f16, bf16, f64}
            (UnaryKind::Hardsigmoid, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardsigmoid_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Hardsigmoid, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardsigmoid_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Hardsigmoid, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardsigmoid_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Hardsigmoid, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardsigmoid_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Hardtanh × {f32, f16, bf16, f64}
            (UnaryKind::Hardtanh, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardtanh_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Hardtanh, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardtanh_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Hardtanh, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardtanh_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Hardtanh, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardtanh_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Neg dtype fill — Neg's trailblazer f32 cell is wired above.
            (UnaryKind::Neg, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_neg_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Neg, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_neg_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Neg, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_neg_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Cbrt × {f32, f16, bf16, f64}
            (UnaryKind::Cbrt, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cbrt_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Cbrt, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cbrt_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Cbrt, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cbrt_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Cbrt, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cbrt_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Exp2 × {f32, f16, bf16, f64}
            (UnaryKind::Exp2, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_exp2_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Exp2, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_exp2_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Exp2, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_exp2_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Exp2, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_exp2_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Log2 × {f32, f16, bf16, f64}
            (UnaryKind::Log2, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log2_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Log2, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log2_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Log2, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log2_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Log2, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log2_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Log10 × {f32, f16, bf16, f64}
            (UnaryKind::Log10, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log10_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Log10, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log10_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Log10, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log10_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Log10, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log10_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Asin × {f32, f16, bf16, f64}
            (UnaryKind::Asin, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_asin_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Asin, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_asin_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Asin, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_asin_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Asin, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_asin_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Acos × {f32, f16, bf16, f64}
            (UnaryKind::Acos, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_acos_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Acos, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_acos_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Acos, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_acos_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Acos, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_acos_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Atan × {f32, f16, bf16, f64}
            (UnaryKind::Atan, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_atan_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Atan, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_atan_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Atan, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_atan_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Atan, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_atan_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Asinh × {f32, f16, bf16, f64}
            (UnaryKind::Asinh, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_asinh_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Asinh, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_asinh_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Asinh, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_asinh_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Asinh, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_asinh_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Acosh × {f32, f16, bf16, f64}
            (UnaryKind::Acosh, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_acosh_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Acosh, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_acosh_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Acosh, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_acosh_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Acosh, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_acosh_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Atanh × {f32, f16, bf16, f64}
            (UnaryKind::Atanh, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_atanh_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Atanh, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_atanh_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Atanh, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_atanh_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Atanh, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_atanh_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Floor × {f32, f16, bf16, f64}
            (UnaryKind::Floor, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_floor_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Floor, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_floor_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Floor, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_floor_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Floor, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_floor_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Ceil × {f32, f16, bf16, f64}
            (UnaryKind::Ceil, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_ceil_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Ceil, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_ceil_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Ceil, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_ceil_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Ceil, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_ceil_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Round × {f32, f16, bf16, f64}
            (UnaryKind::Round, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_round_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Round, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_round_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Round, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_round_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Round, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_round_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Trunc × {f32, f16, bf16, f64}
            (UnaryKind::Trunc, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_trunc_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Trunc, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_trunc_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Trunc, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_trunc_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Trunc, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_trunc_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Frac × {f32, f16, bf16, f64}
            (UnaryKind::Frac, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_frac_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Frac, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_frac_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Frac, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_frac_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Frac, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_frac_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Erf × {f32, f16, bf16, f64}
            (UnaryKind::Erf, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_erf_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Erf, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_erf_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Erf, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_erf_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Erf, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_erf_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Erfc × {f32, f16, bf16, f64}
            (UnaryKind::Erfc, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_erfc_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Erfc, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_erfc_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Erfc, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_erfc_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Erfc, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_erfc_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Lgamma × {f32, f16, bf16, f64}
            (UnaryKind::Lgamma, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_lgamma_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Lgamma, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_lgamma_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Lgamma, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_lgamma_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Lgamma, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_lgamma_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Logit × {f32, f16, bf16, f64}
            (UnaryKind::Logit, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_logit_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Logit, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_logit_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Logit, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_logit_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Logit, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_logit_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Softsign × {f32, f16, bf16, f64}
            (UnaryKind::Softsign, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_softsign_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Softsign, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_softsign_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Softsign, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_softsign_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Softsign, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_softsign_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Tanhshrink × {f32, f16, bf16, f64}
            (UnaryKind::Tanhshrink, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_tanhshrink_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Tanhshrink, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_tanhshrink_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Tanhshrink, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_tanhshrink_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Tanhshrink, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_tanhshrink_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Relu6 × {f32, f16, bf16, f64}
            (UnaryKind::Relu6, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_relu6_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Relu6, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_relu6_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Relu6, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_relu6_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Relu6, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_relu6_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Selu × {f32, f16, bf16, f64}
            (UnaryKind::Selu, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_selu_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Selu, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_selu_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Selu, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_selu_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Selu, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_selu_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // LeakyReLU (α=0.01) × {f32, f16, bf16, f64}
            (UnaryKind::LeakyRelu, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_leaky_relu_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::LeakyRelu, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_leaky_relu_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::LeakyRelu, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_leaky_relu_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::LeakyRelu, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_leaky_relu_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // ELU × {f32, f16, bf16, f64} — α threads through the FFI as
            // of Phase 31. The non-parameterized `UnaryPlan` here keeps
            // the historical α=1.0 PyTorch default; callers wanting a
            // different α call the FFI directly (or wait for the
            // `UnaryParamPlan::Elu` fanout that lands with the next
            // descriptor-parameter session).
            (UnaryKind::Elu, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_elu_f32_run(
                    numel, x_ptr, y_ptr, 1.0f32, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Elu, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_elu_f16_run(
                    numel, x_ptr, y_ptr, 1.0f32, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Elu, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_elu_bf16_run(
                    numel, x_ptr, y_ptr, 1.0f32, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Elu, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_elu_f64_run(
                    numel, x_ptr, y_ptr, 1.0f32, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Hardshrink (λ=0.5) × {f32, f16, bf16, f64}
            (UnaryKind::Hardshrink, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardshrink_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Hardshrink, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardshrink_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Hardshrink, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardshrink_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Hardshrink, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardshrink_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Softshrink (λ=0.5) × {f32, f16, bf16, f64}
            (UnaryKind::Softshrink, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_softshrink_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Softshrink, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_softshrink_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Softshrink, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_softshrink_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Softshrink, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_softshrink_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::UnaryPlan::run reached an unimplemented \
                     (kind, dtype) pair — select() should have caught this",
                ))
            }
        };
        map_status(status)
    }
}

impl<T: Element, const N: usize> UnaryPlan<T, N> {
    /// Launch the strided kernel path. Called by [`Self::run`] when at
    /// least one operand isn't contiguous.
    fn run_strided(
        &self,
        stream_ptr: *mut c_void,
        x_ptr: *const c_void,
        y_ptr: *mut c_void,
        numel: i64,
        args: &UnaryArgs<'_, T, N>,
    ) -> Result<()> {
        let shape = args.y.shape;
        let stride_x = args.x.stride;
        let stride_y = args.y.stride;
        let rank = N as i32;

        let status = match (self.desc.kind, T::KIND) {
            (UnaryKind::Neg, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_neg_f32_strided_run(
                    numel,
                    rank,
                    shape.as_ptr(),
                    stride_x.as_ptr(),
                    stride_y.as_ptr(),
                    x_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            // Abs × {f32, f16, bf16, f64}
            (UnaryKind::Abs, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_abs_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Abs, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_abs_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Abs, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_abs_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Abs, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_abs_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Sign × {f32, f16, bf16, f64}
            (UnaryKind::Sign, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sign_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sign, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sign_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sign, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sign_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sign, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sign_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Reciprocal × {f32, f16, bf16, f64}
            (UnaryKind::Reciprocal, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_reciprocal_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Reciprocal, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_reciprocal_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Reciprocal, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_reciprocal_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Reciprocal, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_reciprocal_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Square × {f32, f16, bf16, f64}
            (UnaryKind::Square, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_square_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Square, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_square_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Square, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_square_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Square, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_square_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Cube × {f32, f16, bf16, f64}
            (UnaryKind::Cube, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cube_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Cube, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cube_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Cube, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cube_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Cube, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cube_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Sqrt × {f32, f16, bf16, f64}
            (UnaryKind::Sqrt, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sqrt_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sqrt, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sqrt_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sqrt, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sqrt_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sqrt, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sqrt_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Rsqrt × {f32, f16, bf16, f64}
            (UnaryKind::Rsqrt, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_rsqrt_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Rsqrt, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_rsqrt_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Rsqrt, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_rsqrt_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Rsqrt, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_rsqrt_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Exp × {f32, f16, bf16, f64}
            (UnaryKind::Exp, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_exp_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Exp, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_exp_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Exp, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_exp_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Exp, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_exp_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Expm1 × {f32, f16, bf16, f64}
            (UnaryKind::Expm1, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_expm1_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Expm1, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_expm1_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Expm1, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_expm1_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Expm1, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_expm1_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Log × {f32, f16, bf16, f64}
            (UnaryKind::Log, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Log, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Log, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Log, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Log1p × {f32, f16, bf16, f64}
            (UnaryKind::Log1p, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log1p_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Log1p, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log1p_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Log1p, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log1p_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Log1p, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log1p_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Sin × {f32, f16, bf16, f64}
            (UnaryKind::Sin, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sin_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sin, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sin_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sin, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sin_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sin, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sin_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Cos × {f32, f16, bf16, f64}
            (UnaryKind::Cos, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cos_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Cos, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cos_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Cos, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cos_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Cos, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cos_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Tan × {f32, f16, bf16, f64}
            (UnaryKind::Tan, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_tan_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Tan, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_tan_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Tan, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_tan_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Tan, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_tan_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Sinh × {f32, f16, bf16, f64}
            (UnaryKind::Sinh, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sinh_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sinh, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sinh_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sinh, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sinh_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sinh, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sinh_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Cosh × {f32, f16, bf16, f64}
            (UnaryKind::Cosh, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cosh_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Cosh, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cosh_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Cosh, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cosh_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Cosh, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cosh_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Tanh × {f32, f16, bf16, f64}
            (UnaryKind::Tanh, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_tanh_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Tanh, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_tanh_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Tanh, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_tanh_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Tanh, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_tanh_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Relu × {f32, f16, bf16, f64}
            (UnaryKind::Relu, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_relu_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Relu, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_relu_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Relu, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_relu_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Relu, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_relu_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Gelu × {f32, f16, bf16, f64}
            (UnaryKind::Gelu, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_gelu_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Gelu, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_gelu_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Gelu, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_gelu_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Gelu, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_gelu_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // GeluTanh × {f32, f16, bf16, f64}
            (UnaryKind::GeluTanh, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_gelu_tanh_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::GeluTanh, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_gelu_tanh_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::GeluTanh, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_gelu_tanh_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::GeluTanh, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_gelu_tanh_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Silu × {f32, f16, bf16, f64}
            (UnaryKind::Silu, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_silu_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Silu, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_silu_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Silu, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_silu_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Silu, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_silu_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Mish × {f32, f16, bf16, f64}
            (UnaryKind::Mish, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_mish_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Mish, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_mish_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Mish, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_mish_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Mish, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_mish_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Sigmoid × {f32, f16, bf16, f64}
            (UnaryKind::Sigmoid, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sigmoid_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sigmoid, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sigmoid_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sigmoid, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sigmoid_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Sigmoid, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_sigmoid_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Softplus × {f32, f16, bf16, f64}
            (UnaryKind::Softplus, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_softplus_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Softplus, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_softplus_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Softplus, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_softplus_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Softplus, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_softplus_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Hardswish × {f32, f16, bf16, f64}
            (UnaryKind::Hardswish, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardswish_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Hardswish, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardswish_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Hardswish, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardswish_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Hardswish, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardswish_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Hardsigmoid × {f32, f16, bf16, f64}
            (UnaryKind::Hardsigmoid, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardsigmoid_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Hardsigmoid, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardsigmoid_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Hardsigmoid, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardsigmoid_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Hardsigmoid, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardsigmoid_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Hardtanh × {f32, f16, bf16, f64}
            (UnaryKind::Hardtanh, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardtanh_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Hardtanh, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardtanh_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Hardtanh, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardtanh_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Hardtanh, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardtanh_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Neg dtype fill — Neg's trailblazer f32 cell is wired above.
            (UnaryKind::Neg, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_neg_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Neg, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_neg_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Neg, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_neg_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Cbrt × {f32, f16, bf16, f64}
            (UnaryKind::Cbrt, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cbrt_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Cbrt, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cbrt_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Cbrt, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cbrt_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Cbrt, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_cbrt_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Exp2 × {f32, f16, bf16, f64}
            (UnaryKind::Exp2, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_exp2_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Exp2, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_exp2_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Exp2, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_exp2_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Exp2, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_exp2_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Log2 × {f32, f16, bf16, f64}
            (UnaryKind::Log2, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log2_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Log2, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log2_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Log2, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log2_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Log2, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log2_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Log10 × {f32, f16, bf16, f64}
            (UnaryKind::Log10, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log10_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Log10, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log10_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Log10, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log10_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Log10, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_log10_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Asin × {f32, f16, bf16, f64}
            (UnaryKind::Asin, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_asin_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Asin, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_asin_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Asin, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_asin_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Asin, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_asin_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Acos × {f32, f16, bf16, f64}
            (UnaryKind::Acos, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_acos_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Acos, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_acos_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Acos, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_acos_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Acos, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_acos_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Atan × {f32, f16, bf16, f64}
            (UnaryKind::Atan, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_atan_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Atan, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_atan_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Atan, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_atan_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Atan, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_atan_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Asinh × {f32, f16, bf16, f64}
            (UnaryKind::Asinh, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_asinh_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Asinh, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_asinh_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Asinh, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_asinh_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Asinh, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_asinh_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Acosh × {f32, f16, bf16, f64}
            (UnaryKind::Acosh, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_acosh_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Acosh, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_acosh_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Acosh, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_acosh_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Acosh, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_acosh_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Atanh × {f32, f16, bf16, f64}
            (UnaryKind::Atanh, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_atanh_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Atanh, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_atanh_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Atanh, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_atanh_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Atanh, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_atanh_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Floor × {f32, f16, bf16, f64}
            (UnaryKind::Floor, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_floor_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Floor, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_floor_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Floor, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_floor_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Floor, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_floor_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Ceil × {f32, f16, bf16, f64}
            (UnaryKind::Ceil, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_ceil_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Ceil, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_ceil_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Ceil, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_ceil_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Ceil, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_ceil_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Round × {f32, f16, bf16, f64}
            (UnaryKind::Round, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_round_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Round, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_round_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Round, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_round_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Round, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_round_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Trunc × {f32, f16, bf16, f64}
            (UnaryKind::Trunc, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_trunc_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Trunc, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_trunc_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Trunc, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_trunc_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Trunc, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_trunc_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Frac × {f32, f16, bf16, f64}
            (UnaryKind::Frac, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_frac_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Frac, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_frac_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Frac, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_frac_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Frac, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_frac_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Erf × {f32, f16, bf16, f64}
            (UnaryKind::Erf, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_erf_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Erf, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_erf_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Erf, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_erf_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Erf, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_erf_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Erfc × {f32, f16, bf16, f64}
            (UnaryKind::Erfc, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_erfc_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Erfc, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_erfc_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Erfc, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_erfc_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Erfc, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_erfc_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Lgamma × {f32, f16, bf16, f64}
            (UnaryKind::Lgamma, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_lgamma_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Lgamma, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_lgamma_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Lgamma, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_lgamma_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Lgamma, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_lgamma_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Logit × {f32, f16, bf16, f64}
            (UnaryKind::Logit, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_logit_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Logit, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_logit_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Logit, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_logit_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Logit, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_logit_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Softsign × {f32, f16, bf16, f64}
            (UnaryKind::Softsign, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_softsign_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Softsign, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_softsign_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Softsign, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_softsign_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Softsign, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_softsign_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Tanhshrink × {f32, f16, bf16, f64}
            (UnaryKind::Tanhshrink, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_tanhshrink_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Tanhshrink, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_tanhshrink_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Tanhshrink, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_tanhshrink_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Tanhshrink, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_tanhshrink_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Relu6 × {f32, f16, bf16, f64}
            (UnaryKind::Relu6, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_relu6_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Relu6, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_relu6_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Relu6, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_relu6_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Relu6, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_relu6_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Selu × {f32, f16, bf16, f64}
            (UnaryKind::Selu, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_selu_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Selu, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_selu_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Selu, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_selu_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Selu, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_selu_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // LeakyReLU (α=0.01) × {f32, f16, bf16, f64}
            (UnaryKind::LeakyRelu, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_leaky_relu_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::LeakyRelu, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_leaky_relu_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::LeakyRelu, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_leaky_relu_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::LeakyRelu, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_leaky_relu_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // ELU × {f32, f16, bf16, f64} — strided sibling; α default 1.0
            // matches the contig path above (Phase 31 FFI threads α
            // through every variant).
            (UnaryKind::Elu, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_elu_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, 1.0f32, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Elu, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_elu_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, 1.0f32, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Elu, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_elu_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, 1.0f32, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Elu, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_elu_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, 1.0f32, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Hardshrink (λ=0.5) × {f32, f16, bf16, f64}
            (UnaryKind::Hardshrink, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardshrink_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Hardshrink, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardshrink_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Hardshrink, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardshrink_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Hardshrink, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_hardshrink_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // Softshrink (λ=0.5) × {f32, f16, bf16, f64}
            (UnaryKind::Softshrink, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_softshrink_f32_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Softshrink, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_softshrink_f16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Softshrink, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_softshrink_bf16_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Softshrink, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_softshrink_f64_strided_run(
                    numel, rank, shape.as_ptr(), stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::UnaryPlan: strided path reached an \
                     unimplemented (kind, dtype) pair — select() should have caught this",
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
