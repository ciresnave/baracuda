//! Parameterized unary elementwise plan.
//!
//! Sibling of [`crate::UnaryPlan`] for ops that carry one or more scalar
//! parameters alongside the tensor input. The plan family fixes the
//! parameter slot count at two `f32` values
//! (`params: [f32; 2]`) so a single descriptor / kernel ABI shape covers
//! every op in this family — ops that need fewer params (e.g.
//! `LeakyRelu(α)` would only consume `p0`) simply ignore the unused
//! slot.
//!
//! Today wired:
//!   * `Threshold` (`y = (x > t) ? x : v`; `t = params[0]`,
//!     `v = params[1]`) across `{f32, f16, bf16, f64}` — FW + BW
//!     (BW lives in [`crate::UnaryParamBackwardPlan`]).
//!   * `PowI` (`y = x^n` integer exponent; `n = params[0] as i32`,
//!     `params[1]` unused) across `{f32, f16, bf16, f64}` — Phase 12.1.
//!     Power-by-squaring; well-defined for negative `x` (no NaN) and
//!     bit-exact for `n = 2` (which collapses to `Square`).
//!
//! The existing single-param activation ops (`LeakyRelu(α)`, `ELU(α)`,
//! `Hardshrink(λ)`, `Softshrink(λ)`) ship through the plain
//! [`crate::UnaryPlan`] today with hardcoded PyTorch defaults; they
//! could later re-emit through this parameterized plan to expose the
//! coefficient as a runtime argument, but doing so isn't required —
//! the two paths can coexist.
//!
//! Trailblazer constraints: contig-only (no strided variant);
//! `x.shape == y.shape == desc.shape`. The kernel does not broadcast.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, UnaryKind, Workspace,
};

/// Descriptor for a parameterized unary elementwise op.
///
/// `shape` is both the input and the output shape. `element` must match
/// `T::KIND` at `select` time. `params` carries op-specific scalars —
/// the layout is fixed by the op's `kind`:
///
/// | Op          | `params[0]` | `params[1]` |
/// |-------------|-------------|-------------|
/// | `Threshold` | `t`         | `v`         |
/// | `PowI`      | `n as f32`  | unused      |
///
/// We chose `[f32; 2]` rather than separate `t: f32, v: f32` fields so
/// the descriptor shape doesn't shift as more 1- or 2-param ops join
/// (LeakyRelu / ELU / Hardshrink / Softshrink could all re-emit here
/// with `params[0] = α` or `params[0] = λ` and `params[1]` ignored).
#[derive(Copy, Clone, Debug)]
pub struct UnaryParamDescriptor<const N: usize> {
    /// Which parameterized unary op to apply.
    pub kind: UnaryKind,
    /// Tensor shape — input and output share it.
    pub shape: [i32; N],
    /// Primary element type. Must match the type parameter `T` of the
    /// containing plan.
    pub element: ElementKind,
    /// Op-specific scalar parameters. Slot semantics depend on `kind`:
    /// for `Threshold` it's `[t, v]`. Parameters are always `f32` on the
    /// FFI; integer / `f64` kernels widen the param losslessly at the
    /// kernel boundary, and half-precision kernels compare in `f32`.
    pub params: [f32; 2],
}

/// Args bundle for a parameterized unary elementwise launch.
pub struct UnaryParamArgs<'a, T: Element, const N: usize> {
    /// Input.
    pub x: TensorRef<'a, T, N>,
    /// Output.
    pub y: TensorMut<'a, T, N>,
}

/// Parameterized unary elementwise plan.
pub struct UnaryParamPlan<T: Element, const N: usize> {
    desc: UnaryParamDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> UnaryParamPlan<T, N> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &UnaryParamDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::UnaryParamPlan: descriptor element != type parameter T",
            ));
        }
        for &d in desc.shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::UnaryParamPlan: shape dims must be non-negative",
                ));
            }
        }

        // Today's wired matrix: {Threshold, PowI} × {f32, f16, bf16, f64}.
        // Future params-bearing ops can extend `kind_in_scope` and add
        // match arms in `run`.
        let kind_in_scope = matches!(desc.kind, UnaryKind::Threshold | UnaryKind::PowI);
        let dtype_in_scope = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !(kind_in_scope && dtype_in_scope) {
            return Err(Error::Unsupported(
                "baracuda-kernels::UnaryParamPlan: today only `{Threshold, PowI} × \
                 {f32, f16, bf16, f64}` is wired; LeakyRelu / ELU / Hardshrink / Softshrink \
                 ship via UnaryPlan with hardcoded PyTorch defaults today; PReLU needs a \
                 distinct (channel-vector) plan.",
            ));
        }

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

    /// Validate args.
    pub fn can_implement(&self, args: &UnaryParamArgs<'_, T, N>) -> Result<()> {
        if args.x.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::UnaryParamPlan: X shape mismatch with descriptor",
            ));
        }
        if args.y.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::UnaryParamPlan: Y shape mismatch with descriptor",
            ));
        }
        if !args.x.is_contiguous() || !args.y.is_contiguous() {
            return Err(Error::Unsupported(
                "baracuda-kernels::UnaryParamPlan: contig-only trailblazer; strided fanout \
                 lands later",
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

    /// Workspace size in bytes.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Identity of the kernel this plan picked.
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
        args: UnaryParamArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.y.numel();
        if numel == 0 {
            return Ok(());
        }
        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let p0 = self.desc.params[0];
        let p1 = self.desc.params[1];

        let status = match (self.desc.kind, T::KIND) {
            (UnaryKind::Threshold, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_threshold_f32_run(
                    numel, x_ptr, y_ptr, p0, p1,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Threshold, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_threshold_f16_run(
                    numel, x_ptr, y_ptr, p0, p1,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Threshold, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_threshold_bf16_run(
                    numel, x_ptr, y_ptr, p0, p1,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Threshold, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_threshold_f64_run(
                    numel, x_ptr, y_ptr, p0, p1,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::PowI, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_powi_f32_run(
                    numel, x_ptr, y_ptr, p0, p1,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::PowI, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_powi_f16_run(
                    numel, x_ptr, y_ptr, p0, p1,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::PowI, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_powi_bf16_run(
                    numel, x_ptr, y_ptr, p0, p1,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::PowI, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_powi_f64_run(
                    numel, x_ptr, y_ptr, p0, p1,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::UnaryParamPlan: dispatcher reached an unimplemented \
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
