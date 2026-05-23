//! Backward plan for the parameterized unary elementwise family.
//!
//! Sibling of [`crate::UnaryParamPlan`] for gradient computation. The
//! kernel ABI mirrors the forward (`params: [f32; 2]` threaded by value)
//! plus a saved forward input `x` — Threshold's BW formula is
//! `dx = (x > t) ? dy : 0`, so we need `x` available at BW time.
//!
//! Today wired:
//!   * `Threshold × {f32, f16, bf16, f64}`. The scalar params `(t, v)`
//!     are constants w.r.t. `x` — no gradient flows to them.
//!   * `PowI × {f32, f16, bf16, f64}` — Phase 12.1. `dx = n · x^(n-1) · dy`
//!     via power-by-squaring; `n` lives in `params[0]` (cast to i32 at
//!     the kernel boundary), `params[1]` unused. Special-cased for
//!     `n == 0` (gradient 0) and `n == 1` (gradient `dy`) inside the
//!     CUDA functor.
//!
//! Layout constraints: `dy.shape == x.shape == dx.shape == desc.shape`.
//!
//! `Threshold` BW remains contig-only today; `PowI` BW got a strided
//! sibling in Phase 14.2 — the run dispatcher checks `is_contiguous()`
//! across `dy` / `x` / `dx` and routes to `*_strided_run` when any of
//! the three is a non-canonical view.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, UnaryKind, Workspace,
};

/// Descriptor for a parameterized unary backward op. Shape and `params`
/// layout match [`crate::UnaryParamDescriptor`].
#[derive(Copy, Clone, Debug)]
pub struct UnaryParamBackwardDescriptor<const N: usize> {
    /// Which forward parameterized unary op this is the backward of.
    pub kind: UnaryKind,
    /// Tensor shape (shared by dy / x / dx).
    pub shape: [i32; N],
    /// Element type.
    pub element: ElementKind,
    /// Op-specific scalar parameters; same layout as the FW descriptor.
    pub params: [f32; 2],
}

/// Args bundle for a parameterized unary backward launch.
pub struct UnaryParamBackwardArgs<'a, T: Element, const N: usize> {
    /// Upstream gradient (input to backward).
    pub dy: TensorRef<'a, T, N>,
    /// Saved forward input `x`. Required by Threshold's BW formula.
    pub x: TensorRef<'a, T, N>,
    /// Gradient w.r.t. the input.
    pub dx: TensorMut<'a, T, N>,
}

/// Parameterized unary backward plan.
pub struct UnaryParamBackwardPlan<T: Element, const N: usize> {
    desc: UnaryParamBackwardDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> UnaryParamBackwardPlan<T, N> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &UnaryParamBackwardDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::UnaryParamBackwardPlan: descriptor element != T",
            ));
        }
        for &d in desc.shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::UnaryParamBackwardPlan: shape dims must be non-negative",
                ));
            }
        }

        let kind_in_scope = matches!(desc.kind, UnaryKind::Threshold | UnaryKind::PowI);
        let dtype_in_scope = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !(kind_in_scope && dtype_in_scope) {
            return Err(Error::Unsupported(
                "baracuda-kernels::UnaryParamBackwardPlan: today only `{Threshold, PowI} × \
                 {f32, f16, bf16, f64}` is wired; future params-bearing BWs land here.",
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
    pub fn can_implement(&self, args: &UnaryParamBackwardArgs<'_, T, N>) -> Result<()> {
        if args.dy.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::UnaryParamBackwardPlan: dy shape mismatch",
            ));
        }
        if args.x.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::UnaryParamBackwardPlan: x shape mismatch",
            ));
        }
        if args.dx.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::UnaryParamBackwardPlan: dx shape mismatch",
            ));
        }
        // PowI BW got a strided sibling in Phase 14.2; Threshold BW
        // remains contig-only until its strided launcher is wired.
        let all_contig =
            args.dy.is_contiguous() && args.x.is_contiguous() && args.dx.is_contiguous();
        if !all_contig && !matches!(self.desc.kind, UnaryKind::PowI) {
            return Err(Error::Unsupported(
                "baracuda-kernels::UnaryParamBackwardPlan: this op is contig-only today; \
                 strided fanout lands later (PowI is the trailblazer in Phase 14.2)",
            ));
        }
        let numel = args.dy.numel();
        let dy_len = args.dy.data.len() as i64;
        let x_len = args.x.data.len() as i64;
        let dx_len = args.dx.data.len() as i64;
        if dy_len < numel || x_len < numel || dx_len < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: dy_len.min(x_len).min(dx_len) as usize,
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
        args: UnaryParamBackwardArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.dy.numel();
        if numel == 0 {
            return Ok(());
        }
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let dx_ptr = args.dx.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let p0 = self.desc.params[0];
        let p1 = self.desc.params[1];

        // Strided fast-fall: only PowI BW has the strided sibling wired
        // today (Phase 14.2). Other kinds were rejected in
        // `can_implement` when any operand was non-contig.
        let all_contig =
            args.dy.is_contiguous() && args.x.is_contiguous() && args.dx.is_contiguous();
        if !all_contig && matches!(self.desc.kind, UnaryKind::PowI) {
            return self.run_strided(
                stream_ptr, dy_ptr, x_ptr, dx_ptr, numel, &args, p0, p1,
            );
        }

        let status = match (self.desc.kind, T::KIND) {
            (UnaryKind::Threshold, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_threshold_backward_f32_run(
                    numel, dy_ptr, x_ptr, dx_ptr, p0, p1,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Threshold, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_threshold_backward_f16_run(
                    numel, dy_ptr, x_ptr, dx_ptr, p0, p1,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Threshold, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_threshold_backward_bf16_run(
                    numel, dy_ptr, x_ptr, dx_ptr, p0, p1,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::Threshold, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_threshold_backward_f64_run(
                    numel, dy_ptr, x_ptr, dx_ptr, p0, p1,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::PowI, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_powi_backward_f32_run(
                    numel, dy_ptr, x_ptr, dx_ptr, p0, p1,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::PowI, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_powi_backward_f16_run(
                    numel, dy_ptr, x_ptr, dx_ptr, p0, p1,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::PowI, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_powi_backward_bf16_run(
                    numel, dy_ptr, x_ptr, dx_ptr, p0, p1,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::PowI, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_powi_backward_f64_run(
                    numel, dy_ptr, x_ptr, dx_ptr, p0, p1,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::UnaryParamBackwardPlan: dispatcher reached an \
                     unimplemented (kind, dtype) pair — select() should have caught this",
                ));
            }
        };
        map_status(status)
    }
}

impl<T: Element, const N: usize> UnaryParamBackwardPlan<T, N> {
    /// Strided dispatcher — called by [`Self::run`] when any of `dy`,
    /// `x`, `dx` is non-contig. Today only `PowI` reaches this path.
    fn run_strided(
        &self,
        stream_ptr: *mut c_void,
        dy_ptr: *const c_void,
        x_ptr: *const c_void,
        dx_ptr: *mut c_void,
        numel: i64,
        args: &UnaryParamBackwardArgs<'_, T, N>,
        p0: f32,
        p1: f32,
    ) -> Result<()> {
        let shape = args.dx.shape;
        let stride_x = args.x.stride;
        let stride_dy = args.dy.stride;
        let stride_dx = args.dx.stride;
        let rank = N as i32;

        let status = match (self.desc.kind, T::KIND) {
            (UnaryKind::PowI, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_powi_backward_f32_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_x.as_ptr(), stride_dy.as_ptr(), stride_dx.as_ptr(),
                    x_ptr, dy_ptr, dx_ptr, p0, p1,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::PowI, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_powi_backward_f16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_x.as_ptr(), stride_dy.as_ptr(), stride_dx.as_ptr(),
                    x_ptr, dy_ptr, dx_ptr, p0, p1,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::PowI, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_powi_backward_bf16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_x.as_ptr(), stride_dy.as_ptr(), stride_dx.as_ptr(),
                    x_ptr, dy_ptr, dx_ptr, p0, p1,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (UnaryKind::PowI, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unary_powi_backward_f64_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_x.as_ptr(), stride_dy.as_ptr(), stride_dx.as_ptr(),
                    x_ptr, dy_ptr, dx_ptr, p0, p1,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::UnaryParamBackwardPlan::run_strided: only PowI is \
                     wired for the strided path today (Phase 14.2 trailblazer)",
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
