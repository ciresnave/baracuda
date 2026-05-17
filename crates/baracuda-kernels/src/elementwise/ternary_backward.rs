//! Backward plan for the ternary elementwise family.
//!
//! Sibling of [`crate::TernaryPlan`] for gradient computation:
//! `(da, db, dc) = backward(dy, a, b, c [, scale])`.
//!
//! Wired matrix: `{Fma, Clamp, Addcmul, Addcdiv} × {f32, f16, bf16,
//! f64}` = 16 cells. Reserved-but-deferred: [`TernaryKind::Where`] —
//! it needs a heterogeneous-dtype backward plan shape (the `cond`
//! input is `u8`, not `T`) and would live in a future
//! `WhereBackwardPlan`.
//!
//! Two kernel families by parameterization:
//! - **Unscaled** (Fma, Clamp). Functor receives `(dy, a, b, c, &da,
//!   &db, &dc)`. Fma needs `a`, `b` algebraically (`c` is a passthrough
//!   for ABI uniformity); Clamp needs all three.
//! - **Scaled** (Addcmul, Addcdiv). Functor receives `(dy, a, b, c,
//!   scale, &da, &db, &dc)`. Both read `b`, `c` algebraically; `a` is
//!   a passthrough.
//!
//! BW formulas:
//! - Fma: `y = a·b + c`. `da = dy·b`, `db = dy·a`, `dc = dy`.
//! - Clamp: `y = min(max(a, b), c)` (b = lo, c = hi). Subgradient:
//!   `da = dy if b≤a≤c else 0`, `db = dy if a<b else 0`, `dc = dy if
//!   a>c else 0`. Boundary ties (a == b or a == c) route gradient to
//!   `a` — matches PyTorch's `torch.clamp` autograd convention.
//! - Addcmul: `y = a + scale·b·c`. `da = dy`, `db = dy·scale·c`,
//!   `dc = dy·scale·b`.
//! - Addcdiv: `y = a + scale·b/c`. `da = dy`, `db = dy·scale/c`,
//!   `dc = -dy·scale·b/c²`.
//!
//! Caller convention: the `Args` struct requires all three saved
//! inputs `a`, `b`, `c` regardless of which the gradient algebraically
//! references — keeps the ABI uniform across the 4 ops. The unused
//! load is one extra coalesced read per cell; negligible versus the
//! 7-pointer launch overhead.
//!
//! Trailblazer constraints: contig-only, same shape across all 7
//! tensors (`dy`, `a`, `b`, `c`, `da`, `db`, `dc`). Strided / broadcast
//! lands in later fanout.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, TernaryKind, Workspace,
};

/// Descriptor for a ternary backward op.
///
/// `scale` is used by parameterized ops (`Addcmul`, `Addcdiv`) — must
/// match the value passed to the forward [`crate::TernaryDescriptor`].
/// Ignored by unparameterized ops (`Fma`, `Clamp`); pass `1.0` for
/// those.
#[derive(Copy, Clone, Debug)]
pub struct TernaryBackwardDescriptor<const N: usize> {
    /// Which forward ternary op this is the backward of.
    pub kind: TernaryKind,
    /// Tensor shape (shared by dy / a / b / c / da / db / dc).
    pub shape: [i32; N],
    /// Element type.
    pub element: ElementKind,
    /// Scalar multiplier for parameterized ops (`Addcmul`, `Addcdiv`).
    /// Unused by `Fma` / `Clamp` — pass `1.0` for those.
    pub scale: f32,
}

/// Args bundle for a ternary backward launch.
///
/// All three saved inputs are required for every wired op. The
/// kernel reads all four (dy, a, b, c) per cell; ops where one
/// save is algebraically unused (Fma's `c`, Addcmul/Addcdiv's `a`)
/// still read it for ABI uniformity.
pub struct TernaryBackwardArgs<'a, T: Element, const N: usize> {
    /// Upstream gradient (input to backward).
    pub dy: TensorRef<'a, T, N>,
    /// Saved forward input `a`.
    pub a: TensorRef<'a, T, N>,
    /// Saved forward input `b`.
    pub b: TensorRef<'a, T, N>,
    /// Saved forward input `c`.
    pub c: TensorRef<'a, T, N>,
    /// Gradient w.r.t. `a`.
    pub da: TensorMut<'a, T, N>,
    /// Gradient w.r.t. `b`.
    pub db: TensorMut<'a, T, N>,
    /// Gradient w.r.t. `c`.
    pub dc: TensorMut<'a, T, N>,
}

/// Ternary backward plan.
pub struct TernaryBackwardPlan<T: Element, const N: usize> {
    desc: TernaryBackwardDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> TernaryBackwardPlan<T, N> {
    /// Pick a kernel for `desc`. Returns [`Error::Unsupported`] for
    /// `Where` (which requires a heterogeneous-dtype plan) or for any
    /// `(kind, T::KIND)` cell outside the wired matrix.
    pub fn select(
        _stream: &Stream,
        desc: &TernaryBackwardDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::TernaryBackwardPlan: descriptor element != type parameter T",
            ));
        }
        for &d in desc.shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::TernaryBackwardPlan: shape dims must be non-negative",
                ));
            }
        }
        if matches!(desc.kind, TernaryKind::Where) {
            return Err(Error::Unsupported(
                "baracuda-kernels::TernaryBackwardPlan: `Where` backward needs a \
                 heterogeneous-dtype plan shape (cond is u8, value tensors are T) — \
                 it will land as a separate `WhereBackwardPlan` in a future session; \
                 this plan only handles the homogeneous-dtype ternary family \
                 (Fma, Clamp, Addcmul, Addcdiv).",
            ));
        }
        let kind_in_scope = matches!(
            desc.kind,
            TernaryKind::Fma | TernaryKind::Clamp | TernaryKind::Addcmul | TernaryKind::Addcdiv
        );
        let dtype_in_scope = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !(kind_in_scope && dtype_in_scope) {
            return Err(Error::Unsupported(
                "baracuda-kernels::TernaryBackwardPlan: this (kind, dtype) cell is not \
                 wired today — the trailblazer covers {Fma, Clamp, Addcmul, Addcdiv} × \
                 {f32, f16, bf16, f64}. Integer / other dtype cells land in later fanout.",
            ));
        }

        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::TernaryElementwise,
            // Use the forward op discriminant. Backward is implied by
            // the plan type itself (TernaryBackwardPlan vs TernaryPlan).
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

    /// Validate args. Trailblazer constraints: contig-only, same shape
    /// across all 7 tensors (dy / a / b / c / da / db / dc).
    pub fn can_implement(&self, args: &TernaryBackwardArgs<'_, T, N>) -> Result<()> {
        if args.dy.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::TernaryBackwardPlan: dy shape mismatch",
            ));
        }
        if args.a.shape != self.desc.shape
            || args.b.shape != self.desc.shape
            || args.c.shape != self.desc.shape
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::TernaryBackwardPlan: saved a/b/c shape mismatch",
            ));
        }
        if args.da.shape != self.desc.shape
            || args.db.shape != self.desc.shape
            || args.dc.shape != self.desc.shape
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::TernaryBackwardPlan: da/db/dc shape mismatch",
            ));
        }
        if !args.dy.is_contiguous()
            || !args.a.is_contiguous()
            || !args.b.is_contiguous()
            || !args.c.is_contiguous()
            || !args.da.is_contiguous()
            || !args.db.is_contiguous()
            || !args.dc.is_contiguous()
        {
            return Err(Error::Unsupported(
                "baracuda-kernels::TernaryBackwardPlan: trailblazer requires contiguous \
                 dy / a / b / c / da / db / dc; strided fanout lands later",
            ));
        }
        let numel = args.dy.numel();
        let needed = numel as usize;
        let lens = [
            args.dy.data.len(),
            args.a.data.len(),
            args.b.data.len(),
            args.c.data.len(),
            args.da.data.len(),
            args.db.data.len(),
            args.dc.data.len(),
        ];
        if let Some(&min_len) = lens.iter().min() {
            if min_len < needed {
                return Err(Error::BufferTooSmall {
                    needed,
                    got: min_len,
                });
            }
        }
        Ok(())
    }

    /// Workspace size in bytes. Always `0` for the trailblazer.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }
    /// Kernel SKU identity.
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
        args: TernaryBackwardArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.dy.numel();
        if numel == 0 {
            return Ok(());
        }
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let a_ptr = args.a.data.as_raw().0 as *const c_void;
        let b_ptr = args.b.data.as_raw().0 as *const c_void;
        let c_ptr = args.c.data.as_raw().0 as *const c_void;
        let da_ptr = args.da.data.as_raw().0 as *mut c_void;
        let db_ptr = args.db.data.as_raw().0 as *mut c_void;
        let dc_ptr = args.dc.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let scale = self.desc.scale;

        let status = match (self.desc.kind, T::KIND) {
            // --- Fma backward (unscaled) -----------------------------
            (TernaryKind::Fma, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_fma_backward_f32_run(
                    numel, dy_ptr, a_ptr, b_ptr, c_ptr, da_ptr, db_ptr, dc_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (TernaryKind::Fma, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_fma_backward_f16_run(
                    numel, dy_ptr, a_ptr, b_ptr, c_ptr, da_ptr, db_ptr, dc_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (TernaryKind::Fma, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_fma_backward_bf16_run(
                    numel, dy_ptr, a_ptr, b_ptr, c_ptr, da_ptr, db_ptr, dc_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (TernaryKind::Fma, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_fma_backward_f64_run(
                    numel, dy_ptr, a_ptr, b_ptr, c_ptr, da_ptr, db_ptr, dc_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // --- Clamp backward (unscaled, mask × dy) ----------------
            (TernaryKind::Clamp, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_clamp_backward_f32_run(
                    numel, dy_ptr, a_ptr, b_ptr, c_ptr, da_ptr, db_ptr, dc_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (TernaryKind::Clamp, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_clamp_backward_f16_run(
                    numel, dy_ptr, a_ptr, b_ptr, c_ptr, da_ptr, db_ptr, dc_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (TernaryKind::Clamp, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_clamp_backward_bf16_run(
                    numel, dy_ptr, a_ptr, b_ptr, c_ptr, da_ptr, db_ptr, dc_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (TernaryKind::Clamp, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_clamp_backward_f64_run(
                    numel, dy_ptr, a_ptr, b_ptr, c_ptr, da_ptr, db_ptr, dc_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // --- Addcmul backward (scaled) ----------------------------
            (TernaryKind::Addcmul, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_addcmul_backward_f32_run(
                    numel, dy_ptr, a_ptr, b_ptr, c_ptr, da_ptr, db_ptr, dc_ptr, scale,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (TernaryKind::Addcmul, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_addcmul_backward_f16_run(
                    numel, dy_ptr, a_ptr, b_ptr, c_ptr, da_ptr, db_ptr, dc_ptr, scale,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (TernaryKind::Addcmul, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_addcmul_backward_bf16_run(
                    numel, dy_ptr, a_ptr, b_ptr, c_ptr, da_ptr, db_ptr, dc_ptr, scale,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (TernaryKind::Addcmul, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_addcmul_backward_f64_run(
                    numel, dy_ptr, a_ptr, b_ptr, c_ptr, da_ptr, db_ptr, dc_ptr, scale,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // --- Addcdiv backward (scaled) ----------------------------
            (TernaryKind::Addcdiv, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_addcdiv_backward_f32_run(
                    numel, dy_ptr, a_ptr, b_ptr, c_ptr, da_ptr, db_ptr, dc_ptr, scale,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (TernaryKind::Addcdiv, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_addcdiv_backward_f16_run(
                    numel, dy_ptr, a_ptr, b_ptr, c_ptr, da_ptr, db_ptr, dc_ptr, scale,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (TernaryKind::Addcdiv, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_addcdiv_backward_bf16_run(
                    numel, dy_ptr, a_ptr, b_ptr, c_ptr, da_ptr, db_ptr, dc_ptr, scale,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (TernaryKind::Addcdiv, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_addcdiv_backward_f64_run(
                    numel, dy_ptr, a_ptr, b_ptr, c_ptr, da_ptr, db_ptr, dc_ptr, scale,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::TernaryBackwardPlan::run reached an \
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
