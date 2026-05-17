//! Heterogeneous-dtype ternary BW plan: `where_backward(cond, dy)`.
//!
//! Sibling of [`crate::WherePlan`] for gradient computation. Forward:
//! `y = where(cond, a, b)` with `cond: u8` and same-dtype `a`/`b`/`y`.
//! Backward (cond is non-differentiable — no `dcond`):
//!
//! - `da = where(cond, dy, 0)` — gradient flows to `a` only where cond is true.
//! - `db = where(cond, 0, dy)` — gradient flows to `b` only where cond is false.
//!
//! Per-cell formula is pure mask + copy — no arithmetic — so output is
//! bit-exact against host reference at every dtype.
//!
//! All 4 FP value dtypes wired: {f32, f16, bf16, f64}. Trailblazer
//! constraints: **contig-only** (no broadcast on `dy` / `da` / `db`).
//! `cond` carries the same heterogeneous-dtype convention as the FW
//! (`u8`, 0 = false). Broadcast support on BW lands later if a use case
//! materializes — the autograd reduction step usually flattens
//! broadcasted gradients upstream of this kernel anyway, so the
//! contig-only trailblazer matches typical caller pipelines.
//!
//! Module name: `where_backward` (the FW lives in `where_op` to dodge
//! Rust's `where` keyword; the BW name is safe because the suffix
//! breaks the keyword match).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a `where_backward` op.
///
/// `shape` is the shared shape of `dy` / `da` / `db`. `element` is the
/// **value** dtype — cond is always `u8`. `element` must match the type
/// parameter `T` of the containing plan at `select` time.
#[derive(Copy, Clone, Debug)]
pub struct WhereBackwardDescriptor<const N: usize> {
    /// Tensor shape (shared by cond / dy / da / db).
    pub shape: [i32; N],
    /// Value element type (dy / da / db dtype; cond is always `u8`).
    pub element: ElementKind,
}

/// Args bundle for a `where_backward` launch.
///
/// `cond` is the FW mask (`u8`, 0 = false). `dy` is the upstream
/// gradient (same dtype as the FW value inputs). `da` and `db` are the
/// gradients w.r.t. the FW `a` and `b` respectively.
pub struct WhereBackwardArgs<'a, T: Element, const N: usize> {
    /// Boolean mask from the forward pass (`0u8` selected `b`,
    /// any other value selected `a`).
    pub cond: TensorRef<'a, u8, N>,
    /// Upstream gradient.
    pub dy: TensorRef<'a, T, N>,
    /// Gradient w.r.t. `a`.
    pub da: TensorMut<'a, T, N>,
    /// Gradient w.r.t. `b`.
    pub db: TensorMut<'a, T, N>,
}

/// `where_backward(cond, dy)` plan with heterogeneous-dtype inputs.
///
/// `T: Element` is the value dtype (`dy` / `da` / `db`). The cond is
/// always `u8`. `const N: usize` is the tensor rank.
pub struct WhereBackwardPlan<T: Element, const N: usize> {
    desc: WhereBackwardDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> WhereBackwardPlan<T, N> {
    /// Pick a kernel for `desc`. Returns [`Error::Unsupported`] if the
    /// value dtype isn't wired today.
    pub fn select(
        _stream: &Stream,
        desc: &WhereBackwardDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::WhereBackwardPlan: descriptor element != type parameter T",
            ));
        }
        for &d in desc.shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::WhereBackwardPlan: shape dims must be non-negative",
                ));
            }
        }

        // All 4 FP value dtypes wired.
        let supported = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::WhereBackwardPlan: value dtype must be one of \
                 {F32, F16, Bf16, F64}",
            ));
        }

        // `where_backward` is a pure mask + copy — no arithmetic —
        // fully deterministic and bit-stable on the same hardware. The
        // MathPrecision tag mirrors the value dtype by convention even
        // though no arithmetic happens.
        let (math_precision, accumulator) = match T::KIND {
            ElementKind::F16 => (MathPrecision::F16, ElementKind::F16),
            ElementKind::Bf16 => (MathPrecision::Bf16, ElementKind::Bf16),
            ElementKind::F64 => (MathPrecision::F64, ElementKind::F64),
            _ => (MathPrecision::F32, ElementKind::F32),
        };
        let precision_guarantee = PrecisionGuarantee {
            math_precision,
            accumulator,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::TernaryElementwise,
            // `op` discriminant: matches `TernaryKind::Where` (= 4).
            // BW is implied by the plan type itself
            // (`WhereBackwardPlan` vs `WherePlan`) — no separate
            // discriminant needed, mirroring the BinaryBackwardPlan
            // convention.
            op: 4,
            element: T::KIND,
            // `aux_element` would capture cond's dtype but ElementKind
            // doesn't carry a `U8` variant today — rely on the
            // `Where`-specific op discriminant for telemetry / cache
            // disambiguation, same as the FW.
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
    pub fn can_implement(&self, args: &WhereBackwardArgs<'_, T, N>) -> Result<()> {
        if args.dy.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::WhereBackwardPlan: dy shape mismatch with descriptor",
            ));
        }
        if args.da.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::WhereBackwardPlan: da shape mismatch with descriptor",
            ));
        }
        if args.db.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::WhereBackwardPlan: db shape mismatch with descriptor",
            ));
        }
        if args.cond.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::WhereBackwardPlan: cond shape mismatch with descriptor \
                 (trailblazer requires full-shape cond; stride-0 broadcasting on cond \
                 lands later)",
            ));
        }

        // Contig-only for trailblazer (cond included — no stride-0 axes).
        if !args.cond.is_contiguous()
            || !args.dy.is_contiguous()
            || !args.da.is_contiguous()
            || !args.db.is_contiguous()
        {
            return Err(Error::Unsupported(
                "baracuda-kernels::WhereBackwardPlan: trailblazer requires contiguous \
                 cond / dy / da / db; strided / broadcast fanout lands later",
            ));
        }

        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::WhereBackwardPlan: tensor rank > 8 not supported",
            ));
        }

        let numel = args.dy.numel();
        let cond_len = args.cond.data.len() as i64;
        let dy_len = args.dy.data.len() as i64;
        let da_len = args.da.data.len() as i64;
        let db_len = args.db.data.len() as i64;
        if dy_len < numel || da_len < numel || db_len < numel || cond_len < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: cond_len.min(dy_len).min(da_len).min(db_len) as usize,
            });
        }
        Ok(())
    }

    /// Workspace size in bytes. Always `0` for the trailblazer.
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
        args: WhereBackwardArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.dy.numel();
        if numel == 0 {
            return Ok(());
        }
        let cond_ptr = args.cond.data.as_raw().0 as *const c_void;
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let da_ptr = args.da.data.as_raw().0 as *mut c_void;
        let db_ptr = args.db.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_where_backward_f32_run(
                    numel,
                    cond_ptr,
                    dy_ptr,
                    da_ptr,
                    db_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_where_backward_f16_run(
                    numel,
                    cond_ptr,
                    dy_ptr,
                    da_ptr,
                    db_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_where_backward_bf16_run(
                    numel,
                    cond_ptr,
                    dy_ptr,
                    da_ptr,
                    db_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_where_backward_f64_run(
                    numel,
                    cond_ptr,
                    dy_ptr,
                    da_ptr,
                    db_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::WhereBackwardPlan::run reached an unimplemented \
                     dtype — select() should have caught this",
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
