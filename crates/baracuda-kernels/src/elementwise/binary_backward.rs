//! Backward plan for the binary elementwise family.
//!
//! Sibling of [`crate::BinaryPlan`] for gradient computation:
//! `(da, db) = backward(dy, [saved tensors per op])`.
//!
//! Today wired: `{Add, Sub, Mul, Div, Maximum, Minimum} × {f32, f16, bf16, f64}`.
//! Add and Sub need no saved tensors; Mul, Div, Maximum, Minimum require
//! the saved forward inputs `a` and `b`:
//! - Add: `(da, db) = (dy, dy)` — no saved
//! - Sub: `(da, db) = (dy, -dy)` — no saved
//! - Mul: `(da, db) = (dy * b, dy * a)` — needs saved `a`, `b`
//! - Div: `(da, db) = (dy / b, -dy * a / b²)` — needs saved `a`, `b`
//! - Maximum / Minimum: saves used purely as comparison references; tie
//!   splits `dy` evenly (PyTorch parity). For Maximum:
//!   `da = where(a==b, dy/2, where(a<b, 0, dy))`,
//!   `db = where(a==b, dy/2, where(b<a, 0, dy))`. Minimum flips `<`/`>`.
//!   NaN inputs propagate `dy` to both (all comparisons false).
//!
//! The `Args` struct carries `a` and `b` as `Option<TensorRef>` so
//! callers omit them for ops that don't need them. The dispatcher
//! validates that needed saves are present.
//!
//! Trailblazer constraints (same shape limits as the forward
//! trailblazer): contig-only (no broadcasting); `dy.shape ==
//! da.shape == db.shape`. Ops with saves additionally require
//! `a.shape == b.shape == dy.shape`.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, BinaryKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a binary backward op.
#[derive(Copy, Clone, Debug)]
pub struct BinaryBackwardDescriptor<const N: usize> {
    /// Which forward binary op this is the backward of.
    pub kind: BinaryKind,
    /// Tensor shape (shared by dy / a / b / da / db).
    pub shape: [i32; N],
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a binary backward launch.
///
/// `a` and `b` are SAVED forward inputs — required by `Mul`, `Div`,
/// `Maximum`, `Minimum` (gradient formula references them) but unused
/// by `Add` / `Sub`. The dispatcher checks that ops needing saves have
/// them supplied.
pub struct BinaryBackwardArgs<'a, T: Element, const N: usize> {
    /// Upstream gradient (input to backward).
    pub dy: TensorRef<'a, T, N>,
    /// Saved forward input `a`. Required by `Mul` / `Div`; ignored otherwise.
    pub a: Option<TensorRef<'a, T, N>>,
    /// Saved forward input `b`. Required by `Mul` / `Div`; ignored otherwise.
    pub b: Option<TensorRef<'a, T, N>>,
    /// Gradient w.r.t. `a`.
    pub da: TensorMut<'a, T, N>,
    /// Gradient w.r.t. `b`.
    pub db: TensorMut<'a, T, N>,
}

/// Binary backward plan.
pub struct BinaryBackwardPlan<T: Element, const N: usize> {
    desc: BinaryBackwardDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

#[inline]
fn op_needs_saves(kind: BinaryKind) -> bool {
    matches!(
        kind,
        BinaryKind::Mul
            | BinaryKind::Div
            | BinaryKind::Pow
            | BinaryKind::Maximum
            | BinaryKind::Minimum
            | BinaryKind::Atan2
            | BinaryKind::Hypot
    )
}

impl<T: Element, const N: usize> BinaryBackwardPlan<T, N> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &BinaryBackwardDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::BinaryBackwardPlan: descriptor element != T",
            ));
        }
        for &d in desc.shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::BinaryBackwardPlan: shape dims must be non-negative",
                ));
            }
        }
        // Phase 3 backward family: {Add, Sub, Mul, Div, Maximum, Minimum} ×
        // {f32, f16, bf16, f64}.
        let supported = matches!(
            (desc.kind, T::KIND),
            (BinaryKind::Add, ElementKind::F32)
                | (BinaryKind::Add, ElementKind::F16)
                | (BinaryKind::Add, ElementKind::Bf16)
                | (BinaryKind::Add, ElementKind::F64)
                | (BinaryKind::Sub, ElementKind::F32)
                | (BinaryKind::Sub, ElementKind::F16)
                | (BinaryKind::Sub, ElementKind::Bf16)
                | (BinaryKind::Sub, ElementKind::F64)
                | (BinaryKind::Mul, ElementKind::F32)
                | (BinaryKind::Mul, ElementKind::F16)
                | (BinaryKind::Mul, ElementKind::Bf16)
                | (BinaryKind::Mul, ElementKind::F64)
                | (BinaryKind::Div, ElementKind::F32)
                | (BinaryKind::Div, ElementKind::F16)
                | (BinaryKind::Div, ElementKind::Bf16)
                | (BinaryKind::Div, ElementKind::F64)
                | (BinaryKind::Maximum, ElementKind::F32)
                | (BinaryKind::Maximum, ElementKind::F16)
                | (BinaryKind::Maximum, ElementKind::Bf16)
                | (BinaryKind::Maximum, ElementKind::F64)
                | (BinaryKind::Minimum, ElementKind::F32)
                | (BinaryKind::Minimum, ElementKind::F16)
                | (BinaryKind::Minimum, ElementKind::Bf16)
                | (BinaryKind::Minimum, ElementKind::F64)
                | (BinaryKind::Pow, ElementKind::F32)
                | (BinaryKind::Pow, ElementKind::F16)
                | (BinaryKind::Pow, ElementKind::Bf16)
                | (BinaryKind::Pow, ElementKind::F64)
                | (BinaryKind::Atan2, ElementKind::F32)
                | (BinaryKind::Atan2, ElementKind::F16)
                | (BinaryKind::Atan2, ElementKind::Bf16)
                | (BinaryKind::Atan2, ElementKind::F64)
                | (BinaryKind::Hypot, ElementKind::F32)
                | (BinaryKind::Hypot, ElementKind::F16)
                | (BinaryKind::Hypot, ElementKind::Bf16)
                | (BinaryKind::Hypot, ElementKind::F64)
        );
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::BinaryBackwardPlan: only \
                 `{Add,Sub,Mul,Div,Maximum,Minimum,Pow,Atan2,Hypot}` × \
                 `{f32, f16, bf16, f64}` are wired today; other (kind, dtype) \
                 pairs (e.g. integer family, Lerp) land in later fanout. Lerp \
                 is reserved-but-deferred pending a parameterized-binary plan \
                 shape.",
            ));
        }

        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::BinaryElementwise,
            // Use the forward op discriminant. Backward is implied by
            // the plan type itself (BinaryBackwardPlan vs BinaryPlan).
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
    pub fn can_implement(&self, args: &BinaryBackwardArgs<'_, T, N>) -> Result<()> {
        if args.dy.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BinaryBackwardPlan: dy shape mismatch",
            ));
        }
        if args.da.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BinaryBackwardPlan: da shape mismatch",
            ));
        }
        if args.db.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BinaryBackwardPlan: db shape mismatch",
            ));
        }
        // Contig-only for trailblazer.
        if !args.dy.is_contiguous() || !args.da.is_contiguous() || !args.db.is_contiguous() {
            return Err(Error::Unsupported(
                "baracuda-kernels::BinaryBackwardPlan: trailblazer requires contiguous \
                 dy / da / db; strided fanout lands later",
            ));
        }
        // Per-op saved-tensor requirements.
        if op_needs_saves(self.desc.kind) {
            let a = args.a.as_ref().ok_or(Error::InvalidProblem(
                "baracuda-kernels::BinaryBackwardPlan: this op requires saved input `a`",
            ))?;
            let b = args.b.as_ref().ok_or(Error::InvalidProblem(
                "baracuda-kernels::BinaryBackwardPlan: this op requires saved input `b`",
            ))?;
            if a.shape != self.desc.shape {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::BinaryBackwardPlan: saved a shape mismatch",
                ));
            }
            if b.shape != self.desc.shape {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::BinaryBackwardPlan: saved b shape mismatch",
                ));
            }
            if !a.is_contiguous() || !b.is_contiguous() {
                return Err(Error::Unsupported(
                    "baracuda-kernels::BinaryBackwardPlan: saved a/b must be contiguous \
                     (strided fanout lands later)",
                ));
            }
            let numel = args.dy.numel() as usize;
            if a.data.len() < numel {
                return Err(Error::BufferTooSmall {
                    needed: numel,
                    got: a.data.len(),
                });
            }
            if b.data.len() < numel {
                return Err(Error::BufferTooSmall {
                    needed: numel,
                    got: b.data.len(),
                });
            }
        }
        let numel = args.dy.numel();
        let dy_len = args.dy.data.len() as i64;
        let da_len = args.da.data.len() as i64;
        let db_len = args.db.data.len() as i64;
        if dy_len < numel || da_len < numel || db_len < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: dy_len.min(da_len).min(db_len) as usize,
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
        args: BinaryBackwardArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.dy.numel();
        if numel == 0 {
            return Ok(());
        }
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let da_ptr = args.da.data.as_raw().0 as *mut c_void;
        let db_ptr = args.db.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let status = match (self.desc.kind, T::KIND) {
            // -------- Add (no saves) --------
            (BinaryKind::Add, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_add_backward_f32_run(
                    numel, dy_ptr, da_ptr, db_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Add, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_add_backward_f16_run(
                    numel, dy_ptr, da_ptr, db_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Add, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_add_backward_bf16_run(
                    numel, dy_ptr, da_ptr, db_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Add, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_add_backward_f64_run(
                    numel, dy_ptr, da_ptr, db_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Sub (no saves) --------
            (BinaryKind::Sub, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_sub_backward_f32_run(
                    numel, dy_ptr, da_ptr, db_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Sub, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_sub_backward_f16_run(
                    numel, dy_ptr, da_ptr, db_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Sub, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_sub_backward_bf16_run(
                    numel, dy_ptr, da_ptr, db_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Sub, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_sub_backward_f64_run(
                    numel, dy_ptr, da_ptr, db_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -------- Mul (saves) --------
            (BinaryKind::Mul, ElementKind::F32) => {
                let (a_ptr, b_ptr) = saved_ptrs(&args);
                unsafe {
                    baracuda_kernels_sys::baracuda_kernels_binary_mul_backward_f32_run(
                        numel, dy_ptr, a_ptr, b_ptr, da_ptr, db_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
            }
            (BinaryKind::Mul, ElementKind::F16) => {
                let (a_ptr, b_ptr) = saved_ptrs(&args);
                unsafe {
                    baracuda_kernels_sys::baracuda_kernels_binary_mul_backward_f16_run(
                        numel, dy_ptr, a_ptr, b_ptr, da_ptr, db_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
            }
            (BinaryKind::Mul, ElementKind::Bf16) => {
                let (a_ptr, b_ptr) = saved_ptrs(&args);
                unsafe {
                    baracuda_kernels_sys::baracuda_kernels_binary_mul_backward_bf16_run(
                        numel, dy_ptr, a_ptr, b_ptr, da_ptr, db_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
            }
            (BinaryKind::Mul, ElementKind::F64) => {
                let (a_ptr, b_ptr) = saved_ptrs(&args);
                unsafe {
                    baracuda_kernels_sys::baracuda_kernels_binary_mul_backward_f64_run(
                        numel, dy_ptr, a_ptr, b_ptr, da_ptr, db_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
            }
            // -------- Div (saves) --------
            (BinaryKind::Div, ElementKind::F32) => {
                let (a_ptr, b_ptr) = saved_ptrs(&args);
                unsafe {
                    baracuda_kernels_sys::baracuda_kernels_binary_div_backward_f32_run(
                        numel, dy_ptr, a_ptr, b_ptr, da_ptr, db_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
            }
            (BinaryKind::Div, ElementKind::F16) => {
                let (a_ptr, b_ptr) = saved_ptrs(&args);
                unsafe {
                    baracuda_kernels_sys::baracuda_kernels_binary_div_backward_f16_run(
                        numel, dy_ptr, a_ptr, b_ptr, da_ptr, db_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
            }
            (BinaryKind::Div, ElementKind::Bf16) => {
                let (a_ptr, b_ptr) = saved_ptrs(&args);
                unsafe {
                    baracuda_kernels_sys::baracuda_kernels_binary_div_backward_bf16_run(
                        numel, dy_ptr, a_ptr, b_ptr, da_ptr, db_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
            }
            (BinaryKind::Div, ElementKind::F64) => {
                let (a_ptr, b_ptr) = saved_ptrs(&args);
                unsafe {
                    baracuda_kernels_sys::baracuda_kernels_binary_div_backward_f64_run(
                        numel, dy_ptr, a_ptr, b_ptr, da_ptr, db_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
            }
            // -------- Maximum (saves used as comparison references) --------
            (BinaryKind::Maximum, ElementKind::F32) => {
                let (a_ptr, b_ptr) = saved_ptrs(&args);
                unsafe {
                    baracuda_kernels_sys::baracuda_kernels_binary_maximum_backward_f32_run(
                        numel, dy_ptr, a_ptr, b_ptr, da_ptr, db_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
            }
            (BinaryKind::Maximum, ElementKind::F16) => {
                let (a_ptr, b_ptr) = saved_ptrs(&args);
                unsafe {
                    baracuda_kernels_sys::baracuda_kernels_binary_maximum_backward_f16_run(
                        numel, dy_ptr, a_ptr, b_ptr, da_ptr, db_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
            }
            (BinaryKind::Maximum, ElementKind::Bf16) => {
                let (a_ptr, b_ptr) = saved_ptrs(&args);
                unsafe {
                    baracuda_kernels_sys::baracuda_kernels_binary_maximum_backward_bf16_run(
                        numel, dy_ptr, a_ptr, b_ptr, da_ptr, db_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
            }
            (BinaryKind::Maximum, ElementKind::F64) => {
                let (a_ptr, b_ptr) = saved_ptrs(&args);
                unsafe {
                    baracuda_kernels_sys::baracuda_kernels_binary_maximum_backward_f64_run(
                        numel, dy_ptr, a_ptr, b_ptr, da_ptr, db_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
            }
            // -------- Minimum (saves used as comparison references) --------
            (BinaryKind::Minimum, ElementKind::F32) => {
                let (a_ptr, b_ptr) = saved_ptrs(&args);
                unsafe {
                    baracuda_kernels_sys::baracuda_kernels_binary_minimum_backward_f32_run(
                        numel, dy_ptr, a_ptr, b_ptr, da_ptr, db_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
            }
            (BinaryKind::Minimum, ElementKind::F16) => {
                let (a_ptr, b_ptr) = saved_ptrs(&args);
                unsafe {
                    baracuda_kernels_sys::baracuda_kernels_binary_minimum_backward_f16_run(
                        numel, dy_ptr, a_ptr, b_ptr, da_ptr, db_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
            }
            (BinaryKind::Minimum, ElementKind::Bf16) => {
                let (a_ptr, b_ptr) = saved_ptrs(&args);
                unsafe {
                    baracuda_kernels_sys::baracuda_kernels_binary_minimum_backward_bf16_run(
                        numel, dy_ptr, a_ptr, b_ptr, da_ptr, db_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
            }
            (BinaryKind::Minimum, ElementKind::F64) => {
                let (a_ptr, b_ptr) = saved_ptrs(&args);
                unsafe {
                    baracuda_kernels_sys::baracuda_kernels_binary_minimum_backward_f64_run(
                        numel, dy_ptr, a_ptr, b_ptr, da_ptr, db_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
            }
            // -------- Pow (saves) --------
            (BinaryKind::Pow, ElementKind::F32) => {
                let (a_ptr, b_ptr) = saved_ptrs(&args);
                unsafe {
                    baracuda_kernels_sys::baracuda_kernels_binary_pow_backward_f32_run(
                        numel, dy_ptr, a_ptr, b_ptr, da_ptr, db_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
            }
            (BinaryKind::Pow, ElementKind::F16) => {
                let (a_ptr, b_ptr) = saved_ptrs(&args);
                unsafe {
                    baracuda_kernels_sys::baracuda_kernels_binary_pow_backward_f16_run(
                        numel, dy_ptr, a_ptr, b_ptr, da_ptr, db_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
            }
            (BinaryKind::Pow, ElementKind::Bf16) => {
                let (a_ptr, b_ptr) = saved_ptrs(&args);
                unsafe {
                    baracuda_kernels_sys::baracuda_kernels_binary_pow_backward_bf16_run(
                        numel, dy_ptr, a_ptr, b_ptr, da_ptr, db_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
            }
            (BinaryKind::Pow, ElementKind::F64) => {
                let (a_ptr, b_ptr) = saved_ptrs(&args);
                unsafe {
                    baracuda_kernels_sys::baracuda_kernels_binary_pow_backward_f64_run(
                        numel, dy_ptr, a_ptr, b_ptr, da_ptr, db_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
            }
            // -------- Atan2 (saves) --------
            (BinaryKind::Atan2, ElementKind::F32) => {
                let (a_ptr, b_ptr) = saved_ptrs(&args);
                unsafe {
                    baracuda_kernels_sys::baracuda_kernels_binary_atan2_backward_f32_run(
                        numel, dy_ptr, a_ptr, b_ptr, da_ptr, db_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
            }
            (BinaryKind::Atan2, ElementKind::F16) => {
                let (a_ptr, b_ptr) = saved_ptrs(&args);
                unsafe {
                    baracuda_kernels_sys::baracuda_kernels_binary_atan2_backward_f16_run(
                        numel, dy_ptr, a_ptr, b_ptr, da_ptr, db_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
            }
            (BinaryKind::Atan2, ElementKind::Bf16) => {
                let (a_ptr, b_ptr) = saved_ptrs(&args);
                unsafe {
                    baracuda_kernels_sys::baracuda_kernels_binary_atan2_backward_bf16_run(
                        numel, dy_ptr, a_ptr, b_ptr, da_ptr, db_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
            }
            (BinaryKind::Atan2, ElementKind::F64) => {
                let (a_ptr, b_ptr) = saved_ptrs(&args);
                unsafe {
                    baracuda_kernels_sys::baracuda_kernels_binary_atan2_backward_f64_run(
                        numel, dy_ptr, a_ptr, b_ptr, da_ptr, db_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
            }
            // -------- Hypot (saves; y reconstructed inside kernel) --------
            (BinaryKind::Hypot, ElementKind::F32) => {
                let (a_ptr, b_ptr) = saved_ptrs(&args);
                unsafe {
                    baracuda_kernels_sys::baracuda_kernels_binary_hypot_backward_f32_run(
                        numel, dy_ptr, a_ptr, b_ptr, da_ptr, db_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
            }
            (BinaryKind::Hypot, ElementKind::F16) => {
                let (a_ptr, b_ptr) = saved_ptrs(&args);
                unsafe {
                    baracuda_kernels_sys::baracuda_kernels_binary_hypot_backward_f16_run(
                        numel, dy_ptr, a_ptr, b_ptr, da_ptr, db_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
            }
            (BinaryKind::Hypot, ElementKind::Bf16) => {
                let (a_ptr, b_ptr) = saved_ptrs(&args);
                unsafe {
                    baracuda_kernels_sys::baracuda_kernels_binary_hypot_backward_bf16_run(
                        numel, dy_ptr, a_ptr, b_ptr, da_ptr, db_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
            }
            (BinaryKind::Hypot, ElementKind::F64) => {
                let (a_ptr, b_ptr) = saved_ptrs(&args);
                unsafe {
                    baracuda_kernels_sys::baracuda_kernels_binary_hypot_backward_f64_run(
                        numel, dy_ptr, a_ptr, b_ptr, da_ptr, db_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
            }
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::BinaryBackwardPlan::run reached an unimplemented \
                     (kind, dtype) pair — select() should have caught this",
                ));
            }
        };
        map_status(status)
    }
}

#[inline]
fn saved_ptrs<T: Element, const N: usize>(
    args: &BinaryBackwardArgs<'_, T, N>,
) -> (*const c_void, *const c_void) {
    // can_implement guarantees Some for ops that reach this path.
    let a = args
        .a
        .as_ref()
        .expect("Mul/Div/Pow/Maximum/Minimum/Atan2/Hypot backward require saved a");
    let b = args
        .b
        .as_ref()
        .expect("Mul/Div/Pow/Maximum/Minimum/Atan2/Hypot backward require saved b");
    (
        a.data.as_raw().0 as *const c_void,
        b.data.as_raw().0 as *const c_void,
    )
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
