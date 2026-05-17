//! Batched-QR dense `Q` / `R` materialization (Milestone 6.14, Piece 2).
//!
//! Consumes the cuBLAS `geqrfBatched` packed output produced by
//! [`super::qr_batched::BatchedQrPlan::run`] and writes out dense
//! per-slot `Q [B, M, M]` and `R [B, K, N]` tensors. `K = min(M, N)`.
//!
//! Two small bespoke kernels do the work:
//!
//! 1. **Upper-triangle copy → R**: per-cell `R[b, i, j] = A_packed[b, i, j]`
//!    if `i ≤ j` else `0`. Pure copy; no arithmetic.
//!
//! 2. **Identity stage → Q**: writes a per-slot `M × M` identity into
//!    the `Q` output buffer. The plan then chains
//!    [`super::ormqr_batched::BatchedOrmqrPlan`] (Side = Left, Op = N)
//!    to overwrite `Q` in place with the dense Q matrix encoded by
//!    the Householder reflectors in `A_packed` + `tau`.
//!
//! Convention: column-major end-to-end, matching the rest of the
//! linalg family. Dtypes: `f32` + `f64` only (inherited from
//! `BatchedQrPlan` and `BatchedOrmqrPlan`).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_sys::{
    baracuda_kernels_batched_qr_materialize_identity_f32_run,
    baracuda_kernels_batched_qr_materialize_identity_f64_run,
    baracuda_kernels_batched_qr_materialize_r_f32_run,
    baracuda_kernels_batched_qr_materialize_r_f64_run,
};
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, LinalgKind, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::ormqr_batched::{
    BatchedOrmqrArgs, BatchedOrmqrDescriptor, BatchedOrmqrOp, BatchedOrmqrPlan, BatchedOrmqrSide,
};

/// Descriptor for batched-QR dense `Q` / `R` materialization.
#[derive(Copy, Clone, Debug)]
pub struct BatchedQrMaterializeDescriptor {
    /// Row count `M` of each input matrix (== rows of `Q`).
    pub m: i32,
    /// Column count `N` of each input matrix (== cols of `R`).
    pub n: i32,
    /// Number of independent batch slots.
    pub batch_size: i32,
    /// Element type. Must be `F32` or `F64`.
    pub element: ElementKind,
}

/// Args bundle for `BatchedQrMaterializePlan::run`.
///
/// `a_packed` and `tau` are the *unmodified* outputs of
/// [`super::qr_batched::BatchedQrPlan::run`]. `q` and `r` are caller-
/// owned destination buffers; both are written by the plan.
pub struct BatchedQrMaterializeArgs<'a, T: Element> {
    /// `geqrf`-packed input from `BatchedQrPlan`: `[batch, M, N]`
    /// column-major.
    pub a_packed: TensorRef<'a, T, 3>,
    /// `geqrf` Householder scalars: `[batch, K]` where `K = min(M, N)`.
    pub tau: TensorRef<'a, T, 2>,
    /// Dense `Q` output: `[batch, M, M]` column-major. Overwritten.
    pub q: TensorMut<'a, T, 3>,
    /// Dense `R` output: `[batch, K, N]` column-major. Strict lower
    /// triangle is zeroed by the plan; the upper triangle is the
    /// non-zero `R` factor.
    pub r: TensorMut<'a, T, 3>,
}

/// Batched-QR dense Q/R materialization plan.
pub struct BatchedQrMaterializePlan<T: Element> {
    desc: BatchedQrMaterializeDescriptor,
    sku: KernelSku,
    // The chained ormqr plan is built lazily on first `run`. We hold an
    // `Option<>` so we can re-use it across launches (it's
    // configuration-only — no GPU resources).
    ormqr: core::cell::OnceCell<BatchedOrmqrPlan<T>>,
    _marker: PhantomData<T>,
}

impl<T: Element> BatchedQrMaterializePlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &BatchedQrMaterializeDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::BatchedQrMaterializePlan: descriptor.element != T::KIND",
            ));
        }
        if !matches!(T::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::BatchedQrMaterializePlan: bespoke kernel wired for f32 + f64 only",
            ));
        }
        if desc.m <= 0 || desc.n <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchedQrMaterializePlan: M, N must be > 0",
            ));
        }
        if desc.batch_size <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchedQrMaterializePlan: batch_size must be > 0",
            ));
        }
        if desc.m < desc.n {
            return Err(Error::Unsupported(
                "baracuda-kernels::BatchedQrMaterializePlan: cuBLAS geqrfBatched requires M >= N",
            ));
        }

        let math_precision = match T::KIND {
            ElementKind::F64 => MathPrecision::F64,
            _ => MathPrecision::F32,
        };
        let precision_guarantee = PrecisionGuarantee {
            math_precision,
            accumulator: T::KIND,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Linalg,
            op: LinalgKind::BatchedQrMaterialize as u16,
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
            ormqr: core::cell::OnceCell::new(),
            _marker: PhantomData,
        })
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

    /// Workspace size in bytes — zero. The chained `BatchedOrmqrPlan`
    /// is itself workspace-free.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Workspace requirement is fixed at zero; reported for cross-plan
    /// API uniformity.
    pub fn query_workspace_size(&self, _stream: &Stream) -> Result<usize> {
        Ok(0)
    }

    fn check_args(&self, args: &BatchedQrMaterializeArgs<'_, T>) -> Result<()> {
        let b = self.desc.batch_size;
        let m = self.desc.m;
        let n = self.desc.n;
        let k = m.min(n);
        if args.a_packed.shape != [b, m, n] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchedQrMaterializePlan: A_packed shape != [batch, M, N]",
            ));
        }
        if args.tau.shape != [b, k] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchedQrMaterializePlan: tau shape != [batch, min(M, N)]",
            ));
        }
        if args.q.shape != [b, m, m] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchedQrMaterializePlan: Q shape != [batch, M, M]",
            ));
        }
        if args.r.shape != [b, k, n] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchedQrMaterializePlan: R shape != [batch, min(M, N), N]",
            ));
        }
        Ok(())
    }

    fn ensure_ormqr(&self, stream: &Stream) -> Result<&BatchedOrmqrPlan<T>> {
        if let Some(p) = self.ormqr.get() {
            return Ok(p);
        }
        let m = self.desc.m;
        let n = self.desc.n;
        let k = m.min(n);
        let desc = BatchedOrmqrDescriptor {
            m,
            n: m,                       // C is the M×M identity → output Q is M×M
            k,
            batch_size: self.desc.batch_size,
            side: BatchedOrmqrSide::Left,
            op: BatchedOrmqrOp::N,
            element: self.desc.element,
        };
        let plan = BatchedOrmqrPlan::<T>::select(stream, &desc, PlanPreference::default())?;
        // OnceCell::set is a no-op on the second concurrent winner, but
        // we serialize plans by holding `&self` exclusively across `run`
        // calls; either branch yields the same plan reference back.
        let _ = self.ormqr.set(plan);
        Ok(self.ormqr.get().expect("ormqr just set"))
    }

    /// Run the materialization pipeline:
    ///
    /// 1. Copy upper triangle of `A_packed` → `R` (bespoke kernel).
    /// 2. Stage an identity into `Q` (bespoke kernel).
    /// 3. Apply Q to the identity via `BatchedOrmqrPlan` (Left, op=N).
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: BatchedQrMaterializeArgs<'_, T>,
    ) -> Result<()> {
        self.check_args(&args)?;
        let b = self.desc.batch_size;
        let m = self.desc.m;
        let n = self.desc.n;
        let k = m.min(n);
        let stream_ptr = stream.as_raw() as *mut c_void;

        // ----- Step 1: upper-triangle copy → R ---------------------------
        let a_ptr = args.a_packed.data.as_raw().0 as *const c_void;
        let r_ptr = args.r.data.as_raw().0 as *mut c_void;
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_batched_qr_materialize_r_f32_run(
                    b, m, n, k, a_ptr, r_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_batched_qr_materialize_r_f64_run(
                    b, m, n, k, a_ptr, r_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::BatchedQrMaterializePlan::run reached an unimplemented dtype",
                ));
            }
        };
        map_status(status)?;

        // ----- Step 2: stage identity into Q -----------------------------
        let q_ptr = args.q.data.as_raw().0 as *mut c_void;
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_batched_qr_materialize_identity_f32_run(
                    b, m, q_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_batched_qr_materialize_identity_f64_run(
                    b, m, q_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => unreachable!(),
        };
        map_status(status)?;

        // ----- Step 3: apply Q via batched-ormqr -------------------------
        // `Q` (the destination, currently the identity) plays the role
        // of `C` in the ormqr call: C := Q · I = Q. ormqr's `c` arg
        // is in-place, so this writes back into our Q buffer.
        let ormqr = self.ensure_ormqr(stream)?;
        let ormqr_args = BatchedOrmqrArgs::<T> {
            a_packed: args.a_packed,
            tau: args.tau,
            c: args.q,
        };
        ormqr.run(stream, Workspace::None, ormqr_args)
    }
}

/// Status-code → Result translation. Mirrors the helper in
/// [`super::ormqr_batched`] — duplicated locally so each module keeps
/// its own private helper rather than re-exporting an internal item.
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
