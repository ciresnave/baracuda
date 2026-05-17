//! Batched-`ormqr` / `unmqr` — apply Householder-encoded `Q`, `Q^T`,
//! or `Q^H` from a [`super::qr_batched::BatchedQrPlan`] packed output
//! to a stack of right-hand-side matrices, fusing all batch slots into
//! one CUDA launch.
//!
//! cuSOLVER's dense `ormqr` / `unmqr` are non-batched (one launch per
//! slot), so for the small-matrix regime — exactly the regime where
//! batched-QR is most useful — launch latency dominates. This bespoke
//! kernel does one launch for the whole batch: `gridDim.x = batch_size`,
//! threads in each block cooperate on the per-reflector projection /
//! outer-product update.
//!
//! **Scope (Milestone 6.18)**:
//! - `side ∈ {Left, Right}` — `C := op(Q) · C` or `C := C · op(Q)`.
//!   For Side = Right, the packed input is square `[B, N, N]` and
//!   `K = N` (`Q` is now `N × N`).
//! - `op ∈ {N, T, C}`. `T` is real-only (plain transpose); `C` is
//!   the conjugate-transpose variant for complex dtypes. Complex +
//!   `op = T` is rejected as mathematically unusual for Householder.
//! - `dtype ∈ {f32, f64, Complex32, Complex64}`. The real dtypes match
//!   [`super::qr_batched`] coverage; the complex dtypes accept inputs
//!   produced by non-batched `cusolverDn{C,Z}geqrf` looped per slot
//!   (Milestone 6.14's `BatchedQrPlan` does not yet ship complex).
//!
//! **Workspace**: zero — the per-reflector projection coefficient
//! vector (length N for Side = Left, M for Side = Right) lives in
//! dynamic shared memory inside the kernel.
//!
//! **Column-major end-to-end**, matching the rest of the linalg
//! family.
//!
//! See the kernel header
//! [`crates/baracuda-kernels-sys/kernels/include/baracuda_batched_ormqr.cuh`]
//! for the algorithm details (block-stride sum reduction → rank-1
//! update, GEMV-rates not GEMM-rates; WY blocking is a future
//! milestone).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_sys::{
    baracuda_kernels_batched_ormqr_complex32_run, baracuda_kernels_batched_ormqr_complex64_run,
    baracuda_kernels_batched_ormqr_f32_run, baracuda_kernels_batched_ormqr_f64_run,
};
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, LinalgKind, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

/// Side of the multiplication for [`BatchedOrmqrPlan`]. Both Left and
/// Right are wired as of Milestone 6.18.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum BatchedOrmqrSide {
    /// `C := op(Q) · C` — `Q` applied from the left. `Q` has shape
    /// `M × M`; the packed input has shape `[B, M, K]`.
    Left = 0,
    /// `C := C · op(Q)` — `Q` applied from the right. `Q` has shape
    /// `N × N`; the packed input has shape `[B, N, N]` and `K = N`.
    Right = 1,
}

/// Transpose / op for [`BatchedOrmqrPlan`]. All three of N, T, C are
/// wired as of Milestone 6.18, subject to dtype constraints:
/// - `T` is real-only (LAPACK contract).
/// - `C` is complex-only (it equals `T` for real, so the real path
///   spells that intent as `T`).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum BatchedOrmqrOp {
    /// Apply `Q` (no transpose).
    N = 0,
    /// Apply `Q^T` (plain transpose). Real-dtype only — for complex
    /// types use [`Self::C`] (conjugate transpose) instead, which is
    /// the meaningful adjoint for unitary `Q`.
    T = 1,
    /// Apply `Q^H` (conjugate transpose). Complex-dtype only — for
    /// real types use [`Self::T`] (it's equivalent and avoids
    /// dispatch ambiguity).
    C = 2,
}

/// Descriptor for a batched-`ormqr` / `unmqr` op.
#[derive(Copy, Clone, Debug)]
pub struct BatchedOrmqrDescriptor {
    /// Row count `M` of each `C` matrix.
    pub m: i32,
    /// Column count `N` of each `C` matrix.
    pub n: i32,
    /// Number of Householder reflectors `K` in each `A_packed`. For
    /// Side = Left, `K = min(M, N_A)` from the originating QR and
    /// must satisfy `0 ≤ K ≤ M`. For Side = Right, `K = N` (the
    /// packed `Q` is `N × N`).
    pub k: i32,
    /// Number of independent slots in the batch.
    pub batch_size: i32,
    /// Side of the multiplication.
    pub side: BatchedOrmqrSide,
    /// Op tag — [`BatchedOrmqrOp::N`] (apply `Q`),
    /// [`BatchedOrmqrOp::T`] (apply `Q^T`, real only), or
    /// [`BatchedOrmqrOp::C`] (apply `Q^H`, complex only).
    pub op: BatchedOrmqrOp,
    /// Element type. Must be one of `F32`, `F64`, `Complex32`, or
    /// `Complex64`.
    pub element: ElementKind,
}

/// Args bundle for a batched-`ormqr` / `unmqr` launch.
///
/// `a_packed` and `tau` are the *unmodified* outputs of
/// [`super::qr_batched::BatchedQrPlan::run`] (for real dtypes) or of
/// per-slot `cusolverDn{C,Z}geqrf` (for complex dtypes); `c` is the
/// right-hand side stack, **overwritten in place** with the result.
///
/// Shape semantics depend on Side:
/// - Side = Left: `a_packed: [batch, M, K]`, `tau: [batch, K]`.
/// - Side = Right: `a_packed: [batch, N, N]`, `tau: [batch, N]`.
pub struct BatchedOrmqrArgs<'a, T: Element> {
    /// `geqrf`-packed input. Strict lower triangle holds the
    /// Householder reflectors; the upper triangle is `R` and is not
    /// read by this op. Shape `[batch, M, K]` for Side = Left,
    /// `[batch, N, N]` for Side = Right.
    pub a_packed: TensorRef<'a, T, 3>,
    /// `geqrf` Householder scalars: `[batch, K]`.
    pub tau: TensorRef<'a, T, 2>,
    /// Right-hand-side matrix stack `[batch, M, N]` column-major.
    /// Overwritten in place with the result.
    pub c: TensorMut<'a, T, 3>,
}

/// Batched-`ormqr` plan.
pub struct BatchedOrmqrPlan<T: Element> {
    desc: BatchedOrmqrDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> BatchedOrmqrPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &BatchedOrmqrDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::BatchedOrmqrPlan: descriptor.element != T::KIND",
            ));
        }
        let is_real = matches!(T::KIND, ElementKind::F32 | ElementKind::F64);
        let is_complex = matches!(T::KIND, ElementKind::Complex32 | ElementKind::Complex64);
        if !(is_real || is_complex) {
            return Err(Error::Unsupported(
                "baracuda-kernels::BatchedOrmqrPlan: dtype must be one of \
                 {f32, f64, Complex32, Complex64}",
            ));
        }
        // Op × dtype gating: T is real-only (LAPACK convention), C is complex-only.
        match (desc.op, is_complex) {
            (BatchedOrmqrOp::T, true) => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::BatchedOrmqrPlan: op = T (plain transpose) is \
                     real-only; use op = C (conjugate transpose) for complex dtypes",
                ));
            }
            (BatchedOrmqrOp::C, false) => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::BatchedOrmqrPlan: op = C (conjugate transpose) is \
                     complex-only; use op = T for real dtypes",
                ));
            }
            _ => {}
        }
        if desc.m <= 0 || desc.n <= 0 || desc.k <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchedOrmqrPlan: M, N, K must be > 0",
            ));
        }
        if desc.batch_size <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchedOrmqrPlan: batch_size must be > 0",
            ));
        }
        match desc.side {
            BatchedOrmqrSide::Left => {
                if desc.k > desc.m {
                    return Err(Error::InvalidProblem(
                        "baracuda-kernels::BatchedOrmqrPlan: side = Left requires K <= M \
                         (LAPACK ormqr/unmqr contract: Q is M × M)",
                    ));
                }
            }
            BatchedOrmqrSide::Right => {
                if desc.k != desc.n {
                    return Err(Error::InvalidProblem(
                        "baracuda-kernels::BatchedOrmqrPlan: side = Right requires K == N \
                         (LAPACK ormqr/unmqr contract: Q is N × N, K = N reflectors)",
                    ));
                }
            }
        }

        let math_precision = match T::KIND {
            ElementKind::F64 | ElementKind::Complex64 => MathPrecision::F64,
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
            op: LinalgKind::BatchedOrmqr as u16,
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

    /// Workspace size in bytes — always zero (the per-reflector
    /// projection vector lives in dynamic shared memory in the kernel).
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Workspace requirement is fixed at zero; reported through the
    /// cross-plan `query_workspace_size` helper for API uniformity.
    pub fn query_workspace_size(&self, _stream: &Stream) -> Result<usize> {
        Ok(0)
    }

    fn check_args(&self, args: &BatchedOrmqrArgs<'_, T>) -> Result<()> {
        let b = self.desc.batch_size;
        let m = self.desc.m;
        let n = self.desc.n;
        let k = self.desc.k;
        // a_packed shape varies by Side: [B, M, K] for Left, [B, N, N] for Right.
        let expected_a_shape = match self.desc.side {
            BatchedOrmqrSide::Left => [b, m, k],
            BatchedOrmqrSide::Right => [b, n, n],
        };
        if args.a_packed.shape != expected_a_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchedOrmqrPlan: A_packed shape mismatch (Left expects \
                 [batch, M, K]; Right expects [batch, N, N])",
            ));
        }
        if args.tau.shape != [b, k] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchedOrmqrPlan: tau shape != [batch, K]",
            ));
        }
        if args.c.shape != [b, m, n] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchedOrmqrPlan: C shape != [batch, M, N]",
            ));
        }
        Ok(())
    }

    /// Run the batched-`ormqr`.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: BatchedOrmqrArgs<'_, T>,
    ) -> Result<()> {
        self.check_args(&args)?;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let a_ptr = args.a_packed.data.as_raw().0 as *const c_void;
        let tau_ptr = args.tau.data.as_raw().0 as *const c_void;
        let c_ptr = args.c.data.as_raw().0 as *mut c_void;
        // Discriminant values for the kernel-side BARACUDA_ORMQR_SIDE_*
        // and BARACUDA_ORMQR_OP_* constants — Left=0/Right=1,
        // N=0/T=1/C=2.
        let side = self.desc.side as i32;
        let op = self.desc.op as i32;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_batched_ormqr_f32_run(
                    self.desc.batch_size,
                    self.desc.m,
                    self.desc.n,
                    self.desc.k,
                    side,
                    op,
                    a_ptr,
                    tau_ptr,
                    c_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_batched_ormqr_f64_run(
                    self.desc.batch_size,
                    self.desc.m,
                    self.desc.n,
                    self.desc.k,
                    side,
                    op,
                    a_ptr,
                    tau_ptr,
                    c_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::Complex32 => unsafe {
                baracuda_kernels_batched_ormqr_complex32_run(
                    self.desc.batch_size,
                    self.desc.m,
                    self.desc.n,
                    self.desc.k,
                    side,
                    op,
                    a_ptr,
                    tau_ptr,
                    c_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::Complex64 => unsafe {
                baracuda_kernels_batched_ormqr_complex64_run(
                    self.desc.batch_size,
                    self.desc.m,
                    self.desc.n,
                    self.desc.k,
                    side,
                    op,
                    a_ptr,
                    tau_ptr,
                    c_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::BatchedOrmqrPlan::run reached an unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}

/// Status-code → Result translation, shared across the bespoke linalg
/// plans in this milestone. Matches the convention used by the
/// attention family ([`crate::attention::map_status`]).
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
