//! Linear solve `A · X = B` via `getrf` + `getrs`.
//!
//! Wraps cuSOLVER's `cusolverDnSgetrf` / `Dgetrf` (LU factorization
//! with partial pivoting) followed by `cusolverDnSgetrs` / `Dgetrs`
//! (triangular substitutions over the packed `LU` + pivot). The plan
//! owns no scratch state across calls — pivot + info are caller-
//! provided, the workspace bytes are reported through
//! [`SolvePlan::workspace_size`] and supplied as `Workspace::Borrowed`.
//!
//! **2-D only** — single `A`, single `B`. No batching today.
//!
//! **In-place semantics**: `A` is overwritten with the packed `LU`
//! factors (cuSOLVER `getrf` convention — `L` in the strict lower
//! triangle with implicit unit diagonal, `U` in the upper triangle).
//! `B` is overwritten with the solution `X`.
//!
//! **Storage convention**: like the rest of the linalg family, the
//! trailblazer passes through cuSOLVER's column-major view of the
//! caller's byte storage. The LU plan documents the same convention —
//! callers that want row-major end-to-end semantics must transpose on
//! either side (a future shape-layout op can fuse the transpose).
//!
//! **Workspace**: cuSOLVER's `getrs` is workspace-free; the entire
//! workspace requirement is the one queried from
//! `cusolverDnSgetrf_bufferSize` / `Dgetrf_bufferSize`.

use core::cell::Cell;
use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_sys::{
    cusolverDnCreate, cusolverDnDestroy, cusolverDnDgetrf, cusolverDnDgetrf_bufferSize,
    cusolverDnDgetrs, cusolverDnHandle_t, cusolverDnSetStream, cusolverDnSgetrf,
    cusolverDnSgetrf_bufferSize, cusolverDnSgetrs, CUBLAS_OP_N,
};
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, LinalgKind, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, Workspace,
};

use super::cholesky::unpack_workspace;

/// Descriptor for a linear-solve.
#[derive(Copy, Clone, Debug)]
pub struct SolveDescriptor {
    /// Order `M` of the (square) coefficient matrix `A`.
    pub m: i32,
    /// Number of right-hand sides — column count of `B` / `X`.
    pub nrhs: i32,
    /// Element type. Must be `F32` or `F64`.
    pub element: ElementKind,
}

/// Args bundle for a linear-solve launch.
///
/// `a` is overwritten in place with the packed `LU` factors produced by
/// `getrf`. `b` is overwritten in place with the solution `X`. `pivot`
/// receives cuSOLVER's 1-based pivot indices (length `M`). `info`
/// receives the single factorization-status word (`0` on success,
/// `k > 0` if `U[k, k] == 0` at step `k`).
pub struct SolveArgs<'a, T: Element> {
    /// Coefficient matrix `[M, M]` (column-major). Overwritten with
    /// packed `LU` in place.
    pub a: TensorMut<'a, T, 2>,
    /// Right-hand side `[M, NRHS]` (column-major) on input; solution
    /// `X` on output.
    pub b: TensorMut<'a, T, 2>,
    /// Pivot vector `[M]` (1-based per LAPACK convention).
    pub pivot: TensorMut<'a, i32, 1>,
    /// Single-cell info: `0` on success.
    pub info: TensorMut<'a, i32, 1>,
}

/// Linear-solve plan — `A · X = B` via `getrf` + `getrs`.
///
/// Two-step pipeline per `run`: `getrf` factors `A` in place to packed
/// `LU` + pivots, then `getrs` solves over the packed factorization.
/// `B` is overwritten with `X`.
///
/// **When to use**: square solve over a general `A`. Use
/// [`super::CholeskyPlan`] + a `trsm` chain when `A` is SPD;
/// [`super::LstSqPlan`] for least-squares.
///
/// **Dtypes**: `f32`, `f64`.
///
/// **Shape**: `[M, M]` × `[M, NRHS]`. 2-D only.
///
/// **Storage**: column-major end-to-end.
///
/// **Workspace**: cuSOLVER `_bufferSize` for `getrf` (queried lazily on
/// first `run`).
///
/// **Precision guarantee**: deterministic; not bit-stable across runs.
///
/// Owns a lazy cuSOLVER handle (`!Sync` / `!Send`); destroyed on `Drop`.
pub struct SolvePlan<T: Element> {
    desc: SolveDescriptor,
    sku: KernelSku,
    handle: Cell<cusolverDnHandle_t>,
    workspace_bytes: Cell<usize>,
    _marker: PhantomData<T>,
}

impl<T: Element> SolvePlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &SolveDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::SolvePlan: descriptor.element != T::KIND",
            ));
        }
        if !matches!(T::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::SolvePlan: cuSOLVER dense solve supports f32 + f64 only",
            ));
        }
        if desc.m <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SolvePlan: m must be > 0",
            ));
        }
        if desc.nrhs <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SolvePlan: nrhs must be > 0",
            ));
        }

        let math_precision = match T::KIND {
            ElementKind::F64 => MathPrecision::F64,
            _ => MathPrecision::F32,
        };
        let precision_guarantee = PrecisionGuarantee {
            math_precision,
            accumulator: T::KIND,
            bit_stable_on_same_hardware: false,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Linalg,
            op: LinalgKind::Solve as u16,
            element: T::KIND,
            aux_element: Some(ElementKind::I32),
            layout: None,
            epilogue: None,
            arch: ArchSku::Sm80,
            backend: BackendKind::Cusolver,
            precision_guarantee,
        };
        Ok(Self {
            desc: *desc,
            sku,
            handle: Cell::new(core::ptr::null_mut()),
            workspace_bytes: Cell::new(0),
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

    /// Workspace size in bytes (the `getrf` requirement; `getrs` is
    /// workspace-free). Lazily populated on first `run`.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        self.workspace_bytes.get()
    }

    /// Materialize the handle and query workspace size.
    pub fn query_workspace_size(&self, _stream: &Stream) -> Result<usize> {
        let h = self.ensure_handle()?;
        let mut lwork: i32 = 0;
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                cusolverDnSgetrf_bufferSize(
                    h,
                    self.desc.m,
                    self.desc.m,
                    core::ptr::null_mut(),
                    self.desc.m,
                    &mut lwork as *mut _,
                )
            },
            ElementKind::F64 => unsafe {
                cusolverDnDgetrf_bufferSize(
                    h,
                    self.desc.m,
                    self.desc.m,
                    core::ptr::null_mut(),
                    self.desc.m,
                    &mut lwork as *mut _,
                )
            },
            _ => unreachable!("select() gates on F32 / F64"),
        };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        let bytes = (lwork as usize) * core::mem::size_of::<T>();
        self.workspace_bytes.set(bytes);
        Ok(bytes)
    }

    fn ensure_handle(&self) -> Result<cusolverDnHandle_t> {
        let h = self.handle.get();
        if !h.is_null() {
            return Ok(h);
        }
        let mut handle: cusolverDnHandle_t = core::ptr::null_mut();
        let status = unsafe { cusolverDnCreate(&mut handle as *mut _) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        self.handle.set(handle);
        Ok(handle)
    }

    fn bind_stream(&self, h: cusolverDnHandle_t, stream: &Stream) -> Result<()> {
        let status = unsafe { cusolverDnSetStream(h, stream.as_raw() as *mut c_void) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        Ok(())
    }

    fn check_args(&self, args: &SolveArgs<'_, T>) -> Result<()> {
        let m = self.desc.m;
        let nrhs = self.desc.nrhs;
        if args.a.shape != [m, m] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SolvePlan: A shape != [M, M]",
            ));
        }
        if args.b.shape != [m, nrhs] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SolvePlan: B shape != [M, NRHS]",
            ));
        }
        if args.pivot.shape != [m] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SolvePlan: pivot shape != [M]",
            ));
        }
        if args.info.shape != [1] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SolvePlan: info shape != [1]",
            ));
        }
        Ok(())
    }
}

// Macro to instantiate run() for f32 / f64.
macro_rules! impl_solve_run {
    ($T:ty, $getrf:ident, $getrs:ident) => {
        impl SolvePlan<$T> {
            /// Run the linear solve.
            pub fn run(
                &self,
                stream: &Stream,
                workspace: Workspace<'_>,
                args: SolveArgs<'_, $T>,
            ) -> Result<()> {
                self.check_args(&args)?;
                let h = self.ensure_handle()?;
                self.bind_stream(h, stream)?;
                let m = self.desc.m;
                let nrhs = self.desc.nrhs;

                let needed = if self.workspace_bytes.get() == 0 {
                    self.query_workspace_size(stream)?
                } else {
                    self.workspace_bytes.get()
                };
                let (ws_ptr, _ws_bytes) = unpack_workspace(workspace, needed)?;

                let a_ptr = args.a.data.as_raw().0 as *mut $T;
                let b_ptr = args.b.data.as_raw().0 as *mut $T;
                let pivot_ptr = args.pivot.data.as_raw().0 as *mut i32;
                let info_ptr = args.info.data.as_raw().0 as *mut i32;

                // 1. getrf — factors A in place, writes pivot + info.
                let status = unsafe {
                    $getrf(h, m, m, a_ptr, m, ws_ptr as *mut $T, pivot_ptr, info_ptr)
                };
                if status != 0 {
                    return Err(Error::CutlassInternal(-status));
                }

                // 2. getrs — solves A · X = B in place over B. trans
                //    == N because storage is end-to-end column-major.
                let status = unsafe {
                    $getrs(
                        h,
                        CUBLAS_OP_N,
                        m,
                        nrhs,
                        a_ptr as *const $T,
                        m,
                        pivot_ptr as *const i32,
                        b_ptr,
                        m,
                        info_ptr,
                    )
                };
                if status != 0 {
                    return Err(Error::CutlassInternal(-status));
                }
                Ok(())
            }
        }
    };
}

impl_solve_run!(f32, cusolverDnSgetrf, cusolverDnSgetrs);
impl_solve_run!(f64, cusolverDnDgetrf, cusolverDnDgetrs);

impl<T: Element> Drop for SolvePlan<T> {
    fn drop(&mut self) {
        let h = self.handle.get();
        if !h.is_null() {
            unsafe {
                let _ = cusolverDnDestroy(h);
            }
            self.handle.set(core::ptr::null_mut());
        }
    }
}
