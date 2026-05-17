//! Least-squares solve `min ||A·x - b||²` (full-rank `A`, `m ≥ n`).
//!
//! Wraps cuSOLVER's mixed-precision iterative-refinement `_gels` family.
//! The plan surfaces the same-precision variants only:
//!
//! - `cusolverDnSSgels` — single-precision input, single-precision compute.
//! - `cusolverDnDDgels` — double-precision input, double-precision compute.
//!
//! Other letter combinations (`SHgels`, `SBgels`, `DSgels`, `DHgels`,
//! `DBgels`) implement mixed-precision strategies (e.g. iterate in `f16`
//! against an `f32` input to accelerate); these are not surfaced here.
//!
//! **Convergence + QR fallback**: `_gels` is iterative; the routine
//! writes a non-negative `niters` on convergence or a negative value on
//! non-convergence (typically because the caller's matrix is poorly
//! conditioned). When non-convergence is reported, the plan can fall
//! back to the standard three-step QR-based solve:
//!
//!   1. `cusolverDn{S,D}geqrf` — factor `A = Q · R` in place (packed).
//!   2. `cusolverDn{S,D}ormqr(side=L, trans=T)` — `B := Q^T · B`.
//!   3. `cublas{S,D}trsm(side=L, uplo=U, trans=N, diag=NU)` — back-
//!      substitute the upper triangle of `R`, solving `R · X = Q^T B`.
//!
//! `X` ends up in the top `N` rows of `B`; the plan then copies that
//! block into the caller's `x` output buffer.
//!
//! Because `_gels` destroys `A` in place on **every** call (the
//! iterative refinement path doesn't restore it on non-convergence),
//! callers that want the fallback path **must** pre-stage a backup
//! copy of `A` and pass it via [`LstSqArgs::a_backup`]. If the backup
//! is `None` and `_gels` fails to converge, the plan returns
//! [`Error::Unsupported`] (preserving the prior behaviour for callers
//! that don't care about the fallback).
//!
//! **Storage**: column-major end-to-end, matching the neighbors.
//!
//! **In-place semantics**: `a` is overwritten in place (factored). `b`
//! is overwritten with the residual / scratch. The solution `x` is
//! written into a **separate caller-provided output buffer** of shape
//! `[N, NRHS]` (cuSOLVER's `_gels` writes the solution to `X`, not
//! over `B` like the LAPACK `gels` API does).
//!
//! **Workspace**: the fallback path may need more workspace than the
//! iterative-refinement path. Callers that want to enable the
//! fallback should size their workspace as
//! `max(plan.workspace_size(), plan.qr_fallback_workspace_size())`.

use core::cell::Cell;
use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_sys::{
    cublasCreate_v2, cublasDestroy_v2, cublasDtrsm, cublasHandle_t, cublasSetStream_v2,
    cublasStrsm, cusolverDnCreate, cusolverDnDDgels, cusolverDnDDgels_bufferSize,
    cusolverDnDestroy, cusolverDnDgeqrf, cusolverDnDgeqrf_bufferSize, cusolverDnDormqr,
    cusolverDnDormqr_bufferSize, cusolverDnHandle_t, cusolverDnSSgels,
    cusolverDnSSgels_bufferSize, cusolverDnSetStream, cusolverDnSgeqrf,
    cusolverDnSgeqrf_bufferSize, cusolverDnSormqr, cusolverDnSormqr_bufferSize,
    CUBLAS_DIAG_NON_UNIT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_SIDE_LEFT,
};
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, LinalgKind, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a least-squares solve.
#[derive(Copy, Clone, Debug)]
pub struct LstSqDescriptor {
    /// Row count `M` of the coefficient matrix `A`. Requires `m >= n`.
    pub m: i32,
    /// Column count `N` of the coefficient matrix `A`.
    pub n: i32,
    /// Number of right-hand sides (column count of `B` / `X`).
    pub nrhs: i32,
    /// Element type. Must be `F32` or `F64`.
    pub element: ElementKind,
}

/// Args bundle for a least-squares launch.
///
/// `a` is overwritten in place (factored). `b` is the input RHS,
/// overwritten with scratch. `x` is the output solution `[N, NRHS]`.
/// `info` receives the per-launch status word (`0` on success).
///
/// **Caller responsibility — A is destroyed on every call**: cuSOLVER's
/// `_gels` overwrites `a` in place regardless of convergence. If the
/// caller wants to enable the QR fallback path when iterative refinement
/// fails to converge (poorly-conditioned matrices), they **must** stage
/// a separate read-only backup of `A` and pass it via
/// [`LstSqArgs::a_backup`]. With `a_backup = None`, non-convergence
/// surfaces as [`Error::Unsupported`].
///
/// Note: cuSOLVER's `_gels` writes the solution to a **separate** `X`
/// buffer (distinct from LAPACK's classic `gels` API which overwrites
/// `B`). The plan surfaces both buffers; callers that want the LAPACK
/// in-place convention can pass the same buffer for `b` and `x` — but
/// must ensure it is at least `max(M, N) * NRHS` elements long.
pub struct LstSqArgs<'a, T: Element> {
    /// Coefficient matrix `[M, N]` (column-major). Overwritten in
    /// place by the factorization.
    pub a: TensorMut<'a, T, 2>,
    /// Right-hand side `[M, NRHS]` (column-major). Overwritten with
    /// scratch on output.
    pub b: TensorMut<'a, T, 2>,
    /// Solution `[N, NRHS]` (column-major). Written by the plan.
    pub x: TensorMut<'a, T, 2>,
    /// Single-cell info: `0` on success.
    pub info: TensorMut<'a, i32, 1>,
    /// Optional read-only backup of `A` (`[M, N]` column-major). If
    /// `Some`, the plan uploads `a_backup → a` before running the QR
    /// fallback (`_gels` destroyed the original `a` in place). If
    /// `None`, non-convergence returns [`Error::Unsupported`].
    pub a_backup: Option<TensorRef<'a, T, 2>>,
}

/// Least-squares plan.
pub struct LstSqPlan<T: Element> {
    desc: LstSqDescriptor,
    sku: KernelSku,
    handle: Cell<cusolverDnHandle_t>,
    cublas_handle: Cell<cublasHandle_t>,
    workspace_bytes: Cell<usize>,
    qr_workspace_bytes: Cell<usize>,
    _marker: PhantomData<T>,
}

impl<T: Element> LstSqPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &LstSqDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::LstSqPlan: descriptor.element != T::KIND",
            ));
        }
        if !matches!(T::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::LstSqPlan: cuSOLVER least-squares supports f32 + f64 only",
            ));
        }
        if desc.m <= 0 || desc.n <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::LstSqPlan: m / n must be > 0",
            ));
        }
        if desc.nrhs <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::LstSqPlan: nrhs must be > 0",
            ));
        }
        if desc.m < desc.n {
            return Err(Error::Unsupported(
                "baracuda-kernels::LstSqPlan: cuSOLVER _gels requires m >= n (full-rank)",
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
            op: LinalgKind::LeastSquares as u16,
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
            cublas_handle: Cell::new(core::ptr::null_mut()),
            workspace_bytes: Cell::new(0),
            qr_workspace_bytes: Cell::new(0),
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

    /// Workspace size in bytes for the iterative-refinement (`_gels`)
    /// path. Lazily populated on first `run` (or
    /// [`Self::query_workspace_size`]).
    #[inline]
    pub fn workspace_size(&self) -> usize {
        self.workspace_bytes.get()
    }

    /// Workspace size in bytes for the QR fallback path. Lazily
    /// populated on first `run` (or
    /// [`Self::query_qr_fallback_workspace_size`]).
    ///
    /// Callers that want the fallback enabled should size their
    /// workspace as
    /// `max(plan.workspace_size(), plan.qr_fallback_workspace_size())`.
    #[inline]
    pub fn qr_fallback_workspace_size(&self) -> usize {
        self.qr_workspace_bytes.get()
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

    fn ensure_cublas_handle(&self) -> Result<cublasHandle_t> {
        let h = self.cublas_handle.get();
        if !h.is_null() {
            return Ok(h);
        }
        let mut handle: cublasHandle_t = core::ptr::null_mut();
        let status = unsafe { cublasCreate_v2(&mut handle as *mut _) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        self.cublas_handle.set(handle);
        Ok(handle)
    }

    fn bind_stream(&self, h: cusolverDnHandle_t, stream: &Stream) -> Result<()> {
        let status = unsafe { cusolverDnSetStream(h, stream.as_raw() as *mut c_void) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        Ok(())
    }

    fn bind_cublas_stream(&self, h: cublasHandle_t, stream: &Stream) -> Result<()> {
        let status = unsafe { cublasSetStream_v2(h, stream.as_raw() as *mut c_void) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        Ok(())
    }

    /// Materialize the handle and query workspace size (in bytes).
    /// `_gels`'s `_bufferSize` is byte-typed (`size_t`), not element-typed.
    pub fn query_workspace_size(&self, _stream: &Stream) -> Result<usize> {
        let h = self.ensure_handle()?;
        let m = self.desc.m;
        let n = self.desc.n;
        let nrhs = self.desc.nrhs;
        let mut lwork_bytes: usize = 0;
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                cusolverDnSSgels_bufferSize(
                    h,
                    m,
                    n,
                    nrhs,
                    core::ptr::null_mut(),
                    m,
                    core::ptr::null_mut(),
                    m,
                    core::ptr::null_mut(),
                    n,
                    core::ptr::null_mut(),
                    &mut lwork_bytes as *mut _,
                )
            },
            ElementKind::F64 => unsafe {
                cusolverDnDDgels_bufferSize(
                    h,
                    m,
                    n,
                    nrhs,
                    core::ptr::null_mut(),
                    m,
                    core::ptr::null_mut(),
                    m,
                    core::ptr::null_mut(),
                    n,
                    core::ptr::null_mut(),
                    &mut lwork_bytes as *mut _,
                )
            },
            _ => unreachable!("select() gates on F32 / F64"),
        };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        self.workspace_bytes.set(lwork_bytes);
        Ok(lwork_bytes)
    }

    /// Materialize the handle and query the QR fallback workspace
    /// size (in bytes). Layout matches [`super::qr::QrPlan`]:
    /// `tau_bytes + max(geqrf_lwork, ormqr_lwork) * sizeof(T)`.
    pub fn query_qr_fallback_workspace_size(&self, _stream: &Stream) -> Result<usize> {
        let h = self.ensure_handle()?;
        let m = self.desc.m;
        let n = self.desc.n;
        let nrhs = self.desc.nrhs;
        let k = m.min(n); // == n since m >= n is enforced by select().
        let mut lwork_geqrf: i32 = 0;
        let mut lwork_ormqr: i32 = 0;
        let status_geqrf = match T::KIND {
            ElementKind::F32 => unsafe {
                cusolverDnSgeqrf_bufferSize(
                    h,
                    m,
                    n,
                    core::ptr::null_mut(),
                    m,
                    &mut lwork_geqrf as *mut _,
                )
            },
            ElementKind::F64 => unsafe {
                cusolverDnDgeqrf_bufferSize(
                    h,
                    m,
                    n,
                    core::ptr::null_mut(),
                    m,
                    &mut lwork_geqrf as *mut _,
                )
            },
            _ => unreachable!(),
        };
        if status_geqrf != 0 {
            return Err(Error::CutlassInternal(-status_geqrf));
        }
        // `ormqr` applies Q^T (k reflectors) to B (m x nrhs).
        let status_ormqr = match T::KIND {
            ElementKind::F32 => unsafe {
                cusolverDnSormqr_bufferSize(
                    h,
                    CUBLAS_SIDE_LEFT,
                    CUBLAS_OP_T,
                    m,
                    nrhs,
                    k,
                    core::ptr::null(),
                    m,
                    core::ptr::null(),
                    core::ptr::null(),
                    m,
                    &mut lwork_ormqr as *mut _,
                )
            },
            ElementKind::F64 => unsafe {
                cusolverDnDormqr_bufferSize(
                    h,
                    CUBLAS_SIDE_LEFT,
                    CUBLAS_OP_T,
                    m,
                    nrhs,
                    k,
                    core::ptr::null(),
                    m,
                    core::ptr::null(),
                    core::ptr::null(),
                    m,
                    &mut lwork_ormqr as *mut _,
                )
            },
            _ => unreachable!(),
        };
        if status_ormqr != 0 {
            return Err(Error::CutlassInternal(-status_ormqr));
        }
        let lwork_max = lwork_geqrf.max(lwork_ormqr) as usize;
        let elem_size = core::mem::size_of::<T>();
        let tau_bytes = (k as usize) * elem_size;
        let bytes = tau_bytes + lwork_max * elem_size;
        self.qr_workspace_bytes.set(bytes);
        Ok(bytes)
    }

    fn check_args(&self, args: &LstSqArgs<'_, T>) -> Result<()> {
        let m = self.desc.m;
        let n = self.desc.n;
        let nrhs = self.desc.nrhs;
        if args.a.shape != [m, n] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::LstSqPlan: A shape != [M, N]",
            ));
        }
        if args.b.shape != [m, nrhs] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::LstSqPlan: B shape != [M, NRHS]",
            ));
        }
        if args.x.shape != [n, nrhs] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::LstSqPlan: X shape != [N, NRHS]",
            ));
        }
        if args.info.shape != [1] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::LstSqPlan: info shape != [1]",
            ));
        }
        if let Some(ref bkp) = args.a_backup {
            if bkp.shape != [m, n] {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::LstSqPlan: a_backup shape != [M, N]",
                ));
            }
        }
        Ok(())
    }
}

macro_rules! impl_lstsq_run {
    ($T:ty, $gels:ident, $geqrf:ident, $ormqr:ident, $trsm:ident) => {
        impl LstSqPlan<$T> {
            /// Run the least-squares solve.
            pub fn run(
                &self,
                stream: &Stream,
                workspace: Workspace<'_>,
                args: LstSqArgs<'_, $T>,
            ) -> Result<()> {
                self.check_args(&args)?;
                let h = self.ensure_handle()?;
                self.bind_stream(h, stream)?;
                let m = self.desc.m;
                let n = self.desc.n;
                let nrhs = self.desc.nrhs;

                let needed = if self.workspace_bytes.get() == 0 {
                    self.query_workspace_size(stream)?
                } else {
                    self.workspace_bytes.get()
                };
                // `_gels` workspace is byte-typed. Unpack as raw bytes;
                // the slice carries u8 cells from the caller, so the
                // length is exactly the byte count.
                let (ws_ptr, ws_bytes) = match workspace {
                    Workspace::None => {
                        if needed == 0 {
                            (core::ptr::null_mut::<u8>() as *mut c_void, 0usize)
                        } else {
                            return Err(Error::WorkspaceTooSmall { needed, got: 0 });
                        }
                    }
                    Workspace::Borrowed(slice) => {
                        let got = slice.len();
                        if got < needed {
                            return Err(Error::WorkspaceTooSmall { needed, got });
                        }
                        (slice.as_raw().0 as *mut c_void, got)
                    }
                };

                let a_ptr = args.a.data.as_raw().0 as *mut $T;
                let b_ptr = args.b.data.as_raw().0 as *mut $T;
                let x_ptr = args.x.data.as_raw().0 as *mut $T;
                let info_ptr = args.info.data.as_raw().0 as *mut i32;

                let mut niters: i32 = 0;
                let status = unsafe {
                    $gels(
                        h,
                        m,
                        n,
                        nrhs,
                        a_ptr,
                        m,
                        b_ptr,
                        m,
                        x_ptr,
                        n,
                        ws_ptr,
                        ws_bytes,
                        &mut niters as *mut _,
                        info_ptr,
                    )
                };
                if status != 0 {
                    return Err(Error::CutlassInternal(-status));
                }
                if niters >= 0 {
                    return Ok(());
                }

                // `niters < 0` — iterative refinement did not converge.
                // Try the QR fallback if the caller staged a backup of A.
                let backup = match args.a_backup {
                    Some(b) => b,
                    None => {
                        return Err(Error::Unsupported(
                            "baracuda-kernels::LstSqPlan: _gels did not converge \
                             (negative niters); pass `a_backup: Some(...)` to \
                             enable the QR fallback path",
                        ));
                    }
                };

                // Re-validate fallback workspace size and unpack the
                // caller's workspace under the fallback's layout.
                let qr_needed = if self.qr_workspace_bytes.get() == 0 {
                    self.query_qr_fallback_workspace_size(stream)?
                } else {
                    self.qr_workspace_bytes.get()
                };
                if ws_bytes < qr_needed {
                    return Err(Error::WorkspaceTooSmall {
                        needed: qr_needed,
                        got: ws_bytes,
                    });
                }
                let elem_size = core::mem::size_of::<$T>();
                let k = m.min(n);
                let tau_bytes = (k as usize) * elem_size;
                let tail_bytes = ws_bytes.saturating_sub(tau_bytes);
                let tau_ptr = ws_ptr as *mut $T;
                let tail_ptr = unsafe { (ws_ptr as *mut u8).add(tau_bytes) as *mut $T };
                let lwork = (tail_bytes / elem_size) as i32;

                // 1. Restore A from caller's backup (D2D copy).
                let a_bytes = (m as usize) * (n as usize) * elem_size;
                let backup_ptr = backup.data.as_raw().0 as *const c_void;
                let a_dev_ptr = args.a.data.as_raw().0 as *mut c_void;
                unsafe {
                    copy_d2d(a_dev_ptr, backup_ptr, a_bytes, stream)?;
                }

                // 2. geqrf — A := packed(Q, R), tau := householder scalars.
                let status = unsafe {
                    $geqrf(h, m, n, a_ptr, m, tau_ptr, tail_ptr, lwork, info_ptr)
                };
                if status != 0 {
                    return Err(Error::CutlassInternal(-status));
                }

                // 3. ormqr (side=LEFT, trans=T) — B := Q^T · B.
                let status = unsafe {
                    $ormqr(
                        h,
                        CUBLAS_SIDE_LEFT,
                        CUBLAS_OP_T,
                        m,
                        nrhs,
                        k,
                        a_ptr as *const $T,
                        m,
                        tau_ptr as *const $T,
                        b_ptr,
                        m,
                        tail_ptr,
                        lwork,
                        info_ptr,
                    )
                };
                if status != 0 {
                    return Err(Error::CutlassInternal(-status));
                }

                // 4. trsm (side=LEFT, uplo=UPPER, trans=N, diag=NON_UNIT)
                //    — solve R · X = (Q^T B)[:N, :] in place over the
                //    top-N rows of B. Use cuBLAS bound to the same
                //    stream as cuSOLVER so we don't race.
                let cublas_h = self.ensure_cublas_handle()?;
                self.bind_cublas_stream(cublas_h, stream)?;
                let alpha: $T = 1.0;
                let status = unsafe {
                    $trsm(
                        cublas_h,
                        CUBLAS_SIDE_LEFT,
                        CUBLAS_FILL_MODE_UPPER,
                        CUBLAS_OP_N,
                        CUBLAS_DIAG_NON_UNIT,
                        n,
                        nrhs,
                        &alpha as *const $T,
                        a_ptr as *const $T,
                        m,
                        b_ptr,
                        m,
                    )
                };
                if status != 0 {
                    return Err(Error::CutlassInternal(-status));
                }

                // 5. Copy the solution (top-N rows of B, ldb = m,
                //    nrhs columns) into the caller's X buffer
                //    (`[N, NRHS]`, ld = n). Column-major copy: per
                //    column, copy `n * elem_size` bytes from
                //    `B + col * m * elem_size` to `X + col * n * elem_size`.
                let col_bytes = (n as usize) * elem_size;
                let b_col_stride_bytes = (m as usize) * elem_size;
                let x_col_stride_bytes = (n as usize) * elem_size;
                for col in 0..(nrhs as usize) {
                    let src = unsafe {
                        (b_ptr as *const u8).add(col * b_col_stride_bytes) as *const c_void
                    };
                    let dst = unsafe {
                        (x_ptr as *mut u8).add(col * x_col_stride_bytes) as *mut c_void
                    };
                    unsafe {
                        copy_d2d(dst, src, col_bytes, stream)?;
                    }
                }

                Ok(())
            }
        }
    };
}

impl_lstsq_run!(f32, cusolverDnSSgels, cusolverDnSgeqrf, cusolverDnSormqr, cublasStrsm);
impl_lstsq_run!(f64, cusolverDnDDgels, cusolverDnDgeqrf, cusolverDnDormqr, cublasDtrsm);

impl<T: Element> Drop for LstSqPlan<T> {
    fn drop(&mut self) {
        let h = self.handle.get();
        if !h.is_null() {
            unsafe {
                let _ = cusolverDnDestroy(h);
            }
            self.handle.set(core::ptr::null_mut());
        }
        let h = self.cublas_handle.get();
        if !h.is_null() {
            unsafe {
                let _ = cublasDestroy_v2(h);
            }
            self.cublas_handle.set(core::ptr::null_mut());
        }
    }
}

// ----- Device-to-device async memcpy used by the QR fallback -------

unsafe fn copy_d2d(
    dst: *mut c_void,
    src: *const c_void,
    bytes: usize,
    stream: &Stream,
) -> Result<()> {
    #[allow(non_camel_case_types)]
    type CUresult = i32;
    unsafe extern "system" {
        fn cuMemcpyDtoDAsync_v2(
            dst_device: u64,
            src_device: u64,
            byte_count: usize,
            h_stream: *mut c_void,
        ) -> CUresult;
    }
    let status = unsafe {
        cuMemcpyDtoDAsync_v2(dst as u64, src as u64, bytes, stream.as_raw() as *mut c_void)
    };
    if status != 0 {
        return Err(Error::CutlassInternal(-status));
    }
    Ok(())
}
