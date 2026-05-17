//! Symmetric / Hermitian eigendecomposition — `A · v = λ · v` with `A`
//! real-symmetric or complex-Hermitian, returning real eigenvalues and
//! orthonormal eigenvectors.
//!
//! Wraps cuSOLVER's divide-and-conquer eigh routines:
//! - `cusolverDnSsyevd` / `Dsyevd` — real symmetric (`f32` / `f64`).
//! - `cusolverDnCheevd` / `Zheevd` — complex Hermitian (`Complex32` /
//!   `Complex64`); eigenvalues are real even for Hermitian input.
//!
//! **2-D only** — single matrix per launch. cuSOLVER's dense API has no
//! batched `syevd` / `heevd`.
//!
//! **In-place semantics**: `A` is overwritten with the eigenvector
//! matrix (column-major). `W` receives the eigenvalues sorted in
//! ascending order. The opposite triangle of the input is not read.
//!
//! **Storage convention**: column-major end-to-end (matches the rest of
//! the linalg family — Cholesky / LU / QR / SVD). A symmetric / Hermitian
//! matrix has the same byte storage in either row-major or column-major
//! view, so the `uplo` flag from the caller is passed through to cuSOLVER
//! verbatim (no row-major adapter flip; cf. Cholesky which does flip
//! because its caller-facing factor is asymmetric).

use core::cell::Cell;
use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_sys::{
    cuComplex, cuDoubleComplex, cusolverDnCheevd, cusolverDnCheevd_bufferSize, cusolverDnCreate,
    cusolverDnDestroy, cusolverDnDsyevd, cusolverDnDsyevd_bufferSize, cusolverDnHandle_t,
    cusolverDnSetStream, cusolverDnSsyevd, cusolverDnSsyevd_bufferSize, cusolverDnZheevd,
    cusolverDnZheevd_bufferSize, CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER,
    CUSOLVER_EIG_MODE_VECTOR,
};
use baracuda_kernels_types::{
    ArchSku, BackendKind, Complex32, Complex64, Element, ElementKind, FillMode, KernelSku,
    LinalgKind, MathPrecision, OpCategory, PlanPreference, PrecisionGuarantee, TensorMut,
    Workspace,
};

use super::cholesky::unpack_workspace;

/// Descriptor for a symmetric / Hermitian eigendecomposition.
#[derive(Copy, Clone, Debug)]
pub struct EighDescriptor {
    /// Order `N` of the (square) input matrix.
    pub n: i32,
    /// Triangle of `A` read by cuSOLVER. The opposite triangle's
    /// contents are ignored.
    pub uplo: FillMode,
    /// Element type. Must be `F32`, `F64`, `Complex32`, or `Complex64`.
    pub element: ElementKind,
}

/// Args bundle for a symmetric / Hermitian eigh launch.
///
/// `a` is **overwritten in place** with the eigenvector matrix (column-
/// major); column `j` is the eigenvector for `w[j]`. `w` receives the
/// `n` real eigenvalues sorted ascending.
///
/// `TW` is the real eigenvalue element type — `f32` when `T = f32` or
/// `Complex32`, `f64` when `T = f64` or `Complex64`. The plan validates
/// `TW == T::Scalar` at runtime. Surfacing `TW` as a second generic
/// (rather than just `T::Scalar`) sidesteps the
/// `T::Scalar: DeviceRepr` propagation gap in the [`Element`] trait —
/// the constraint lands on the concrete `TW` instead.
pub struct EighArgs<'a, T: Element, TW: Element> {
    /// Input symmetric / Hermitian matrix `[N, N]` column-major. Only
    /// the triangle selected by `descriptor.uplo` is read. Overwritten
    /// in place with the eigenvectors.
    pub a: TensorMut<'a, T, 2>,
    /// Eigenvalues `[N]` (always real, sorted ascending). `TW` must be
    /// the real-scalar sibling of `T` — i.e. `f32` for `T = f32` /
    /// `Complex32`, `f64` for `T = f64` / `Complex64`.
    pub w: TensorMut<'a, TW, 1>,
    /// Single-cell info: `0` on success, `k > 0` if the algorithm
    /// failed to converge at step `k`, `k < 0` if the `-k`-th argument
    /// was invalid.
    pub info: TensorMut<'a, i32, 1>,
}

/// Symmetric / Hermitian eigh plan.
///
/// Owns a lazy cuSOLVER handle (`!Sync` / `!Send`). The handle is
/// destroyed on `Drop`.
pub struct EighPlan<T: Element> {
    desc: EighDescriptor,
    sku: KernelSku,
    handle: Cell<cusolverDnHandle_t>,
    workspace_bytes: Cell<usize>,
    _marker: PhantomData<T>,
}

impl<T: Element> EighPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &EighDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::EighPlan: descriptor.element != T::KIND",
            ));
        }
        if !matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F64 | ElementKind::Complex32 | ElementKind::Complex64
        ) {
            return Err(Error::Unsupported(
                "baracuda-kernels::EighPlan: supports f32 / f64 / Complex32 / Complex64 only",
            ));
        }
        if desc.n <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::EighPlan: n must be > 0",
            ));
        }

        let math_precision = match T::KIND {
            ElementKind::F64 | ElementKind::Complex64 => MathPrecision::F64,
            _ => MathPrecision::F32,
        };
        let precision_guarantee = PrecisionGuarantee {
            math_precision,
            accumulator: T::KIND,
            // cuSOLVER's reduction / sweep order is implementation-
            // defined; reductions inside syevd / heevd vary with launch
            // parameters. Conservative.
            bit_stable_on_same_hardware: false,
            // Deterministic in the single-run-from-one-thread sense (no
            // atomic accumulation visible to the caller).
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Linalg,
            op: LinalgKind::Eigh as u16,
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

    /// Workspace size in bytes. Lazily populated on first `run` /
    /// [`Self::query_workspace_size`].
    #[inline]
    pub fn workspace_size(&self) -> usize {
        self.workspace_bytes.get()
    }

    /// Materialize the handle and query workspace size.
    pub fn query_workspace_size(&self, _stream: &Stream) -> Result<usize> {
        let h = self.ensure_handle()?;
        let n = self.desc.n;
        let uplo = self.cusolver_uplo();
        let mut lwork: i32 = 0;
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                cusolverDnSsyevd_bufferSize(
                    h,
                    CUSOLVER_EIG_MODE_VECTOR,
                    uplo,
                    n,
                    core::ptr::null(),
                    n,
                    core::ptr::null(),
                    &mut lwork as *mut _,
                )
            },
            ElementKind::F64 => unsafe {
                cusolverDnDsyevd_bufferSize(
                    h,
                    CUSOLVER_EIG_MODE_VECTOR,
                    uplo,
                    n,
                    core::ptr::null(),
                    n,
                    core::ptr::null(),
                    &mut lwork as *mut _,
                )
            },
            ElementKind::Complex32 => unsafe {
                cusolverDnCheevd_bufferSize(
                    h,
                    CUSOLVER_EIG_MODE_VECTOR,
                    uplo,
                    n,
                    core::ptr::null(),
                    n,
                    core::ptr::null(),
                    &mut lwork as *mut _,
                )
            },
            ElementKind::Complex64 => unsafe {
                cusolverDnZheevd_bufferSize(
                    h,
                    CUSOLVER_EIG_MODE_VECTOR,
                    uplo,
                    n,
                    core::ptr::null(),
                    n,
                    core::ptr::null(),
                    &mut lwork as *mut _,
                )
            },
            _ => unreachable!("select() gates on F32 / F64 / Complex32 / Complex64"),
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

    /// Map the caller's `FillMode` to a cuSOLVER `uplo` value. Eigh is
    /// pass-through (no row-major adapter flip — see module docs).
    #[inline]
    fn cusolver_uplo(&self) -> i32 {
        match self.desc.uplo {
            FillMode::Lower => CUBLAS_FILL_MODE_LOWER,
            FillMode::Upper => CUBLAS_FILL_MODE_UPPER,
        }
    }

    fn check_args<TW: Element>(&self, args: &EighArgs<'_, T, TW>) -> Result<()> {
        let n = self.desc.n;
        if args.a.shape != [n, n] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::EighPlan: A shape != [N, N]",
            ));
        }
        if args.w.shape != [n] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::EighPlan: W shape != [N]",
            ));
        }
        if args.info.shape != [1] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::EighPlan: info shape != [1]",
            ));
        }
        // Cross-check: TW must be the real-scalar sibling of T.
        let expected = real_scalar_kind::<T>();
        if TW::KIND != expected {
            return Err(Error::Unsupported(
                "baracuda-kernels::EighPlan: TW != real-scalar sibling of T \
                 (f32/Complex32 → f32, f64/Complex64 → f64)",
            ));
        }
        Ok(())
    }
}

/// Returns the real-scalar element-kind sibling of `T` — the type of
/// the eigenvalue tensor `W` in [`EighArgs`].
#[inline]
fn real_scalar_kind<T: Element>() -> ElementKind {
    match T::KIND {
        ElementKind::F32 | ElementKind::Complex32 => ElementKind::F32,
        ElementKind::F64 | ElementKind::Complex64 => ElementKind::F64,
        _ => unreachable!("select() gates on F32 / F64 / Complex32 / Complex64"),
    }
}

// ----- f32 ------------------------------------------------------------------

impl EighPlan<f32> {
    /// Run the symmetric eigendecomposition (f32). Eigenvalues land in
    /// `args.w` as `f32`.
    pub fn run(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: EighArgs<'_, f32, f32>,
    ) -> Result<()> {
        self.check_args(&args)?;
        let h = self.ensure_handle()?;
        self.bind_stream(h, stream)?;
        let n = self.desc.n;
        let uplo = self.cusolver_uplo();
        let needed = if self.workspace_bytes.get() == 0 {
            self.query_workspace_size(stream)?
        } else {
            self.workspace_bytes.get()
        };
        let (ws_ptr, ws_bytes) = unpack_workspace(workspace, needed)?;
        let lwork = (ws_bytes / core::mem::size_of::<f32>()) as i32;

        let a_ptr = args.a.data.as_raw().0 as *mut f32;
        let w_ptr = args.w.data.as_raw().0 as *mut f32;
        let info_ptr = args.info.data.as_raw().0 as *mut i32;

        let status = unsafe {
            cusolverDnSsyevd(
                h,
                CUSOLVER_EIG_MODE_VECTOR,
                uplo,
                n,
                a_ptr,
                n,
                w_ptr,
                ws_ptr as *mut f32,
                lwork,
                info_ptr,
            )
        };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        Ok(())
    }
}

// ----- f64 ------------------------------------------------------------------

impl EighPlan<f64> {
    /// Run the symmetric eigendecomposition (f64). Eigenvalues land in
    /// `args.w` as `f64`.
    pub fn run(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: EighArgs<'_, f64, f64>,
    ) -> Result<()> {
        self.check_args(&args)?;
        let h = self.ensure_handle()?;
        self.bind_stream(h, stream)?;
        let n = self.desc.n;
        let uplo = self.cusolver_uplo();
        let needed = if self.workspace_bytes.get() == 0 {
            self.query_workspace_size(stream)?
        } else {
            self.workspace_bytes.get()
        };
        let (ws_ptr, ws_bytes) = unpack_workspace(workspace, needed)?;
        let lwork = (ws_bytes / core::mem::size_of::<f64>()) as i32;

        let a_ptr = args.a.data.as_raw().0 as *mut f64;
        let w_ptr = args.w.data.as_raw().0 as *mut f64;
        let info_ptr = args.info.data.as_raw().0 as *mut i32;

        let status = unsafe {
            cusolverDnDsyevd(
                h,
                CUSOLVER_EIG_MODE_VECTOR,
                uplo,
                n,
                a_ptr,
                n,
                w_ptr,
                ws_ptr as *mut f64,
                lwork,
                info_ptr,
            )
        };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        Ok(())
    }
}

// ----- Complex32 (Hermitian) -----------------------------------------------

impl EighPlan<Complex32> {
    /// Run the Hermitian eigendecomposition (Complex32). Eigenvalues
    /// land in `args.w` as real `f32` (the Hermitian eigenvalue
    /// spectrum is always real).
    pub fn run(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: EighArgs<'_, Complex32, f32>,
    ) -> Result<()> {
        self.check_args(&args)?;
        let h = self.ensure_handle()?;
        self.bind_stream(h, stream)?;
        let n = self.desc.n;
        let uplo = self.cusolver_uplo();
        let needed = if self.workspace_bytes.get() == 0 {
            self.query_workspace_size(stream)?
        } else {
            self.workspace_bytes.get()
        };
        let (ws_ptr, ws_bytes) = unpack_workspace(workspace, needed)?;
        let lwork = (ws_bytes / core::mem::size_of::<Complex32>()) as i32;

        let a_ptr = args.a.data.as_raw().0 as *mut cuComplex;
        // W is real-valued even for Hermitian input.
        let w_ptr = args.w.data.as_raw().0 as *mut f32;
        let info_ptr = args.info.data.as_raw().0 as *mut i32;

        let status = unsafe {
            cusolverDnCheevd(
                h,
                CUSOLVER_EIG_MODE_VECTOR,
                uplo,
                n,
                a_ptr,
                n,
                w_ptr,
                ws_ptr as *mut cuComplex,
                lwork,
                info_ptr,
            )
        };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        Ok(())
    }
}

// ----- Complex64 (Hermitian) -----------------------------------------------

impl EighPlan<Complex64> {
    /// Run the Hermitian eigendecomposition (Complex64). Eigenvalues
    /// land in `args.w` as real `f64`.
    pub fn run(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: EighArgs<'_, Complex64, f64>,
    ) -> Result<()> {
        self.check_args(&args)?;
        let h = self.ensure_handle()?;
        self.bind_stream(h, stream)?;
        let n = self.desc.n;
        let uplo = self.cusolver_uplo();
        let needed = if self.workspace_bytes.get() == 0 {
            self.query_workspace_size(stream)?
        } else {
            self.workspace_bytes.get()
        };
        let (ws_ptr, ws_bytes) = unpack_workspace(workspace, needed)?;
        let lwork = (ws_bytes / core::mem::size_of::<Complex64>()) as i32;

        let a_ptr = args.a.data.as_raw().0 as *mut cuDoubleComplex;
        let w_ptr = args.w.data.as_raw().0 as *mut f64;
        let info_ptr = args.info.data.as_raw().0 as *mut i32;

        let status = unsafe {
            cusolverDnZheevd(
                h,
                CUSOLVER_EIG_MODE_VECTOR,
                uplo,
                n,
                a_ptr,
                n,
                w_ptr,
                ws_ptr as *mut cuDoubleComplex,
                lwork,
                info_ptr,
            )
        };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        Ok(())
    }
}

impl<T: Element> Drop for EighPlan<T> {
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
