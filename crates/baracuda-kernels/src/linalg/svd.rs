//! Singular value decomposition — `A = U · diag(S) · V^T`.
//!
//! Wraps cuSOLVER's `cusolverDnSgesvd` / `Dgesvd`.
//!
//! **2-D only** — cuSOLVER's dense API has no batched `gesvd`.
//!
//! **cuSOLVER constraint**: `gesvd` requires `m >= n`. Callers that
//! need `m < n` must transpose the input before invoking the plan
//! (the same column-major-pass-through convention as [`super::qr`]
//! applies — see that module for the discussion).
//!
//! **Outputs**:
//! - `full_matrices = true`: `U: [M, M]`, `V^T: [N, N]`.
//! - `full_matrices = false` (thin): `U: [M, K]`, `V^T: [K, N]` with
//!   `K = min(M, N)`.
//! - `s`: always `[K]`.

use core::cell::Cell;
use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_sys::{
    cusolverDnCreate, cusolverDnDestroy, cusolverDnDgesvd, cusolverDnDgesvd_bufferSize,
    cusolverDnHandle_t, cusolverDnSetStream, cusolverDnSgesvd, cusolverDnSgesvd_bufferSize,
};
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, LinalgKind, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, Workspace,
};

use super::cholesky::unpack_workspace;

/// Descriptor for an SVD.
#[derive(Copy, Clone, Debug)]
pub struct SvdDescriptor {
    /// Row count `M` of the input matrix (column-major; `m >= n`).
    pub m: i32,
    /// Column count `N` of the input matrix.
    pub n: i32,
    /// `true` → compute full `U` (`[M, M]`) and full `V^T` (`[N, N]`).
    /// `false` → compute thin `U` (`[M, K]`) and thin `V^T` (`[K, N]`).
    pub full_matrices: bool,
    /// Element type. Must be `F32` or `F64`.
    pub element: ElementKind,
}

/// Args bundle for an SVD launch.
///
/// `a` is **overwritten** by cuSOLVER's `gesvd` during the call. The
/// post-`run` contents of `a` are not meaningful (the routine uses it
/// as scratch).
pub struct SvdArgs<'a, T: Element> {
    /// Input matrix `[M, N]` column-major. Overwritten in place.
    pub a: TensorMut<'a, T, 2>,
    /// Singular values `[K]` where `K = min(M, N)`.
    pub s: TensorMut<'a, T, 1>,
    /// Left singular vectors. Shape `[M, M]` (full) or `[M, K]` (thin).
    pub u: TensorMut<'a, T, 2>,
    /// Right singular vectors (transposed). Shape `[N, N]` (full) or
    /// `[K, N]` (thin).
    pub vt: TensorMut<'a, T, 2>,
    /// Single-cell info: `0` on success.
    pub info: TensorMut<'a, i32, 1>,
}

/// Singular value decomposition plan — `A = U · diag(S) · V^T`.
///
/// Wraps cuSOLVER's `gesvd` (bidiagonal reduction + QR sweeps). Use
/// [`super::BatchedSvdPlan`] for batched square problems, or
/// [`super::BatchedSvdaPlan`] for batched rectangular / truncated SVD.
///
/// **Dtypes**: `f32`, `f64` only.
///
/// **Shape**: `[M, N]` with `m >= n` (cuSOLVER constraint — transpose
/// before invoking if you need `m < n`). `full_matrices` toggles
/// between full (`U: [M, M]`, `V^T: [N, N]`) and thin
/// (`U: [M, K]`, `V^T: [K, N]`) shapes where `K = min(M, N)`.
///
/// **Storage**: column-major end-to-end.
///
/// **Workspace**: cuSOLVER `_bufferSize` (queried lazily on first
/// `run`).
///
/// **Precision guarantee**: deterministic; not bit-stable across runs.
///
/// Owns a lazy cuSOLVER handle (`!Sync` / `!Send`); destroyed on `Drop`.
pub struct SvdPlan<T: Element> {
    desc: SvdDescriptor,
    sku: KernelSku,
    handle: Cell<cusolverDnHandle_t>,
    workspace_bytes: Cell<usize>,
    _marker: PhantomData<T>,
}

impl<T: Element> SvdPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(_stream: &Stream, desc: &SvdDescriptor, _pref: PlanPreference) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::SvdPlan: descriptor.element != T::KIND",
            ));
        }
        if !matches!(T::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::SvdPlan: cuSOLVER dense SVD supports f32 + f64 only",
            ));
        }
        if desc.m <= 0 || desc.n <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SvdPlan: m / n must be > 0",
            ));
        }
        if desc.m < desc.n {
            return Err(Error::Unsupported(
                "baracuda-kernels::SvdPlan: cuSOLVER gesvd requires m >= n",
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
            op: LinalgKind::Svd as u16,
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

    /// Workspace size in bytes. Lazily populated on first `run`.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        self.workspace_bytes.get()
    }

    /// Query and cache the cuSOLVER workspace size.
    pub fn query_workspace_size(&self, _stream: &Stream) -> Result<usize> {
        let h = self.ensure_handle()?;
        let mut lwork: i32 = 0;
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                cusolverDnSgesvd_bufferSize(h, self.desc.m, self.desc.n, &mut lwork as *mut _)
            },
            ElementKind::F64 => unsafe {
                cusolverDnDgesvd_bufferSize(h, self.desc.m, self.desc.n, &mut lwork as *mut _)
            },
            _ => unreachable!(),
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

    fn check_args(&self, args: &SvdArgs<'_, T>) -> Result<()> {
        let m = self.desc.m;
        let n = self.desc.n;
        let k = m.min(n);
        if args.a.shape != [m, n] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SvdPlan: A shape != [M, N]",
            ));
        }
        if args.s.shape != [k] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SvdPlan: S shape != [min(M, N)]",
            ));
        }
        let (u_cols, vt_rows) = if self.desc.full_matrices {
            (m, n)
        } else {
            (k, k)
        };
        if args.u.shape != [m, u_cols] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SvdPlan: U shape != [M, M] (full) or [M, K] (thin)",
            ));
        }
        if args.vt.shape != [vt_rows, n] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SvdPlan: V^T shape != [N, N] (full) or [K, N] (thin)",
            ));
        }
        if args.info.shape != [1] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SvdPlan: info shape != [1]",
            ));
        }
        Ok(())
    }
}

impl SvdPlan<f32> {
    /// Run SVD (f32).
    pub fn run(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: SvdArgs<'_, f32>,
    ) -> Result<()> {
        self.check_args(&args)?;
        let h = self.ensure_handle()?;
        self.bind_stream(h, stream)?;
        let m = self.desc.m;
        let n = self.desc.n;
        let needed = if self.workspace_bytes.get() == 0 {
            self.query_workspace_size(stream)?
        } else {
            self.workspace_bytes.get()
        };
        let (ws_ptr, ws_bytes) = unpack_workspace(workspace, needed)?;
        let lwork = (ws_bytes / core::mem::size_of::<f32>()) as i32;

        let jobu = if self.desc.full_matrices { b'A' } else { b'S' };
        let jobv = if self.desc.full_matrices { b'A' } else { b'S' };

        let a_ptr = args.a.data.as_raw().0 as *mut f32;
        let s_ptr = args.s.data.as_raw().0 as *mut f32;
        let u_ptr = args.u.data.as_raw().0 as *mut f32;
        let vt_ptr = args.vt.data.as_raw().0 as *mut f32;
        let info_ptr = args.info.data.as_raw().0 as *mut i32;
        let ldu = m;
        let ldvt = if self.desc.full_matrices { n } else { m.min(n) };

        let status = unsafe {
            cusolverDnSgesvd(
                h,
                jobu,
                jobv,
                m,
                n,
                a_ptr,
                m,
                s_ptr,
                u_ptr,
                ldu,
                vt_ptr,
                ldvt,
                ws_ptr as *mut f32,
                lwork,
                core::ptr::null_mut(),
                info_ptr,
            )
        };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        Ok(())
    }
}

impl SvdPlan<f64> {
    /// Run SVD (f64).
    pub fn run(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: SvdArgs<'_, f64>,
    ) -> Result<()> {
        self.check_args(&args)?;
        let h = self.ensure_handle()?;
        self.bind_stream(h, stream)?;
        let m = self.desc.m;
        let n = self.desc.n;
        let needed = if self.workspace_bytes.get() == 0 {
            self.query_workspace_size(stream)?
        } else {
            self.workspace_bytes.get()
        };
        let (ws_ptr, ws_bytes) = unpack_workspace(workspace, needed)?;
        let lwork = (ws_bytes / core::mem::size_of::<f64>()) as i32;

        let jobu = if self.desc.full_matrices { b'A' } else { b'S' };
        let jobv = if self.desc.full_matrices { b'A' } else { b'S' };

        let a_ptr = args.a.data.as_raw().0 as *mut f64;
        let s_ptr = args.s.data.as_raw().0 as *mut f64;
        let u_ptr = args.u.data.as_raw().0 as *mut f64;
        let vt_ptr = args.vt.data.as_raw().0 as *mut f64;
        let info_ptr = args.info.data.as_raw().0 as *mut i32;
        let ldu = m;
        let ldvt = if self.desc.full_matrices { n } else { m.min(n) };

        let status = unsafe {
            cusolverDnDgesvd(
                h,
                jobu,
                jobv,
                m,
                n,
                a_ptr,
                m,
                s_ptr,
                u_ptr,
                ldu,
                vt_ptr,
                ldvt,
                ws_ptr as *mut f64,
                lwork,
                core::ptr::null_mut(),
                info_ptr,
            )
        };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        Ok(())
    }
}

impl<T: Element> Drop for SvdPlan<T> {
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
