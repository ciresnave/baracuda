//! Batched singular value decomposition — Jacobi method.
//!
//! Wraps cuSOLVER's `cusolverDnSgesvdjBatched` / `DgesvdjBatched`. Unlike
//! the non-batched [`super::svd::SvdPlan`] which uses `gesvd` (R-bidiag
//! + QR-sweep), this plan uses the **one-sided Jacobi** method
//! (`gesvdj`), which cuSOLVER batches across independent matrices.
//!
//! **Constraints**:
//! - cuSOLVER's batched-Jacobi-SVD requires **square** input matrices
//!   (`m == n`). Rectangular batched SVD is achievable via
//!   `gesvdaStridedBatched` (approximate-SVD) — deferred.
//! - The routine returns `V` directly (not `V^T`). Callers that want
//!   `V^T` apply the transpose themselves.
//! - Dtypes: `f32` + `f64` only.
//!
//! **Storage**: column-major end-to-end, matching the neighbors.
//!
//! **Convergence**: Jacobi-SVD is iterative; the default `gesvdjInfo_t`
//! parameter object sets `tol = 1e-7` (f32) / `1e-12` (f64) with
//! `max_sweeps = 100`. Per-matrix `info[b] > 0` indicates the
//! `b`-th matrix did not converge within the sweep limit; callers
//! should inspect `info` after the call.

use core::cell::Cell;
use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_sys::{
    cusolverDnCreate, cusolverDnCreateGesvdjInfo, cusolverDnDestroy, cusolverDnDestroyGesvdjInfo,
    cusolverDnDgesvdjBatched, cusolverDnDgesvdjBatched_bufferSize, cusolverDnHandle_t,
    cusolverDnSetStream, cusolverDnSgesvdjBatched, cusolverDnSgesvdjBatched_bufferSize,
    gesvdjInfo_t, CUSOLVER_EIG_MODE_NOVECTOR, CUSOLVER_EIG_MODE_VECTOR,
};
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, LinalgKind, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, Workspace,
};

use super::cholesky::unpack_workspace;

/// Descriptor for a batched SVD.
#[derive(Copy, Clone, Debug)]
pub struct BatchedSvdDescriptor {
    /// Matrix size `N` of each square matrix in the batch. cuSOLVER's
    /// Jacobi-batched SVD only accepts square matrices.
    pub matrix_size: i32,
    /// Number of independent matrices in the batch.
    pub batch_size: i32,
    /// `true` requests singular vectors (`U` + `V`); `false` computes
    /// only the singular values (`S`).
    pub compute_vectors: bool,
    /// Element type. Must be `F32` or `F64`.
    pub element: ElementKind,
}

/// Args bundle for a batched-SVD launch.
///
/// `a` is **overwritten** by cuSOLVER during the call (used as scratch
/// for the Jacobi sweeps). When `compute_vectors == false`, the `u` /
/// `v` tensor fields are unused (the caller may pass zero-sized
/// tensors — the plan does not read them).
pub struct BatchedSvdArgs<'a, T: Element> {
    /// Input stack `[batch, N, N]` column-major. Overwritten in place.
    pub a: TensorMut<'a, T, 3>,
    /// Singular values `[batch, N]`.
    pub s: TensorMut<'a, T, 2>,
    /// Left singular vectors `[batch, N, N]` (column-major). Only
    /// written when `compute_vectors == true`.
    pub u: TensorMut<'a, T, 3>,
    /// Right singular vectors `[batch, N, N]` (column-major — note this
    /// is `V`, not `V^T`). Only written when `compute_vectors == true`.
    pub v: TensorMut<'a, T, 3>,
    /// Per-matrix info `[batch]`: `0` on success, `> 0` if non-converged.
    pub info: TensorMut<'a, i32, 1>,
}

/// Batched SVD plan (Jacobi).
pub struct BatchedSvdPlan<T: Element> {
    desc: BatchedSvdDescriptor,
    sku: KernelSku,
    handle: Cell<cusolverDnHandle_t>,
    params: Cell<gesvdjInfo_t>,
    workspace_bytes: Cell<usize>,
    _marker: PhantomData<T>,
}

impl<T: Element> BatchedSvdPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &BatchedSvdDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::BatchedSvdPlan: descriptor.element != T::KIND",
            ));
        }
        if !matches!(T::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::BatchedSvdPlan: cuSOLVER batched SVD supports f32 + f64 only",
            ));
        }
        if desc.matrix_size <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchedSvdPlan: matrix_size must be > 0",
            ));
        }
        if desc.batch_size <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchedSvdPlan: batch_size must be > 0",
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
            op: LinalgKind::BatchedSvd as u16,
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
            params: Cell::new(core::ptr::null_mut()),
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

    fn jobz(&self) -> i32 {
        if self.desc.compute_vectors {
            CUSOLVER_EIG_MODE_VECTOR
        } else {
            CUSOLVER_EIG_MODE_NOVECTOR
        }
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

    fn ensure_params(&self) -> Result<gesvdjInfo_t> {
        let p = self.params.get();
        if !p.is_null() {
            return Ok(p);
        }
        let mut params: gesvdjInfo_t = core::ptr::null_mut();
        let status = unsafe { cusolverDnCreateGesvdjInfo(&mut params as *mut _) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        self.params.set(params);
        Ok(params)
    }

    fn bind_stream(&self, h: cusolverDnHandle_t, stream: &Stream) -> Result<()> {
        let status = unsafe { cusolverDnSetStream(h, stream.as_raw() as *mut c_void) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        Ok(())
    }

    /// Materialize the handle + params and query workspace size.
    pub fn query_workspace_size(&self, _stream: &Stream) -> Result<usize> {
        let h = self.ensure_handle()?;
        let p = self.ensure_params()?;
        let n = self.desc.matrix_size;
        let b = self.desc.batch_size;
        let mut lwork: i32 = 0;
        let jobz = self.jobz();
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                cusolverDnSgesvdjBatched_bufferSize(
                    h,
                    jobz,
                    n,
                    n,
                    core::ptr::null(),
                    n,
                    core::ptr::null(),
                    core::ptr::null(),
                    n,
                    core::ptr::null(),
                    n,
                    &mut lwork as *mut _,
                    p,
                    b,
                )
            },
            ElementKind::F64 => unsafe {
                cusolverDnDgesvdjBatched_bufferSize(
                    h,
                    jobz,
                    n,
                    n,
                    core::ptr::null(),
                    n,
                    core::ptr::null(),
                    core::ptr::null(),
                    n,
                    core::ptr::null(),
                    n,
                    &mut lwork as *mut _,
                    p,
                    b,
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

    fn check_args(&self, args: &BatchedSvdArgs<'_, T>) -> Result<()> {
        let b = self.desc.batch_size;
        let n = self.desc.matrix_size;
        if args.a.shape != [b, n, n] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchedSvdPlan: A shape != [batch, N, N]",
            ));
        }
        if args.s.shape != [b, n] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchedSvdPlan: S shape != [batch, N]",
            ));
        }
        if self.desc.compute_vectors {
            if args.u.shape != [b, n, n] {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::BatchedSvdPlan: U shape != [batch, N, N]",
                ));
            }
            if args.v.shape != [b, n, n] {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::BatchedSvdPlan: V shape != [batch, N, N]",
                ));
            }
        }
        if args.info.shape != [b] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchedSvdPlan: info shape != [batch]",
            ));
        }
        Ok(())
    }
}

macro_rules! impl_batched_svd_run {
    ($T:ty, $gesvdj_batched:ident) => {
        impl BatchedSvdPlan<$T> {
            /// Run the batched Jacobi SVD.
            pub fn run(
                &self,
                stream: &Stream,
                workspace: Workspace<'_>,
                args: BatchedSvdArgs<'_, $T>,
            ) -> Result<()> {
                self.check_args(&args)?;
                let h = self.ensure_handle()?;
                let p = self.ensure_params()?;
                self.bind_stream(h, stream)?;
                let n = self.desc.matrix_size;
                let b = self.desc.batch_size;

                let needed = if self.workspace_bytes.get() == 0 {
                    self.query_workspace_size(stream)?
                } else {
                    self.workspace_bytes.get()
                };
                let (ws_ptr, ws_bytes) = unpack_workspace(workspace, needed)?;
                let lwork = (ws_bytes / core::mem::size_of::<$T>()) as i32;

                let a_ptr = args.a.data.as_raw().0 as *mut $T;
                let s_ptr = args.s.data.as_raw().0 as *mut $T;
                let u_ptr = if self.desc.compute_vectors {
                    args.u.data.as_raw().0 as *mut $T
                } else {
                    core::ptr::null_mut()
                };
                let v_ptr = if self.desc.compute_vectors {
                    args.v.data.as_raw().0 as *mut $T
                } else {
                    core::ptr::null_mut()
                };
                let info_ptr = args.info.data.as_raw().0 as *mut i32;

                let status = unsafe {
                    $gesvdj_batched(
                        h,
                        self.jobz(),
                        n,
                        n,
                        a_ptr,
                        n,
                        s_ptr,
                        u_ptr,
                        n,
                        v_ptr,
                        n,
                        ws_ptr as *mut $T,
                        lwork,
                        info_ptr,
                        p,
                        b,
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

impl_batched_svd_run!(f32, cusolverDnSgesvdjBatched);
impl_batched_svd_run!(f64, cusolverDnDgesvdjBatched);

impl<T: Element> Drop for BatchedSvdPlan<T> {
    fn drop(&mut self) {
        let p = self.params.get();
        if !p.is_null() {
            unsafe {
                let _ = cusolverDnDestroyGesvdjInfo(p);
            }
            self.params.set(core::ptr::null_mut());
        }
        let h = self.handle.get();
        if !h.is_null() {
            unsafe {
                let _ = cusolverDnDestroy(h);
            }
            self.handle.set(core::ptr::null_mut());
        }
    }
}
