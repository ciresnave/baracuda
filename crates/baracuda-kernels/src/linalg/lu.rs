//! LU factorization with partial pivoting — `P · A = L · U`.
//!
//! Wraps cuSOLVER's `cusolverDnSgetrf` / `Dgetrf` (non-batched, supports
//! rectangular `[M, N]`) and `cusolverDnSgetrfBatched` /
//! `DgetrfBatched` (square-only batched).
//!
//! **In-place**: cuSOLVER overwrites the input matrix with the packed
//! `LU` factors (L stored in the strict lower triangle with implicit
//! unit diagonal, U in the upper triangle including the diagonal). The
//! `pivot` tensor receives the per-step row swaps (1-based per LAPACK
//! convention).
//!
//! **Row-major adapter**: see [`super`] for the column-major bridge.
//! The caller's row-major `[M, N]` input is interpreted as cuSOLVER's
//! column-major `[N, M]`. The resulting `LU` is the factorization of
//! `A^T` in cuSOLVER's view; reconstruction in the row-major buffer
//! requires the same interpretation throughout.

use core::cell::Cell;
use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_sys::{
    cusolverDnCreate, cusolverDnDestroy, cusolverDnDgetrf, cusolverDnDgetrf_bufferSize,
    cusolverDnHandle_t, cusolverDnSetStream, cusolverDnSgetrf, cusolverDnSgetrf_bufferSize,
};
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, LinalgKind, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, Workspace,
};

use super::cholesky::unpack_workspace;

/// Descriptor for an LU factorization.
#[derive(Copy, Clone, Debug)]
pub struct LuDescriptor {
    /// Row count `M` of each input matrix.
    pub m: i32,
    /// Column count `N` of each input matrix.
    pub n: i32,
    /// Number of independent matrices. Only `1` is wired in this
    /// milestone — cuSOLVER's dense API does not expose a batched
    /// `getrf` (the batched routine lives in cuBLAS). Future
    /// milestones can route batched LU through `cublasSgetrfBatched`.
    pub batch_size: i32,
    /// Element type. Must be `F32` or `F64`.
    pub element: ElementKind,
}

/// Args bundle for an LU launch.
pub struct LuArgs<'a, T: Element> {
    /// Input / output matrix stack `[batch, M, N]` row-major contiguous.
    /// Overwritten with packed `LU` in place.
    pub a: TensorMut<'a, T, 3>,
    /// Pivot tensor `[batch, min(M, N)]`. cuSOLVER returns 1-based
    /// indices per the LAPACK convention.
    pub pivot: TensorMut<'a, i32, 2>,
    /// Per-matrix info: `0` on success, `k > 0` if the factorization
    /// found `U[k, k] == 0` at step `k`.
    pub info: TensorMut<'a, i32, 1>,
}

/// LU factorization plan.
pub struct LuPlan<T: Element> {
    desc: LuDescriptor,
    sku: KernelSku,
    handle: Cell<cusolverDnHandle_t>,
    workspace_bytes: Cell<usize>,
    _marker: PhantomData<T>,
}

impl<T: Element> LuPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &LuDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::LuPlan: descriptor.element != T::KIND",
            ));
        }
        if !matches!(T::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::LuPlan: cuSOLVER dense LU supports f32 + f64 only",
            ));
        }
        if desc.m <= 0 || desc.n <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::LuPlan: m / n must be > 0",
            ));
        }
        if desc.batch_size != 1 {
            return Err(Error::Unsupported(
                "baracuda-kernels::LuPlan: only batch_size == 1 is wired today \
                 (cuSOLVER dense `getrf` is non-batched; cuBLAS batched LU is deferred)",
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
            op: LinalgKind::Lu as u16,
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

    /// Materialize the handle and query workspace size.
    pub fn query_workspace_size(&self, _stream: &Stream) -> Result<usize> {
        let h = self.ensure_handle()?;
        let mut lwork: i32 = 0;
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                cusolverDnSgetrf_bufferSize(
                    h,
                    self.desc.m,
                    self.desc.n,
                    core::ptr::null_mut(),
                    self.desc.m,
                    &mut lwork as *mut _,
                )
            },
            ElementKind::F64 => unsafe {
                cusolverDnDgetrf_bufferSize(
                    h,
                    self.desc.m,
                    self.desc.n,
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

    fn check_args(&self, args: &LuArgs<'_, T>) -> Result<()> {
        let exp_a = [self.desc.batch_size, self.desc.m, self.desc.n];
        if args.a.shape != exp_a {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::LuPlan: A shape != [batch, M, N]",
            ));
        }
        let k = self.desc.m.min(self.desc.n);
        if args.pivot.shape != [self.desc.batch_size, k] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::LuPlan: pivot shape != [batch, min(M, N)]",
            ));
        }
        if args.info.shape != [self.desc.batch_size] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::LuPlan: info shape != [batch]",
            ));
        }
        let needed_a = (self.desc.batch_size as i64) * (self.desc.m as i64) * (self.desc.n as i64);
        if (args.a.data.len() as i64) < needed_a {
            return Err(Error::BufferTooSmall {
                needed: needed_a as usize,
                got: args.a.data.len(),
            });
        }
        Ok(())
    }
}

// ----- f32 -----

impl LuPlan<f32> {
    /// Run the LU factorization (f32).
    pub fn run(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: LuArgs<'_, f32>,
    ) -> Result<()> {
        self.check_args(&args)?;
        let h = self.ensure_handle()?;
        self.bind_stream(h, stream)?;
        let a_ptr = args.a.data.as_raw().0 as *mut f32;
        let pivot_ptr = args.pivot.data.as_raw().0 as *mut i32;
        let info_ptr = args.info.data.as_raw().0 as *mut i32;
        let m = self.desc.m;
        let n = self.desc.n;

        let needed = if self.workspace_bytes.get() == 0 {
            self.query_workspace_size(stream)?
        } else {
            self.workspace_bytes.get()
        };
        let (ws_ptr, _ws_bytes) = unpack_workspace(workspace, needed)?;
        let status = unsafe {
            cusolverDnSgetrf(h, m, n, a_ptr, m, ws_ptr as *mut f32, pivot_ptr, info_ptr)
        };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        Ok(())
    }
}

// ----- f64 -----

impl LuPlan<f64> {
    /// Run the LU factorization (f64).
    pub fn run(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: LuArgs<'_, f64>,
    ) -> Result<()> {
        self.check_args(&args)?;
        let h = self.ensure_handle()?;
        self.bind_stream(h, stream)?;
        let a_ptr = args.a.data.as_raw().0 as *mut f64;
        let pivot_ptr = args.pivot.data.as_raw().0 as *mut i32;
        let info_ptr = args.info.data.as_raw().0 as *mut i32;
        let m = self.desc.m;
        let n = self.desc.n;

        let needed = if self.workspace_bytes.get() == 0 {
            self.query_workspace_size(stream)?
        } else {
            self.workspace_bytes.get()
        };
        let (ws_ptr, _ws_bytes) = unpack_workspace(workspace, needed)?;
        let status = unsafe {
            cusolverDnDgetrf(h, m, n, a_ptr, m, ws_ptr as *mut f64, pivot_ptr, info_ptr)
        };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        Ok(())
    }
}

impl<T: Element> Drop for LuPlan<T> {
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
