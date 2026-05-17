//! Batched QR factorization — `A_b = Q_b · R_b` per batch slot.
//!
//! Wraps **cuBLAS**'s `cublasSgeqrfBatched` / `DgeqrfBatched` (note:
//! despite belonging to the linalg family conceptually, batched-QR lives
//! in cuBLAS, not cuSOLVER — cuSOLVER-Dn has no batched-geqrf entry
//! point). Unlike the non-batched [`super::qr::QrPlan`] which materializes
//! a dense `Q` via `ormqr` over an identity, this plan **only produces**
//! cuBLAS's native packed-output: each `Aarray[b]` is overwritten in
//! place with the `geqrf`-packed `R` (upper triangle) + Householder
//! reflectors (strict lower triangle), and `TauArray[b]` receives the
//! Householder scalars (length `K = min(M, N)`). Callers that need a
//! dense `Q` / `R` per batch slot can post-process via the non-batched
//! plan in a follow-up sweep, or apply `ormqr` themselves.
//!
//! **Dtypes**: `f32` + `f64` only — cuBLAS's batched-geqrf API does
//! not expose `f16` / `bf16`.
//!
//! **Storage convention**: column-major end-to-end, matching the
//! non-batched neighbor.
//!
//! **Workspace layout**: the batched routine is itself workspace-free
//! (cuBLAS allocates internally), but the plan needs a small piece of
//! caller workspace to lift two device-pointer arrays (`Aarray[]` and
//! `TauArray[]`) onto the device. The plan reports `2 * batch_size *
//! sizeof(void*)` bytes through [`BatchedQrPlan::workspace_size`].

use core::cell::Cell;
use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_sys::{
    cublasCgeqrfBatched, cublasCreate_v2, cublasDestroy_v2, cublasDgeqrfBatched, cublasHandle_t,
    cublasSetStream_v2, cublasSgeqrfBatched, cublasZgeqrfBatched,
};
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, LinalgKind, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, Workspace,
};

use super::cholesky::{copy_h2d, unpack_workspace};

/// Descriptor for a batched QR factorization.
#[derive(Copy, Clone, Debug)]
pub struct BatchedQrDescriptor {
    /// Row count `M` of each matrix in the batch.
    pub m: i32,
    /// Column count `N` of each matrix in the batch.
    pub n: i32,
    /// Number of independent matrices in the batch.
    pub batch_size: i32,
    /// Element type. Must be `F32` or `F64`.
    pub element: ElementKind,
}

/// Args bundle for a batched-QR launch.
///
/// `a` is **overwritten in place** with cuBLAS's packed-output (`R`
/// upper + Householder reflectors lower). `tau` receives the
/// Householder scalars, length `K = min(M, N)` per batch slot.
///
/// cuBLAS's batched-geqrf contract returns a single host-side
/// argument-validity flag (NOT a per-slot device array); any non-zero
/// value surfaces as [`Error::CutlassInternal`] from `run`.
pub struct BatchedQrArgs<'a, T: Element> {
    /// Input batch stack `[batch, M, N]` column-major contiguous.
    /// Overwritten with cuBLAS `geqrf`-packed output per batch slot.
    pub a: TensorMut<'a, T, 3>,
    /// Householder scalars `[batch, K]` where `K = min(M, N)`.
    pub tau: TensorMut<'a, T, 2>,
}

/// Batched QR factorization plan.
pub struct BatchedQrPlan<T: Element> {
    desc: BatchedQrDescriptor,
    sku: KernelSku,
    handle: Cell<cublasHandle_t>,
    workspace_bytes: Cell<usize>,
    _marker: PhantomData<T>,
}

impl<T: Element> BatchedQrPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &BatchedQrDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::BatchedQrPlan: descriptor.element != T::KIND",
            ));
        }
        if !matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F64 | ElementKind::Complex32 | ElementKind::Complex64
        ) {
            return Err(Error::Unsupported(
                "baracuda-kernels::BatchedQrPlan: cuBLAS batched QR supports f32 / f64 / \
                 Complex32 / Complex64",
            ));
        }
        if desc.m <= 0 || desc.n <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchedQrPlan: m / n must be > 0",
            ));
        }
        if desc.batch_size <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchedQrPlan: batch_size must be > 0",
            ));
        }
        if desc.m < desc.n {
            return Err(Error::Unsupported(
                "baracuda-kernels::BatchedQrPlan: cuSOLVER geqrfBatched requires m >= n",
            ));
        }

        let math_precision = match T::KIND {
            ElementKind::F64 | ElementKind::Complex64 => MathPrecision::F64,
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
            op: LinalgKind::BatchedQr as u16,
            element: T::KIND,
            aux_element: Some(ElementKind::I32),
            layout: None,
            epilogue: None,
            arch: ArchSku::Sm80,
            backend: BackendKind::Cublas,
            precision_guarantee,
        };
        // The batched routine is workspace-free internally but we need
        // device-resident pointer arrays for Aarray + TauArray, each
        // `batch_size * sizeof(*mut T)` bytes.
        let ptr_bytes = 2 * (desc.batch_size as usize) * core::mem::size_of::<u64>();
        Ok(Self {
            desc: *desc,
            sku,
            handle: Cell::new(core::ptr::null_mut()),
            workspace_bytes: Cell::new(ptr_bytes),
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

    /// Workspace size in bytes — two device-resident pointer arrays
    /// (`Aarray[]`, `TauArray[]`).
    #[inline]
    pub fn workspace_size(&self) -> usize {
        self.workspace_bytes.get()
    }

    /// Returns the workspace requirement; populated at `select` time so
    /// this is just a getter that matches the cross-plan API shape.
    pub fn query_workspace_size(&self, _stream: &Stream) -> Result<usize> {
        Ok(self.workspace_bytes.get())
    }

    fn ensure_handle(&self) -> Result<cublasHandle_t> {
        let h = self.handle.get();
        if !h.is_null() {
            return Ok(h);
        }
        let mut handle: cublasHandle_t = core::ptr::null_mut();
        let status = unsafe { cublasCreate_v2(&mut handle as *mut _) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        self.handle.set(handle);
        Ok(handle)
    }

    fn bind_stream(&self, h: cublasHandle_t, stream: &Stream) -> Result<()> {
        let status = unsafe { cublasSetStream_v2(h, stream.as_raw() as *mut c_void) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        Ok(())
    }

    fn check_args(&self, args: &BatchedQrArgs<'_, T>) -> Result<()> {
        let b = self.desc.batch_size;
        let m = self.desc.m;
        let n = self.desc.n;
        let k = m.min(n);
        if args.a.shape != [b, m, n] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchedQrPlan: A shape != [batch, M, N]",
            ));
        }
        if args.tau.shape != [b, k] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchedQrPlan: tau shape != [batch, min(M, N)]",
            ));
        }
        Ok(())
    }
}

macro_rules! impl_batched_qr_run {
    ($T:ty, $geqrf_batched:ident, $Cublas:ty) => {
        impl BatchedQrPlan<$T> {
            /// Run the batched QR factorization.
            pub fn run(
                &self,
                stream: &Stream,
                workspace: Workspace<'_>,
                args: BatchedQrArgs<'_, $T>,
            ) -> Result<()> {
                self.check_args(&args)?;
                let h = self.ensure_handle()?;
                self.bind_stream(h, stream)?;
                let b = self.desc.batch_size;
                let m = self.desc.m;
                let n = self.desc.n;
                let k = m.min(n);

                let needed = self.workspace_bytes.get();
                let (ws_ptr, ws_bytes) = unpack_workspace(workspace, needed)?;
                if ws_bytes < needed {
                    return Err(Error::WorkspaceTooSmall {
                        needed,
                        got: ws_bytes,
                    });
                }

                // Build host arrays of per-slot device pointers, then
                // copy them into the caller's workspace (lifted to device).
                let a_base = args.a.data.as_raw().0;
                let tau_base = args.tau.data.as_raw().0;
                let elem_size = core::mem::size_of::<$T>() as u64;
                let a_stride = (m as u64) * (n as u64) * elem_size;
                let tau_stride = (k as u64) * elem_size;
                let bu = b as usize;
                let mut host_a_ptrs: Vec<u64> = Vec::with_capacity(bu);
                let mut host_tau_ptrs: Vec<u64> = Vec::with_capacity(bu);
                for i in 0..b {
                    host_a_ptrs.push(a_base + (i as u64) * a_stride);
                    host_tau_ptrs.push(tau_base + (i as u64) * tau_stride);
                }
                let a_array_bytes = bu * core::mem::size_of::<u64>();
                let a_array_ptr = ws_ptr;
                let tau_array_ptr = unsafe { (ws_ptr as *mut u8).add(a_array_bytes) as *mut c_void };
                unsafe {
                    copy_h2d(
                        a_array_ptr,
                        host_a_ptrs.as_ptr() as *const c_void,
                        a_array_bytes,
                        stream,
                    )?;
                    copy_h2d(
                        tau_array_ptr,
                        host_tau_ptrs.as_ptr() as *const c_void,
                        a_array_bytes,
                        stream,
                    )?;
                }

                let mut host_info: i32 = 0;
                let status = unsafe {
                    $geqrf_batched(
                        h,
                        m,
                        n,
                        a_array_ptr as *mut *mut $Cublas,
                        m,
                        tau_array_ptr as *mut *mut $Cublas,
                        &mut host_info as *mut i32,
                        b,
                    )
                };
                if status != 0 {
                    return Err(Error::CutlassInternal(-status));
                }
                if host_info != 0 {
                    return Err(Error::CutlassInternal(-host_info));
                }
                Ok(())
            }
        }
    };
}

impl_batched_qr_run!(f32, cublasSgeqrfBatched, f32);
impl_batched_qr_run!(f64, cublasDgeqrfBatched, f64);
impl_batched_qr_run!(
    baracuda_kernels_types::Complex32,
    cublasCgeqrfBatched,
    baracuda_kernels_sys::cuComplex
);
impl_batched_qr_run!(
    baracuda_kernels_types::Complex64,
    cublasZgeqrfBatched,
    baracuda_kernels_sys::cuDoubleComplex
);

impl<T: Element> Drop for BatchedQrPlan<T> {
    fn drop(&mut self) {
        let h = self.handle.get();
        if !h.is_null() {
            unsafe {
                let _ = cublasDestroy_v2(h);
            }
            self.handle.set(core::ptr::null_mut());
        }
    }
}
