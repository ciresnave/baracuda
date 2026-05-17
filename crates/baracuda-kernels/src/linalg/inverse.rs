//! Matrix inverse `A^{-1}` via `getrf` + `getrs` over an identity RHS.
//!
//! cuSOLVER's dense API does not expose a direct `getri` equivalent
//! (the closest is `cusolverRfBatchSolve`, which lives in the
//! refactorization library and targets sparse-direct solvers). The
//! standard dense workaround is the LAPACK `getrf` + `getrs` pair with
//! `B = I` — exactly what this plan does. The cost is one extra dense
//! solve over an `M × M` identity instead of the specialized
//! `getri`-style back-substitution, but the wall-time difference is
//! negligible for the matrix sizes baracuda targets in this milestone.
//!
//! **2-D only** — single matrix. No batching today.
//!
//! **Semantics**:
//! - `a` is overwritten in place with the packed `LU` factors. Callers
//!   that need the original `A` preserved should copy before invoking.
//! - `inv` is overwritten in place with `A^{-1}`. The plan stages an
//!   identity matrix into `inv` at the start of `run` (host-side build
//!   + async H2D), then `getrs` solves `A · X = I` in place over `inv`
//!   giving `X = A^{-1}`.
//!
//! **Storage convention**: column-major end-to-end (matches the rest
//! of the linalg family's column-major trailblazers — LU / QR / SVD).
//! Because the identity matrix is symmetric, the row-major / column-
//! major distinction for `inv`'s initial fill is irrelevant — what
//! matters is that the byte storage of `inv` is interpreted column-
//! major by cuSOLVER, consistent with how `a` is interpreted.
//!
//! **Workspace**: same as [`super::solve::SolvePlan`] — only `getrf`'s
//! requirement (cuSOLVER `getrs` is workspace-free).

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

/// Descriptor for a matrix inverse.
#[derive(Copy, Clone, Debug)]
pub struct InverseDescriptor {
    /// Order `M` of the (square) input matrix.
    pub m: i32,
    /// Element type. Must be `F32` or `F64`.
    pub element: ElementKind,
}

/// Args bundle for an inverse launch.
///
/// `a` is overwritten in place with the packed `LU` factors. `inv` is
/// overwritten in place with `A^{-1}` — its contents on entry are
/// ignored (the plan stages an identity into it at the start of
/// `run`). `pivot` receives cuSOLVER's 1-based pivot indices.
pub struct InverseArgs<'a, T: Element> {
    /// Input matrix `[M, M]` (column-major). Overwritten with packed
    /// `LU` in place.
    pub a: TensorMut<'a, T, 2>,
    /// Output `[M, M]` (column-major). Receives `A^{-1}`. Contents on
    /// entry are ignored — the plan rewrites it.
    pub inv: TensorMut<'a, T, 2>,
    /// Pivot vector `[M]` (1-based per LAPACK convention).
    pub pivot: TensorMut<'a, i32, 1>,
    /// Single-cell info: `0` on success.
    pub info: TensorMut<'a, i32, 1>,
}

/// Matrix-inverse plan.
pub struct InversePlan<T: Element> {
    desc: InverseDescriptor,
    sku: KernelSku,
    handle: Cell<cusolverDnHandle_t>,
    workspace_bytes: Cell<usize>,
    _marker: PhantomData<T>,
}

impl<T: Element> InversePlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &InverseDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::InversePlan: descriptor.element != T::KIND",
            ));
        }
        if !matches!(T::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::InversePlan: cuSOLVER dense inverse supports f32 + f64 only",
            ));
        }
        if desc.m <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::InversePlan: m must be > 0",
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
            op: LinalgKind::Inverse as u16,
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

    fn check_args(&self, args: &InverseArgs<'_, T>) -> Result<()> {
        let m = self.desc.m;
        if args.a.shape != [m, m] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::InversePlan: A shape != [M, M]",
            ));
        }
        if args.inv.shape != [m, m] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::InversePlan: inv shape != [M, M]",
            ));
        }
        if args.pivot.shape != [m] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::InversePlan: pivot shape != [M]",
            ));
        }
        if args.info.shape != [1] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::InversePlan: info shape != [1]",
            ));
        }
        Ok(())
    }
}

// Macro to instantiate run() for f32 / f64.
macro_rules! impl_inverse_run {
    ($T:ty, $getrf:ident, $getrs:ident, $one:expr) => {
        impl InversePlan<$T> {
            /// Run the matrix inverse.
            pub fn run(
                &self,
                stream: &Stream,
                workspace: Workspace<'_>,
                args: InverseArgs<'_, $T>,
            ) -> Result<()> {
                self.check_args(&args)?;
                let h = self.ensure_handle()?;
                self.bind_stream(h, stream)?;
                let m = self.desc.m;

                let needed = if self.workspace_bytes.get() == 0 {
                    self.query_workspace_size(stream)?
                } else {
                    self.workspace_bytes.get()
                };
                let (ws_ptr, _ws_bytes) = unpack_workspace(workspace, needed)?;

                let a_ptr = args.a.data.as_raw().0 as *mut $T;
                let inv_ptr = args.inv.data.as_raw().0 as *mut $T;
                let pivot_ptr = args.pivot.data.as_raw().0 as *mut i32;
                let info_ptr = args.info.data.as_raw().0 as *mut i32;

                // 1. getrf — factors A in place, writes pivot + info.
                let status = unsafe {
                    $getrf(h, m, m, a_ptr, m, ws_ptr as *mut $T, pivot_ptr, info_ptr)
                };
                if status != 0 {
                    return Err(Error::CutlassInternal(-status));
                }

                // 2. Stage an identity matrix into `inv`. Host-side
                //    build then async H2D — same pattern QR uses for its
                //    `q` identity stage.
                let m_sz = (m as usize) * (m as usize);
                let mut host_id: Vec<$T> = vec![<$T as Default>::default(); m_sz];
                for i in 0..(m as usize) {
                    host_id[i * (m as usize) + i] = $one;
                }
                let elem_size = core::mem::size_of::<$T>();
                let bytes = m_sz * elem_size;
                unsafe {
                    copy_h2d(
                        inv_ptr as *mut c_void,
                        host_id.as_ptr() as *const c_void,
                        bytes,
                        stream,
                    )?;
                }

                // 3. getrs — solves A · X = I in place over `inv`. trans
                //    == N because storage is end-to-end column-major.
                let status = unsafe {
                    $getrs(
                        h,
                        CUBLAS_OP_N,
                        m,
                        m,
                        a_ptr as *const $T,
                        m,
                        pivot_ptr as *const i32,
                        inv_ptr,
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

impl_inverse_run!(f32, cusolverDnSgetrf, cusolverDnSgetrs, 1.0f32);
impl_inverse_run!(f64, cusolverDnDgetrf, cusolverDnDgetrs, 1.0f64);

impl<T: Element> Drop for InversePlan<T> {
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

// ----- H2D helper used to stage the identity matrix ---------------------

unsafe fn copy_h2d(
    dst: *mut c_void,
    src: *const c_void,
    bytes: usize,
    stream: &Stream,
) -> Result<()> {
    #[allow(non_camel_case_types)]
    type CUresult = i32;
    unsafe extern "system" {
        fn cuMemcpyHtoDAsync_v2(
            dst_device: u64,
            src_host: *const c_void,
            byte_count: usize,
            h_stream: *mut c_void,
        ) -> CUresult;
    }
    let status =
        unsafe { cuMemcpyHtoDAsync_v2(dst as u64, src, bytes, stream.as_raw() as *mut c_void) };
    if status != 0 {
        return Err(Error::CutlassInternal(-status));
    }
    // Sync so the host identity buffer can be freed when this function
    // returns. The getrs call that follows expects the device buffer to
    // be populated.
    stream.synchronize().map_err(Error::Driver)?;
    Ok(())
}
