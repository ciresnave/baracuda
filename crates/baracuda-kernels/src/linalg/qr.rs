//! QR factorization — `A = Q · R`.
//!
//! Wraps cuSOLVER's `cusolverDnSgeqrf` / `Dgeqrf` (Householder
//! reflectors + tau) followed by `cusolverDnSormqr` / `Dormqr` to
//! materialize `Q` as a dense matrix by applying `Q` to an identity.
//!
//! **2-D only** — cuSOLVER's dense API has no batched `geqrf`.
//!
//! **Outputs** (full mode; `K = min(M, N)`):
//! - `Q`: `[M, M]`
//! - `R`: `[M, N]` (upper-triangular in column-major; the strict lower
//!   triangle is not touched on the GPU side by this plan — callers
//!   who need a strict R must zero it post-hoc).
//!
//! **Storage convention**: cuSOLVER is column-major. To keep the
//! trailblazer simple, this plan **passes through** the column-major
//! interpretation — the caller's buffer is interpreted column-major
//! end-to-end. The descriptor's `m` / `n` are cuSOLVER's M / N.
//! Reconstruction `Q · R == A` therefore holds in the column-major
//! view of the byte storage (which is also the only view cuSOLVER
//! produces). For row-major inputs, callers may transpose before /
//! after via a future shape-layout op, or use square symmetric
//! matrices in tests where row / column views coincide.
//!
//! **Workspace** layout: the plan needs space for
//! 1. `tau` (length `K` × `sizeof(T)`)
//! 2. `geqrf`'s `lwork` (queried via `_bufferSize`)
//! 3. `ormqr`'s `lwork` (queried via `_bufferSize`)
//!
//! The plan reports `tau + max(geqrf_lwork, ormqr_lwork) * sizeof(T)`.
//! `geqrf` and `ormqr` share the workspace tail because they run
//! sequentially on the same stream.

use core::cell::Cell;
use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_sys::{
    cusolverDnCreate, cusolverDnDestroy, cusolverDnDgeqrf, cusolverDnDgeqrf_bufferSize,
    cusolverDnDormqr, cusolverDnDormqr_bufferSize, cusolverDnHandle_t, cusolverDnSetStream,
    cusolverDnSgeqrf, cusolverDnSgeqrf_bufferSize, cusolverDnSormqr, cusolverDnSormqr_bufferSize,
    CUBLAS_OP_N, CUBLAS_SIDE_LEFT,
};
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, LinalgKind, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, Workspace,
};

use super::cholesky::unpack_workspace;

/// Descriptor for a QR factorization.
#[derive(Copy, Clone, Debug)]
pub struct QrDescriptor {
    /// Row count `M` of the input matrix (cuSOLVER column-major).
    pub m: i32,
    /// Column count `N` of the input matrix.
    pub n: i32,
    /// Element type. Must be `F32` or `F64`.
    pub element: ElementKind,
}

/// Args bundle for a QR launch.
///
/// `a` is **overwritten in place** by `geqrf`. The plan reads it again
/// during `ormqr` to apply `Q` to an identity, so the post-`run`
/// contents of `a` are the cuSOLVER `geqrf` output (Householder
/// reflectors + R packed). `q` receives the dense `Q` (`[M, M]`); `r`
/// is written by a small host-side trampoline (see plan docs).
pub struct QrArgs<'a, T: Element> {
    /// Input matrix `[M, N]` column-major. Overwritten in place by
    /// `geqrf`.
    pub a: TensorMut<'a, T, 2>,
    /// Q output, `[M, M]` column-major.
    pub q: TensorMut<'a, T, 2>,
    /// R output, `[M, N]` column-major. Upper-triangular on return;
    /// strict lower triangle is zeroed.
    pub r: TensorMut<'a, T, 2>,
    /// Single-cell info: `0` on success.
    pub info: TensorMut<'a, i32, 1>,
}

/// QR factorization plan.
pub struct QrPlan<T: Element> {
    desc: QrDescriptor,
    sku: KernelSku,
    handle: Cell<cusolverDnHandle_t>,
    workspace_bytes: Cell<usize>,
    _marker: PhantomData<T>,
}

impl<T: Element> QrPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(_stream: &Stream, desc: &QrDescriptor, _pref: PlanPreference) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::QrPlan: descriptor.element != T::KIND",
            ));
        }
        if !matches!(T::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::QrPlan: cuSOLVER dense QR supports f32 + f64 only",
            ));
        }
        if desc.m <= 0 || desc.n <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::QrPlan: m / n must be > 0",
            ));
        }
        if desc.m < desc.n {
            return Err(Error::Unsupported(
                "baracuda-kernels::QrPlan: cuSOLVER geqrf requires m >= n",
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
            op: LinalgKind::Qr as u16,
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
    /// Layout: `tau_bytes + max(geqrf_lwork, ormqr_lwork) * sizeof(T)`.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        self.workspace_bytes.get()
    }

    /// Materialize the handle and query workspace size.
    pub fn query_workspace_size(&self, _stream: &Stream) -> Result<usize> {
        let h = self.ensure_handle()?;
        let m = self.desc.m;
        let n = self.desc.n;
        let k = m.min(n);
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
        let status_ormqr = match T::KIND {
            ElementKind::F32 => unsafe {
                cusolverDnSormqr_bufferSize(
                    h,
                    CUBLAS_SIDE_LEFT,
                    CUBLAS_OP_N,
                    m,
                    m,
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
                    CUBLAS_OP_N,
                    m,
                    m,
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

    fn check_args(&self, args: &QrArgs<'_, T>) -> Result<()> {
        let m = self.desc.m;
        let n = self.desc.n;
        if args.a.shape != [m, n] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::QrPlan: A shape != [M, N]",
            ));
        }
        if args.q.shape != [m, m] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::QrPlan: Q shape != [M, M]",
            ));
        }
        if args.r.shape != [m, n] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::QrPlan: R shape != [M, N]",
            ));
        }
        if args.info.shape != [1] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::QrPlan: info shape != [1]",
            ));
        }
        Ok(())
    }
}

// Macro to instantiate run() for f32 / f64. The bodies are nearly
// identical modulo the dtype-tagged cuSOLVER entry points.
//
// Implementation note: cuSOLVER's `geqrf` packs `R` (upper triangle)
// + Householder vectors (strict lower triangle) into `A` in place,
// and writes the `tau` Householder scalars to a separate buffer.
// Extracting a dense `R` and a dense `Q` requires two follow-up
// steps:
//
//   - **R**: copy `A` into the caller's `r` buffer and zero the
//     strict lower triangle. We round-trip through the host because
//     baracuda's public driver API exposes async H2D on slice-mut
//     views but no D2H; this milestone calls the CUDA driver API
//     directly via the `cuda` system library (statically linked from
//     baracuda-kernels-sys/build.rs). The matrices in this milestone
//     are small (typically ≤ 64×64), so the host round-trip is
//     cheap. A future tuning pass can replace this with a one-block
//     triangular-copy kernel on the device.
//
//   - **Q**: stage an identity matrix into `q`, then `ormqr` overwrites
//     it in place with `Q · I = Q`.
macro_rules! impl_qr_run {
    ($T:ty, $geqrf:ident, $ormqr:ident) => {
        impl QrPlan<$T> {
            /// Run the QR factorization.
            pub fn run(
                &self,
                stream: &Stream,
                workspace: Workspace<'_>,
                args: QrArgs<'_, $T>,
            ) -> Result<()> {
                self.check_args(&args)?;
                let h = self.ensure_handle()?;
                self.bind_stream(h, stream)?;
                let m = self.desc.m;
                let n = self.desc.n;
                let k = m.min(n);

                let needed = if self.workspace_bytes.get() == 0 {
                    self.query_workspace_size(stream)?
                } else {
                    self.workspace_bytes.get()
                };
                let (ws_ptr, ws_bytes) = unpack_workspace(workspace, needed)?;
                let elem_size = core::mem::size_of::<$T>();
                let tau_bytes = (k as usize) * elem_size;
                let tail_bytes = ws_bytes.saturating_sub(tau_bytes);
                let tau_ptr = ws_ptr as *mut $T;
                let tail_ptr = unsafe { (ws_ptr as *mut u8).add(tau_bytes) as *mut $T };
                let lwork = (tail_bytes / elem_size) as i32;

                let a_ptr = args.a.data.as_raw().0 as *mut $T;
                let info_ptr = args.info.data.as_raw().0 as *mut i32;

                // 1. geqrf — overwrites `a` with R (upper) + Householder
                //    vectors (strict lower), populates `tau`.
                let status = unsafe {
                    $geqrf(h, m, n, a_ptr, m, tau_ptr, tail_ptr, lwork, info_ptr)
                };
                if status != 0 {
                    return Err(Error::CutlassInternal(-status));
                }

                // 2. D2H the post-`geqrf` `a` into a host vector, zero
                //    the strict lower triangle, H2D into `r`.
                let elem_count = (m as usize) * (n as usize);
                let bytes = elem_count * elem_size;
                let mut host_a: Vec<$T> = vec![<$T as Default>::default(); elem_count];
                unsafe {
                    copy_d2h(host_a.as_mut_ptr() as *mut c_void, a_ptr as *const c_void, bytes, stream)?;
                }
                stream.synchronize().map_err(Error::Driver)?;
                for j in 0..(n as usize) {
                    for i in 0..(m as usize) {
                        if i > j {
                            host_a[j * (m as usize) + i] = <$T as Default>::default();
                        }
                    }
                }
                let r_ptr = args.r.data.as_raw().0 as *mut $T;
                unsafe {
                    copy_h2d_typed(r_ptr as *mut c_void, host_a.as_ptr() as *const c_void, bytes, stream)?;
                }

                // 3. Stage an identity into `q`, then `ormqr` overwrites
                //    `q` in place with `Q · I = Q`.
                let m_sz = (m as usize) * (m as usize);
                let mut host_id: Vec<$T> = vec![<$T as Default>::default(); m_sz];
                fill_identity(&mut host_id, m as usize);
                let q_ptr = args.q.data.as_raw().0 as *mut $T;
                let q_bytes = m_sz * elem_size;
                unsafe {
                    copy_h2d_typed(q_ptr as *mut c_void, host_id.as_ptr() as *const c_void, q_bytes, stream)?;
                }
                stream.synchronize().map_err(Error::Driver)?;

                let status = unsafe {
                    $ormqr(
                        h,
                        CUBLAS_SIDE_LEFT,
                        CUBLAS_OP_N,
                        m, m, k,
                        a_ptr as *const $T,
                        m,
                        tau_ptr as *const $T,
                        q_ptr,
                        m,
                        tail_ptr,
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
    };
}

impl_qr_run!(f32, cusolverDnSgeqrf, cusolverDnSormqr);
impl_qr_run!(f64, cusolverDnDgeqrf, cusolverDnDormqr);

impl<T: Element> Drop for QrPlan<T> {
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

// ----- D2H / H2D helpers used by the trailblazer QR pipeline -----

unsafe fn copy_d2h(
    dst: *mut c_void,
    src: *const c_void,
    bytes: usize,
    stream: &Stream,
) -> Result<()> {
    #[allow(non_camel_case_types)]
    type CUresult = i32;
    unsafe extern "system" {
        fn cuMemcpyDtoHAsync_v2(
            dst_host: *mut c_void,
            src_device: u64,
            byte_count: usize,
            h_stream: *mut c_void,
        ) -> CUresult;
    }
    let status = unsafe {
        cuMemcpyDtoHAsync_v2(dst, src as u64, bytes, stream.as_raw() as *mut c_void)
    };
    if status != 0 {
        return Err(Error::CutlassInternal(-status));
    }
    Ok(())
}

unsafe fn copy_h2d_typed(
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
    let status = unsafe {
        cuMemcpyHtoDAsync_v2(dst as u64, src, bytes, stream.as_raw() as *mut c_void)
    };
    if status != 0 {
        return Err(Error::CutlassInternal(-status));
    }
    Ok(())
}

/// Identity-filler trait — implemented for `f32` / `f64` to avoid a
/// `From<u8>` bound that those types don't have in stable Rust.
trait IdentityFill: Default + Copy {
    const ONE: Self;
}
impl IdentityFill for f32 {
    const ONE: f32 = 1.0;
}
impl IdentityFill for f64 {
    const ONE: f64 = 1.0;
}

fn fill_identity<T: IdentityFill>(buf: &mut [T], n: usize) {
    for i in 0..n {
        for j in 0..n {
            buf[i * n + j] = if i == j { T::ONE } else { T::default() };
        }
    }
}
