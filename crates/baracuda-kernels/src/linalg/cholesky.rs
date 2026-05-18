//! Cholesky factorization — `A = L · L^T` (lower) or `A = U^T · U` (upper).
//!
//! Wraps cuSOLVER's `cusolverDnSpotrf` / `Dpotrf` (non-batched) and
//! `cusolverDnSpotrfBatched` / `DpotrfBatched`. Input must be symmetric
//! positive-definite — the kernel reports a positive `info` value
//! pointing at the smallest leading minor where the factorization
//! failed if the input is not SPD.
//!
//! **In-place**: cuSOLVER overwrites the input matrix with the factor.
//! The opposite triangle of the input is left untouched by cuSOLVER —
//! callers that need a strictly-triangular output should zero it
//! post-hoc.
//!
//! **Row-major adapter**: the descriptor's `lower: bool` reflects the
//! caller's row-major semantics. The plan flips it before handing the
//! `uplo` arg to cuSOLVER (row-major lower-L ≡ column-major upper-U
//! over the same byte storage).

use core::cell::Cell;
use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_sys::{
    cusolverDnCreate, cusolverDnDestroy, cusolverDnDpotrf, cusolverDnDpotrfBatched,
    cusolverDnDpotrf_bufferSize, cusolverDnHandle_t, cusolverDnSetStream, cusolverDnSpotrf,
    cusolverDnSpotrfBatched, cusolverDnSpotrf_bufferSize, CUBLAS_FILL_MODE_LOWER,
    CUBLAS_FILL_MODE_UPPER,
};
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, LinalgKind, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, Workspace,
};

/// Descriptor for a Cholesky factorization.
#[derive(Copy, Clone, Debug)]
pub struct CholeskyDescriptor {
    /// Matrix size `N` (input shape is `[batch, N, N]` row-major).
    pub matrix_size: i32,
    /// Number of independent matrices to factor in one launch. `1` runs
    /// the non-batched cuSOLVER path; values `> 1` route through the
    /// batched `*potrfBatched` API.
    pub batch_size: i32,
    /// `true` requests the lower-triangular factor (row-major
    /// convention); `false` requests upper. The plan flips this when
    /// handing the `uplo` arg to cuSOLVER (column-major).
    pub lower: bool,
    /// Element type. Must be `F32` or `F64`.
    pub element: ElementKind,
}

/// Args bundle for a Cholesky launch.
///
/// `a` is **overwritten in place** with the requested triangular factor.
/// The opposite triangle's contents are not meaningful after the call.
pub struct CholeskyArgs<'a, T: Element> {
    /// Input / output matrix stack `[batch, N, N]` row-major contiguous.
    /// Overwritten with `L` (or `U`) in place.
    pub a: TensorMut<'a, T, 3>,
    /// Per-matrix factorization info, one `i32` per batch element.
    /// `info[i] == 0` on success; `info[i] == k > 0` if the leading
    /// `k`-by-`k` minor of matrix `i` is not positive-definite (the
    /// factorization halted at that minor).
    pub info: TensorMut<'a, i32, 1>,
}

/// Cholesky factorization plan — `A = L · L^T` (lower) or `U^T · U`
/// (upper).
///
/// **When to use**: factor an SPD matrix into a triangular factor; the
/// follow-on solve is a pair of triangular substitutions (use
/// [`super::SolvePlan`] for the general `getrf` / `getrs` solve, or
/// chain a `trsm` for the SPD case).
///
/// **Dtypes**: `f32`, `f64` only — cuSOLVER's dense `*potrf` family
/// does not expose `f16` / `bf16` / complex (Hermitian Cholesky is a
/// future-milestone deferral).
///
/// **Shape**: `[batch, N, N]`. `batch_size == 1` routes through the
/// non-batched `potrf`; `batch_size > 1` routes through
/// `*potrfBatched`.
///
/// **Workspace**: non-batched path needs cuSOLVER's `_bufferSize`
/// scratch (queried lazily on first `run`; see
/// [`Self::workspace_size`] / [`Self::query_workspace_size`]). Batched
/// path needs `batch_size * 8` bytes for the device-side array of
/// pointers.
///
/// **Precision guarantee**: deterministic (single-stream cuSOLVER), but
/// **not** bit-stable across runs — cuSOLVER's reduction order is
/// implementation-defined.
///
/// Owns a lazy cuSOLVER handle (`!Sync` / `!Send`); destroyed on `Drop`.
pub struct CholeskyPlan<T: Element> {
    desc: CholeskyDescriptor,
    sku: KernelSku,
    handle: Cell<cusolverDnHandle_t>,
    workspace_bytes: Cell<usize>,
    _marker: PhantomData<T>,
}

impl<T: Element> CholeskyPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &CholeskyDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::CholeskyPlan: descriptor.element != T::KIND",
            ));
        }
        if !matches!(T::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::CholeskyPlan: cuSOLVER dense Cholesky supports f32 + f64 only",
            ));
        }
        if desc.matrix_size <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CholeskyPlan: matrix_size must be > 0",
            ));
        }
        if desc.batch_size <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CholeskyPlan: batch_size must be > 0",
            ));
        }

        let math_precision = match T::KIND {
            ElementKind::F64 => MathPrecision::F64,
            _ => MathPrecision::F32,
        };
        let precision_guarantee = PrecisionGuarantee {
            math_precision,
            accumulator: T::KIND,
            // cuSOLVER's reduction order is implementation-defined; we
            // can't promise bit-stability across runs (different launch
            // parameters / block scheduling can swap the addition order
            // inside potrf). Mark conservative.
            bit_stable_on_same_hardware: false,
            // The library is deterministic in the single-run-from-one-
            // thread sense (no random scheduling decisions, no atomic
            // accumulation visible to the caller).
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Linalg,
            op: LinalgKind::Cholesky as u16,
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

    /// Workspace size in bytes.
    ///
    /// Returns `0` before the first `run` (the cuSOLVER `_bufferSize`
    /// query requires a live handle — created lazily on first `run`).
    /// Call [`Self::query_workspace_size`] to perform the query
    /// explicitly. For batched ops cuSOLVER allocates workspace
    /// internally, so this returns `0` for `batch_size > 1`.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        self.workspace_bytes.get()
    }

    /// Materialize the cuSOLVER handle if necessary and run the
    /// workspace-size query. Returns the same value as
    /// [`Self::workspace_size`] would after a `run`.
    pub fn query_workspace_size(&self, _stream: &Stream) -> Result<usize> {
        if self.desc.batch_size > 1 {
            // Batched APIs do not take a workspace.
            return Ok(0);
        }
        let h = self.ensure_handle()?;
        let n = self.desc.matrix_size;
        let uplo = self.cusolver_uplo();
        let mut lwork: i32 = 0;
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                cusolverDnSpotrf_bufferSize(
                    h,
                    uplo,
                    n,
                    core::ptr::null_mut(),
                    n,
                    &mut lwork as *mut _,
                )
            },
            ElementKind::F64 => unsafe {
                cusolverDnDpotrf_bufferSize(
                    h,
                    uplo,
                    n,
                    core::ptr::null_mut(),
                    n,
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

    /// Map the row-major `lower` flag to a column-major cuSOLVER `uplo`.
    /// Row-major lower-L over storage `S` ≡ column-major upper-U over
    /// the same `S`, so we flip the flag.
    #[inline]
    fn cusolver_uplo(&self) -> i32 {
        if self.desc.lower {
            CUBLAS_FILL_MODE_UPPER
        } else {
            CUBLAS_FILL_MODE_LOWER
        }
    }

    /// Lazily create the cuSOLVER handle. Idempotent.
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

    /// Bind the handle to the caller's stream. cuSOLVER associates each
    /// handle with at most one stream at a time; rebinding on every run
    /// lets the plan be reused across streams.
    fn bind_stream(&self, h: cusolverDnHandle_t, stream: &Stream) -> Result<()> {
        let stream_ptr = stream.as_raw() as *mut c_void;
        let status = unsafe { cusolverDnSetStream(h, stream_ptr) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        Ok(())
    }

    fn check_args(&self, args: &CholeskyArgs<'_, T>) -> Result<()> {
        let expected_shape = [self.desc.batch_size, self.desc.matrix_size, self.desc.matrix_size];
        if args.a.shape != expected_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CholeskyPlan: A shape != [batch, N, N]",
            ));
        }
        if args.info.shape != [self.desc.batch_size] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CholeskyPlan: info shape != [batch]",
            ));
        }
        let needed_a = (self.desc.batch_size as i64)
            * (self.desc.matrix_size as i64)
            * (self.desc.matrix_size as i64);
        if (args.a.data.len() as i64) < needed_a {
            return Err(Error::BufferTooSmall {
                needed: needed_a as usize,
                got: args.a.data.len(),
            });
        }
        if (args.info.data.len() as i64) < self.desc.batch_size as i64 {
            return Err(Error::BufferTooSmall {
                needed: self.desc.batch_size as usize,
                got: args.info.data.len(),
            });
        }
        Ok(())
    }
}

// ----- f32 ------------------------------------------------------------------

impl CholeskyPlan<f32> {
    /// Run the Cholesky factorization (f32).
    pub fn run(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: CholeskyArgs<'_, f32>,
    ) -> Result<()> {
        self.check_args(&args)?;
        let h = self.ensure_handle()?;
        self.bind_stream(h, stream)?;
        let n = self.desc.matrix_size;
        let uplo = self.cusolver_uplo();
        let a_ptr = args.a.data.as_raw().0 as *mut f32;
        let info_ptr = args.info.data.as_raw().0 as *mut i32;
        let stride: usize = (n as usize) * (n as usize);

        if self.desc.batch_size == 1 {
            // Query (and cache) the workspace size if necessary.
            let needed = if self.workspace_bytes.get() == 0 {
                self.query_workspace_size(stream)?
            } else {
                self.workspace_bytes.get()
            };
            let (ws_ptr, ws_bytes) = unpack_workspace(workspace, needed)?;
            let lwork = (ws_bytes / core::mem::size_of::<f32>()) as i32;
            let status = unsafe {
                cusolverDnSpotrf(h, uplo, n, a_ptr, n, ws_ptr as *mut f32, lwork, info_ptr)
            };
            if status != 0 {
                return Err(Error::CutlassInternal(-status));
            }
            Ok(())
        } else {
            // Batched. cuSOLVER expects a device-resident array of
            // pointers — we construct it from per-matrix pointers
            // computed via slice offsets, lifting the array onto the
            // device by storing it in caller-supplied workspace (we
            // request enough room for the pointer array in
            // `workspace_size`).
            let ptr_bytes = (self.desc.batch_size as usize) * core::mem::size_of::<u64>();
            let (ws_ptr, ws_bytes) = unpack_workspace(workspace, ptr_bytes)?;
            if ws_bytes < ptr_bytes {
                return Err(Error::WorkspaceTooSmall {
                    needed: ptr_bytes,
                    got: ws_bytes,
                });
            }
            // Build host-side array of device pointers, then upload.
            let base = args.a.data.as_raw().0;
            let mut host_ptrs: Vec<u64> = Vec::with_capacity(self.desc.batch_size as usize);
            for b in 0..self.desc.batch_size {
                host_ptrs.push(base + (b as u64) * (stride as u64) * (core::mem::size_of::<f32>() as u64));
            }
            unsafe { copy_h2d(ws_ptr, host_ptrs.as_ptr() as *const c_void, ptr_bytes, stream)?; }
            let status = unsafe {
                cusolverDnSpotrfBatched(
                    h,
                    uplo,
                    n,
                    ws_ptr as *mut *mut f32,
                    n,
                    info_ptr,
                    self.desc.batch_size,
                )
            };
            if status != 0 {
                return Err(Error::CutlassInternal(-status));
            }
            Ok(())
        }
    }
}

// ----- f64 ------------------------------------------------------------------

impl CholeskyPlan<f64> {
    /// Run the Cholesky factorization (f64).
    pub fn run(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: CholeskyArgs<'_, f64>,
    ) -> Result<()> {
        self.check_args(&args)?;
        let h = self.ensure_handle()?;
        self.bind_stream(h, stream)?;
        let n = self.desc.matrix_size;
        let uplo = self.cusolver_uplo();
        let a_ptr = args.a.data.as_raw().0 as *mut f64;
        let info_ptr = args.info.data.as_raw().0 as *mut i32;
        let stride: usize = (n as usize) * (n as usize);

        if self.desc.batch_size == 1 {
            let needed = if self.workspace_bytes.get() == 0 {
                self.query_workspace_size(stream)?
            } else {
                self.workspace_bytes.get()
            };
            let (ws_ptr, ws_bytes) = unpack_workspace(workspace, needed)?;
            let lwork = (ws_bytes / core::mem::size_of::<f64>()) as i32;
            let status = unsafe {
                cusolverDnDpotrf(h, uplo, n, a_ptr, n, ws_ptr as *mut f64, lwork, info_ptr)
            };
            if status != 0 {
                return Err(Error::CutlassInternal(-status));
            }
            Ok(())
        } else {
            let ptr_bytes = (self.desc.batch_size as usize) * core::mem::size_of::<u64>();
            let (ws_ptr, ws_bytes) = unpack_workspace(workspace, ptr_bytes)?;
            if ws_bytes < ptr_bytes {
                return Err(Error::WorkspaceTooSmall {
                    needed: ptr_bytes,
                    got: ws_bytes,
                });
            }
            let base = args.a.data.as_raw().0;
            let mut host_ptrs: Vec<u64> = Vec::with_capacity(self.desc.batch_size as usize);
            for b in 0..self.desc.batch_size {
                host_ptrs.push(base + (b as u64) * (stride as u64) * (core::mem::size_of::<f64>() as u64));
            }
            unsafe { copy_h2d(ws_ptr, host_ptrs.as_ptr() as *const c_void, ptr_bytes, stream)?; }
            let status = unsafe {
                cusolverDnDpotrfBatched(
                    h,
                    uplo,
                    n,
                    ws_ptr as *mut *mut f64,
                    n,
                    info_ptr,
                    self.desc.batch_size,
                )
            };
            if status != 0 {
                return Err(Error::CutlassInternal(-status));
            }
            Ok(())
        }
    }
}

impl<T: Element> Drop for CholeskyPlan<T> {
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

// Shared helpers — kept module-private; the LU / QR / SVD plans
// re-implement them locally rather than share through a sub-module
// because the workspace and pointer-array contract differs slightly
// per op. The plan-layer adapter is intentionally inlined.

pub(crate) fn unpack_workspace<'a>(
    workspace: Workspace<'a>,
    needed: usize,
) -> Result<(*mut c_void, usize)> {
    match workspace {
        Workspace::None => {
            if needed == 0 {
                Ok((core::ptr::null_mut(), 0))
            } else {
                Err(Error::WorkspaceTooSmall { needed, got: 0 })
            }
        }
        Workspace::Borrowed(slice) => {
            let got = slice.len();
            if got < needed {
                return Err(Error::WorkspaceTooSmall { needed, got });
            }
            Ok((slice.as_raw().0 as *mut c_void, got))
        }
    }
}

/// Upload `host_bytes` bytes from a host buffer into a device buffer,
/// synchronously w.r.t. `stream`. Used by the batched plans to lift
/// the array-of-pointers onto the device.
///
/// This is a tiny helper that pokes `cuMemcpyHtoDAsync` through the
/// baracuda-driver. We surface it here because the linalg batched
/// path is the first user of "stage a small host array on the device
/// in workspace" — if more op families need it, it can graduate to a
/// shared utility.
///
/// # Safety
/// `dst` must point to at least `bytes` device-writable bytes; `src`
/// to at least `bytes` host-readable bytes. `stream` must be live.
pub(crate) unsafe fn copy_h2d(
    dst: *mut c_void,
    src: *const c_void,
    bytes: usize,
    stream: &Stream,
) -> Result<()> {
    // baracuda-driver does not expose a public raw memcpy helper, so
    // we call the CUDA driver API directly. Linkage to `cuda` (CUDA
    // Driver API) is already pulled in transitively by the
    // baracuda-driver dependency, so the symbol resolves at link time.
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
    // Sync so the host buffer can be safely freed when this function
    // returns. The batched cuSOLVER call that follows expects the
    // device array to be populated.
    stream
        .synchronize()
        .map_err(|e| Error::Driver(e))?;
    Ok(())
}
