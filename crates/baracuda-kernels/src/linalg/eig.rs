//! General (non-symmetric) eigendecomposition — `A · v = λ · v` where
//! `A` may have complex eigenvalues even for real input.
//!
//! Wraps cuSOLVER's 64-bit-index `cusolverDnXgeev` (cuSOLVER 11.7+ /
//! CUDA 12.6+). The single Xgeev entry point handles all four input
//! dtypes (`f32` / `f64` / `Complex32` / `Complex64`) via the
//! `cudaDataType` argument.
//!
//! ## LAPACK-style output convention (NOT always-complex)
//!
//! cuSOLVER Xgeev follows LAPACK's `dgeev` / `sgeev` output convention
//! exactly — there is **no "always-complex output" mode**, and asking
//! for one via `dataTypeW = CUDA_C_*` for real input causes
//! `cusolverDnXgeev_bufferSize` to fail with
//! `CUSOLVER_STATUS_INVALID_VALUE`. The supported configurations are:
//!
//! | dtypeA       | dtypeW       | dtypeVL/VR   | computeType  |
//! |--------------|--------------|--------------|--------------|
//! | CUDA_R_32F   | CUDA_R_32F   | CUDA_R_32F   | CUDA_R_32F   |
//! | CUDA_R_64F   | CUDA_R_64F   | CUDA_R_64F   | CUDA_R_64F   |
//! | CUDA_C_32F   | CUDA_C_32F   | CUDA_C_32F   | CUDA_C_32F   |
//! | CUDA_C_64F   | CUDA_C_64F   | CUDA_C_64F   | CUDA_C_64F   |
//!
//! ### W layout
//! - **Real input (`f32` / `f64`)**: `W` is sized `[2 * N]`. The first
//!   `N` elements are the real parts (`wr`), the last `N` are the
//!   imaginary parts (`wi`). Complex eigenvalues appear as
//!   conjugate pairs at adjacent indices: `wr[k] + i·wi[k]` and
//!   `wr[k+1] + i·wi[k+1]` with `wi[k+1] = -wi[k]`.
//! - **Complex input (`Complex32` / `Complex64`)**: `W` is sized `[N]`
//!   and contains complex eigenvalues directly.
//!
//! ### VL/VR layout
//! - **Real input**: column-major `[N, N]`, real-typed. For complex
//!   eigenpair `(wr[k], wi[k])` with `wi[k] > 0`, columns `k` and `k+1`
//!   together encode the complex eigenvector: column `k` is the real
//!   part, column `k+1` is the imaginary part. So the eigenvector for
//!   eigenvalue `wr[k] + i·wi[k]` is `V[:, k] + i·V[:, k+1]`, and for
//!   `wr[k+1] + i·wi[k+1] = wr[k] - i·wi[k]` it's `V[:, k] - i·V[:, k+1]`.
//! - **Complex input**: column-major `[N, N]`, complex-typed. Each
//!   column is one complex eigenvector directly.
//!
//! ## Other notes
//!
//! - **In-place semantics**: `A` is destroyed (used as scratch by the
//!   Schur-decomposition algorithm).
//! - **2-D only**: single matrix per launch.
//! - **Column-major end-to-end** (matches the rest of the linalg family).
//! - **NaN input**: cuSOLVER's Xgeev is known to return
//!   `CUSOLVER_INTERNAL_ERROR` (status code 7) when the input contains
//!   NaNs. Callers responsible for sanitizing.
//!
//! ## cusolverDnXgeev calling-convention notes
//!
//! 1. Requires a `cusolverDnParams_t` opaque settings struct, lazily
//!    created on first `run` (same lifetime pattern as the cuSOLVER
//!    handle).
//! 2. Sizes are `int64_t`, not `int`.
//! 3. The buffer-size query returns **two** byte counts (host workspace
//!    + device workspace). The plan allocates the device workspace from
//!    `Workspace::Borrowed` and the host workspace from a `Vec<u8>` it
//!    owns transparently.

use core::cell::Cell;
use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_sys::{
    cudaDataType, cusolverDnCreate, cusolverDnCreateParams, cusolverDnDestroy,
    cusolverDnDestroyParams, cusolverDnHandle_t, cusolverDnParams_t, cusolverDnSetStream,
    cusolverDnXgeev, cusolverDnXgeev_bufferSize, CUDA_C_32F, CUDA_C_64F, CUDA_R_32F, CUDA_R_64F,
    CUSOLVER_EIG_MODE_NOVECTOR, CUSOLVER_EIG_MODE_VECTOR,
};
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, LinalgKind, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, Workspace,
};

use super::cholesky::unpack_workspace;

/// Descriptor for a general (non-symmetric) eigendecomposition.
#[derive(Copy, Clone, Debug)]
pub struct EigDescriptor {
    /// Order `N` of the (square) input matrix.
    pub n: i32,
    /// `true` to compute left eigenvectors `VL`. `false` to skip — `VL`
    /// may be `None` in [`EigArgs`].
    pub compute_left: bool,
    /// `true` to compute right eigenvectors `VR`. `false` to skip —
    /// `VR` may be `None`.
    pub compute_right: bool,
    /// Input element type. `F32` / `F64` / `Complex32` / `Complex64`.
    /// The output dtype matches the input dtype — see module-level docs
    /// for the LAPACK packed-real convention used when input is real.
    pub element: ElementKind,
}

/// Args bundle for a general eig launch.
///
/// **All output tensors take the same element type `T` as the input**
/// (matches cuSOLVER Xgeev's API exactly). The interpretation depends on
/// whether `T` is real or complex — see module-level docs.
pub struct EigArgs<'a, T: Element> {
    /// Input matrix `[N, N]` column-major. Destroyed in place.
    pub a: TensorMut<'a, T, 2>,
    /// Eigenvalues.
    /// - Real `T`: shape `[2 * N]` — `wr` (real parts) then `wi` (imag).
    /// - Complex `T`: shape `[N]` — complex eigenvalues directly.
    pub w: TensorMut<'a, T, 1>,
    /// Left eigenvectors `[N, N]` column-major. Required when
    /// `descriptor.compute_left == true`. For real `T`, complex pairs
    /// are LAPACK-packed across adjacent columns (see module docs).
    pub vl: Option<TensorMut<'a, T, 2>>,
    /// Right eigenvectors `[N, N]` column-major. Required when
    /// `descriptor.compute_right == true`. Same LAPACK packing rules as
    /// `VL` for real `T`.
    pub vr: Option<TensorMut<'a, T, 2>>,
    /// Single-cell info: `0` on success; non-zero per cuSOLVER's
    /// `geev` info contract (k > 0 means the QR algorithm failed to
    /// compute all eigenvalues; only `wr[k:]` / `wi[k:]` are valid).
    pub info: TensorMut<'a, i32, 1>,
}

/// General eig plan. Type parameter `T` is the input element type —
/// outputs use the same `T` per cuSOLVER Xgeev's convention.
pub struct EigPlan<T: Element> {
    desc: EigDescriptor,
    sku: KernelSku,
    handle: Cell<cusolverDnHandle_t>,
    params: Cell<cusolverDnParams_t>,
    /// Device workspace byte count (queried lazily on first `run`).
    workspace_bytes_device: Cell<usize>,
    /// Host workspace byte count (queried lazily on first `run`).
    workspace_bytes_host: Cell<usize>,
    _marker: PhantomData<T>,
}

impl<T: Element> EigPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &EigDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::EigPlan: descriptor.element != T::KIND",
            ));
        }
        if !matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F64 | ElementKind::Complex32 | ElementKind::Complex64
        ) {
            return Err(Error::Unsupported(
                "baracuda-kernels::EigPlan: supports f32 / f64 / Complex32 / Complex64 only",
            ));
        }
        if desc.n <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::EigPlan: n must be > 0",
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
            op: LinalgKind::Eig as u16,
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
            workspace_bytes_device: Cell::new(0),
            workspace_bytes_host: Cell::new(0),
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

    /// Device workspace size in bytes (cuSOLVER Xgeev's
    /// `workspaceInBytesOnDevice`). Populated lazily by
    /// [`query_workspace_size`].
    #[inline]
    pub fn workspace_size(&self) -> usize {
        self.workspace_bytes_device.get()
    }

    /// Host workspace size in bytes (cuSOLVER Xgeev's
    /// `workspaceInBytesOnHost`). Populated lazily by
    /// [`query_workspace_size`]. Allocated transparently by the plan on
    /// each `run` from a `Vec<u8>` — not exposed via `Workspace`.
    #[inline]
    pub fn host_workspace_size(&self) -> usize {
        self.workspace_bytes_host.get()
    }

    /// Materialize the handle + params and run the workspace-size query.
    pub fn query_workspace_size(&self, _stream: &Stream) -> Result<usize> {
        let h = self.ensure_handle()?;
        let params = self.ensure_params()?;
        let n: i64 = self.desc.n as i64;
        let jobvl = if self.desc.compute_left {
            CUSOLVER_EIG_MODE_VECTOR
        } else {
            CUSOLVER_EIG_MODE_NOVECTOR
        };
        let jobvr = if self.desc.compute_right {
            CUSOLVER_EIG_MODE_VECTOR
        } else {
            CUSOLVER_EIG_MODE_NOVECTOR
        };
        let dtype = dtype_tag::<T>();
        let mut ws_device: usize = 0;
        let mut ws_host: usize = 0;
        let status = unsafe {
            cusolverDnXgeev_bufferSize(
                h,
                params,
                jobvl,
                jobvr,
                n,
                dtype,
                core::ptr::null(),
                n,
                dtype,
                core::ptr::null(),
                dtype,
                core::ptr::null(),
                n,
                dtype,
                core::ptr::null(),
                n,
                dtype,
                &mut ws_device as *mut _,
                &mut ws_host as *mut _,
            )
        };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        self.workspace_bytes_device.set(ws_device);
        self.workspace_bytes_host.set(ws_host);
        Ok(ws_device)
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

    fn ensure_params(&self) -> Result<cusolverDnParams_t> {
        let p = self.params.get();
        if !p.is_null() {
            return Ok(p);
        }
        let mut params: cusolverDnParams_t = core::ptr::null_mut();
        let status = unsafe { cusolverDnCreateParams(&mut params as *mut _) };
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

    fn check_args(&self, args: &EigArgs<'_, T>) -> Result<()> {
        let n = self.desc.n;
        if args.a.shape != [n, n] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::EigPlan: A shape != [N, N]",
            ));
        }
        // W shape depends on input dtype: real input packs wr + wi into
        // 2N reals; complex input has N complex.
        let w_len = w_packed_len::<T>(n);
        if args.w.shape != [w_len] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::EigPlan: W shape != [2*N] (real input) or [N] (complex input)",
            ));
        }
        if args.info.shape != [1] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::EigPlan: info shape != [1]",
            ));
        }
        if self.desc.compute_left {
            match args.vl.as_ref() {
                Some(vl) if vl.shape == [n, n] => {}
                Some(_) => {
                    return Err(Error::InvalidProblem(
                        "baracuda-kernels::EigPlan: VL shape != [N, N]",
                    ));
                }
                None => {
                    return Err(Error::InvalidProblem(
                        "baracuda-kernels::EigPlan: VL is None but compute_left == true",
                    ));
                }
            }
        }
        if self.desc.compute_right {
            match args.vr.as_ref() {
                Some(vr) if vr.shape == [n, n] => {}
                Some(_) => {
                    return Err(Error::InvalidProblem(
                        "baracuda-kernels::EigPlan: VR shape != [N, N]",
                    ));
                }
                None => {
                    return Err(Error::InvalidProblem(
                        "baracuda-kernels::EigPlan: VR is None but compute_right == true",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Run the eigendecomposition.
    pub fn run(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: EigArgs<'_, T>,
    ) -> Result<()> {
        self.check_args(&args)?;
        let h = self.ensure_handle()?;
        let params = self.ensure_params()?;
        self.bind_stream(h, stream)?;
        let n: i64 = self.desc.n as i64;
        if self.workspace_bytes_device.get() == 0 {
            self.query_workspace_size(stream)?;
        }
        let needed_device = self.workspace_bytes_device.get();
        let needed_host = self.workspace_bytes_host.get();
        let (ws_dev_ptr, _ws_bytes) = unpack_workspace(workspace, needed_device)?;

        let mut host_ws: Vec<u8> = if needed_host > 0 {
            vec![0u8; needed_host]
        } else {
            Vec::new()
        };
        let host_ws_ptr = if needed_host > 0 {
            host_ws.as_mut_ptr() as *mut c_void
        } else {
            core::ptr::null_mut()
        };

        let jobvl = if self.desc.compute_left {
            CUSOLVER_EIG_MODE_VECTOR
        } else {
            CUSOLVER_EIG_MODE_NOVECTOR
        };
        let jobvr = if self.desc.compute_right {
            CUSOLVER_EIG_MODE_VECTOR
        } else {
            CUSOLVER_EIG_MODE_NOVECTOR
        };
        let dtype = dtype_tag::<T>();

        let a_ptr = args.a.data.as_raw().0 as *mut c_void;
        let w_ptr = args.w.data.as_raw().0 as *mut c_void;
        let vl_ptr = args
            .vl
            .as_ref()
            .map(|v| v.data.as_raw().0 as *mut c_void)
            .unwrap_or(core::ptr::null_mut());
        let vr_ptr = args
            .vr
            .as_ref()
            .map(|v| v.data.as_raw().0 as *mut c_void)
            .unwrap_or(core::ptr::null_mut());
        let info_ptr = args.info.data.as_raw().0 as *mut i32;

        let status = unsafe {
            cusolverDnXgeev(
                h,
                params,
                jobvl,
                jobvr,
                n,
                dtype,
                a_ptr,
                n,
                dtype,
                w_ptr,
                dtype,
                vl_ptr,
                n,
                dtype,
                vr_ptr,
                n,
                dtype,
                ws_dev_ptr,
                needed_device,
                host_ws_ptr,
                needed_host,
                info_ptr,
            )
        };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        Ok(())
    }
}

impl<T: Element> Drop for EigPlan<T> {
    fn drop(&mut self) {
        let p = self.params.get();
        if !p.is_null() {
            unsafe {
                let _ = cusolverDnDestroyParams(p);
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

/// The single `cudaDataType` tag used for every Xgeev arg (A, W, VL,
/// VR, computeType all match the input dtype).
#[inline]
fn dtype_tag<T: Element>() -> cudaDataType {
    match T::KIND {
        ElementKind::F32 => CUDA_R_32F,
        ElementKind::F64 => CUDA_R_64F,
        ElementKind::Complex32 => CUDA_C_32F,
        ElementKind::Complex64 => CUDA_C_64F,
        _ => unreachable!("select() gates on F32 / F64 / Complex32 / Complex64"),
    }
}

/// Required `W` length given `N` and input dtype:
/// - real input → `2 * N` (`wr` + `wi` packed)
/// - complex input → `N` (complex eigenvalues directly)
#[inline]
fn w_packed_len<T: Element>(n: i32) -> i32 {
    match T::KIND {
        ElementKind::F32 | ElementKind::F64 => 2 * n,
        ElementKind::Complex32 | ElementKind::Complex64 => n,
        _ => unreachable!("select() gates on F32 / F64 / Complex32 / Complex64"),
    }
}
