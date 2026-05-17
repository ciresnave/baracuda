//! Rectangular batched approximate-SVD via `gesvdaStridedBatched`.
//!
//! Wraps cuSOLVER's `cusolverDnSgesvdaStridedBatched` /
//! `DgesvdaStridedBatched`. Unlike the sibling
//! [`super::svd_batched::BatchedSvdPlan`] — which uses Jacobi
//! (`gesvdjBatched`) and is **square-only** — this plan accepts
//! arbitrary `m × n` per batch slot.
//!
//! **API shape differences vs. Jacobi-batched**:
//!
//! - Batch addressing is via **element-strides** between slots (not the
//!   pointer-array variant — slots must lie in a single packed buffer
//!   with uniform stride).
//! - A `rank` parameter (`1 ≤ rank ≤ min(m, n)`) selects how many
//!   singular triplets to compute; pass `min(m, n)` for the full thin
//!   SVD or a smaller value for truncated SVD.
//! - Per-slot residual Frobenius norms are written to a caller-supplied
//!   **host** array (`h_R_nrmF`, `f64[batch_size]`).
//! - Workspace `lwork` from `_bufferSize` is measured in **elements**,
//!   not bytes — the plan multiplies by `sizeof(T)` to get the byte
//!   count for the `Workspace`.
//!
//! **Storage**: column-major end-to-end, matching the neighbors. The
//! caller's packed `[B, M, N]` buffer has element-stride `M·N` between
//! batch slots; similarly `S` strides by `rank`, `U` by `M·rank`, and
//! `V` by `N·rank`. cuSOLVER returns `V` directly (not `V^T`).
//!
//! **Dtypes**: `f32` + `f64` only.

use core::cell::Cell;
use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_sys::{
    cusolverDnCreate, cusolverDnDestroy, cusolverDnDgesvdaStridedBatched,
    cusolverDnDgesvdaStridedBatched_bufferSize, cusolverDnHandle_t, cusolverDnSetStream,
    cusolverDnSgesvdaStridedBatched, cusolverDnSgesvdaStridedBatched_bufferSize,
    CUSOLVER_EIG_MODE_NOVECTOR, CUSOLVER_EIG_MODE_VECTOR,
};
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, LinalgKind, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, Workspace,
};

use super::cholesky::unpack_workspace;

/// Descriptor for a rectangular batched approximate-SVD.
#[derive(Copy, Clone, Debug)]
pub struct BatchedSvdaDescriptor {
    /// Row count `M` of each matrix in the batch (column-major).
    pub m: i32,
    /// Column count `N` of each matrix in the batch.
    pub n: i32,
    /// Requested rank `1 ≤ rank ≤ min(M, N)`. Pass `min(M, N)` for the
    /// full thin SVD; pass a smaller value for truncated SVD.
    pub rank: i32,
    /// Number of independent matrices in the batch.
    pub batch_size: i32,
    /// `true` requests singular vectors (`U` + `V`) via
    /// `CUSOLVER_EIG_MODE_VECTOR`; `false` computes only singular
    /// values via `CUSOLVER_EIG_MODE_NOVECTOR`.
    pub compute_vectors: bool,
    /// Element type. Must be `F32` or `F64`.
    pub element: ElementKind,
}

/// Args bundle for a rectangular batched-SVD launch.
///
/// When `compute_vectors == true`, the `u` and `v` tensors **must** be
/// provided (they receive the left / right singular vectors). When
/// `false`, both may be `None`.
///
/// `residuals` is a **host** buffer of `batch_size` `f64`s that
/// cuSOLVER writes per-slot residual Frobenius norms into. Pass `None`
/// if you don't care — the plan will allocate a scratch `Vec<f64>` for
/// the call and drop it.
pub struct BatchedSvdaArgs<'a, T: Element> {
    /// Input stack `[batch, M, N]` (column-major per slot). Read-only
    /// for cuSOLVER's `gesvdaStridedBatched`.
    pub a: TensorMut<'a, T, 3>,
    /// Singular values `[batch, rank]`.
    pub s: TensorMut<'a, T, 2>,
    /// Left singular vectors `[batch, M, rank]` (column-major). Required
    /// when `compute_vectors == true`.
    pub u: Option<TensorMut<'a, T, 3>>,
    /// Right singular vectors `[batch, N, rank]` (column-major — note
    /// this is `V`, not `V^T`). Required when `compute_vectors == true`.
    pub v: Option<TensorMut<'a, T, 3>>,
    /// Per-matrix info `[batch]`: `0` on success.
    pub info: TensorMut<'a, i32, 1>,
    /// Optional caller-supplied host buffer (`f64[batch]`) that receives
    /// per-slot residual Frobenius norms. `None` discards them.
    pub residuals: Option<&'a mut [f64]>,
}

/// Rectangular batched-SVD plan (`gesvdaStridedBatched`).
pub struct BatchedSvdaPlan<T: Element> {
    desc: BatchedSvdaDescriptor,
    sku: KernelSku,
    handle: Cell<cusolverDnHandle_t>,
    workspace_bytes: Cell<usize>,
    _marker: PhantomData<T>,
}

impl<T: Element> BatchedSvdaPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &BatchedSvdaDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::BatchedSvdaPlan: descriptor.element != T::KIND",
            ));
        }
        if !matches!(T::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::BatchedSvdaPlan: cuSOLVER gesvdaStridedBatched supports f32 + f64 only",
            ));
        }
        if desc.m <= 0 || desc.n <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchedSvdaPlan: m / n must be > 0",
            ));
        }
        if desc.batch_size <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchedSvdaPlan: batch_size must be > 0",
            ));
        }
        let kmax = desc.m.min(desc.n);
        if desc.rank < 1 || desc.rank > kmax {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchedSvdaPlan: rank must satisfy 1 <= rank <= min(m, n)",
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
            op: LinalgKind::BatchedSvda as u16,
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

    fn bind_stream(&self, h: cusolverDnHandle_t, stream: &Stream) -> Result<()> {
        let status = unsafe { cusolverDnSetStream(h, stream.as_raw() as *mut c_void) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        Ok(())
    }

    /// Materialize the handle and query workspace size (in bytes —
    /// cuSOLVER returns `lwork` in **elements**, this method converts).
    pub fn query_workspace_size(&self, _stream: &Stream) -> Result<usize> {
        let h = self.ensure_handle()?;
        let m = self.desc.m;
        let n = self.desc.n;
        let rank = self.desc.rank;
        let b = self.desc.batch_size;
        let stride_a = (m as i64) * (n as i64);
        let stride_s = rank as i64;
        let stride_u = (m as i64) * (rank as i64);
        let stride_v = (n as i64) * (rank as i64);
        let mut lwork: i32 = 0;
        let jobz = self.jobz();
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                cusolverDnSgesvdaStridedBatched_bufferSize(
                    h,
                    jobz,
                    rank,
                    m,
                    n,
                    core::ptr::null(),
                    m,
                    stride_a,
                    core::ptr::null(),
                    stride_s,
                    core::ptr::null(),
                    m,
                    stride_u,
                    core::ptr::null(),
                    n,
                    stride_v,
                    &mut lwork as *mut _,
                    b,
                )
            },
            ElementKind::F64 => unsafe {
                cusolverDnDgesvdaStridedBatched_bufferSize(
                    h,
                    jobz,
                    rank,
                    m,
                    n,
                    core::ptr::null(),
                    m,
                    stride_a,
                    core::ptr::null(),
                    stride_s,
                    core::ptr::null(),
                    m,
                    stride_u,
                    core::ptr::null(),
                    n,
                    stride_v,
                    &mut lwork as *mut _,
                    b,
                )
            },
            _ => unreachable!("select() gates on F32 / F64"),
        };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        // `lwork` is in **elements**, not bytes — multiply by sizeof(T).
        let bytes = (lwork as usize) * core::mem::size_of::<T>();
        self.workspace_bytes.set(bytes);
        Ok(bytes)
    }

    fn check_args(&self, args: &BatchedSvdaArgs<'_, T>) -> Result<()> {
        let b = self.desc.batch_size;
        let m = self.desc.m;
        let n = self.desc.n;
        let rank = self.desc.rank;
        if args.a.shape != [b, m, n] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchedSvdaPlan: A shape != [batch, M, N]",
            ));
        }
        if args.s.shape != [b, rank] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchedSvdaPlan: S shape != [batch, rank]",
            ));
        }
        if self.desc.compute_vectors {
            match args.u.as_ref() {
                Some(u) => {
                    if u.shape != [b, m, rank] {
                        return Err(Error::InvalidProblem(
                            "baracuda-kernels::BatchedSvdaPlan: U shape != [batch, M, rank]",
                        ));
                    }
                }
                None => {
                    return Err(Error::InvalidProblem(
                        "baracuda-kernels::BatchedSvdaPlan: U required when compute_vectors",
                    ));
                }
            }
            match args.v.as_ref() {
                Some(v) => {
                    if v.shape != [b, n, rank] {
                        return Err(Error::InvalidProblem(
                            "baracuda-kernels::BatchedSvdaPlan: V shape != [batch, N, rank]",
                        ));
                    }
                }
                None => {
                    return Err(Error::InvalidProblem(
                        "baracuda-kernels::BatchedSvdaPlan: V required when compute_vectors",
                    ));
                }
            }
        }
        if args.info.shape != [b] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchedSvdaPlan: info shape != [batch]",
            ));
        }
        if let Some(r) = args.residuals.as_ref() {
            if r.len() != b as usize {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::BatchedSvdaPlan: residuals len != batch",
                ));
            }
        }
        Ok(())
    }
}

macro_rules! impl_batched_svda_run {
    ($T:ty, $gesvda_batched:ident) => {
        impl BatchedSvdaPlan<$T> {
            /// Run the rectangular batched approximate-SVD.
            pub fn run(
                &self,
                stream: &Stream,
                workspace: Workspace<'_>,
                mut args: BatchedSvdaArgs<'_, $T>,
            ) -> Result<()> {
                self.check_args(&args)?;
                let h = self.ensure_handle()?;
                self.bind_stream(h, stream)?;
                let m = self.desc.m;
                let n = self.desc.n;
                let rank = self.desc.rank;
                let b = self.desc.batch_size;
                let stride_a = (m as i64) * (n as i64);
                let stride_s = rank as i64;
                let stride_u = (m as i64) * (rank as i64);
                let stride_v = (n as i64) * (rank as i64);

                let needed = if self.workspace_bytes.get() == 0 {
                    self.query_workspace_size(stream)?
                } else {
                    self.workspace_bytes.get()
                };
                let (ws_ptr, ws_bytes) = unpack_workspace(workspace, needed)?;
                // `lwork` is in elements; convert from the byte count we
                // unpacked from the Workspace.
                let lwork = (ws_bytes / core::mem::size_of::<$T>()) as i32;

                let a_ptr = args.a.data.as_raw().0 as *const $T;
                let s_ptr = args.s.data.as_raw().0 as *mut $T;
                let (u_ptr, ldu) = if self.desc.compute_vectors {
                    let u_ref = args.u.as_mut().expect("check_args ensures U is Some");
                    (u_ref.data.as_raw().0 as *mut $T, m)
                } else {
                    // cuSOLVER tolerates null U / V when jobz == NOVECTOR;
                    // pass a benign `ldu` (>= max(1, m)) since the API
                    // validates it even when the pointer is unused.
                    (core::ptr::null_mut(), m)
                };
                let (v_ptr, ldv) = if self.desc.compute_vectors {
                    let v_ref = args.v.as_mut().expect("check_args ensures V is Some");
                    (v_ref.data.as_raw().0 as *mut $T, n)
                } else {
                    (core::ptr::null_mut(), n)
                };
                let info_ptr = args.info.data.as_raw().0 as *mut i32;

                // Per-slot residual norms — caller buffer or local scratch.
                let mut local_scratch: Vec<f64>;
                let residual_ptr: *mut f64 = match args.residuals.as_mut() {
                    Some(r) => r.as_mut_ptr(),
                    None => {
                        local_scratch = vec![0f64; b as usize];
                        local_scratch.as_mut_ptr()
                    }
                };

                let status = unsafe {
                    $gesvda_batched(
                        h,
                        self.jobz(),
                        rank,
                        m,
                        n,
                        a_ptr,
                        m,
                        stride_a,
                        s_ptr,
                        stride_s,
                        u_ptr,
                        ldu,
                        stride_u,
                        v_ptr,
                        ldv,
                        stride_v,
                        ws_ptr as *mut $T,
                        lwork,
                        info_ptr,
                        residual_ptr,
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

impl_batched_svda_run!(f32, cusolverDnSgesvdaStridedBatched);
impl_batched_svda_run!(f64, cusolverDnDgesvdaStridedBatched);

impl<T: Element> Drop for BatchedSvdaPlan<T> {
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
