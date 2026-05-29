//! # baracuda-megatron
//!
//! Megatron-LM-style tensor-parallel (TP) primitives on top of
//! [`baracuda-cublas`] (local GEMM) and [`baracuda-nccl`] (cross-rank
//! collectives). Pure-composition crate — **no new CUDA kernels**.
//!
//! ## Algorithmic reference
//!
//! Shoeybi, Patwary, Puri, LeGresley, Casper, and Catanzaro,
//! "Megatron-LM: Training Multi-Billion Parameter Language Models Using
//! Model Parallelism", arXiv:1909.08053 (2019). The upstream
//! [NVIDIA Megatron-LM repository](https://github.com/NVIDIA/Megatron-LM)
//! is Apache-2.0; **no source is vendored** — the kernel primitives are
//! reused from the rest of the baracuda stack and composed here.
//!
//! ## What this crate is
//!
//! Two foundational TP plans for splitting a `Linear` layer's weight
//! matrix `W` across `N` ranks of a NCCL communicator:
//!
//! | Type | Splits along | Local input | Local GEMM | Collective |
//! |---|---|---|---|---|
//! | [`ColumnParallelLinearPlan`] | output (`out_features`) | full `X` | `Y_local = X · W_local^T` `[B, out/N]` | `all_gather` → `Y` `[B, out]` |
//! | [`RowParallelLinearPlan`]    | input (`in_features`)   | sharded `X_local` `[B, in/N]` | `Y_partial = X_local · W_local^T` `[B, out]` | `all_reduce(Sum)` → `Y` `[B, out]` |
//!
//! Both backward passes mirror the forward pattern (the collective on
//! one side becomes a no-op on the other side, mediated by the
//! sharded-vs-replicated input/output convention).
//!
//! ## Why a separate crate
//!
//! Mirrors the [`baracuda-optim`] / [`baracuda-transformer-engine`]
//! pattern from Phase 49 / Phase 55 — deliberate scope expansion that
//! lives in its own crate so:
//!
//! - Inference-only / single-GPU consumers (e.g. Fuel) **don't pay**
//!   the dep surface cost — they simply don't depend on this crate.
//! - The `megatron_tp` cargo feature on `baracuda-kernels` re-exports
//!   these plans into the unified facade when a downstream wants the
//!   full distributed-training surface.
//!
//! ## Scope (Tier 1)
//!
//! - **Dtypes**: `f32` (always), `f16` + `bf16` behind the `half-crate`
//!   cargo feature.
//! - **Modes**: forward + backward (inference + training).
//! - **Bias**: API accepts an optional bias arg; Tier 2 will wire it
//!   via a baracuda-kernels `Affine` composition. Phase 57 returns
//!   `InvalidArgument` if a bias is passed (callers can do bias-add
//!   themselves between calls — note that on RowParallel, the bias
//!   must be added **after** the `all_reduce` so it doesn't get
//!   summed `N` times).
//! - **Single-rank degenerate case**: when `world_size == 1`, every
//!   collective short-circuits to a stream-ordered D2D copy and the
//!   plan behaves bit-equivalently to a standard `Linear` layer.
//!   Used by the in-process smoke tests on single-GPU dev hardware.
//!
//! ## Out of scope (deferred)
//!
//! - **Async overlap** (Hopper TMA + `comm_gemm_overlap` territory) —
//!   sm_89 hardware blocked; future polish phase.
//! - **Sequence parallelism** — Phase 56 (Ring Attention)'s domain.
//! - **Pipeline parallelism** — orchestration-heavy; future phase.
//! - **VocabParallelEmbedding** — Megatron-specific; future polish.
//! - **Distributed gradient accumulation** — Phase 58+.
//! - **Expert parallelism (MoE)** — separate distributed phase.

#![warn(missing_debug_implementations)]
#![deny(missing_docs)]

use core::marker::PhantomData;

use baracuda_cublas::{gemm, Handle, Op};
use baracuda_driver::{DeviceBuffer, Stream};
use baracuda_nccl::{Communicator, NcclScalar, RedOp};
use thiserror::Error;

#[cfg(feature = "half-crate")]
use half::{bf16, f16};

// ============================================================================
// Error / Result
// ============================================================================

/// Error category surfaced by the Megatron TP plans.
///
/// `#[non_exhaustive]` per the baracuda Phase 28 audit — variants may
/// be added as the surface grows (e.g. autograd-checkpointing hooks
/// once those land).
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// Caller-side validation failed before launch (shape mismatch,
    /// `out_features` not divisible by `world_size`, …).
    #[error("megatron argument invalid: {0}")]
    InvalidArgument(&'static str),
    /// Buffer length didn't match the expected `batch * dim` product.
    #[error("buffer length mismatch: {what}: got {got}, expected {expected}")]
    BufferLengthMismatch {
        /// Which buffer was the wrong size (e.g. `"x"`, `"y_local"`).
        what: &'static str,
        /// Actual element count.
        got: usize,
        /// Expected element count.
        expected: usize,
    },
    /// A cuBLAS call returned non-success status — typically a stream
    /// binding or invalid-argument error.
    #[error("cuBLAS GEMM failed: {0}")]
    Cublas(#[from] baracuda_cublas::Error),
    /// A NCCL collective returned non-success status — typically a
    /// communicator-aborted or stream-aborted error.
    #[error("NCCL collective failed: {0:?}")]
    Nccl(baracuda_nccl::Error),
    /// A driver-level error (stream binding, buffer allocation, etc.).
    #[error("CUDA driver error: {0:?}")]
    Driver(baracuda_driver::Error),
}

impl From<baracuda_nccl::Error> for Error {
    fn from(e: baracuda_nccl::Error) -> Self {
        Error::Nccl(e)
    }
}

impl From<baracuda_driver::Error> for Error {
    fn from(e: baracuda_driver::Error) -> Self {
        Error::Driver(e)
    }
}

/// `Result` alias used throughout the crate.
pub type Result<T> = core::result::Result<T, Error>;

// ============================================================================
// TensorParallelContext
// ============================================================================

/// Per-Plan binding of a NCCL [`Communicator`] + the Linear-layer shape
/// metadata that participates in the tensor-parallel split.
///
/// The context borrows the communicator (so the caller controls its
/// lifetime — the typical pattern is one `Communicator` per process
/// shared by many TP layers) and caches the per-rank shard sizes
/// derived from `world_size`.
///
/// **Divisibility contract**: `out_features` must be divisible by
/// `world_size` for column parallelism; `in_features` must be
/// divisible by `world_size` for row parallelism. Both are checked
/// at plan-construction time, not here — a single context can host
/// both kinds of plans as long as the relevant dim divides cleanly.
#[derive(Debug)]
pub struct TensorParallelContext<'comm> {
    comm: &'comm Communicator,
    in_features: i32,
    out_features: i32,
    rank: i32,
    world_size: i32,
}

impl<'comm> TensorParallelContext<'comm> {
    /// Bind a Linear layer's `[in_features, out_features]` shape to a
    /// NCCL communicator. The `rank` and `world_size` are cached from
    /// the communicator at construction.
    pub fn new(comm: &'comm Communicator, in_features: i32, out_features: i32) -> Self {
        Self {
            comm,
            in_features,
            out_features,
            rank: comm.rank(),
            world_size: comm.world_size(),
        }
    }

    /// Underlying NCCL communicator.
    #[inline]
    pub fn communicator(&self) -> &'comm Communicator {
        self.comm
    }

    /// This rank's index within the TP group (0..world_size).
    #[inline]
    pub fn rank(&self) -> i32 {
        self.rank
    }

    /// Total number of ranks in the TP group.
    #[inline]
    pub fn world_size(&self) -> i32 {
        self.world_size
    }

    /// Full (unpartitioned) `in_features` dimension.
    #[inline]
    pub fn in_features(&self) -> i32 {
        self.in_features
    }

    /// Full (unpartitioned) `out_features` dimension.
    #[inline]
    pub fn out_features(&self) -> i32 {
        self.out_features
    }

    /// Per-rank slice of the `out_features` dimension —
    /// `out_features / world_size`. Used by [`ColumnParallelLinearPlan`]
    /// to size `W_local: [out_features/N, in_features]`.
    ///
    /// # Panics
    ///
    /// Panics if `out_features` is not divisible by `world_size`.
    /// Plan constructors validate this with a structured error; this
    /// helper is for diagnostics + smoke-test ergonomics.
    #[inline]
    pub fn partitioned_out_features(&self) -> i32 {
        assert!(
            self.out_features % self.world_size == 0,
            "out_features ({}) must be divisible by world_size ({})",
            self.out_features,
            self.world_size
        );
        self.out_features / self.world_size
    }

    /// Per-rank slice of the `in_features` dimension —
    /// `in_features / world_size`. Used by [`RowParallelLinearPlan`]
    /// to size `W_local: [out_features, in_features/N]`.
    ///
    /// # Panics
    ///
    /// Panics if `in_features` is not divisible by `world_size`.
    #[inline]
    pub fn partitioned_in_features(&self) -> i32 {
        assert!(
            self.in_features % self.world_size == 0,
            "in_features ({}) must be divisible by world_size ({})",
            self.in_features,
            self.world_size
        );
        self.in_features / self.world_size
    }
}

// ============================================================================
// MegatronGemmScalar — sealed trait covering the supported dtypes
// ============================================================================

/// Element type for [`ColumnParallelLinearPlan`] / [`RowParallelLinearPlan`].
///
/// Sealed (no downstream impls). Always implemented for `f32`;
/// `half::f16` + `half::bf16` impls require the `half-crate` cargo
/// feature.
///
/// The compute path is dtype-erased at the boundary — `f32` dispatches
/// to `cublasSgemm`, `f16` / `bf16` dispatch to `cublasGemmEx` with a
/// `Compute32F` accumulator (matching the rest of baracuda's
/// half-precision GEMM convention; see `baracuda-cutlass` Phase 30).
pub trait MegatronGemmScalar:
    NcclScalar + Copy + Send + Sync + sealed::Sealed + 'static
{
    /// Dispatch a row-major GEMM `D = α · A · B^T + β · D` for this
    /// dtype, using the cuBLAS column-major-from-row-major trick.
    ///
    /// `m` × `n` × `k` are the logical row-major shapes (output is
    /// `[m, n]`, A is `[m, k]`, B is `[n, k]`).
    ///
    /// # Safety
    ///
    /// `a`, `b`, `d` are device buffers sized at least `m*k`, `n*k`,
    /// `m*n` respectively. `handle` is bound to a stream owned by the
    /// caller. Caller must guarantee `m`, `n`, `k > 0`.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn row_major_gemm_nt(
        handle: &Handle,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: &DeviceBuffer<Self>,
        b: &DeviceBuffer<Self>,
        beta: f32,
        d: &mut DeviceBuffer<Self>,
    ) -> Result<()>;

    /// Same as [`row_major_gemm_nt`](Self::row_major_gemm_nt) but with
    /// the second operand un-transposed: `D = α · A · B + β · D`,
    /// shapes `D=[m,n]`, `A=[m,k]`, `B=[k,n]`.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn row_major_gemm_nn(
        handle: &Handle,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: &DeviceBuffer<Self>,
        b: &DeviceBuffer<Self>,
        beta: f32,
        d: &mut DeviceBuffer<Self>,
    ) -> Result<()>;

    /// Same as [`row_major_gemm_nt`](Self::row_major_gemm_nt) but with
    /// the first operand transposed: `D = α · A^T · B + β · D`, shapes
    /// `D=[m,n]`, `A=[k,m]`, `B=[k,n]`. Used in the BW paths for
    /// computing `dW = dY^T @ X`.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn row_major_gemm_tn(
        handle: &Handle,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: &DeviceBuffer<Self>,
        b: &DeviceBuffer<Self>,
        beta: f32,
        d: &mut DeviceBuffer<Self>,
    ) -> Result<()>;
}

mod sealed {
    /// Sealed marker — downstream crates cannot add new dtype impls.
    pub trait Sealed {}
}

// f32 impl — cublasSgemm direct path.
impl MegatronGemmScalar for f32 {
    unsafe fn row_major_gemm_nt(
        handle: &Handle,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        beta: f32,
        d: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        // Row-major D[m,n] = α · A[m,k] · B[n,k]^T + β · D
        // cuBLAS sees memory column-major:
        //   A row-major [m,k] (lda=k) → cuBLAS [k,m] (lda=k; A^T-stored)
        //   B row-major [n,k] (ldb=k) → cuBLAS [k,n] (ldb=k; B^T-stored)
        //   D row-major [m,n] (ldd=n) → cuBLAS [n,m] (ldd=n; D^T)
        //
        // Target: D[i,j] = Σ_p A[i,p] · B[j,p]
        // In cuBLAS view: D^T[j,i] = Σ_p B[j,p] · A[i,p]
        //
        // So we issue gemm with (m_blas, n_blas, k) = (n, m, k):
        //   first  operand = B (cuBLAS sees [k,n] with ldb=k)
        //     → need shape (n, k) for the GEMM → Op::T on B
        //   second operand = A (cuBLAS sees [k,m] with lda=k)
        //     → need shape (k, m) for the GEMM → Op::N on A
        gemm(
            handle,
            Op::T,
            Op::N,
            n,
            m,
            k,
            alpha,
            b,
            k,
            a,
            k,
            beta,
            d,
            n,
        )
        .map_err(Error::from)
    }

    unsafe fn row_major_gemm_nn(
        handle: &Handle,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        beta: f32,
        d: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        // Row-major D[m,n] = α · A[m,k] · B[k,n] + β · D
        // cuBLAS view:
        //   A row-major [m,k] (lda=k) → cuBLAS [k,m] (A^T-stored)
        //   B row-major [k,n] (ldb=n) → cuBLAS [n,k] (B^T-stored)
        //   D row-major [m,n] (ldd=n) → cuBLAS [n,m] (D^T)
        //
        // Target D[i,j] = Σ_p A[i,p] · B[p,j]
        // In cuBLAS view: D^T[j,i] = Σ_p A[i,p] · B[p,j]
        //
        // For GEMM (m_blas=n, n_blas=m, k_shared=k):
        //   first  operand = B (cuBLAS [n,k]) → need (n,k) — Op::N
        //   second operand = A (cuBLAS [k,m]) → need (k,m) — Op::N
        gemm(
            handle,
            Op::N,
            Op::N,
            n,
            m,
            k,
            alpha,
            b,
            n,
            a,
            k,
            beta,
            d,
            n,
        )
        .map_err(Error::from)
    }

    unsafe fn row_major_gemm_tn(
        handle: &Handle,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        beta: f32,
        d: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        // Row-major D[m,n] = α · A[k,m]^T · B[k,n] + β · D
        //                  = α · Σ_p A[p,i] · B[p,j]
        // cuBLAS view:
        //   A row-major [k,m] (lda=m) → cuBLAS [m,k] (A^T-stored)
        //   B row-major [k,n] (ldb=n) → cuBLAS [n,k] (B^T-stored)
        //   D row-major [m,n] (ldd=n) → cuBLAS [n,m] (D^T)
        //
        // Target D[i,j] = Σ_p A[p,i] · B[p,j]
        // In cuBLAS view: D^T[j,i] = Σ_p A_stored[i,p] · B_stored[j,p]
        //
        // GEMM (m_blas=n, n_blas=m, k_shared=k):
        //   first  operand = B (cuBLAS [n,k]) → (n,k) — Op::N
        //   second operand = A (cuBLAS [m,k]) → (k,m) — Op::T
        gemm(
            handle,
            Op::N,
            Op::T,
            n,
            m,
            k,
            alpha,
            b,
            n,
            a,
            m,
            beta,
            d,
            n,
        )
        .map_err(Error::from)
    }
}

impl sealed::Sealed for f32 {}

// Half-precision impls — cublasGemmEx with Compute32F accumulator.
#[cfg(feature = "half-crate")]
mod half_impls {
    use super::*;
    use baracuda_cublas::{cublasComputeType_t, cudaDataType_t};

    // CUBLAS_GEMM_DEFAULT_TENSOR_OP — the well-known tensor-op-preferring
    // algo selector. Matches the baracuda-cutlass cuBLAS fast-path
    // convention (Phase 30).
    const CUBLAS_GEMM_ALGO: i32 = 99;

    /// Common gemmEx dispatcher for half-precision row-major GEMM.
    ///
    /// `op_first` / `op_second` are the cuBLAS op codes for the
    /// COLUMN-MAJOR-swapped operand order (first = B, second = A).
    /// `cublas_lda` / `cublas_ldb` follow the same convention.
    ///
    /// # Safety
    ///
    /// All pointers must be valid device pointers of the declared
    /// dtype, sized per the cuBLAS contract for the given m/n/k.
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm_ex_half<T: MegatronGemmScalar>(
        handle: &Handle,
        op_first: Op,
        op_second: Op,
        m_blas: i32,
        n_blas: i32,
        k: i32,
        alpha: f32,
        first_ptr: *const T,
        cublas_lda: i32,
        second_ptr: *const T,
        cublas_ldb: i32,
        beta: f32,
        d_ptr: *mut T,
        ldd: i32,
        dtype_tag: cudaDataType_t,
    ) -> Result<()> {
        // SAFETY: per the function's `# Safety` contract above —
        // caller asserts all pointers are valid device pointers of the
        // declared dtype with sizes matching cuBLAS's m/n/k contract.
        unsafe {
            baracuda_cublas::gemm_ex(
                handle,
                op_first,
                op_second,
                m_blas,
                n_blas,
                k,
                &alpha as *const f32 as *const core::ffi::c_void,
                first_ptr as *const core::ffi::c_void,
                dtype_tag,
                cublas_lda,
                second_ptr as *const core::ffi::c_void,
                dtype_tag,
                cublas_ldb,
                &beta as *const f32 as *const core::ffi::c_void,
                d_ptr as *mut core::ffi::c_void,
                dtype_tag,
                ldd,
                cublasComputeType_t::Compute32F,
                CUBLAS_GEMM_ALGO,
            )
        }
        .map_err(Error::from)
    }

    impl MegatronGemmScalar for f16 {
        unsafe fn row_major_gemm_nt(
            handle: &Handle,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: &DeviceBuffer<f16>,
            b: &DeviceBuffer<f16>,
            beta: f32,
            d: &mut DeviceBuffer<f16>,
        ) -> Result<()> {
            // SAFETY: contract inherited from the trait method's `# Safety`
            // — caller asserts shape / dtype / liveness of all operands.
            unsafe {
                gemm_ex_half::<f16>(
                    handle,
                    Op::T,
                    Op::N,
                    n,
                    m,
                    k,
                    alpha,
                    b.as_raw().0 as *const f16,
                    k,
                    a.as_raw().0 as *const f16,
                    k,
                    beta,
                    d.as_raw().0 as *mut f16,
                    n,
                    cudaDataType_t::R_16F,
                )
            }
        }

        unsafe fn row_major_gemm_nn(
            handle: &Handle,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: &DeviceBuffer<f16>,
            b: &DeviceBuffer<f16>,
            beta: f32,
            d: &mut DeviceBuffer<f16>,
        ) -> Result<()> {
            // SAFETY: see Self::row_major_gemm_nt above.
            unsafe {
                gemm_ex_half::<f16>(
                    handle,
                    Op::N,
                    Op::N,
                    n,
                    m,
                    k,
                    alpha,
                    b.as_raw().0 as *const f16,
                    n,
                    a.as_raw().0 as *const f16,
                    k,
                    beta,
                    d.as_raw().0 as *mut f16,
                    n,
                    cudaDataType_t::R_16F,
                )
            }
        }

        unsafe fn row_major_gemm_tn(
            handle: &Handle,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: &DeviceBuffer<f16>,
            b: &DeviceBuffer<f16>,
            beta: f32,
            d: &mut DeviceBuffer<f16>,
        ) -> Result<()> {
            // SAFETY: see Self::row_major_gemm_nt above.
            unsafe {
                gemm_ex_half::<f16>(
                    handle,
                    Op::N,
                    Op::T,
                    n,
                    m,
                    k,
                    alpha,
                    b.as_raw().0 as *const f16,
                    n,
                    a.as_raw().0 as *const f16,
                    m,
                    beta,
                    d.as_raw().0 as *mut f16,
                    n,
                    cudaDataType_t::R_16F,
                )
            }
        }
    }

    impl sealed::Sealed for f16 {}

    impl MegatronGemmScalar for bf16 {
        unsafe fn row_major_gemm_nt(
            handle: &Handle,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: &DeviceBuffer<bf16>,
            b: &DeviceBuffer<bf16>,
            beta: f32,
            d: &mut DeviceBuffer<bf16>,
        ) -> Result<()> {
            // SAFETY: contract inherited from the trait method's `# Safety`.
            unsafe {
                gemm_ex_half::<bf16>(
                    handle,
                    Op::T,
                    Op::N,
                    n,
                    m,
                    k,
                    alpha,
                    b.as_raw().0 as *const bf16,
                    k,
                    a.as_raw().0 as *const bf16,
                    k,
                    beta,
                    d.as_raw().0 as *mut bf16,
                    n,
                    cudaDataType_t::R_16BF,
                )
            }
        }

        unsafe fn row_major_gemm_nn(
            handle: &Handle,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: &DeviceBuffer<bf16>,
            b: &DeviceBuffer<bf16>,
            beta: f32,
            d: &mut DeviceBuffer<bf16>,
        ) -> Result<()> {
            // SAFETY: see Self::row_major_gemm_nt above.
            unsafe {
                gemm_ex_half::<bf16>(
                    handle,
                    Op::N,
                    Op::N,
                    n,
                    m,
                    k,
                    alpha,
                    b.as_raw().0 as *const bf16,
                    n,
                    a.as_raw().0 as *const bf16,
                    k,
                    beta,
                    d.as_raw().0 as *mut bf16,
                    n,
                    cudaDataType_t::R_16BF,
                )
            }
        }

        unsafe fn row_major_gemm_tn(
            handle: &Handle,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: &DeviceBuffer<bf16>,
            b: &DeviceBuffer<bf16>,
            beta: f32,
            d: &mut DeviceBuffer<bf16>,
        ) -> Result<()> {
            // SAFETY: see Self::row_major_gemm_nt above.
            unsafe {
                gemm_ex_half::<bf16>(
                    handle,
                    Op::N,
                    Op::T,
                    n,
                    m,
                    k,
                    alpha,
                    b.as_raw().0 as *const bf16,
                    n,
                    a.as_raw().0 as *const bf16,
                    m,
                    beta,
                    d.as_raw().0 as *mut bf16,
                    n,
                    cudaDataType_t::R_16BF,
                )
            }
        }
    }

    impl sealed::Sealed for bf16 {}
}

// ============================================================================
// ColumnParallelLinearPlan
// ============================================================================

/// Tensor-parallel Linear layer that **splits the weight matrix along
/// the OUTPUT dimension**. Each rank holds `W_local: [out_features/N, in_features]`
/// (PyTorch convention — rows are output neurons).
///
/// ## Forward pass
///
/// ```text
/// X        : [batch, in_features]            (replicated across ranks)
/// W_local  : [out_features/N, in_features]   (this rank's shard)
/// b_local  : [out_features/N]                (optional bias shard; Tier 2)
///
/// Y_local  = X @ W_local^T + b_local         (local GEMM)
///          : [batch, out_features/N]
///
/// Y        = all_gather(Y_local)             (NCCL collective)
///          : [N, batch, out_features/N]      (gathered shape; see note)
/// ```
///
/// ## Backward pass
///
/// ```text
/// dY       : [batch, out_features]           (replicated; gradient w.r.t. Y)
/// dY_local = dY[:, rank*out/N : (rank+1)*out/N]   (caller-side slice)
///          : [batch, out_features/N]
///
/// dX_partial = dY_local @ W_local            (local GEMM, partial sum)
///            : [batch, in_features]
/// dX         = all_reduce(dX_partial, Sum)   (cross-rank sum)
///
/// dW_local   = dY_local^T @ X                (local GEMM)
///            : [out_features/N, in_features]
/// db_local   = sum(dY_local, axis=0)         (optional; caller-side)
/// ```
///
/// The `dX` all-reduce is the cross-rank synchronization point — every
/// rank ends up with the same `dX: [batch, in_features]` after the
/// collective. `dW_local` stays sharded (each rank updates its own
/// portion via the optimizer).
pub struct ColumnParallelLinearPlan<'comm, T: MegatronGemmScalar> {
    tpctx_in_features: i32,
    tpctx_out_features: i32,
    tpctx_world_size: i32,
    tpctx_rank: i32,
    /// Cached `out_features / world_size` — the per-rank shard size.
    out_per_rank: i32,
    /// Caller-declared batch size. Plans are batch-shape-bound at
    /// construction (matches the rest of baracuda's plan convention).
    batch: i32,
    /// Owned cuBLAS handle bound to the caller's stream at run-time.
    handle: Handle,
    /// Borrow of the NCCL communicator — for the all_gather / all_reduce.
    comm: &'comm Communicator,
    _marker: PhantomData<T>,
}

impl<T: MegatronGemmScalar> core::fmt::Debug for ColumnParallelLinearPlan<'_, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ColumnParallelLinearPlan")
            .field("in_features", &self.tpctx_in_features)
            .field("out_features", &self.tpctx_out_features)
            .field("out_per_rank", &self.out_per_rank)
            .field("batch", &self.batch)
            .field("rank", &self.tpctx_rank)
            .field("world_size", &self.tpctx_world_size)
            .field("dtype", &core::any::type_name::<T>())
            .finish()
    }
}

impl<'comm, T: MegatronGemmScalar> ColumnParallelLinearPlan<'comm, T> {
    /// Construct a Column-parallel Linear plan bound to the given TP
    /// context for a particular batch size.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] if `out_features` is not
    /// divisible by `world_size`, or if `batch` / `in_features` /
    /// `out_features` / `world_size` are non-positive.
    pub fn new(tpctx: &TensorParallelContext<'comm>, batch: i32) -> Result<Self> {
        if batch <= 0 {
            return Err(Error::InvalidArgument("batch must be > 0"));
        }
        if tpctx.in_features <= 0 || tpctx.out_features <= 0 {
            return Err(Error::InvalidArgument(
                "in_features and out_features must be > 0",
            ));
        }
        if tpctx.world_size <= 0 {
            return Err(Error::InvalidArgument("world_size must be > 0"));
        }
        if tpctx.out_features % tpctx.world_size != 0 {
            return Err(Error::InvalidArgument(
                "ColumnParallelLinear: out_features must be divisible by world_size",
            ));
        }
        let handle = Handle::new()?;
        Ok(Self {
            tpctx_in_features: tpctx.in_features,
            tpctx_out_features: tpctx.out_features,
            tpctx_world_size: tpctx.world_size,
            tpctx_rank: tpctx.rank,
            out_per_rank: tpctx.out_features / tpctx.world_size,
            batch,
            handle,
            comm: tpctx.comm,
            _marker: PhantomData,
        })
    }

    /// Per-rank shard of the output dimension.
    #[inline]
    pub fn out_per_rank(&self) -> i32 {
        self.out_per_rank
    }

    /// This rank's index within the TP group.
    #[inline]
    pub fn rank(&self) -> i32 {
        self.tpctx_rank
    }

    /// Total number of ranks in the TP group.
    #[inline]
    pub fn world_size(&self) -> i32 {
        self.tpctx_world_size
    }

    /// Forward pass.
    ///
    /// - `x`         : `[batch, in_features]` (replicated across ranks)
    /// - `w_local`   : `[out_features/N, in_features]` (this rank's shard, row-major)
    /// - `bias_local`: `Some([out_features/N])` (this rank's shard; Tier 2 — pass `None`)
    /// - `y_local`   : `[batch, out_features/N]` — written by the local GEMM
    /// - `y_full`    : `[world_size * batch * out_features/N]` — written by
    ///   `all_gather(y_local)`. NCCL concatenates per-rank sendbufs in
    ///   rank order: `y_full = [y_local_rank0; y_local_rank1; …; y_local_rankN-1]`.
    ///   For the common `batch == 1` (LLM decode) case this is equivalent
    ///   to `y_full[r*out/N : (r+1)*out/N] = y_local_of_rank_r` — exactly
    ///   the `[1, out_features]` shape callers expect.
    ///
    /// For `batch > 1` the post-AllGather layout has each rank's
    /// `[batch, out/N]` block contiguous in `y_full` (shape
    /// `[world_size, batch, out/N]`); callers that need true
    /// `[batch, out_features]` row-major must apply a strided
    /// permutation (cheap — `Contiguize` from baracuda-kernels). This
    /// matches Megatron-LM's `_gather_along_last_dim` contract.
    ///
    /// Both `y_local` and `y_full` are caller-owned; the plan never
    /// allocates internally. `y_local` is exposed so callers that need
    /// it (e.g. checkpointing the un-gathered pre-activation for a
    /// later backward) can keep it around.
    pub fn forward(
        &self,
        stream: &Stream,
        x: &DeviceBuffer<T>,
        w_local: &DeviceBuffer<T>,
        bias_local: Option<&DeviceBuffer<T>>,
        y_local: &mut DeviceBuffer<T>,
        y_full: &mut DeviceBuffer<T>,
    ) -> Result<()> {
        let batch = self.batch;
        let in_f = self.tpctx_in_features;
        let out_n = self.out_per_rank;
        let world_size = self.tpctx_world_size;

        check_len("x", x.len(), (batch as usize) * (in_f as usize))?;
        check_len(
            "w_local",
            w_local.len(),
            (out_n as usize) * (in_f as usize),
        )?;
        check_len(
            "y_local",
            y_local.len(),
            (batch as usize) * (out_n as usize),
        )?;
        check_len(
            "y_full",
            y_full.len(),
            (world_size as usize) * (batch as usize) * (out_n as usize),
        )?;
        if let Some(b) = bias_local {
            check_len("bias_local", b.len(), out_n as usize)?;
        }

        if bias_local.is_some() {
            return Err(Error::InvalidArgument(
                "ColumnParallelLinear: bias is Tier 2 — pass None for Phase 57; \
                 caller can run baracuda-kernels Affine after forward()",
            ));
        }

        // Bind the cuBLAS handle to the caller's stream.
        self.handle.set_stream(stream)?;

        // y_local = x @ w_local^T  (row-major: [batch, in_f] · [out_n, in_f]^T)
        //   m_logical = batch, n_logical = out_n, k_logical = in_f.
        // SAFETY: shapes validated above; handle bound to `stream`.
        unsafe {
            T::row_major_gemm_nt(
                &self.handle,
                batch,
                out_n,
                in_f,
                1.0,
                x,
                w_local,
                0.0,
                y_local,
            )?;
        }

        // All-gather y_local into y_full across ranks.
        // sendcount = batch * out_n elements per rank.
        // y_full must be sized world_size * sendcount.
        let send_count = (batch as usize) * (out_n as usize);
        self.comm.all_gather(y_local, y_full, send_count, stream)?;
        Ok(())
    }

    /// Backward pass.
    ///
    /// - `x`           : `[batch, in_features]` (from the FW pass; replicated)
    /// - `w_local`     : `[out_features/N, in_features]` (this rank's shard)
    /// - `dy_local`    : `[batch, out_features/N]` (this rank's slice of dY;
    ///   caller is responsible for slicing the full `dY` if they have it,
    ///   matching the `[rank*out/N : (rank+1)*out/N]` range)
    /// - `dx_partial`  : `[batch, in_features]` — written by the local GEMM
    ///   before the all-reduce
    /// - `dx`          : `[batch, in_features]` — written by `all_reduce(dx_partial)`
    /// - `dw_local`    : `[out_features/N, in_features]` — written; stays sharded
    pub fn backward(
        &self,
        stream: &Stream,
        x: &DeviceBuffer<T>,
        w_local: &DeviceBuffer<T>,
        dy_local: &DeviceBuffer<T>,
        dx_partial: &mut DeviceBuffer<T>,
        dx: &mut DeviceBuffer<T>,
        dw_local: &mut DeviceBuffer<T>,
    ) -> Result<()> {
        let batch = self.batch;
        let in_f = self.tpctx_in_features;
        let out_n = self.out_per_rank;

        check_len("x", x.len(), (batch as usize) * (in_f as usize))?;
        check_len(
            "w_local",
            w_local.len(),
            (out_n as usize) * (in_f as usize),
        )?;
        check_len(
            "dy_local",
            dy_local.len(),
            (batch as usize) * (out_n as usize),
        )?;
        check_len(
            "dx_partial",
            dx_partial.len(),
            (batch as usize) * (in_f as usize),
        )?;
        check_len("dx", dx.len(), (batch as usize) * (in_f as usize))?;
        check_len(
            "dw_local",
            dw_local.len(),
            (out_n as usize) * (in_f as usize),
        )?;

        self.handle.set_stream(stream)?;

        // dx_partial = dy_local @ w_local
        //   shapes: [batch, out_n] · [out_n, in_f] = [batch, in_f]
        //   m=batch, n=in_f, k=out_n
        // SAFETY: shapes validated; handle bound.
        unsafe {
            T::row_major_gemm_nn(
                &self.handle,
                batch,
                in_f,
                out_n,
                1.0,
                dy_local,
                w_local,
                0.0,
                dx_partial,
            )?;
        }

        // dw_local = dy_local^T @ x
        //   shapes: [out_n, batch] · [batch, in_f] = [out_n, in_f]
        //   m=out_n, n=in_f, k=batch
        // SAFETY: shapes validated; handle bound.
        unsafe {
            T::row_major_gemm_tn(
                &self.handle,
                out_n,
                in_f,
                batch,
                1.0,
                dy_local,
                x,
                0.0,
                dw_local,
            )?;
        }

        // All-reduce dx_partial across ranks → dx (Sum).
        // For column-parallel BW, every rank's dx_partial is its share
        // of `dX = Σ_r dY[:, r*out/N:(r+1)*out/N] @ W_r`. Summing them
        // yields the full dX, then NCCL replicates it back to every rank.
        self.comm.all_reduce(dx_partial, dx, RedOp::Sum, stream)?;
        Ok(())
    }
}

// ============================================================================
// RowParallelLinearPlan
// ============================================================================

/// Tensor-parallel Linear layer that **splits the weight matrix along
/// the INPUT dimension**. Each rank holds `W_local: [out_features, in_features/N]`,
/// and consumes a pre-sharded `X_local: [batch, in_features/N]`.
///
/// ## Forward pass
///
/// ```text
/// X_local  : [batch, in_features/N]          (this rank's input shard)
/// W_local  : [out_features, in_features/N]   (this rank's weight shard)
/// b        : [out_features]                  (optional bias, replicated;
///                                             add AFTER the all_reduce)
///
/// Y_partial = X_local @ W_local^T            (local GEMM)
///           : [batch, out_features]
///
/// Y         = all_reduce(Y_partial, Sum)
///           : [batch, out_features]          (replicated across ranks)
/// ```
///
/// Megatron pairs `ColumnParallelLinear` (output split) with
/// `RowParallelLinear` (input split) back-to-back inside an MLP /
/// attention block so the intermediate hidden state lives in
/// distributed (sharded) form and only the boundaries hit a NCCL
/// collective.
///
/// ## Backward pass
///
/// ```text
/// dY        : [batch, out_features]          (replicated; gradient w.r.t. Y)
///
/// dX_local  = dY @ W_local                   (local GEMM, no comm needed —
///           : [batch, in_features/N]          dY is already replicated)
///
/// dW_local  = dY^T @ X_local                 (local GEMM)
///           : [out_features, in_features/N]
/// db        = sum(dY, axis=0)                (optional; caller-side)
/// ```
///
/// Note the **inverted collective pattern** vs `ColumnParallelLinear`:
/// the FW pass needs an all-reduce; the BW pass needs none (dY is
/// already replicated, so each rank computes its own `dX_local` slice
/// locally). This is the design point of the Megatron pairing — only
/// one collective per layer-pair, not two.
pub struct RowParallelLinearPlan<'comm, T: MegatronGemmScalar> {
    tpctx_in_features: i32,
    tpctx_out_features: i32,
    tpctx_world_size: i32,
    tpctx_rank: i32,
    in_per_rank: i32,
    batch: i32,
    handle: Handle,
    comm: &'comm Communicator,
    _marker: PhantomData<T>,
}

impl<T: MegatronGemmScalar> core::fmt::Debug for RowParallelLinearPlan<'_, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RowParallelLinearPlan")
            .field("in_features", &self.tpctx_in_features)
            .field("out_features", &self.tpctx_out_features)
            .field("in_per_rank", &self.in_per_rank)
            .field("batch", &self.batch)
            .field("rank", &self.tpctx_rank)
            .field("world_size", &self.tpctx_world_size)
            .field("dtype", &core::any::type_name::<T>())
            .finish()
    }
}

impl<'comm, T: MegatronGemmScalar> RowParallelLinearPlan<'comm, T> {
    /// Construct a Row-parallel Linear plan bound to the given TP
    /// context for a particular batch size.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] if `in_features` is not
    /// divisible by `world_size`, or if `batch` / `in_features` /
    /// `out_features` / `world_size` are non-positive.
    pub fn new(tpctx: &TensorParallelContext<'comm>, batch: i32) -> Result<Self> {
        if batch <= 0 {
            return Err(Error::InvalidArgument("batch must be > 0"));
        }
        if tpctx.in_features <= 0 || tpctx.out_features <= 0 {
            return Err(Error::InvalidArgument(
                "in_features and out_features must be > 0",
            ));
        }
        if tpctx.world_size <= 0 {
            return Err(Error::InvalidArgument("world_size must be > 0"));
        }
        if tpctx.in_features % tpctx.world_size != 0 {
            return Err(Error::InvalidArgument(
                "RowParallelLinear: in_features must be divisible by world_size",
            ));
        }
        let handle = Handle::new()?;
        Ok(Self {
            tpctx_in_features: tpctx.in_features,
            tpctx_out_features: tpctx.out_features,
            tpctx_world_size: tpctx.world_size,
            tpctx_rank: tpctx.rank,
            in_per_rank: tpctx.in_features / tpctx.world_size,
            batch,
            handle,
            comm: tpctx.comm,
            _marker: PhantomData,
        })
    }

    /// Per-rank shard of the input dimension.
    #[inline]
    pub fn in_per_rank(&self) -> i32 {
        self.in_per_rank
    }

    /// This rank's index within the TP group.
    #[inline]
    pub fn rank(&self) -> i32 {
        self.tpctx_rank
    }

    /// Total number of ranks in the TP group.
    #[inline]
    pub fn world_size(&self) -> i32 {
        self.tpctx_world_size
    }

    /// Forward pass.
    ///
    /// - `x_local`   : `[batch, in_features/N]` (this rank's input shard)
    /// - `w_local`   : `[out_features, in_features/N]` (this rank's weight shard)
    /// - `bias`      : `Some([out_features])` (replicated bias; Tier 2 — pass `None`)
    /// - `y_partial` : `[batch, out_features]` — written by the local GEMM
    /// - `y`         : `[batch, out_features]` — written by `all_reduce(y_partial)`
    pub fn forward(
        &self,
        stream: &Stream,
        x_local: &DeviceBuffer<T>,
        w_local: &DeviceBuffer<T>,
        bias: Option<&DeviceBuffer<T>>,
        y_partial: &mut DeviceBuffer<T>,
        y: &mut DeviceBuffer<T>,
    ) -> Result<()> {
        let batch = self.batch;
        let in_n = self.in_per_rank;
        let out_f = self.tpctx_out_features;

        check_len(
            "x_local",
            x_local.len(),
            (batch as usize) * (in_n as usize),
        )?;
        check_len(
            "w_local",
            w_local.len(),
            (out_f as usize) * (in_n as usize),
        )?;
        check_len(
            "y_partial",
            y_partial.len(),
            (batch as usize) * (out_f as usize),
        )?;
        check_len("y", y.len(), (batch as usize) * (out_f as usize))?;
        if let Some(b) = bias {
            check_len("bias", b.len(), out_f as usize)?;
        }

        if bias.is_some() {
            return Err(Error::InvalidArgument(
                "RowParallelLinear: bias is Tier 2 — pass None for Phase 57; \
                 caller can run baracuda-kernels Affine after forward() to add the \
                 replicated bias (do it post-all_reduce so it doesn't get summed N times)",
            ));
        }

        self.handle.set_stream(stream)?;

        // y_partial = x_local @ w_local^T
        //   shapes: [batch, in_n] · [out_f, in_n]^T = [batch, out_f]
        //   m=batch, n=out_f, k=in_n
        // SAFETY: shapes validated; handle bound.
        unsafe {
            T::row_major_gemm_nt(
                &self.handle,
                batch,
                out_f,
                in_n,
                1.0,
                x_local,
                w_local,
                0.0,
                y_partial,
            )?;
        }

        // All-reduce y_partial → y across ranks.
        self.comm.all_reduce(y_partial, y, RedOp::Sum, stream)?;
        Ok(())
    }

    /// Backward pass.
    ///
    /// - `x_local`   : `[batch, in_features/N]` (this rank's input shard from FW)
    /// - `w_local`   : `[out_features, in_features/N]`
    /// - `dy`        : `[batch, out_features]` (replicated; gradient w.r.t. Y)
    /// - `dx_local`  : `[batch, in_features/N]` — written; this rank's input-grad shard
    /// - `dw_local`  : `[out_features, in_features/N]` — written; stays sharded
    ///
    /// **No collectives in this BW**: `dY` is replicated coming in, so
    /// each rank computes its own `dX_local` slice purely locally.
    /// This is the asymmetry that makes Column+Row pairing efficient:
    /// only one collective per layer-pair (the FW all-reduce here, or
    /// equivalently the FW all-gather of `ColumnParallel`).
    pub fn backward(
        &self,
        stream: &Stream,
        x_local: &DeviceBuffer<T>,
        w_local: &DeviceBuffer<T>,
        dy: &DeviceBuffer<T>,
        dx_local: &mut DeviceBuffer<T>,
        dw_local: &mut DeviceBuffer<T>,
    ) -> Result<()> {
        let batch = self.batch;
        let in_n = self.in_per_rank;
        let out_f = self.tpctx_out_features;

        check_len(
            "x_local",
            x_local.len(),
            (batch as usize) * (in_n as usize),
        )?;
        check_len(
            "w_local",
            w_local.len(),
            (out_f as usize) * (in_n as usize),
        )?;
        check_len("dy", dy.len(), (batch as usize) * (out_f as usize))?;
        check_len(
            "dx_local",
            dx_local.len(),
            (batch as usize) * (in_n as usize),
        )?;
        check_len(
            "dw_local",
            dw_local.len(),
            (out_f as usize) * (in_n as usize),
        )?;

        self.handle.set_stream(stream)?;

        // dx_local = dy @ w_local
        //   shapes: [batch, out_f] · [out_f, in_n] = [batch, in_n]
        //   m=batch, n=in_n, k=out_f
        // SAFETY: shapes validated; handle bound.
        unsafe {
            T::row_major_gemm_nn(
                &self.handle,
                batch,
                in_n,
                out_f,
                1.0,
                dy,
                w_local,
                0.0,
                dx_local,
            )?;
        }

        // dw_local = dy^T @ x_local
        //   shapes: [out_f, batch] · [batch, in_n] = [out_f, in_n]
        //   m=out_f, n=in_n, k=batch
        // SAFETY: shapes validated; handle bound.
        unsafe {
            T::row_major_gemm_tn(
                &self.handle,
                out_f,
                in_n,
                batch,
                1.0,
                dy,
                x_local,
                0.0,
                dw_local,
            )?;
        }

        // No collective — dX is sharded by design (caller pairs with
        // the upstream ColumnParallel whose BW will all-reduce the dX
        // sum back).
        Ok(())
    }
}

// ============================================================================
// Helpers
// ============================================================================

#[inline]
fn check_len(what: &'static str, got: usize, expected: usize) -> Result<()> {
    if got != expected {
        return Err(Error::BufferLengthMismatch {
            what,
            got,
            expected,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Pure-host unit tests — no GPU / NCCL required. Compile-only
    //! sanity checks on the trait bounds.
    use super::*;

    /// Compile-time check that `f32` satisfies the dtype trait bounds.
    #[allow(dead_code)]
    fn require_megatron_scalar<T: MegatronGemmScalar>() {}

    #[test]
    fn f32_is_megatron_scalar() {
        require_megatron_scalar::<f32>();
    }

    #[cfg(feature = "half-crate")]
    #[test]
    fn half_dtypes_are_megatron_scalars() {
        require_megatron_scalar::<half::f16>();
        require_megatron_scalar::<half::bf16>();
    }
}
