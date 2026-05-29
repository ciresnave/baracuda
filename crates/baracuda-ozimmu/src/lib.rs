//! # baracuda-ozimmu
//!
//! Safe Rust wrapper for baracuda's clean-fork of
//! [ozIMMU](https://github.com/enp1s0/ozIMMU) — Hiroyuki Ootomo's
//! Ozaki-scheme FP64 GEMM library. Provides an RAII [`Handle`] type
//! and a drop-in [`Handle::dgemm`] entry that lets baracuda's
//! `GemmPlan` route FP64 matrix-multiplies through `S²` int8 tensor-
//! core matmuls when the caller opts in via
//! `PlanPreference::prefer_backend = Some(BackendKind::Ozaki { slices: ... })`.
//!
//! ## Accuracy contract
//!
//! **NOT bit-equivalent to native DGEMM.** ozIMMU at `S = 8` reaches
//! "comparable to DGEMM" accuracy on well-conditioned inputs (see the
//! original paper for the formal mantissa-loss analysis); at `S = 3`
//! it's intentionally low-precision and at `S = 18` it's slow. Use
//! [`OzakiSlices::S8`] as the default; raise `S` for ill-conditioned
//! workloads, or use [`OzakiSlices::Auto`] to let ozIMMU pick
//! dynamically based on each input's mantissa-loss histogram.
//!
//! For workloads that need the bit-exact DGEMM contract, do **not**
//! use this crate — baracuda's default FP64 GEMM path stays on
//! CUTLASS / cuBLAS DGEMM and gives that guarantee.
//!
//! ## Determinism
//!
//! Given the same hardware + same `OzakiSlices` setting, ozIMMU is
//! bit-reproducible across launches (the int8 tensor-core path itself
//! is deterministic; the upstream library does not use atomics on
//! the accumulate stage). Switching `OzakiSlices` is a numerical
//! change and produces different — but still bit-reproducible — output.
//!
//! ## Workspace
//!
//! ozIMMU owns its own working-memory allocator (`cudaMalloc` /
//! `cudaMallocAsync` under the hood, picked at [`Handle::new_with_mode`]
//! time). The first [`Handle::dgemm`] call grows the scratch to the
//! size required by the problem shape; subsequent calls re-use it.
//! Use [`Handle::reallocate_working_memory_bytes`] to pre-grow the
//! scratch if you want the one-time allocation cost paid before the
//! hot path.
//!
//! The workspace lifecycle is **not** integrated with baracuda's
//! stream-ordered allocator in this alpha. A future polish phase may
//! add that bridge if profiling shows the ozIMMU-internal allocator
//! contending with the rest of baracuda for VRAM at scale.
//!
//! ## Acknowledgments
//!
//! ozIMMU is the work of Hiroyuki Ootomo, Katsuhisa Ozaki & Rio Yokota
//! ("DGEMM on Integer Matrix Multiplication Unit", *IJHPCA* 2024;
//! [arXiv:2306.11975](https://arxiv.org/abs/2306.11975)). The original
//! reference implementation is MIT-licensed; baracuda Phase 44b
//! internalized the sources under `crates/baracuda-ozimmu-sys/cuda/`
//! — see that crate's `ATTRIBUTION.md` for the full provenance
//! story, the algorithm references, and the unmodified MIT license
//! text.

#![deny(missing_docs)]

use core::ffi::c_void;
use core::ptr;

use baracuda_driver::Stream;
use baracuda_ozimmu_sys as sys;

use thiserror::Error;

/// Error category surfaced by [`Handle`] operations.
///
/// Wraps both Rust-side validation failures and the integer status
/// codes returned by the underlying ozIMMU C entry points.
///
/// `#[non_exhaustive]` per the baracuda Phase 28 audit — new variants
/// may land as the safe surface grows (e.g. workspace-bridge failures
/// once the stream-ordered allocator integration ships).
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// `mtk::ozimmu::create` returned non-zero. The wrapped integer is
    /// the raw status code; ozIMMU does not promise a stable mapping
    /// across versions but `0` always means success.
    #[error("ozIMMU handle creation failed (status {0})")]
    CreateFailed(i32),
    /// `mtk::ozimmu::destroy` returned non-zero. This is rare in
    /// practice — `destroy` mainly frees `cudaMalloc`'d scratch and
    /// `cublasDestroy`s the embedded handle.
    #[error("ozIMMU handle destruction failed (status {0})")]
    DestroyFailed(i32),
    /// `baracuda_ozimmu_dgemm` returned non-zero. The wrapped value is
    /// the propagated `cublasStatus_t` from the internal cuBLAS GEMM
    /// chain (e.g. `13` = `CUBLAS_STATUS_EXECUTION_FAILED`).
    #[error("ozIMMU dgemm launch failed (status {0})")]
    DgemmFailed(i32),
    /// Host-side argument validation failed before launch. The string
    /// is a short message describing which precondition was violated.
    #[error("ozIMMU argument invalid: {0}")]
    InvalidArgument(&'static str),
}

/// `Result` alias used throughout the crate.
pub type Result<T> = core::result::Result<T, Error>;

/// How many int8 slices ozIMMU splits each FP64 matrix into.
///
/// The Ozaki scheme synthesizes one FP64 GEMM out of `S²` int8
/// tensor-core matmuls. Higher `S` → better accuracy, more compute.
/// At `S = 8` accuracy is "comparable to DGEMM" on well-conditioned
/// inputs (the upstream paper's recommended default); at `S = 18` it
/// recovers most of the remaining IEEE-754 mantissa precision but
/// costs proportionally more.
///
/// [`OzakiSlices::Auto`] lets ozIMMU pick `S` per-call based on the
/// inputs' mantissa-loss histogram (the runtime threshold is
/// configurable via the env var `OZIMMU_MANTISSA_LOSS_THRESHOLD` or
/// the `set_auto_mantissa_loss_threshold` upstream API, which the
/// baracuda wrapper does not currently expose).
///
/// `#[non_exhaustive]` per the baracuda Phase 28 audit — variants are
/// stable but the upstream library could add new compute modes.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum OzakiSlices {
    /// 3-slice — fastest, lowest accuracy. Not recommended outside
    /// research / sweeps.
    S3,
    /// 4-slice.
    S4,
    /// 5-slice.
    S5,
    /// 6-slice.
    S6,
    /// 7-slice.
    S7,
    /// 8-slice — "comparable-to-DGEMM-accuracy" sweet spot.
    /// This is the safe-wrapper default.
    S8,
    /// 9-slice.
    S9,
    /// 10-slice.
    S10,
    /// 11-slice.
    S11,
    /// 12-slice — typical "high-accuracy" choice for ill-conditioned
    /// matrices.
    S12,
    /// 13-slice.
    S13,
    /// 14-slice.
    S14,
    /// 15-slice.
    S15,
    /// 16-slice.
    S16,
    /// 17-slice.
    S17,
    /// 18-slice — most accurate, most expensive. Approaches native
    /// DGEMM accuracy on most workloads.
    S18,
    /// Per-call dynamic selection based on the inputs' mantissa-loss
    /// histogram. Adds one extra small kernel launch per `dgemm`
    /// (the histogram pass) but lets ozIMMU spend cycles
    /// proportional to the actual accuracy demand.
    Auto,
}

impl Default for OzakiSlices {
    /// `S8` — the "comparable-to-DGEMM" sweet spot recommended by the
    /// upstream paper.
    fn default() -> Self {
        OzakiSlices::S8
    }
}

impl OzakiSlices {
    /// Cast to the integer `compute_mode_t` the FFI shim expects.
    pub fn to_compute_mode(self) -> sys::ComputeMode {
        match self {
            OzakiSlices::S3 => sys::COMPUTE_MODE_FP64_INT8_3,
            OzakiSlices::S4 => sys::COMPUTE_MODE_FP64_INT8_4,
            OzakiSlices::S5 => sys::COMPUTE_MODE_FP64_INT8_5,
            OzakiSlices::S6 => sys::COMPUTE_MODE_FP64_INT8_6,
            OzakiSlices::S7 => sys::COMPUTE_MODE_FP64_INT8_7,
            OzakiSlices::S8 => sys::COMPUTE_MODE_FP64_INT8_8,
            OzakiSlices::S9 => sys::COMPUTE_MODE_FP64_INT8_9,
            OzakiSlices::S10 => sys::COMPUTE_MODE_FP64_INT8_10,
            OzakiSlices::S11 => sys::COMPUTE_MODE_FP64_INT8_11,
            OzakiSlices::S12 => sys::COMPUTE_MODE_FP64_INT8_12,
            OzakiSlices::S13 => sys::COMPUTE_MODE_FP64_INT8_13,
            OzakiSlices::S14 => sys::COMPUTE_MODE_FP64_INT8_14,
            OzakiSlices::S15 => sys::COMPUTE_MODE_FP64_INT8_15,
            OzakiSlices::S16 => sys::COMPUTE_MODE_FP64_INT8_16,
            OzakiSlices::S17 => sys::COMPUTE_MODE_FP64_INT8_17,
            OzakiSlices::S18 => sys::COMPUTE_MODE_FP64_INT8_18,
            OzakiSlices::Auto => sys::COMPUTE_MODE_FP64_INT8_AUTO,
        }
    }

    /// Numeric slice count (3..=18), or `None` for [`Auto`].
    ///
    /// Useful for logging / telemetry. `None` doesn't mean "no slices" —
    /// it means "decided at run time by the library."
    pub fn slice_count(self) -> Option<u32> {
        match self {
            OzakiSlices::S3 => Some(3),
            OzakiSlices::S4 => Some(4),
            OzakiSlices::S5 => Some(5),
            OzakiSlices::S6 => Some(6),
            OzakiSlices::S7 => Some(7),
            OzakiSlices::S8 => Some(8),
            OzakiSlices::S9 => Some(9),
            OzakiSlices::S10 => Some(10),
            OzakiSlices::S11 => Some(11),
            OzakiSlices::S12 => Some(12),
            OzakiSlices::S13 => Some(13),
            OzakiSlices::S14 => Some(14),
            OzakiSlices::S15 => Some(15),
            OzakiSlices::S16 => Some(16),
            OzakiSlices::S17 => Some(17),
            OzakiSlices::S18 => Some(18),
            OzakiSlices::Auto => None,
        }
    }
}

/// Operand transpose flag — mirrors `cublasOperation_t`.
///
/// Same convention as `baracuda-cublas::Op::{N, T}`. The upstream
/// ozIMMU `mtk::ozimmu::operation_t` enum only has `op_n` / `op_t` —
/// the conjugate-transpose `op_c` isn't supported on the real path
/// (the Ozaki scheme is real-only; complex64 is out of scope for
/// the Phase 44 integration).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Op {
    /// No transpose.
    N,
    /// Transpose.
    T,
}

impl Op {
    /// Cast to the FFI's `op_a` / `op_b` integer (0 = N, 1 = T).
    pub fn to_ffi(self) -> i32 {
        match self {
            Op::N => 0,
            Op::T => 1,
        }
    }
}

/// Whether the handle allocates working memory synchronously or via
/// `cudaMallocAsync`.
///
/// Sync is the safe default (matches the upstream `malloc_sync`).
/// Async is meaningful only when the bound stream is using the
/// memory-pool allocator and the caller wants the scratch grown
/// without an implicit stream-wide sync.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Default)]
pub enum MallocMode {
    /// `cudaMalloc` — synchronous, default.
    #[default]
    Sync,
    /// `cudaMallocAsync` — bound to the handle's stream's memory pool.
    Async,
}

/// Numerical guarantee surfaced by [`Handle::precision_guarantee`].
///
/// Lightweight mirror of `baracuda_kernels_types::PrecisionGuarantee`
/// without taking a workspace dep on it from this leaf crate. The
/// `baracuda-kernels` integration translates this into the workspace's
/// shared type.
///
/// **The accuracy claim is workload-dependent.** ozIMMU at `S >= 8` is
/// "comparable to DGEMM" on well-conditioned inputs (per the upstream
/// paper); at `S < 8` accuracy degrades quickly. Always validate
/// against your own workload before substituting for `cublasDgemm`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct PrecisionGuarantee {
    /// Bit-reproducible across runs on the same hardware with the
    /// same inputs and the same `OzakiSlices` choice. ozIMMU does not
    /// use atomics on the int8 → fp64 accumulate stage.
    pub bit_stable_on_same_hardware: bool,
    /// Bit-reproducible within a single process / thread — i.e. no
    /// internal randomness. Always `true` for non-`Auto` modes;
    /// `Auto` is also deterministic given the same inputs (the
    /// histogram pass is deterministic).
    pub deterministic: bool,
}

impl PrecisionGuarantee {
    /// The guarantee any non-`Auto` `OzakiSlices` provides.
    pub const fn standard() -> Self {
        Self {
            bit_stable_on_same_hardware: true,
            deterministic: true,
        }
    }
}

/// RAII handle to an ozIMMU session.
///
/// Owns:
///
/// - the embedded cuBLAS handle (created at [`Handle::new`] time);
/// - the working-memory scratch (grown lazily on the first
///   [`Handle::dgemm`] launch);
/// - a copy of the bound CUDA stream.
///
/// `Drop` calls `mtk::ozimmu::destroy`, which frees both the cuBLAS
/// handle and the scratch buffer. Subsequent `dgemm` launches that
/// have already been enqueued on the stream are not affected by drop
/// because the scratch lives long enough — cuBLAS captures the
/// pointer at launch time and the upstream `destroy` does not
/// `cudaFree` until the stream sync the caller is expected to do
/// before drop.
///
/// **Not `Sync`**: ozIMMU handles wrap a cuBLAS handle, which is
/// `!Sync` per the cuBLAS contract (one host thread at a time). The
/// PhantomData below records that.
pub struct Handle {
    raw: sys::OzimmuHandleT,
    _not_sync: core::marker::PhantomData<*mut ()>,
}

impl Handle {
    /// Create a new handle in [`MallocMode::Sync`] mode.
    pub fn new() -> Result<Self> {
        Self::new_with_mode(MallocMode::Sync)
    }

    /// Create a new handle with an explicit allocator mode.
    pub fn new_with_mode(mode: MallocMode) -> Result<Self> {
        let mut raw: sys::OzimmuHandleT = ptr::null_mut();
        let async_flag: i32 = match mode {
            MallocMode::Sync => 0,
            MallocMode::Async => 1,
        };
        // SAFETY: `out_handle` is a valid pointer to writable storage
        // for the duration of the call.
        let status = unsafe { sys::baracuda_ozimmu_create(&mut raw, async_flag) };
        if status != 0 {
            return Err(Error::CreateFailed(status));
        }
        if raw.is_null() {
            return Err(Error::CreateFailed(-1));
        }
        Ok(Self {
            raw,
            _not_sync: core::marker::PhantomData,
        })
    }

    /// Bind a CUDA stream to this handle.
    ///
    /// Subsequent [`Self::dgemm`] launches enqueue on the supplied
    /// stream. Re-binding is supported and is the usual pattern when
    /// the handle is held thread-local but used across multiple
    /// per-request streams. The bind is sticky across calls.
    pub fn set_stream(&self, stream: &Stream) {
        let raw_stream = stream.as_raw() as *mut c_void;
        // SAFETY: handle is non-null (constructor enforces); raw_stream
        // is a valid CUstream alias for the stream's lifetime.
        unsafe {
            sys::baracuda_ozimmu_set_cuda_stream(self.raw, raw_stream);
        }
    }

    /// Pre-grow the working-memory scratch to at least `bytes`.
    ///
    /// Returns the new total scratch size (in bytes) when the
    /// allocator grew, or `0` when the existing scratch was already
    /// large enough. Idempotent.
    pub fn reallocate_working_memory_bytes(&self, bytes: usize) -> usize {
        // SAFETY: handle is non-null.
        unsafe { sys::baracuda_ozimmu_reallocate_working_memory_bytes(self.raw, bytes) }
    }

    /// Numerical guarantee this handle's `dgemm` provides.
    pub const fn precision_guarantee() -> PrecisionGuarantee {
        PrecisionGuarantee::standard()
    }

    /// FP64 GEMM via the Ozaki scheme.
    ///
    /// `compute_mode` controls the slice count `S` (see
    /// [`OzakiSlices`]). Layout convention is cuBLAS column-major,
    /// matching `cublasDgemm` — callers integrating into baracuda's
    /// row-major-first `GemmPlan` need to apply the standard
    /// `D^T = (op_b B)^T · (op_a A)^T` mapping themselves (see the
    /// `baracuda-kernels` `GemmPlan::run_ozimmu` dispatch for the
    /// reference call site).
    ///
    /// All pointer arguments are device-resident `f64`s. `alpha` /
    /// `beta` are host scalars (matches cuBLAS).
    ///
    /// # Safety
    ///
    /// `a_ptr`, `b_ptr`, `c_ptr` must be valid device pointers into
    /// FP64 buffers sized for the (op, m, n, k, ld) tuple, and must
    /// live for the duration of the launch (the call is asynchronous
    /// on the bound stream — the caller is responsible for not
    /// freeing the underlying memory before a stream sync).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn dgemm(
        &self,
        op_a: Op,
        op_b: Op,
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a_ptr: *const f64,
        lda: usize,
        b_ptr: *const f64,
        ldb: usize,
        beta: f64,
        c_ptr: *mut f64,
        ldc: usize,
        compute_mode: OzakiSlices,
    ) -> Result<()> {
        if m == 0 || n == 0 || k == 0 {
            return Err(Error::InvalidArgument("m, n, k must be > 0"));
        }
        let min_lda = if op_a == Op::N { m } else { k };
        let min_ldb = if op_b == Op::N { k } else { n };
        if lda < min_lda {
            return Err(Error::InvalidArgument(
                "lda smaller than the leading-dim minimum for op_a",
            ));
        }
        if ldb < min_ldb {
            return Err(Error::InvalidArgument(
                "ldb smaller than the leading-dim minimum for op_b",
            ));
        }
        if ldc < m {
            return Err(Error::InvalidArgument("ldc must be >= m"));
        }
        let status = unsafe {
            sys::baracuda_ozimmu_dgemm(
                self.raw,
                op_a.to_ffi(),
                op_b.to_ffi(),
                m,
                n,
                k,
                &alpha as *const f64,
                a_ptr,
                lda,
                b_ptr,
                ldb,
                &beta as *const f64,
                c_ptr,
                ldc,
                compute_mode.to_compute_mode(),
            )
        };
        if status != 0 {
            return Err(Error::DgemmFailed(status));
        }
        Ok(())
    }

    /// Raw FFI handle — useful for ABI-bridging into other crates
    /// (e.g. `baracuda-kernels`'s GemmPlan dispatch). Callers must
    /// not call `baracuda_ozimmu_destroy` on the returned pointer;
    /// the wrapping `Handle` retains ownership.
    pub fn as_raw(&self) -> sys::OzimmuHandleT {
        self.raw
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: `self.raw` was produced by `_create` and we
            // haven't already destroyed it. The status return is
            // discarded — there's nothing actionable a Drop can do
            // if `destroy` fails. A stray non-zero would typically
            // mean the caller didn't sync the stream before dropping,
            // which is a usage error caught by `cuda-memcheck`.
            unsafe {
                let _ = sys::baracuda_ozimmu_destroy(self.raw);
            }
            self.raw = ptr::null_mut();
        }
    }
}

// `Handle` is intentionally `!Sync` (see the struct docs). `Send` is
// safe — the underlying ozIMMU handle holds device-side state, not
// thread-local state, and `set_stream` re-binds to whichever stream
// the new-owner thread provides.
unsafe impl Send for Handle {}
