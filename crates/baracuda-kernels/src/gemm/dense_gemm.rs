//! Dense floating-point GEMM plan — cuBLAS-backed (Phase 74).
//!
//! The plain dense `f32 / f64 / f16 / bf16` GEMM family that the rest
//! of the `gemm` module conspicuously lacked: every other plan here is
//! quantized, sparse, or packed (`int_gemm`, `fp8_gemm`, `int4_*`,
//! `bin_gemm`, `sparse24`), while plain dense GEMM lived only in
//! `baracuda-cutlass`. That split forced downstream consumers (Fuel)
//! to keep their own cuBLAS MatMul wrapper — the last non-baracuda
//! CUDA code on their side. This plan (and its flat C twin,
//! `baracuda_kernels_gemm_dense_*` in `baracuda-kernels-sys`) closes
//! that gap.
//!
//! ## Relationship to `baracuda_cutlass::GemmPlan`
//!
//! Deliberately complementary, not a replacement:
//!
//! - **This plan** is cuBLAS-backed, covers RRR / RCR / **CRR**
//!   layouts, folds strided-batch into the descriptor, and its f32
//!   path is true IEEE binary32 (cuBLAS default math mode — no TF32).
//! - **`GemmPlan`** is CUTLASS-first (with the Phase 30 cuBLAS
//!   decode-regime heuristic), adds fused `Bias*` epilogues and the
//!   Ozaki f64 backend, but has no CRR and batches only via the
//!   separate `BatchedGemmPlan` (Rcr × f16/bf16 only).
//!
//! Callers wanting CUTLASS routing or fused epilogues should use
//! `GemmPlan`; callers wanting uniform dense coverage with maximum
//! layout/stride flexibility (autograd graphs, framework backends)
//! belong here. If a CUTLASS fast path later supersedes cuBLAS for
//! some SKU, it slots in behind this same plan (and behind the same
//! FFI symbols) without caller-visible changes.
//!
//! ## Layout vocabulary
//!
//! This plan needs CRR (column-major A — the `xᵀ · dy` grad-weight
//! shape), which [`LayoutSku`](baracuda_kernels_types::LayoutSku)
//! doesn't have. Rather than growing that (deliberately exhaustive,
//! hot-path-matched) enum in the same phase, the plan carries its own
//! [`DenseGemmLayout`]. Unifying the two vocabularies is tracked in
//! the ROADMAP layout-planner item.
//!
//! ## Numerics
//!
//! f16 / bf16 accumulate in f32 (`CUBLAS_COMPUTE_32F`), matching the
//! reduce family's convention. f64 accumulates in f64. Determinism
//! follows cuBLAS's published guarantee: bitwise-identical across runs
//! for the same (toolkit version, GPU architecture, SM count, shape).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    BackendKind, Element, ElementKind, MathPrecision, MatrixMut, MatrixRef, PlanPreference,
    PrecisionGuarantee, ScalarType, Workspace,
};

/// Operand layout for [`DenseGemmPlan`]. `D` is always row-major
/// `[M, N]`.
///
/// The integer tags match the `layout` parameter of the
/// `baracuda_kernels_gemm_dense_*` FFI symbols (RRR = 0, RCR = 1,
/// CRR = 2) — keep the two in sync.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum DenseGemmLayout {
    /// `A` row-major `[M, K]` (`ld ≥ K`), `B` row-major `[K, N]`
    /// (`ld ≥ N`). The "plain matmul" shape.
    Rrr,
    /// `A` row-major `[M, K]` (`ld ≥ K`), `B` column-major `[K, N]`
    /// (`ld ≥ K`). The "transposed weight" shape (`x · Wᵀ` with `W`
    /// stored `[N, K]` row-major).
    Rcr,
    /// `A` column-major `[M, K]` (`ld ≥ M`), `B` row-major `[K, N]`
    /// (`ld ≥ N`). The "transposed activation" shape (`xᵀ · dy`,
    /// grad-weight). Unique to this family — no CUTLASS SKU covers it.
    Crr,
}

impl DenseGemmLayout {
    /// FFI tag for the `baracuda_kernels_gemm_dense_*` symbols.
    #[inline]
    fn ffi_tag(self) -> i32 {
        match self {
            DenseGemmLayout::Rrr => 0,
            DenseGemmLayout::Rcr => 1,
            DenseGemmLayout::Crr => 2,
        }
    }

    /// `(min_lda, min_ldb)` in elements for a `(M, N, K)` problem.
    #[inline]
    fn min_lds(self, m: i32, n: i32, k: i32) -> (i64, i64) {
        match self {
            DenseGemmLayout::Rrr => (k as i64, n as i64),
            DenseGemmLayout::Rcr => (k as i64, k as i64),
            DenseGemmLayout::Crr => (m as i64, n as i64),
        }
    }
}

/// Problem shape handed to [`DenseGemmPlan::select`].
#[derive(Copy, Clone, Debug)]
pub struct DenseGemmDescriptor {
    /// Output row count (per batch slot).
    pub m: i32,
    /// Output column count (per batch slot).
    pub n: i32,
    /// Reduction depth (per batch slot).
    pub k: i32,
    /// Number of batch slots. `1` for a single GEMM; `0` is a valid
    /// empty problem (no-op `run`).
    pub batch: i32,
    /// Operand layout.
    pub layout: DenseGemmLayout,
}

/// Per-launch arguments for a [`DenseGemmPlan::run`] call.
///
/// Computes `D[g] = α · A[g] · B[g] + β · D[g]` per batch slot `g`.
/// There is no separate `C` operand: `β ≠ 0` accumulates into the
/// existing contents of `D` (read-modify-write — callers wanting
/// `D = α·A·B + β·C` with `C ≠ D` copy `C` into `D` first).
///
/// `stride_*` are batch strides in **elements** (`ptr + g * stride`),
/// ignored when `desc.batch == 1`. `stride_a` / `stride_b` may be `0`
/// to broadcast one matrix across all slots; `stride_d` must be
/// non-zero when `batch > 1` (overlapping outputs race).
///
/// Each [`MatrixRef`]'s `rows` / `cols` are the **logical** dims
/// (`A: [M, K]`, `B: [K, N]`, `D: [M, N]`) regardless of layout; `ld`
/// follows the storage order ([`MatrixRef`]'s convention: row-stride
/// for row-major operands, column-stride for column-major ones).
#[derive(Debug)]
pub struct DenseGemmArgs<'a, T: Element> {
    /// Left input — base pointer for batch slot 0.
    pub a: MatrixRef<'a, T>,
    /// Element offset between consecutive `A` slots (`0` = broadcast).
    pub stride_a: i64,
    /// Right input — base pointer for batch slot 0.
    pub b: MatrixRef<'a, T>,
    /// Element offset between consecutive `B` slots (`0` = broadcast).
    pub stride_b: i64,
    /// Output (and `β`-accumulation source) — base pointer for slot 0.
    pub d: MatrixMut<'a, T>,
    /// Element offset between consecutive `D` slots.
    pub stride_d: i64,
    /// Multiplier on the matrix-multiply accumulator. `f32` for
    /// f16/bf16/f32, `f64` for f64 (via `T::Scalar`).
    pub alpha: T::Scalar,
    /// Multiplier on the existing contents of `D`.
    pub beta: T::Scalar,
}

/// Dense floating-point GEMM plan (cuBLAS-backed). See the module docs
/// for scope and the relationship to `baracuda_cutlass::GemmPlan`.
pub struct DenseGemmPlan<T: Element> {
    desc: DenseGemmDescriptor,
    _marker: PhantomData<T>,
}

impl<T: Element> DenseGemmPlan<T> {
    /// Pick a dense GEMM kernel for `desc`.
    ///
    /// Backend is always [`BackendKind::Cublas`] in v1;
    /// `pref.prefer_backend` may name it (or be `None`) — any other
    /// backend request is [`Error::Unsupported`].
    pub fn select(
        _stream: &Stream,
        desc: &DenseGemmDescriptor,
        pref: PlanPreference,
    ) -> Result<Self> {
        if !matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F64 | ElementKind::F16 | ElementKind::Bf16
        ) {
            return Err(Error::Unsupported(
                "baracuda-kernels::DenseGemmPlan: dense GEMM covers f32 / f64 / f16 / \
                 bf16 only (for bit-stable strict-f32 SIMT math use \
                 baracuda_cutlass::GemmPlan<F32Strict>)",
            ));
        }
        if desc.m < 0 || desc.n < 0 || desc.k < 0 || desc.batch < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::DenseGemmPlan: M, N, K, batch must be non-negative",
            ));
        }
        match pref.prefer_backend {
            None | Some(BackendKind::Cublas) => {}
            Some(_) => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::DenseGemmPlan: v1 is cuBLAS-backed only; \
                     leave PlanPreference::prefer_backend unset or set it to \
                     BackendKind::Cublas",
                ));
            }
        }
        Ok(Self {
            desc: *desc,
            _marker: PhantomData,
        })
    }

    /// Validate that this plan can launch with `args`.
    pub fn can_implement(&self, args: &DenseGemmArgs<'_, T>) -> Result<()> {
        let d = &self.desc;
        if args.a.rows != d.m || args.a.cols != d.k {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::DenseGemmPlan: A logical shape mismatch with \
                 descriptor (M, K)",
            ));
        }
        if args.b.rows != d.k || args.b.cols != d.n {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::DenseGemmPlan: B logical shape mismatch with \
                 descriptor (K, N)",
            ));
        }
        if args.d.rows != d.m || args.d.cols != d.n {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::DenseGemmPlan: D shape mismatch with descriptor \
                 (M, N)",
            ));
        }
        // Leading-dim minimums — keep in sync with the FFI facade's
        // `validate` (gemm_dense_cublas_facade.rs), which re-checks at
        // run time.
        let (min_lda, min_ldb) = d.layout.min_lds(d.m, d.n, d.k);
        if args.a.ld < min_lda.max(1) {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::DenseGemmPlan: lda below the layout's minimum",
            ));
        }
        if args.b.ld < min_ldb.max(1) {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::DenseGemmPlan: ldb below the layout's minimum",
            ));
        }
        if args.d.ld < (d.n as i64).max(1) {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::DenseGemmPlan: ldd below N",
            ));
        }
        let i32_max = i32::MAX as i64;
        if args.a.ld > i32_max || args.b.ld > i32_max || args.d.ld > i32_max {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::DenseGemmPlan: leading dimensions must fit in i32 \
                 (cuBLAS limit)",
            ));
        }
        if d.batch > 1 && args.stride_d == 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::DenseGemmPlan: stride_d must be non-zero when \
                 batch > 1 (overlapping outputs race)",
            ));
        }
        // Negative batch strides can never index in-bounds: the
        // MatrixRef base pointer IS the slice start, so any negative
        // reach walks before the allocation.
        if d.batch > 1 && (args.stride_a < 0 || args.stride_b < 0 || args.stride_d < 0) {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::DenseGemmPlan: negative batch strides walk before \
                 the buffer base (MatrixRef has no base offset)",
            ));
        }

        // Buffer-bounds: device-slice lengths must cover the per-slot
        // storage footprint plus the (batch-1)·stride reach — the
        // unsafe FFI facade below has no length information, so this
        // typed layer is the soundness boundary (same contract as
        // `baracuda_cutlass::GemmPlan` / `BatchedGemmPlan`).
        if d.m > 0 && d.n > 0 && d.batch > 0 {
            // Storage footprint in elements of one slot: row-major
            // `[rows, cols]` with leading dim `ld` touches
            // `(rows-1)·ld + cols`; a col-major operand is the same
            // formula with rows/cols swapped.
            let footprint = |rows: i64, cols: i64, ld: i64| -> i64 {
                if rows == 0 || cols == 0 {
                    0
                } else {
                    (rows - 1) * ld + cols
                }
            };
            let (m, n, k) = (d.m as i64, d.n as i64, d.k as i64);
            let a_slot = match d.layout {
                DenseGemmLayout::Rrr | DenseGemmLayout::Rcr => footprint(m, k, args.a.ld),
                DenseGemmLayout::Crr => footprint(k, m, args.a.ld),
            };
            let b_slot = match d.layout {
                DenseGemmLayout::Rrr | DenseGemmLayout::Crr => footprint(k, n, args.b.ld),
                DenseGemmLayout::Rcr => footprint(n, k, args.b.ld),
            };
            let d_slot = footprint(m, n, args.d.ld);
            let reach = |slot: i64, stride: i64| -> Result<i64> {
                if d.batch == 1 || stride == 0 {
                    return Ok(slot);
                }
                (d.batch as i64 - 1)
                    .checked_mul(stride)
                    .and_then(|extra| slot.checked_add(extra))
                    .ok_or(Error::InvalidProblem(
                        "baracuda-kernels::DenseGemmPlan: batch-stride reach overflows i64",
                    ))
            };
            let check = |needed: i64, got: usize| -> Result<()> {
                if (got as i64) < needed {
                    return Err(Error::BufferTooSmall {
                        needed: needed as usize,
                        got,
                    });
                }
                Ok(())
            };
            check(reach(a_slot, args.stride_a)?, args.a.data.len())?;
            check(reach(b_slot, args.stride_b)?, args.b.data.len())?;
            check(reach(d_slot, args.stride_d)?, args.d.data.len())?;
        }
        Ok(())
    }

    /// Workspace size in bytes — always `0` (cuBLAS manages its own
    /// per-handle workspace internally).
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// The library backend this plan launches through. Always
    /// [`BackendKind::Cublas`] in v1.
    #[inline]
    pub fn backend(&self) -> BackendKind {
        BackendKind::Cublas
    }

    /// The operand layout this plan was selected for.
    ///
    /// Note: no `GemmSku` accessor on this plan — `GemmSku::layout` is
    /// a [`LayoutSku`](baracuda_kernels_types::LayoutSku), which can't
    /// represent [`DenseGemmLayout::Crr`] yet (see the module docs).
    #[inline]
    pub fn layout(&self) -> DenseGemmLayout {
        self.desc.layout
    }

    /// Numerical guarantees for this plan's launches.
    ///
    /// `bit_stable_on_same_hardware` / `deterministic` are reported
    /// `false` — conservatively. cuBLAS's published reproducibility
    /// guarantee (identical bits across runs for the same toolkit
    /// version, GPU architecture, SM count, and shape) holds **only
    /// while a single CUDA stream is active**; with concurrent GEMMs
    /// on multiple streams the library may select different internal
    /// implementations run-to-run. The shared-handle pool in the FFI
    /// facade cannot enforce the single-stream condition, so the plan
    /// doesn't promise what it can't keep. Callers that control their
    /// stream discipline (one active stream) DO get cuBLAS's bitwise
    /// run-to-run stability in practice.
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        let (math_precision, accumulator) = match T::KIND {
            ElementKind::F16 => (MathPrecision::F16, ElementKind::F32),
            ElementKind::Bf16 => (MathPrecision::Bf16, ElementKind::F32),
            ElementKind::F64 => (MathPrecision::F64, ElementKind::F64),
            // f32 — cuBLAS default math mode: true IEEE binary32
            // multiply-add, NOT TF32 (unlike GemmPlan<f32>'s CUTLASS
            // tensor-core SKU). Process-wide NVIDIA_TF32_OVERRIDE=1
            // would silently force TF32 — see the facade module docs.
            _ => (MathPrecision::F32, ElementKind::F32),
        };
        PrecisionGuarantee {
            math_precision,
            accumulator,
            bit_stable_on_same_hardware: false,
            deterministic: false,
        }
    }

    /// Identity of the kernel this plan picked, in the generic
    /// [`KernelSku`](baracuda_kernels_types::KernelSku) vocabulary.
    ///
    /// `layout` is `Some` for [`DenseGemmLayout::Rrr`] / [`Rcr`]
    /// (mapped onto [`LayoutSku`](baracuda_kernels_types::LayoutSku))
    /// and `None` for [`Crr`] — `LayoutSku` has no col-major-A variant
    /// yet (see the module docs); use [`Self::layout`] for the
    /// lossless value.
    ///
    /// [`Rcr`]: DenseGemmLayout::Rcr
    /// [`Crr`]: DenseGemmLayout::Crr
    pub fn sku(&self) -> baracuda_kernels_types::KernelSku {
        use baracuda_kernels_types::{KernelSku, LayoutSku, OpCategory};
        KernelSku {
            category: OpCategory::Gemm,
            // Category-local op tag: 0 = dense. (The quantized GEMM
            // plans identify through `GemmSku`, not `KernelSku`, so
            // this namespace currently has the one entry.)
            op: 0,
            element: T::KIND,
            aux_element: None,
            layout: match self.desc.layout {
                DenseGemmLayout::Rrr => Some(LayoutSku::Rrr),
                DenseGemmLayout::Rcr => Some(LayoutSku::Rcr),
                DenseGemmLayout::Crr => None,
            },
            epilogue: None,
            arch: baracuda_kernels_types::ArchSku::Sm80,
            backend: BackendKind::Cublas,
            precision_guarantee: self.precision_guarantee(),
        }
    }

    /// Launch.
    pub fn run(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: DenseGemmArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        // cuBLAS needs no caller workspace; accept-and-ignore matches
        // the FFI facade's reserved-parameter contract.
        let _ = workspace;

        let d = &self.desc;
        let layout = d.layout.ffi_tag();
        let a_ptr = args.a.data.as_raw().0 as *const c_void;
        let b_ptr = args.b.data.as_raw().0 as *const c_void;
        let d_ptr = args.d.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gemm_dense_f32_run(
                    d.m, d.n, d.k, d.batch, layout,
                    args.alpha.to_f32(), args.beta.to_f32(),
                    a_ptr, args.a.ld, args.stride_a,
                    b_ptr, args.b.ld, args.stride_b,
                    d_ptr, args.d.ld, args.stride_d,
                    core::ptr::null_mut(), 0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gemm_dense_f64_run(
                    d.m, d.n, d.k, d.batch, layout,
                    args.alpha.to_f64(), args.beta.to_f64(),
                    a_ptr, args.a.ld, args.stride_a,
                    b_ptr, args.b.ld, args.stride_b,
                    d_ptr, args.d.ld, args.stride_d,
                    core::ptr::null_mut(), 0,
                    stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gemm_dense_f16_run(
                    d.m, d.n, d.k, d.batch, layout,
                    args.alpha.to_f32(), args.beta.to_f32(),
                    a_ptr, args.a.ld, args.stride_a,
                    b_ptr, args.b.ld, args.stride_b,
                    d_ptr, args.d.ld, args.stride_d,
                    core::ptr::null_mut(), 0,
                    stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gemm_dense_bf16_run(
                    d.m, d.n, d.k, d.batch, layout,
                    args.alpha.to_f32(), args.beta.to_f32(),
                    a_ptr, args.a.ld, args.stride_a,
                    b_ptr, args.b.ld, args.stride_b,
                    d_ptr, args.d.ld, args.stride_d,
                    core::ptr::null_mut(), 0,
                    stream_ptr,
                )
            },
            // Unreachable: `select` rejected every other element kind.
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::DenseGemmPlan: unreachable dtype dispatch arm",
                ));
            }
        };

        match status {
            0 => Ok(()),
            2 => Err(Error::InvalidProblem(
                "baracuda-kernels-sys dense GEMM facade reported an invalid problem",
            )),
            n => Err(Error::CutlassInternal(n)),
        }
    }
}
