//! Capture-safe dense m=1 GEMV plan (bespoke kernel).
//!
//! The m == 1 companion to [`DenseGemmPlan`](super::dense_gemm::DenseGemmPlan).
//! Where `DenseGemmPlan` is cuBLAS-backed (faster, batched, but NOT
//! capture-safe — cuBLAS may pick a different internal implementation
//! between launches), this plan drives a plain bespoke CUDA kernel
//! (`baracuda_kernels_gemv_dense_m1_*` in `baracuda-kernels-sys`) that
//! has no vendor internal state, so a captured CUDA graph replays
//! byte-identical by construction. It exists to unblock Fuel's
//! CUDA-graph `CapturedRun`: on the capture path only, Fuel routes the
//! decode-step matmuls (all m == 1) here; uncaptured realizes keep
//! `DenseGemmPlan` (throughput wins, capture-safety is moot).
//!
//! ## Scope
//!
//! - `m == 1` only (the decode / GEMV regime). Speculative-decode
//!   `K > 1` and prefill stay on the batched `DenseGemmPlan` path.
//! - RRR (row-major A / B / D) only — the single layout Fuel's capture
//!   envelope needs.
//! - `f32` / `f16` / `bf16`. No `f64` (the sys family has none — decode
//!   is never f64).
//! - Strided-batch with `stride_b == 0` broadcast: the GQA case (one
//!   shared B across `batch = n_heads` query heads).
//!
//! ## Numerics
//!
//! The whole dot product runs in **f32** (`f32` is true IEEE — NOT
//! TF32; `f16` / `bf16` widen to f32 on load and narrow once on store).
//! The reduction is a fixed-order serial K-loop with one thread per
//! output column — no atomics, no warp shuffle — so results are
//! bit-identical run-to-run on the same hardware and, crucially,
//! byte-identical across captured-graph replays. That determinism (not
//! throughput) is the whole point.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    BackendKind, Element, ElementKind, MathPrecision, MatrixMut, MatrixRef, PlanPreference,
    PrecisionGuarantee, ScalarType, Workspace,
};

/// RRR FFI layout tag (must match the `layout` parameter of the
/// `baracuda_kernels_gemv_dense_m1_*` symbols — RRR = 0). RRR is the
/// only layout this plan implements.
const LAYOUT_RRR: i32 = 0;

/// Problem shape handed to [`GemvDensePlan::select`].
///
/// Mirrors [`DenseGemmDescriptor`](super::dense_gemm::DenseGemmDescriptor)
/// restricted to the GEMV regime: `m` must be `1`, and the layout is
/// always RRR (not carried — see the module docs).
#[derive(Copy, Clone, Debug)]
pub struct GemvDenseDescriptor {
    /// Output row count. Must be `1` (this is the m == 1 GEMV).
    pub m: i32,
    /// Output column count (per batch slot).
    pub n: i32,
    /// Reduction depth (per batch slot).
    pub k: i32,
    /// Number of batch slots. `1` for a single GEMV; `batch = n_heads`
    /// for the GQA scores / ·v matmuls.
    pub batch: i32,
}

/// Per-launch arguments for a [`GemvDensePlan::run`] call.
///
/// Computes `D[g, 0, j] = α · Σ_kk A[g, 0, kk] · B[g, kk, j]
/// + β · D[g, 0, j]` per batch slot `g` (RRR row-major). There is no
/// separate `C` operand: `β ≠ 0` accumulates into the existing contents
/// of `D` (read-modify-write); `β == 0` is write-only (`D` may be
/// uninitialized).
///
/// `stride_*` are batch strides in **elements** (`ptr + g · stride`),
/// ignored when `desc.batch == 1`. `stride_a` / `stride_b` may be `0` to
/// broadcast one matrix across all slots (`stride_b == 0` is the GQA rhs
/// broadcast); `stride_d` must be non-zero when `batch > 1` (overlapping
/// outputs race).
///
/// Each [`MatrixRef`]'s `rows` / `cols` are the logical dims
/// (`A: [1, K]`, `B: [K, N]`, `D: [1, N]`); `ld` is the row stride.
#[derive(Debug)]
pub struct GemvDenseArgs<'a, T: Element> {
    /// Left input (row vector) — base pointer for batch slot 0.
    pub a: MatrixRef<'a, T>,
    /// Element offset between consecutive `A` slots (`0` = broadcast).
    pub stride_a: i64,
    /// Right input `[K, N]` — base pointer for batch slot 0.
    pub b: MatrixRef<'a, T>,
    /// Element offset between consecutive `B` slots (`0` = GQA broadcast).
    pub stride_b: i64,
    /// Output (and `β`-accumulation source) — base pointer for slot 0.
    pub d: MatrixMut<'a, T>,
    /// Element offset between consecutive `D` slots.
    pub stride_d: i64,
    /// Multiplier on the matrix-multiply accumulator (`f32` for
    /// f16/bf16/f32, via `T::Scalar`).
    pub alpha: T::Scalar,
    /// Multiplier on the existing contents of `D`.
    pub beta: T::Scalar,
}

/// Capture-safe dense m=1 GEMV plan (bespoke kernel). See the module
/// docs for scope and the relationship to `DenseGemmPlan`.
#[derive(Debug)]
pub struct GemvDensePlan<T: Element> {
    desc: GemvDenseDescriptor,
    _marker: PhantomData<T>,
}

impl<T: Element> GemvDensePlan<T> {
    /// Pick the GEMV kernel for `desc`.
    ///
    /// Backend is always [`BackendKind::Bespoke`]; `pref.prefer_backend`
    /// may name it (or be `None`) — any other backend request is
    /// [`Error::Unsupported`]. Restricted to `f32` / `f16` / `bf16`.
    pub fn select(
        _stream: &Stream,
        desc: &GemvDenseDescriptor,
        pref: PlanPreference,
    ) -> Result<Self> {
        if !matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16
        ) {
            return Err(Error::Unsupported(
                "baracuda-kernels::GemvDensePlan: capture-safe GEMV covers f32 / \
                 f16 / bf16 only (decode dtypes; use DenseGemmPlan for f64 or the \
                 batched m > 1 path)",
            ));
        }
        if desc.m < 0 || desc.n < 0 || desc.k < 0 || desc.batch < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GemvDensePlan: M, N, K, batch must be non-negative",
            ));
        }
        match pref.prefer_backend {
            None | Some(BackendKind::Bespoke) => {}
            Some(_) => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::GemvDensePlan: bespoke-kernel-backed only; \
                     leave PlanPreference::prefer_backend unset or set it to \
                     BackendKind::Bespoke",
                ));
            }
        }
        Ok(Self {
            desc: *desc,
            _marker: PhantomData,
        })
    }

    /// Validate that this plan can launch with `args`.
    pub fn can_implement(&self, args: &GemvDenseArgs<'_, T>) -> Result<()> {
        let d = &self.desc;
        // This family is the m == 1 GEMV specialization.
        if d.m != 1 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GemvDensePlan: M must be 1 (m > 1 belongs on \
                 the batched DenseGemmPlan path)",
            ));
        }
        if args.a.rows != d.m || args.a.cols != d.k {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GemvDensePlan: A logical shape mismatch with \
                 descriptor (M=1, K)",
            ));
        }
        if args.b.rows != d.k || args.b.cols != d.n {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GemvDensePlan: B logical shape mismatch with \
                 descriptor (K, N)",
            ));
        }
        if args.d.rows != d.m || args.d.cols != d.n {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GemvDensePlan: D shape mismatch with descriptor \
                 (M=1, N)",
            ));
        }
        // RRR leading-dim minimums (elements): A [1, K] ld ≥ K, B [K, N]
        // ld ≥ N, D [1, N] ld ≥ N. `lda` / `ldd` are honored though
        // unused at m == 1 (single A / D row). Keep in sync with the
        // .cu `gemv_validate`, which re-checks at run time.
        if args.a.ld < (d.k as i64).max(1) {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GemvDensePlan: lda below K",
            ));
        }
        if args.b.ld < (d.n as i64).max(1) {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GemvDensePlan: ldb below N",
            ));
        }
        if args.d.ld < (d.n as i64).max(1) {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GemvDensePlan: ldd below N",
            ));
        }
        if d.batch > 1 && args.stride_d == 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GemvDensePlan: stride_d must be non-zero when \
                 batch > 1 (overlapping outputs race)",
            ));
        }
        // Negative batch strides can never index in-bounds: the
        // MatrixRef base pointer IS the slice start, so any negative
        // reach walks before the allocation.
        if d.batch > 1 && (args.stride_a < 0 || args.stride_b < 0 || args.stride_d < 0) {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GemvDensePlan: negative batch strides walk before \
                 the buffer base (MatrixRef has no base offset)",
            ));
        }

        // Buffer-bounds: device-slice lengths must cover the per-slot
        // storage footprint plus the (batch-1)·stride reach — the
        // unsafe FFI layer has no length information, so this typed layer
        // is the soundness boundary (same contract as `DenseGemmPlan`).
        if d.m > 0 && d.n > 0 && d.batch > 0 {
            // Row-major `[rows, cols]` with leading dim `ld` touches
            // `(rows-1)·ld + cols`. CHECKED like `reach` below — an unchecked
            // multiply here could wrap negative on a pathological `rows·ld` and
            // silently pass the `check` below (this is the soundness boundary, so
            // every arithmetic step guards). Unreachable at decode shapes, but the
            // sibling `reach` proves the intent; keep them consistent.
            let footprint = |rows: i64, cols: i64, ld: i64| -> Result<i64> {
                if rows == 0 || cols == 0 {
                    return Ok(0);
                }
                (rows - 1)
                    .checked_mul(ld)
                    .and_then(|p| p.checked_add(cols))
                    .ok_or(Error::InvalidProblem(
                        "baracuda-kernels::GemvDensePlan: operand footprint (rows-1)*ld+cols \
                         overflows i64",
                    ))
            };
            let (m, n, k) = (d.m as i64, d.n as i64, d.k as i64);
            let a_slot = footprint(m, k, args.a.ld)?; // A [1, K]
            let b_slot = footprint(k, n, args.b.ld)?; // B [K, N]
            let d_slot = footprint(m, n, args.d.ld)?; // D [1, N]
            let reach = |slot: i64, stride: i64| -> Result<i64> {
                if d.batch == 1 || stride == 0 {
                    return Ok(slot);
                }
                (d.batch as i64 - 1)
                    .checked_mul(stride)
                    .and_then(|extra| slot.checked_add(extra))
                    .ok_or(Error::InvalidProblem(
                        "baracuda-kernels::GemvDensePlan: batch-stride reach overflows i64",
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

    /// Workspace size in bytes — always `0` (no split-K / staging at
    /// m == 1).
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// The library backend this plan launches through — always
    /// [`BackendKind::Bespoke`].
    #[inline]
    pub fn backend(&self) -> BackendKind {
        BackendKind::Bespoke
    }

    /// Numerical guarantees for this plan's launches.
    ///
    /// Unlike [`DenseGemmPlan`](super::dense_gemm::DenseGemmPlan) (which
    /// reports `false` — cuBLAS's internal-implementation choice is
    /// stream-scoped), this kernel's fixed-order serial reduction with
    /// one thread per output column is bit-identical run-to-run on the
    /// same hardware AND byte-identical across captured-graph replays,
    /// so both flags are `true`. The multiply-add runs in **f32** for
    /// every SKU (CUDA cores, no tensor cores) — f16 / bf16 inputs are
    /// widened to f32 — so `math_precision` is `F32` throughout, which
    /// is the honest description (and higher accuracy than the
    /// tensor-core path).
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        }
    }

    /// Launch.
    pub fn run(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: GemvDenseArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        // No caller workspace at m == 1; accept-and-ignore matches the
        // FFI's reserved-parameter contract.
        let _ = workspace;

        let d = &self.desc;
        let a_ptr = args.a.data.as_raw().0 as *const c_void;
        let b_ptr = args.b.data.as_raw().0 as *const c_void;
        let d_ptr = args.d.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw();
        let alpha = args.alpha.to_f32();
        let beta = args.beta.to_f32();

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gemv_dense_m1_f32_run(
                    d.m,
                    d.n,
                    d.k,
                    d.batch,
                    LAYOUT_RRR,
                    alpha,
                    beta,
                    a_ptr,
                    args.a.ld,
                    args.stride_a,
                    b_ptr,
                    args.b.ld,
                    args.stride_b,
                    d_ptr,
                    args.d.ld,
                    args.stride_d,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gemv_dense_m1_f16_run(
                    d.m,
                    d.n,
                    d.k,
                    d.batch,
                    LAYOUT_RRR,
                    alpha,
                    beta,
                    a_ptr,
                    args.a.ld,
                    args.stride_a,
                    b_ptr,
                    args.b.ld,
                    args.stride_b,
                    d_ptr,
                    args.d.ld,
                    args.stride_d,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gemv_dense_m1_bf16_run(
                    d.m,
                    d.n,
                    d.k,
                    d.batch,
                    LAYOUT_RRR,
                    alpha,
                    beta,
                    a_ptr,
                    args.a.ld,
                    args.stride_a,
                    b_ptr,
                    args.b.ld,
                    args.stride_b,
                    d_ptr,
                    args.d.ld,
                    args.stride_d,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            // Unreachable: `select` rejected every other element kind.
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::GemvDensePlan: unreachable dtype dispatch arm",
                ));
            }
        };

        match status {
            0 => Ok(()),
            2 => Err(Error::InvalidProblem(
                "baracuda-kernels-sys dense GEMV kernel reported an invalid problem",
            )),
            n => Err(Error::CutlassInternal(n)),
        }
    }
}
