//! # baracuda-ozimmu-sys
//!
//! Raw FFI bindings + static-link wiring for baracuda's clean-fork of
//! [ozIMMU](https://github.com/enp1s0/ozIMMU) — Hiroyuki Ootomo's
//! Ozaki-scheme FP64 GEMM library. ozIMMU synthesizes a DGEMM from
//! `S²` int8 tensor-core matmuls — on consumer Ada (sm_89), where
//! there are no FP64 tensor cores, this is ~10–20× faster than native
//! DGEMM at the cost of leaving the bit-exact-DGEMM accuracy contract.
//!
//! This crate is the FFI shim — the safe wrapper lives in
//! [`baracuda-ozimmu`]. Most baracuda callers will not touch this
//! crate directly; they opt into the Ozaki backend via
//! `PlanPreference::prefer_backend = Some(BackendKind::Ozaki { slices: ... })`
//! on a [`baracuda_kernels::GemmPlan`] (gated behind the `ozimmu`
//! cargo feature on `baracuda-kernels`).
//!
//! ## What this crate exposes
//!
//! A small flat C ABI defined by `cuda/baracuda_shim.cu`. Direct
//! bindgen on the C++ header was avoided because the public API uses
//! `std::vector<std::tuple<...>>` for one of the
//! `reallocate_working_memory` overloads — bindgen-friendly only at
//! the cost of dragging the whole `std::` namespace into the
//! generated bindings. The flat C wrapper handles only the ops the
//! safe layer needs (`create` / `destroy` / `set_stream` /
//! `reallocate_bytes` / `dgemm`).
//!
//! ## Provenance
//!
//! Phase 44 (alpha.56) vendored ozIMMU upstream under
//! `vendor/ozimmu/` + a `cutf` git submodule. Phase 44b (alpha.57)
//! retired both: the algorithm is in the literature (Ootomo / Ozaki /
//! Yokota, "DGEMM on Integer Matrix Multiplication Unit", IJHPCA
//! 2024, [arXiv:2306.11975](https://arxiv.org/abs/2306.11975)), the
//! original MIT-licensed reference implementation is preserved under
//! `ATTRIBUTION.md`, and the sources now live under `cuda/` as
//! first-class baracuda code. See `ATTRIBUTION.md` at the crate
//! root for the full story.

#![no_std]
#![deny(missing_docs)]

use core::ffi::c_int;

/// Opaque handle to an ozIMMU session.
///
/// Carries a cuBLAS handle, the working-memory scratch pointer + size,
/// and a copy of the bound stream. Construct via
/// [`baracuda_ozimmu_create`] and release via [`baracuda_ozimmu_destroy`].
/// The pointer points to a `mtk::ozimmu::handle` struct under the hood;
/// the Rust type is intentionally opaque (`#[repr(C)]` on a struct with
/// a single PhantomData private field would be equivalent but adds no
/// information for the FFI surface).
#[repr(C)]
pub struct OzimmuHandle {
    _opaque: [u8; 0],
}

/// Pointer alias for the opaque handle — matches the C++ side's
/// `mtk::ozimmu::handle_t` typedef.
pub type OzimmuHandleT = *mut OzimmuHandle;

/// Integer cast of `mtk::ozimmu::compute_mode_t`. Values match the
/// upstream enum:
///
/// - `0`  — `sgemm` (host f32 path; not exposed to Rust callers)
/// - `1`  — `dgemm` (native DGEMM passthrough; reference fallback)
/// - `2..=17` — `fp64_int8_3` .. `fp64_int8_18` (S = 3 .. 18 slices)
/// - `18` — `fp64_int8_auto` (runtime selection based on the
///   handle's `auto_mantissa_loss_threshold`)
///
/// The safe wrapper in [`baracuda-ozimmu`] re-exports these as an
/// `OzakiSlices` enum.
pub type ComputeMode = c_int;

/// `mtk::ozimmu::dgemm` — reference passthrough to native cuBLAS DGEMM.
pub const COMPUTE_MODE_DGEMM: ComputeMode = 1;
/// Ozaki S=3 slice count.
pub const COMPUTE_MODE_FP64_INT8_3: ComputeMode = 2;
/// Ozaki S=4 slice count.
pub const COMPUTE_MODE_FP64_INT8_4: ComputeMode = 3;
/// Ozaki S=5 slice count.
pub const COMPUTE_MODE_FP64_INT8_5: ComputeMode = 4;
/// Ozaki S=6 slice count.
pub const COMPUTE_MODE_FP64_INT8_6: ComputeMode = 5;
/// Ozaki S=7 slice count.
pub const COMPUTE_MODE_FP64_INT8_7: ComputeMode = 6;
/// Ozaki S=8 slice count — the "comparable-to-DGEMM-accuracy"
/// sweet spot recommended by the upstream paper for general workloads.
pub const COMPUTE_MODE_FP64_INT8_8: ComputeMode = 7;
/// Ozaki S=9 slice count.
pub const COMPUTE_MODE_FP64_INT8_9: ComputeMode = 8;
/// Ozaki S=10 slice count.
pub const COMPUTE_MODE_FP64_INT8_10: ComputeMode = 9;
/// Ozaki S=11 slice count.
pub const COMPUTE_MODE_FP64_INT8_11: ComputeMode = 10;
/// Ozaki S=12 slice count.
pub const COMPUTE_MODE_FP64_INT8_12: ComputeMode = 11;
/// Ozaki S=13 slice count.
pub const COMPUTE_MODE_FP64_INT8_13: ComputeMode = 12;
/// Ozaki S=14 slice count.
pub const COMPUTE_MODE_FP64_INT8_14: ComputeMode = 13;
/// Ozaki S=15 slice count.
pub const COMPUTE_MODE_FP64_INT8_15: ComputeMode = 14;
/// Ozaki S=16 slice count.
pub const COMPUTE_MODE_FP64_INT8_16: ComputeMode = 15;
/// Ozaki S=17 slice count.
pub const COMPUTE_MODE_FP64_INT8_17: ComputeMode = 16;
/// Ozaki S=18 slice count — most accurate, most expensive.
pub const COMPUTE_MODE_FP64_INT8_18: ComputeMode = 17;
/// `mtk::ozimmu::fp64_int8_auto` — picks S at run time based on the
/// handle's `auto_mantissa_loss_threshold` (default 0).
pub const COMPUTE_MODE_FP64_INT8_AUTO: ComputeMode = 18;

/// Phase 44c variant flag — base ozIMMU (Ootomo / Ozaki / Yokota
/// 2023, IJHPCA 2024). Bit-identical to the historical
/// `baracuda_ozimmu_dgemm` path; default when callers don't opt in.
pub const OZIMMU_VARIANT_BASE: c_int = 0;
/// Phase 44c variant flag — EF (group-wise error-free summation).
/// Reduces FP64 accumulation overhead vs Base by chaining int8
/// GEMMs into the same int32 accumulator (`beta_i = 1`) and flushing
/// to f64 once per `2^(31 - 2*bits - ceil(log2(k)))`-sized group.
/// Same accuracy as Base.
pub const OZIMMU_VARIANT_EF: c_int = 1;
/// Phase 44c variant flag — RN (nearest-rounding split). Replaces
/// the truncation-style int8 extraction with `(a + t) - t`
/// round-to-nearest. ~2 extra effective bits per slice at the same
/// throughput as Base.
pub const OZIMMU_VARIANT_RN: c_int = 2;
/// Phase 44c variant flag — H (EF + RN combined). Best accuracy /
/// perf tradeoff per the upstream perf paper.
pub const OZIMMU_VARIANT_H: c_int = 3;

unsafe extern "C" {
    /// Create an ozIMMU handle. `malloc_mode_async` is `0` (sync)
    /// or `1` (async, only useful when the bound stream uses
    /// `cudaMallocAsync`-compatible allocations).
    ///
    /// Returns `0` on success, non-zero on failure (the upstream
    /// `mtk::ozimmu::create` returns 0 on success).
    pub fn baracuda_ozimmu_create(
        out_handle: *mut OzimmuHandleT,
        malloc_mode_async: c_int,
    ) -> c_int;

    /// Destroy an ozIMMU handle. Idempotent on null input is **not**
    /// guaranteed — pass only handles produced by
    /// [`baracuda_ozimmu_create`].
    ///
    /// Returns `0` on success.
    pub fn baracuda_ozimmu_destroy(handle: OzimmuHandleT) -> c_int;

    /// Bind a CUDA stream to the handle. Subsequent
    /// [`baracuda_ozimmu_dgemm`] calls enqueue on `stream`; the cuBLAS
    /// handle owned by the ozIMMU session is re-bound too.
    ///
    /// `stream` is a raw `cudaStream_t` (an opaque pointer alias).
    pub fn baracuda_ozimmu_set_cuda_stream(
        handle: OzimmuHandleT,
        stream: *mut core::ffi::c_void,
    );

    /// Re-allocate the handle's working-memory scratch to at least
    /// `size_in_bytes`. Returns the new total size (in bytes) if the
    /// allocator grew, or `0` if the existing scratch was already
    /// large enough.
    ///
    /// Callers normally don't need to call this directly — the first
    /// `dgemm` launch grows the scratch on demand. Useful when a
    /// caller knows the worst-case problem size up front and wants to
    /// pay the one-time `cudaMalloc` cost outside the hot path.
    pub fn baracuda_ozimmu_reallocate_working_memory_bytes(
        handle: OzimmuHandleT,
        size_in_bytes: usize,
    ) -> usize;

    /// Phase 44c — variant-aware FP64 GEMM via the Ozaki scheme.
    ///
    /// Identical to [`baracuda_ozimmu_dgemm`] except for the trailing
    /// `variant` parameter, which selects a perf-enhancement variant
    /// ported from RIKEN-RCCS/accelerator_for_ozIMMU:
    ///
    ///   - [`OZIMMU_VARIANT_BASE`] (= 0) — original ozIMMU (Ootomo /
    ///     Ozaki / Yokota 2023). Bit-identical to
    ///     [`baracuda_ozimmu_dgemm`] output.
    ///   - [`OZIMMU_VARIANT_EF`]   (= 1) — group-wise error-free
    ///     summation. Reduces the number of int32 → f64
    ///     materialization passes by chaining consecutive
    ///     `cublasGemmEx` calls into the same int32 accumulator. Same
    ///     accuracy as Base, ~5–15% faster on perf-paper benchmarks.
    ///   - [`OZIMMU_VARIANT_RN`]   (= 2) — nearest-rounding splitter.
    ///     Replaces the upstream truncation-style int8 extraction with
    ///     the `(a + t) - t` round-to-nearest trick (Uchino / Ozaki /
    ///     Imamura 2024, §3.2). Same speed as Base, ~2 extra effective
    ///     bits of mantissa precision per slice. Use to drop one
    ///     slice for the same accuracy.
    ///   - [`OZIMMU_VARIANT_H`]    (= 3) — EF + RN combined.
    ///
    /// n-blocking (chunk large-N int8 GEMMs into 8192-wide pieces) is
    /// applied automatically by `matmul_core` regardless of the
    /// variant flag.
    ///
    /// `compute_mode` follows the same convention as the base call;
    /// the variant is independent of the slice count.
    pub fn baracuda_ozimmu_dgemm_with_variant(
        handle: OzimmuHandleT,
        op_a: c_int,
        op_b: c_int,
        m: usize,
        n: usize,
        k: usize,
        alpha: *const f64,
        a_ptr: *const f64,
        lda: usize,
        b_ptr: *const f64,
        ldb: usize,
        beta: *const f64,
        c_ptr: *mut f64,
        ldc: usize,
        compute_mode: c_int,
        variant: c_int,
    ) -> c_int;

    /// Run an FP64 GEMM via the Ozaki scheme.
    ///
    /// Arguments mirror `cublasDgemm` but with the addition of an
    /// explicit `compute_mode` discriminant. Layout convention: cuBLAS
    /// column-major, same as `cublasDgemm`. `op_a` / `op_b` are
    /// `0` (= N) or `1` (= T).
    ///
    /// `alpha`, `beta` are host-pointer FP64 scalars (same shape as
    /// `cublasDgemm`'s `alpha` / `beta`). `a_ptr` / `b_ptr` / `c_ptr`
    /// are device-resident FP64 buffers. `lda` / `ldb` / `ldc` are
    /// leading dimensions in elements.
    ///
    /// `compute_mode` is one of the `COMPUTE_MODE_*` constants. For
    /// most callers `COMPUTE_MODE_FP64_INT8_8` is the right default;
    /// `COMPUTE_MODE_FP64_INT8_AUTO` lets ozIMMU pick S dynamically
    /// at the cost of one extra split-stat kernel per call.
    ///
    /// Returns `0` on success. Non-zero status maps to the upstream
    /// `cublasStatus_t` returned by the internal cuBLAS call chain.
    pub fn baracuda_ozimmu_dgemm(
        handle: OzimmuHandleT,
        op_a: c_int,
        op_b: c_int,
        m: usize,
        n: usize,
        k: usize,
        alpha: *const f64,
        a_ptr: *const f64,
        lda: usize,
        b_ptr: *const f64,
        ldb: usize,
        beta: *const f64,
        c_ptr: *mut f64,
        ldc: usize,
        compute_mode: c_int,
    ) -> c_int;
}
